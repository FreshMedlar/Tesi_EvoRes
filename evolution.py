"""
evolution.py
------------
Evolutionary algorithm that optimises the NeuronGene population.

Strategy:  (μ + λ) truncation-selection evolution with:
    1. Fitness sharing         — penalise clustering in gene space
    2. Adaptive σ              — grow/shrink mutation noise to keep diversity
                                 in a target band
    3. Stagnation injection    — replace worst individuals with fresh random
                                 genes when best fitness doesn't improve for
                                 `stagnation_patience` generations

Each generation:
    1.  Build EvoReservoirNetwork from current gene population.
    2.  Train readout (ridge regression) on a training batch.
    3.  Evaluate raw fitness on a held-out validation batch.
    4.  Apply fitness sharing to the raw scores.
    5.  Select top μ genes by shared fitness.
    6.  Adapt σ based on current population diversity.
    7.  Produce λ offspring via Gaussian mutation with current σ.
    8.  Inject random individuals if stagnation is detected.
"""

from __future__ import annotations
import copy
import math
import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass, field

from neuron import NeuronGene, NeuronGeneConfig
from network import EvoReservoirNetwork


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvoConfig:
    # Population
    population_size:  int   = 16    # total P (must be even for antithetic)
    n_elites:         int   = 4     # μ — kept unchanged each generation

    # Mutation noise
    mutation_sigma:       float = 0.10  # σ — initial, then adapted
    sigma_min:            float = 0.01  # floor for adaptive σ
    sigma_max:            float = 0.50  # ceiling for adaptive σ
    sigma_adapt_rate:     float = 0.10  # how fast σ tracks target diversity
    diversity_floor:      float = 0.15  # below → increase σ
    diversity_ceiling:    float = 0.80  # above → decrease σ

    # Fitness sharing
    sharing_radius:       float = 0.40  # σ_share — cutoff distance in gene space
    sharing_alpha:        float = 1.0   # shape exponent (1 = linear niche)
    sharing_enabled:      bool  = True

    # Stagnation injection
    stagnation_patience:  int   = 25    # gens without improvement before inject
    injection_fraction:   float = 0.25  # fraction of population to replace

    # Reservoir
    reservoir_size:   int   = 256
    density:          float = 0.1
    spectral_radius:  float = 0.9
    dt:               float = 1.0
    lr_plasticity:    float = 1e-3
    rewire_every:     int   = 10
    ridge_alpha:      float = 1e-3

    # Training
    n_generations:    int   = 200
    eval_every:       int   = 10
    device:           str   = "cpu"

    # Gene
    gene_cfg: NeuronGeneConfig = field(default_factory=NeuronGeneConfig)


# ---------------------------------------------------------------------------
# Diversity helpers
# ---------------------------------------------------------------------------

def _gene_vectors(population: List[NeuronGene]) -> torch.Tensor:
    """Stack gene parameter vectors into [P, D]."""
    return torch.stack([g.to_vector() for g in population])


def _pairwise_distances(vecs: torch.Tensor) -> torch.Tensor:
    """L2 pairwise distances [P, P]."""
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    sq = (vecs ** 2).sum(dim=1, keepdim=True)          # [P, 1]
    dist2 = sq + sq.T - 2.0 * (vecs @ vecs.T)         # [P, P]
    dist2 = dist2.clamp(min=0.0)
    return dist2.sqrt()


def _mean_diversity(vecs: torch.Tensor) -> float:
    """Mean pairwise L2 distance across the population."""
    P = vecs.shape[0]
    if P < 2:
        return 0.0
    dists = _pairwise_distances(vecs)
    # Upper triangle only (exclude diagonal)
    mask = torch.ones(P, P, dtype=torch.bool).triu(diagonal=1)
    return dists[mask].mean().item()


def _fitness_sharing(
    fitnesses: torch.Tensor,      # [P]
    vecs:      torch.Tensor,      # [P, D]
    radius:    float,
    alpha:     float,
) -> torch.Tensor:
    """
    Apply fitness sharing to raw fitnesses.

    Shared fitness = raw_fitness / niche_count
    niche_count_i  = Σ_j  sh(d_ij)
    sh(d)          = 1 - (d / radius)^alpha   if d < radius, else 0
    """
    dists = _pairwise_distances(vecs)          # [P, P]
    ratio = (dists / radius).clamp(max=1.0)
    sh    = (1.0 - ratio ** alpha).clamp(min=0.0)  # [P, P] — includes diagonal (=1)
    niche = sh.sum(dim=1)                      # [P]
    return fitnesses / niche.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Evolutionary engine
# ---------------------------------------------------------------------------

class EvolutionEngine:
    """
    (μ + λ) evolutionary loop with fitness sharing, adaptive σ,
    and stagnation injection.

    Parameters
    ----------
    cfg             :  EvoConfig
    input_size      :  dimensionality of input at each timestep
    output_size     :  number of output classes
    train_loader    :  callable() → (x_seq [T,B,I], targets [T,B])
    val_loader      :  callable() → (x_seq [T,B,I], targets [T,B])
    log_fn          :  optional callback(gen, stats_dict, best_gene)
    """

    def __init__(
        self,
        cfg:          EvoConfig,
        input_size:   int,
        output_size:  int,
        train_loader: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        val_loader:   Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        log_fn:       Optional[Callable] = None,
    ):
        self.cfg          = cfg
        self.input_size   = input_size
        self.output_size  = output_size
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.log_fn       = log_fn or (lambda *a, **kw: None)
        self.device       = torch.device(cfg.device)

        # Initialise gene population (on CPU for gene arithmetic)
        self.population: List[NeuronGene] = [
            NeuronGene(cfg.gene_cfg) for _ in range(cfg.population_size)
        ]

        # Adaptive σ (mutable)
        self.sigma = cfg.mutation_sigma

        # Tracking
        self.best_gene:     Optional[NeuronGene] = None
        self.best_fitness:  float = -math.inf
        self._stagnation:   int   = 0       # gens since last improvement
        self.history:       List[dict] = []

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> NeuronGene:
        """Run evolution and return the best gene found."""
        cfg = self.cfg

        for gen in range(cfg.n_generations):
            # ---- Build network ----
            genes = [g.to(self.device) for g in self.population]
            net   = EvoReservoirNetwork(
                genes           = genes,
                input_size      = self.input_size,
                output_size     = self.output_size,
                reservoir_size  = cfg.reservoir_size,
                density         = cfg.density,
                dt              = cfg.dt,
                lr_plasticity   = cfg.lr_plasticity,
                rewire_every    = cfg.rewire_every,
                spectral_radius = cfg.spectral_radius,
                ridge_alpha     = cfg.ridge_alpha,
                device          = cfg.device,
            ).to(self.device)

            # ---- Train readout ----
            x_train, y_train = self.train_loader()
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)

            net.reset(x_train.shape[1])
            with torch.no_grad():
                net.fit_readout(x_train, y_train)

            # ---- Evaluate raw fitness ----
            x_val, y_val = self.val_loader()
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)

            net.reset(x_val.shape[1])
            with torch.no_grad():
                raw_fit = net.fitness(x_val, y_val)   # [P]

            raw_fit_cpu = raw_fit.cpu()

            # ---- Gene vectors for diversity ops ----
            vecs = _gene_vectors(self.population)      # [P, D]

            # ---- Fitness sharing ----
            if cfg.sharing_enabled:
                shared_fit = _fitness_sharing(
                    raw_fit_cpu, vecs, cfg.sharing_radius, cfg.sharing_alpha
                )
            else:
                shared_fit = raw_fit_cpu

            # ---- Population diversity ----
            diversity = _mean_diversity(vecs)

            # ---- Select elites by SHARED fitness ----
            ranked = torch.argsort(shared_fit, descending=True).tolist()
            elites = [self.population[i].clone() for i in ranked[:cfg.n_elites]]

            # ---- Track global best by RAW fitness ----
            raw_list   = raw_fit_cpu.tolist()
            gen_best   = max(raw_list)
            gen_mean   = sum(raw_list) / len(raw_list)
            best_idx   = ranked[0]  # by shared; raw best might differ slightly

            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                # Find the individual with highest raw fitness
                raw_best_idx = int(torch.argmax(raw_fit_cpu).item())
                self.best_gene = self.population[raw_best_idx].clone()
                self._stagnation = 0
            else:
                self._stagnation += 1

            # ---- Adapt σ ----
            self.sigma = self._adapt_sigma(diversity)

            # ---- Produce offspring ----
            new_population: List[NeuronGene] = list(elites)
            n_offspring = cfg.population_size - cfg.n_elites
            for k in range(n_offspring):
                parent = elites[k % cfg.n_elites].clone()
                parent.mutate_(sigma=self.sigma)
                new_population.append(parent)

            # ---- Stagnation injection ----
            new_population = self._inject_if_stagnant(new_population)

            self.population = new_population

            # ---- Logging ----
            record = {
                "generation":   gen,
                "best_fitness": gen_best,
                "mean_fitness": gen_mean,
                "diversity":    diversity,
                "sigma":        self.sigma,
                "stagnation":   self._stagnation,
            }
            self.history.append(record)

            if gen % cfg.eval_every == 0:
                print(
                    f"Gen {gen:4d} | "
                    f"best={gen_best:.4f} | "
                    f"mean={gen_mean:.4f} | "
                    f"div={diversity:.3f} | "
                    f"σ={self.sigma:.4f} | "
                    f"stag={self._stagnation:3d} | "
                    f"{elites[0]}"
                )
                self.log_fn(gen, record, elites[0])

        print(f"\nEvolution complete.  Best fitness: {self.best_fitness:.4f}")
        print(f"Best gene: {self.best_gene}")
        return self.best_gene

    # ------------------------------------------------------------------ #
    # Adaptive σ
    # ------------------------------------------------------------------ #

    def _adapt_sigma(self, diversity: float) -> float:
        """
        Adjust σ so that population diversity stays in
        [diversity_floor, diversity_ceiling].

        Below floor  → increase σ (need more exploration).
        Above ceiling → decrease σ (population is spread, exploit more).
        In band      → nudge σ toward current value (soft update).
        """
        cfg = self.cfg
        sigma = self.sigma

        if diversity < cfg.diversity_floor:
            # Scale up: move sigma toward sigma_max
            target = cfg.sigma_max
        elif diversity > cfg.diversity_ceiling:
            # Scale down: move sigma toward sigma_min
            target = cfg.sigma_min
        else:
            # In band — keep current sigma
            return sigma

        # Exponential moving average toward target
        sigma = sigma + cfg.sigma_adapt_rate * (target - sigma)
        return float(max(cfg.sigma_min, min(cfg.sigma_max, sigma)))

    # ------------------------------------------------------------------ #
    # Stagnation injection
    # ------------------------------------------------------------------ #

    def _inject_if_stagnant(
        self,
        population: List[NeuronGene],
    ) -> List[NeuronGene]:
        """
        If stagnation exceeds patience, replace the bottom
        injection_fraction individuals with fresh random genes.
        Elites are never replaced (they sit at the front of the list).
        """
        cfg = self.cfg
        if self._stagnation < cfg.stagnation_patience:
            return population

        n_inject = max(1, int(cfg.population_size * cfg.injection_fraction))
        # Replace the last n_inject slots (worst by shared fitness, post-elites)
        for i in range(n_inject):
            idx = len(population) - 1 - i
            if idx >= cfg.n_elites:              # never overwrite elites
                population[idx] = NeuronGene(cfg.gene_cfg)

        # Reset stagnation counter so we don't inject every generation
        self._stagnation = 0

        print(f"  ↺  Injected {n_inject} fresh random genes.")
        return population

    # ------------------------------------------------------------------ #
    # Antithetic (mirrored) evaluation helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def antithetic_mutate(
        gene:  NeuronGene,
        sigma: float,
    ) -> Tuple[NeuronGene, NeuronGene]:
        """
        Return a (+ε, −ε) pair of gene variants for antithetic sampling.
        The noise vector ε is shared, reducing estimation variance.
        """
        vec   = gene.to_vector()
        noise = torch.randn_like(vec) * sigma

        pos_gene = gene.clone(); pos_gene.from_vector_(vec + noise)
        neg_gene = gene.clone(); neg_gene.from_vector_(vec - noise)
        return pos_gene, neg_gene
