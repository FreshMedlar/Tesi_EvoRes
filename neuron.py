"""
neuron.py
---------
Defines the *evolvable base neuron gene*.

A NeuronGene is a compact parameter vector that encodes:
  1. A **weight-plasticity rule** — how synaptic strengths are updated
     online (Hebbian / anti-Hebbian / BCM-like, parameterised).
  2. A **rewiring rule** — how dormant/weak synapses are pruned and new
     ones are formed, controlled by activity thresholds and probability
     coefficients.

One NeuronGene is used to build an entire reservoir: every reservoir
neuron shares the *same* plasticity / rewiring coefficients (the gene),
but has its *own* private state (membrane potential, firing rate, etc.).

Multiple NeuronGene variants form the *evolution population*.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Gene encoding
# ---------------------------------------------------------------------------

@dataclass
class NeuronGeneConfig:
    """Hyper-parameters that are *fixed* (not evolved)."""
    n_plasticity_terms: int = 5   # number of Hebbian-style correlation terms
    n_rewire_params:    int = 4   # params controlling rewiring probability


class NeuronGene(nn.Module):
    """
    Evolvable gene that parameterises the plasticity & rewiring rules of
    every neuron inside a reservoir.

    Parameters (all evolved by the outer evolutionary loop):
    --------------------------------------------------------
    plasticity_coeffs  [n_plasticity_terms]
        Coefficients (η_k) in the generalised Hebbian update:
            Δw_ij = Σ_k  η_k · ϕ_k(pre_i, post_j, w_ij)
        where ϕ_k are fixed basis functions (see PlasticityBasis).

    tau_m  (scalar, log-space)
        Membrane time constant (milliseconds, conceptually).

    activation_gain  (scalar)
        Scales the tanh nonlinearity.

    rewire_prune_thresh  (scalar)
        Absolute weight threshold below which a synapse is a pruning
        candidate.

    rewire_growth_prob  (scalar, sigmoid-space)
        Base probability of spontaneously growing a new synapse per step.

    rewire_activity_bias  (scalar)
        How much post-synaptic activity increases growth probability.

    rewire_weight_init_std  (scalar, softplus-space)
        Standard deviation of newly grown synapse weights.
    """

    def __init__(self, cfg: NeuronGeneConfig = NeuronGeneConfig()):
        super().__init__()
        self.cfg = cfg

        # ---- Plasticity rule parameters --------------------------------
        # η_k coefficients for the Hebbian basis
        self.plasticity_coeffs = nn.Parameter(
            torch.zeros(cfg.n_plasticity_terms)
        )

        # Membrane leakage (unconstrained; softplus applied on use)
        self._log_tau_m = nn.Parameter(torch.tensor(math.log(10.0)))

        # Gain of activation function
        self._activation_gain_raw = nn.Parameter(torch.tensor(0.0))  # → softplus

        # ---- Rewiring rule parameters ----------------------------------
        # All stored in raw (unconstrained) form; transformed on use.
        self._rewire_prune_thresh_raw   = nn.Parameter(torch.tensor(-1.0))  # → softplus
        self._rewire_growth_prob_raw    = nn.Parameter(torch.tensor(-3.0))  # → sigmoid
        self._rewire_activity_bias_raw  = nn.Parameter(torch.tensor(0.0))   # raw
        self._rewire_weight_init_std_raw= nn.Parameter(torch.tensor(-1.0))  # → softplus

    # ------------------------------------------------------------------ #
    # Derived (constrained) properties
    # ------------------------------------------------------------------ #

    @property
    def tau_m(self) -> torch.Tensor:
        """Membrane time constant > 0."""
        return torch.exp(self._log_tau_m).clamp(min=1.0, max=1000.0)

    @property
    def activation_gain(self) -> torch.Tensor:
        """Activation gain > 0."""
        return torch.nn.functional.softplus(self._activation_gain_raw).clamp(min=1e-3)

    @property
    def rewire_prune_thresh(self) -> torch.Tensor:
        """Weight magnitude below which a synapse can be pruned (> 0)."""
        return torch.nn.functional.softplus(self._rewire_prune_thresh_raw).clamp(min=1e-6)

    @property
    def rewire_growth_prob(self) -> torch.Tensor:
        """Probability in (0, 1) of growing a new synapse per timestep."""
        return torch.sigmoid(self._rewire_growth_prob_raw)

    @property
    def rewire_activity_bias(self) -> torch.Tensor:
        """Modulates how post-synaptic activity scales growth probability."""
        return self._rewire_activity_bias_raw  # unconstrained

    @property
    def rewire_weight_init_std(self) -> torch.Tensor:
        """Std-dev of newly initialised synapse weights (> 0)."""
        return torch.nn.functional.softplus(self._rewire_weight_init_std_raw).clamp(min=1e-6)

    # ------------------------------------------------------------------ #
    # Parameter vector  (for evolutionary operators)
    # ------------------------------------------------------------------ #

    def to_vector(self) -> torch.Tensor:
        """Flatten all raw parameters into a single 1-D tensor."""
        parts = [p.data.view(-1) for p in self.parameters()]
        return torch.cat(parts)

    @torch.no_grad()
    def from_vector_(self, vec: torch.Tensor) -> None:
        """Load raw parameters from a flat vector (in-place)."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(vec[offset:offset + numel].view_as(p))
            offset += numel

    @staticmethod
    def gene_dim(cfg: NeuronGeneConfig = NeuronGeneConfig()) -> int:
        """Total number of evolvable scalars."""
        return (
            cfg.n_plasticity_terms
            + 1   # _log_tau_m
            + 1   # _activation_gain_raw
            + 1   # _rewire_prune_thresh_raw
            + 1   # _rewire_growth_prob_raw
            + 1   # _rewire_activity_bias_raw
            + 1   # _rewire_weight_init_std_raw
        )

    def clone(self) -> "NeuronGene":
        """Return a deep copy with the same parameter values."""
        child = NeuronGene(self.cfg)
        child.from_vector_(self.to_vector().clone())
        return child

    def mutate_(self, sigma: float = 0.05) -> None:
        """Gaussian mutation in-place."""
        vec = self.to_vector()
        vec.add_(torch.randn_like(vec) * sigma)
        self.from_vector_(vec)

    def __repr__(self) -> str:
        return (
            f"NeuronGene("
            f"τ_m={self.tau_m.item():.2f}, "
            f"gain={self.activation_gain.item():.3f}, "
            f"prune_thr={self.rewire_prune_thresh.item():.4f}, "
            f"grow_p={self.rewire_growth_prob.item():.4f})"
        )


# ---------------------------------------------------------------------------
# Plasticity basis functions
# ---------------------------------------------------------------------------

class PlasticityBasis:
    """
    A collection of fixed, local Hebbian-style correlation terms.

    For a synapse w_ij connecting pre-neuron i (rate x_i) to
    post-neuron j (rate x_j), the basis evaluates:

        ϕ_0 = x_pre · x_post         (pure Hebb)
        ϕ_1 = x_pre · (1 - x_post)  (anti-Hebb post)
        ϕ_2 = (1 - x_pre) · x_post  (anti-Hebb pre)
        ϕ_3 = -w                      (weight decay)
        ϕ_4 = x_post² · w            (BCM-like sliding threshold)

    The update rule is:  Δw = lr_plasticity · Σ_k η_k · ϕ_k

    All computations are batched over [pop, batch, neurons].
    """

    N_TERMS = 5  # must match NeuronGeneConfig.n_plasticity_terms

    @staticmethod
    def evaluate(
        x_pre:  torch.Tensor,   # [pop, batch, n_pre]
        x_post: torch.Tensor,   # [pop, batch, n_post]
        W:      torch.Tensor,   # [pop, n_post, n_pre]
    ) -> torch.Tensor:
        """
        Returns basis values of shape [pop, batch, n_post, n_pre, N_TERMS].
        Memory-intensive for large reservoirs; consider chunked evaluation.
        """
        # x_pre:  [P, B, Npre]  → broadcast to [P, B, Npost, Npre]
        # x_post: [P, B, Npost] → broadcast to [P, B, Npost, Npre]
        # W:      [P, Npost, Npre] → [P, 1, Npost, Npre]

        xpre  = x_pre.unsqueeze(2)   # [P, B, 1,     Npre]
        xpost = x_post.unsqueeze(3)  # [P, B, Npost, 1   ]
        w_exp = W.unsqueeze(1)       # [P, 1, Npost, Npre]

        phi0 = xpre * xpost                        # Hebb
        phi1 = xpre * (1.0 - xpost)               # anti-Hebb post
        phi2 = (1.0 - xpre) * xpost               # anti-Hebb pre
        phi3 = -w_exp.expand_as(phi0)             # weight decay
        phi4 = (xpost ** 2) * w_exp.expand_as(phi0)  # BCM-like

        return torch.stack([phi0, phi1, phi2, phi3, phi4], dim=-1)
