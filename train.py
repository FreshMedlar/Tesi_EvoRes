"""
train.py
--------
Entry-point for training the EvoReservoir system on a character-level
language modelling task (TinyShakespeare by default, but any text works).

Architecture recap
~~~~~~~~~~~~~~~~~~
  Char tokens (one-hot)  →  Input layer (W_in)
         ↓
  Reservoir  [N neurons, plastic & rewiring]
         ↓
  Readout W_out  (trained by ridge regression each generation)
         ↓
  Char logits

Evolutionary loop
~~~~~~~~~~~~~~~~~
  P NeuronGene variants are evaluated in parallel each generation.
  The gene controls the reservoir's plasticity & rewiring rules.
  Fitness = negative cross-entropy on a held-out validation batch.
"""

import os
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from neuron import NeuronGene, NeuronGeneConfig
from evolution import EvoConfig, EvolutionEngine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_path:  str = "tinyshakespeare.txt"
    seq_len:    int = 64
    batch_size: int = 32

    # Evolution (forwarded to EvoConfig)
    population_size:  int   = 16
    n_elites:         int   = 4
    mutation_sigma:   float = 0.10   # higher start; adaptive σ will tune it
    n_generations:    int   = 300
    eval_every:       int   = 10

    # Diversity / adaptive σ
    sigma_min:            float = 0.01
    sigma_max:            float = 0.50
    sigma_adapt_rate:     float = 0.10
    diversity_floor:      float = 0.15
    diversity_ceiling:    float = 0.80
    sharing_radius:       float = 0.40
    sharing_alpha:        float = 1.0
    sharing_enabled:      bool  = True

    # Stagnation injection
    stagnation_patience:  int   = 25
    injection_fraction:   float = 0.25

    # Reservoir
    reservoir_size:  int   = 512
    density:         float = 0.1
    spectral_radius: float = 0.9
    dt:              float = 1.0
    lr_plasticity:   float = 5e-4
    rewire_every:    int   = 16
    ridge_alpha:     float = 1e-3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_text(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars    = sorted(set(text))
    vocab    = len(chars)
    stoi     = {c: i for i, c in enumerate(chars)}
    itos     = {i: c for i, c in enumerate(chars)}
    data     = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab, stoi, itos


def make_loader(data: torch.Tensor, seq_len: int, batch_size: int, vocab_size: int, device: str):
    """
    Returns a callable that, when called, yields a fresh random batch.

    Returns
    -------
    x_seq   : [T, B, vocab_size]  one-hot encoded input
    targets : [T, B]              integer token indices (next-char prediction)
    """
    def loader():
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        x_idx  = torch.stack([data[i:i+seq_len]   for i in ix])   # [B, T]
        y_idx  = torch.stack([data[i+1:i+seq_len+1] for i in ix]) # [B, T]

        # One-hot: [B, T, V] → permute → [T, B, V]
        x_oh = F.one_hot(x_idx, vocab_size).float().permute(1, 0, 2)
        y    = y_idx.permute(1, 0)   # [T, B]
        return x_oh.to(device), y.to(device)

    return loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = TrainConfig()

    print(f"Device : {cfg.device}")
    print(f"Loading data from '{cfg.data_path}' ...")
    data, vocab_size, stoi, itos = load_text(cfg.data_path)

    n_train   = int(0.9 * len(data))
    train_raw = data[:n_train]
    val_raw   = data[n_train:]

    print(f"Vocab size : {vocab_size}")
    print(f"Train tokens : {len(train_raw):,}  |  Val tokens : {len(val_raw):,}")
    print(f"Reservoir size : {cfg.reservoir_size}  |  Density : {cfg.density}")
    print(f"Population : {cfg.population_size}  |  Elites : {cfg.n_elites}")
    print()

    train_loader = make_loader(train_raw, cfg.seq_len, cfg.batch_size, vocab_size, cfg.device)
    val_loader   = make_loader(val_raw,   cfg.seq_len, cfg.batch_size, vocab_size, cfg.device)

    # Build evo config
    evo_cfg = EvoConfig(
        population_size      = cfg.population_size,
        n_elites             = cfg.n_elites,
        mutation_sigma       = cfg.mutation_sigma,
        sigma_min            = cfg.sigma_min,
        sigma_max            = cfg.sigma_max,
        sigma_adapt_rate     = cfg.sigma_adapt_rate,
        diversity_floor      = cfg.diversity_floor,
        diversity_ceiling    = cfg.diversity_ceiling,
        sharing_radius       = cfg.sharing_radius,
        sharing_alpha        = cfg.sharing_alpha,
        sharing_enabled      = cfg.sharing_enabled,
        stagnation_patience  = cfg.stagnation_patience,
        injection_fraction   = cfg.injection_fraction,
        reservoir_size       = cfg.reservoir_size,
        density              = cfg.density,
        spectral_radius      = cfg.spectral_radius,
        dt                   = cfg.dt,
        lr_plasticity        = cfg.lr_plasticity,
        rewire_every         = cfg.rewire_every,
        ridge_alpha          = cfg.ridge_alpha,
        n_generations        = cfg.n_generations,
        eval_every           = cfg.eval_every,
        device               = cfg.device,
        gene_cfg             = NeuronGeneConfig(),
    )

    # Log callback (extend with wandb/tensorboard as needed)
    def log_fn(gen, stats, gene):
        pass   # plug in wandb.log(stats) here if desired

    engine = EvolutionEngine(
        cfg          = evo_cfg,
        input_size   = vocab_size,
        output_size  = vocab_size,
        train_loader = train_loader,
        val_loader   = val_loader,
        log_fn       = log_fn,
    )

    best_gene = engine.run()

    # Save best gene
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "gene_vector":   best_gene.to_vector(),
        "best_fitness":  engine.best_fitness,
        "history":       engine.history,
        "vocab_size":    vocab_size,
        "stoi":          stoi,
        "itos":          itos,
    }
    torch.save(ckpt, "checkpoints/best_gene.pt")
    print("Saved best gene to checkpoints/best_gene.pt")


if __name__ == "__main__":
    main()
