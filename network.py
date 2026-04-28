"""
network.py
----------
Full reservoir computing network:
    Input  →  Reservoir  →  Readout (trainable)

The **readout** is a *linear* layer trained by ridge regression (or
online gradient descent), fitting the reservoir states to the target
outputs.  The reservoir weights themselves are NOT trained by gradient
descent — they evolve via the outer evolutionary loop.

Population-parallel evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All P gene variants (population members) are evaluated in the same
forward pass.  The readout is therefore also P-parallel:
    W_out : [P, output_size, N]

After the reservoir forward pass, the output is:
    y_hat = h_all @ W_out^T    →   [T, P, B, output_size]

Readout training (ridge regression on reservoir states)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Given collected states  H  [T*B, N] and targets  Y  [T*B, output_size],
the closed-form ridge solution is:
    W_out = Y^T H (H^T H + λI)^{-1}

This is done independently for each population member.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from neuron import NeuronGene, NeuronGeneConfig
from reservoir import Reservoir


# ---------------------------------------------------------------------------
# Readout (trained layer)
# ---------------------------------------------------------------------------

class PopulationReadout(nn.Module):
    """
    A population-parallel linear readout.

    Parameters
    ----------
    population_size : int   (P)
    reservoir_size  : int   (N)
    output_size     : int   (O)
    ridge_alpha     : float  regularisation for ridge regression
    """

    def __init__(
        self,
        population_size: int,
        reservoir_size:  int,
        output_size:     int,
        ridge_alpha:     float = 1e-3,
    ):
        super().__init__()
        self.P           = population_size
        self.N           = reservoir_size
        self.O           = output_size
        self.ridge_alpha = ridge_alpha

        # Readout weights [P, O, N]
        self.register_buffer(
            "W_out",
            torch.zeros(population_size, output_size, reservoir_size),
        )
        self._trained = False

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : [T, P, B, N]

        Returns
        -------
        logits : [T, P, B, O]
        """
        # states: [T, P, B, N] → einsum → [T, P, B, O]
        return torch.einsum("tpbn,pon->tpbo", states, self.W_out)

    @torch.no_grad()
    def fit(
        self,
        states:  torch.Tensor,   # [T, P, B, N]
        targets: torch.Tensor,   # [T, B]  (integer class labels)
        n_classes: int,
    ) -> None:
        """
        Fit the readout for each population member by ridge regression.

        States from the first washout_frac of the sequence are discarded
        to let the reservoir settle.
        """
        T, P, B, N = states.shape

        # One-hot encode targets: [T, B, O]
        targets_oh = F.one_hot(targets, n_classes).float()   # [T, B, O]

        # Flatten time × batch → samples
        H = states.permute(1, 0, 2, 3).reshape(P, T * B, N)   # [P, T*B, N]
        Y = targets_oh.reshape(T * B, n_classes)               # [T*B, O]

        # Ridge: W = (H^T H + λI)^{-1} H^T Y   (per population member)
        alpha = self.ridge_alpha
        I_N   = torch.eye(N, device=states.device)

        for p in range(P):
            Hp  = H[p]                               # [T*B, N]
            HtH = Hp.T @ Hp + alpha * I_N           # [N, N]
            HtY = Hp.T @ Y                           # [N, O]
            try:
                W_p = torch.linalg.solve(HtH, HtY).T  # [O, N]
            except Exception:
                # Fallback: pseudo-inverse
                W_p = (torch.linalg.pinv(HtH) @ HtY).T
            self.W_out[p] = W_p

        self._trained = True


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class EvoReservoirNetwork(nn.Module):
    """
    Complete reservoir network:
        InputLayer  →  Reservoir  →  Readout

    Designed to be run with a population of NeuronGene variants.

    Usage
    -----
    1.  Initialise with a list of NeuronGene objects (one per pop member).
    2.  Call ``reset()`` before each new episode / batch.
    3.  Call ``forward(x_seq)`` to get predictions.
    4.  Call ``fit_readout(states, targets)`` to train the readout.
    5.  Fitness is computed from the readout predictions.

    The reservoir (W_rec, W_in) is *not* shared across population members;
    each member has its own independently-initialised weights that evolve
    via the gene's plasticity / rewiring rules.
    """

    def __init__(
        self,
        genes:           list[NeuronGene],   # length P
        input_size:      int,
        output_size:     int,
        reservoir_size:  int  = 512,
        density:         float = 0.1,
        dt:              float = 1.0,
        lr_plasticity:   float = 1e-3,
        rewire_every:    int   = 10,
        spectral_radius: float = 0.9,
        ridge_alpha:     float = 1e-3,
        device:          str   = "cpu",
    ):
        super().__init__()

        self.P            = len(genes)
        self.I            = input_size
        self.O            = output_size
        self.N            = reservoir_size
        self.device       = device

        # We use the *first* gene to build the reservoir structure.
        # In the evolutionary loop the caller builds one Reservoir per
        # gene variant, but here we keep a single Reservoir object whose
        # gene can be swapped.  For true population-parallelism the
        # Reservoir itself holds P-parallel weight tensors.
        #
        # To handle P *different* genes we keep a list of Reservoir
        # objects — one per population member — and stack their outputs.
        # This avoids the complexity of a P-vectorised gene lookup while
        # keeping the reservoir dynamics vectorised over (B, N).
        #
        # For very large P, a fully-fused approach is possible but
        # significantly more complex.

        self.reservoirs = nn.ModuleList([
            Reservoir(
                gene=gene,
                population_size=1,       # each reservoir handles 1 gene
                reservoir_size=reservoir_size,
                input_size=input_size,
                density=density,
                dt=dt,
                lr_plasticity=lr_plasticity,
                rewire_every=rewire_every,
                spectral_radius=spectral_radius,
                device=device,
            )
            for gene in genes
        ])

        self.readout = PopulationReadout(
            population_size=self.P,
            reservoir_size=reservoir_size,
            output_size=output_size,
            ridge_alpha=ridge_alpha,
        )

    def reset(self, batch_size: int) -> None:
        """Reset hidden states of all population reservoirs."""
        for res in self.reservoirs:
            res.reset_state(batch_size)

    def forward(
        self,
        x_seq: torch.Tensor,            # [T, B, I]
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run all population reservoirs over the input sequence.

        Returns
        -------
        logits : [T, P, B, O]
        states : [T, P, B, N]   (only if return_states=True)
        """
        T, B, _ = x_seq.shape
        all_states = []

        # Run each reservoir independently, then stack
        for res in self.reservoirs:
            # states_p: [T, 1, B, N]
            states_p = res.forward(x_seq)   # [T, 1, B, N]
            all_states.append(states_p)

        # [T, P, B, N]
        states = torch.cat(all_states, dim=1)

        logits = self.readout(states)   # [T, P, B, O]

        if return_states:
            return logits, states
        return logits, None

    def fit_readout(
        self,
        x_seq:    torch.Tensor,   # [T, B, I]
        targets:  torch.Tensor,   # [T, B]  integer labels
    ) -> torch.Tensor:
        """
        Collect reservoir states and fit the readout by ridge regression.

        Returns
        -------
        states : [T, P, B, N]
        """
        with torch.no_grad():
            _, states = self.forward(x_seq, return_states=True)

        self.readout.fit(states, targets, n_classes=self.O)
        return states

    def compute_loss(
        self,
        x_seq:   torch.Tensor,   # [T, B, I]
        targets: torch.Tensor,   # [T, B]
    ) -> torch.Tensor:
        """
        Cross-entropy loss per population member.

        Returns
        -------
        losses : [P]   (mean over T and B)
        """
        logits, _ = self.forward(x_seq)   # [T, P, B, O]
        T, P, B, O = logits.shape

        # Flatten T*B for each P
        logits_flat  = logits.permute(1, 0, 2, 3).reshape(P, T * B, O)
        targets_flat = targets.reshape(T * B).unsqueeze(0).expand(P, -1)  # [P, T*B]

        losses = torch.stack([
            F.cross_entropy(logits_flat[p], targets_flat[p])
            for p in range(P)
        ])
        return losses   # [P]

    def fitness(
        self,
        x_seq:   torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Fitness = negative loss (higher is better)."""
        return -self.compute_loss(x_seq, targets)
