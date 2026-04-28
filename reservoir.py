"""
reservoir.py
------------
Builds and runs one **plastic, rewiring reservoir** from a NeuronGene.

The reservoir follows the standard Echo State Network (ESN) topology:
    Input layer  →  Reservoir  →  (Output layer trained separately)

Key mechanics driven by the NeuronGene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Activation / leakage**: membrane leaks with time constant τ_m.
  The continuous-time leaky integrator is discretised as:
      α = dt / τ_m   (leak coefficient)
      h_new = (1 − α) h  +  α · gain · tanh(W_rec h + W_in x + bias)

* **Online weight plasticity**: after every step, W_rec is nudged by
  the Hebbian update encoded in the gene's plasticity_coeffs:
      ΔW = lr_plasticity · Σ_k η_k · ϕ_k(pre, post, W)
  Only *existing* synapses (mask == 1) are updated.

* **Rewiring**: periodically, weak synapses are pruned and new ones
  grown according to the gene's rewiring parameters:
      - Prune:  abs(w_ij) < prune_thresh   (and coin-flip)
      - Grow:   p_grow = growth_prob + activity_bias · x_post_mean

All batch members and population members are kept in the *same*
tensors, so evaluation is fully vectorised.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from neuron import NeuronGene, PlasticityBasis


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------

class Reservoir(nn.Module):
    """
    A single plastic, rewiring reservoir parameterised by one NeuronGene.

    Shapes
    ------
    P = population_size  (number of gene variants evaluated in parallel)
    B = batch_size
    N = reservoir_size   (number of recurrent neurons)
    I = input_size

    Buffers (not evolved, reset each episode)
    -----------------------------------------
    h       [P, B, N]   hidden state (real-valued firing rates)
    W_rec   [P, N, N]   recurrent weight matrix  (sparse, masked)
    W_in    [P, N, I]   input projection weights  (fixed after init)
    mask    [P, N, N]   binary connectivity mask  (updated by rewiring)
    bias    [P, N]      per-neuron bias
    """

    def __init__(
        self,
        gene:           NeuronGene,
        population_size: int,
        reservoir_size:  int,
        input_size:      int,
        density:         float = 0.1,
        dt:              float = 1.0,
        lr_plasticity:   float = 1e-3,
        rewire_every:    int   = 10,
        spectral_radius: float = 0.9,
        device:          str   = "cpu",
    ):
        super().__init__()

        self.gene            = gene
        self.P               = population_size
        self.N               = reservoir_size
        self.I               = input_size
        self.density         = density
        self.dt              = dt
        self.lr_plasticity   = lr_plasticity
        self.rewire_every    = rewire_every
        self.spectral_radius = spectral_radius
        self.device          = device
        self._step           = 0

        # ----------------------------------------------------------------
        # Initialise weights (shared structure, but independent per gene)
        # ----------------------------------------------------------------
        W_rec, mask = self._init_recurrent(population_size, reservoir_size, density, spectral_radius)
        W_in        = self._init_input(population_size, reservoir_size, input_size)
        bias        = torch.zeros(population_size, reservoir_size)

        # Register as buffers (moved with .to(device), not trained by grad)
        self.register_buffer("W_rec", W_rec)
        self.register_buffer("W_in",  W_in)
        self.register_buffer("mask",  mask)
        self.register_buffer("bias",  bias)

        # Hidden state
        self.register_buffer("h", torch.zeros(population_size, 1, reservoir_size))

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _init_recurrent(
        P: int, N: int, density: float, spectral_radius: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse random recurrent weights scaled to target spectral radius."""
        W = torch.randn(P, N, N) * 0.1
        mask = (torch.rand(P, N, N) < density).float()
        # No self-connections
        eye = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(P, -1, -1)
        mask[eye] = 0.0
        W = W * mask

        # Scale each population member to target spectral radius
        for p in range(P):
            eigs = torch.linalg.eigvals(W[p])
            rho  = torch.max(torch.abs(eigs)).real.item()
            if rho > 1e-8:
                W[p] = W[p] * (spectral_radius / rho)

        return W, mask

    @staticmethod
    def _init_input(P: int, N: int, I: int) -> torch.Tensor:
        """Dense random input projection, scaled by 1/sqrt(I)."""
        return torch.randn(P, N, I) / math.sqrt(I)

    # ------------------------------------------------------------------ #
    # Reset state
    # ------------------------------------------------------------------ #

    def reset_state(self, batch_size: int) -> None:
        """Zero the hidden state for a new sequence."""
        self.h = torch.zeros(self.P, batch_size, self.N, device=self.device)
        self._step = 0

    # ------------------------------------------------------------------ #
    # Forward step
    # ------------------------------------------------------------------ #

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        One time-step of the reservoir.

        Parameters
        ----------
        x : [B, I] or [P, B, I]
            Input at this time-step.  If [B, I], it is broadcast to all P.

        Returns
        -------
        h : [P, B, N]
            New hidden state.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.P, -1, -1)  # [P, B, I]

        gene = self.gene

        # Leak coefficient α = dt / τ_m  (clamped to [0,1])
        alpha  = (self.dt / gene.tau_m).clamp(0.01, 0.99)
        gain   = gene.activation_gain

        # Recurrent input:  [P, B, N]
        # h: [P, B, N],  W_rec: [P, N, N]
        rec_in = torch.bmm(self.h, self.W_rec.transpose(1, 2))   # [P, B, N]

        # Input projection:  [P, B, N]
        # x: [P, B, I],  W_in: [P, N, I]
        inp_in = torch.bmm(x, self.W_in.transpose(1, 2))          # [P, B, N]

        pre_act = rec_in + inp_in + self.bias.unsqueeze(1)         # [P, B, N]

        # Leaky integrator update
        h_new = (1.0 - alpha) * self.h + alpha * gain * torch.tanh(pre_act)

        # Online weight plasticity (only existing synapses)
        if self.lr_plasticity > 0.0:
            self._apply_plasticity(self.h, h_new)

        self.h = h_new
        self._step += 1

        # Periodic rewiring
        if self.rewire_every > 0 and self._step % self.rewire_every == 0:
            self._apply_rewiring(h_new)

        return h_new  # [P, B, N]

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Run the reservoir over a full sequence.

        Parameters
        ----------
        x_seq : [T, B, I]
            Sequence of inputs.

        Returns
        -------
        states : [T, P, B, N]
            Hidden states at every timestep.
        """
        T = x_seq.shape[0]
        states = []
        for t in range(T):
            h = self.step(x_seq[t])      # [P, B, N]
            states.append(h)
        return torch.stack(states, dim=0)  # [T, P, B, N]

    # ------------------------------------------------------------------ #
    # Plasticity rule
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _apply_plasticity(self, h_pre: torch.Tensor, h_post: torch.Tensor) -> None:
        """
        Update W_rec using the gene's Hebbian rule.

        h_pre  [P, B, N]  — state *before* the step  (pre-synaptic rates)
        h_post [P, B, N]  — state *after*  the step  (post-synaptic rates)

        The update is averaged over the batch dimension B.
        """
        gene = self.gene
        eta  = gene.plasticity_coeffs   # [n_terms]

        # Average over batch to get mean rates  [P, N]
        xpre  = h_pre.mean(dim=1)    # [P, N]
        xpost = h_post.mean(dim=1)   # [P, N]

        # Evaluate basis [P, N_post, N_pre, n_terms] — expensive for large N;
        # use a closed-form linear combination instead for efficiency.
        #
        # ΔW_ij = η_0 x_pre_i x_post_j
        #       + η_1 x_pre_i (1 - x_post_j)
        #       + η_2 (1 - x_pre_i) x_post_j
        #       + η_3 (- W_ij)
        #       + η_4 (x_post_j²  W_ij)
        #
        # We can factor this as an outer-product sum.

        e0, e1, e2, e3, e4 = (eta[k] for k in range(5))

        # xpost: [P, N] → [P, Npost, 1]
        # xpre:  [P, N] → [P, 1, Npre]
        xpost_col = xpost.unsqueeze(2)   # [P, N, 1]
        xpre_row  = xpre.unsqueeze(1)    # [P, 1, N]

        dW = (
            e0 * (xpost_col * xpre_row)                          # Hebb
          + e1 * (xpre_row * (1.0 - xpost_col))                  # anti-Hebb post
          + e2 * ((1.0 - xpre_row) * xpost_col)                  # anti-Hebb pre
          + e3 * (-self.W_rec)                                    # weight decay
          + e4 * ((xpost_col ** 2) * self.W_rec)                 # BCM-like
        )

        # Apply only to existing synapses
        self.W_rec.add_(self.lr_plasticity * dW * self.mask)

    # ------------------------------------------------------------------ #
    # Rewiring rule
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _apply_rewiring(self, h: torch.Tensor) -> None:
        """
        Prune weak synapses and grow new ones.

        h : [P, B, N]  — current hidden state (batch-averaged for activity)
        """
        gene = self.gene

        prune_thresh    = gene.rewire_prune_thresh     # scalar
        growth_prob_base= gene.rewire_growth_prob      # scalar in (0,1)
        act_bias        = gene.rewire_activity_bias    # scalar
        w_init_std      = gene.rewire_weight_init_std  # scalar > 0

        # ---------- Prune ----------
        weak      = torch.abs(self.W_rec) < prune_thresh   # [P, N, N]
        prune_coin= torch.rand_like(self.W_rec) < 0.5      # random chance
        to_prune  = weak & prune_coin & self.mask.bool()

        self.W_rec[to_prune] = 0.0
        self.mask[to_prune]  = 0.0

        # ---------- Grow ----------
        # Activity of post-synaptic neurons (averaged over batch) [P, N]
        activity = h.mean(dim=1).abs()   # [P, N]

        # Growth probability per post-neuron: p_ij = sigmoid(base + bias * act_j)
        # Broadcast: [P, N, 1] → [P, N, N]
        act_col = activity.unsqueeze(2)                       # [P, N, 1]
        p_grow  = torch.sigmoid(
            growth_prob_base.logit() + act_bias * act_col
        ).expand_as(self.W_rec)                               # [P, N, N]

        # Candidate positions: currently unconnected, non-self
        vacant = (1.0 - self.mask)
        # No self-connections
        eye = torch.eye(self.N, device=self.device, dtype=torch.bool)
        eye = eye.unsqueeze(0).expand(self.P, -1, -1)
        vacant[eye] = 0.0

        # Stochastic growth
        coin   = torch.rand_like(self.W_rec) < (p_grow * vacant)
        new_w  = torch.randn_like(self.W_rec) * w_init_std

        self.W_rec[coin] = new_w[coin]
        self.mask[coin]  = 1.0

        # Enforce density ceiling (prune weakest if over budget)
        max_edges = int(self.N * self.N * self.density * 1.5)
        n_edges   = self.mask.sum(dim=(1, 2))                  # [P]
        for p in range(self.P):
            if n_edges[p].item() > max_edges:
                flat_w  = self.W_rec[p].abs().view(-1)
                flat_m  = self.mask[p].view(-1)
                # Sort connected weights
                vals, idx = flat_w[flat_m.bool()].sort()
                excess  = int(n_edges[p].item()) - max_edges
                prune_idx = idx[:excess]
                # Recover global indices
                connected_idx = flat_m.bool().nonzero(as_tuple=True)[0]
                for pi in connected_idx[prune_idx]:
                    self.W_rec[p].view(-1)[pi] = 0.0
                    self.mask[p].view(-1)[pi]  = 0.0
