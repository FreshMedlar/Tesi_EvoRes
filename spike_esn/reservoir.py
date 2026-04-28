"""
Spike Reservoir — sparse recurrent network with spike-current input processing.

Implements Section III-B of the paper:
  1. Generate internal weight matrix W_res with sparsity η and spectral radius ρ  (Eq. 8)
  2. Generate input weight matrix W_in                                            (Alg. 1, step 8)
  3. Compute spike-based input current f_spike via exponential kernel              (Eq. 9)
  4. Update reservoir internal state x(t) with tanh activation                     (Eq. 9)
  5. Collect all states into the state collection matrix X                         (Eq. 10)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SpikeReservoir:
    """Spike reservoir with exponential synaptic current model.

    Parameters
    ----------
    N_res : int
        Number of neurons in the reservoir.
    N_sam : int
        Length of the spike sequence (temporal dimension).
    rho : float
        Spectral radius — controls the echo state property. Must be < 1
        for the reservoir to have fading memory.
    eta : float
        Sparsity of the reservoir weight matrix (fraction of non-zero entries).
    psi : float
        Time constant of synaptic currents (ψ in the paper).
        Controls the magnitude and decay of the exponential current kernel.
    input_scaling : float
        Scaling factor applied to W_in (set to 0.8 in the paper).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        N_res: int = 100,
        N_sam: int = 100,
        rho: float = 0.9,
        eta: float = 0.1,
        psi: float = 5000.0,
        input_scaling: float = 0.8,
        seed: int | None = None,
    ) -> None:
        self.N_res = N_res
        self.N_sam = N_sam
        self.rho = rho
        self.eta = eta
        self.psi = psi
        self.input_scaling = input_scaling
        self.rng = np.random.default_rng(seed)

        # Initialise weight matrices
        self.W_in = self._init_input_weights()
        self.W_res = self._init_reservoir_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_input_weights(self) -> NDArray[np.float64]:
        """Generate W_in ∈ ℝ^{N_res × N_sam} from Uniform(−1, 1), scaled."""
        W_in = self.rng.uniform(-1, 1, size=(self.N_res, self.N_sam))
        return W_in * self.input_scaling

    def _init_reservoir_weights(self) -> NDArray[np.float64]:
        """Generate W_res with sparsity η and spectral radius ρ  (Eq. 8).

        Steps:
          1. Sample W from Uniform(−1, 1).
          2. Apply sparsity mask (keep only η fraction of entries).
          3. Scale so that spectral radius equals ρ.
        """
        N = self.N_res

        # Random matrix in [-1, 1]
        W = self.rng.uniform(-1, 1, size=(N, N))

        # Apply sparsity mask
        mask = self.rng.random(size=(N, N)) < self.eta
        W = W * mask

        # Compute maximum eigenvalue
        eigenvalues = np.linalg.eigvals(W)
        lambda_max = np.max(np.abs(eigenvalues))

        if lambda_max == 0:
            # Degenerate case — re-initialise with slightly denser matrix
            return self._init_reservoir_weights()

        # Eq. 8 — scale to desired spectral radius
        W_res = self.rho * (W / lambda_max)
        return W_res

    # ------------------------------------------------------------------
    # Spike current computation  (Eq. 9 — f_spike)
    # ------------------------------------------------------------------
    def compute_spike_current(
        self, spike_seq: NDArray[np.int8]
    ) -> NDArray[np.float64]:
        """Compute the spike-based input current vector f_spike.

        Processes the input in blocks of self.N_sam to support multi-channel
        (one-hot) encoding correctly. Each channel gets its own local
        timeline from 1 to N_sam.
        """
        N_total = len(spike_seq)
        N_sam = self.N_sam
        n_channels = N_total // N_sam
        
        # Reshape to (n_channels, N_sam)
        spikes_2d = spike_seq.reshape(n_channels, N_sam)
        f_spike_2d = np.zeros((n_channels, N_sam), dtype=np.float64)
        
        # t_seq runs from 1 to N_sam (1-indexed)
        t_seq = np.arange(1, N_sam + 1, dtype=np.float64)

        for c in range(n_channels):
            spike_positions = np.where(spikes_2d[c] == 1)[0] + 1
            if len(spike_positions) > 0:
                # Eq. 9 — Vectorised over all spike times in this channel
                diffs = t_seq[:, None] - spike_positions[None, :]
                f_spike_2d[c] = np.sum(np.exp(-diffs / self.psi), axis=1)
        
        return f_spike_2d.flatten()

    # ------------------------------------------------------------------
    # Reservoir state update  (Eq. 9, first line)
    # ------------------------------------------------------------------
    def update_state(
        self,
        f_spike: NDArray[np.float64],
        x_prev: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute x(t) = tanh(W_in · f_spike + W_res · x(t−1)).

        Parameters
        ----------
        f_spike : ndarray of shape (N_sam,)
            Spike-based input current vector.
        x_prev : ndarray of shape (N_res,)
            Previous reservoir state.

        Returns
        -------
        x_new : ndarray of shape (N_res,)
            Updated reservoir state.
        """
        return np.tanh(self.W_in @ f_spike + self.W_res @ x_prev)

    # ------------------------------------------------------------------
    # Harvest states from a full spike-encoded series  (Eq. 10)
    # ------------------------------------------------------------------
    def harvest_states(
        self,
        spike_matrix: NDArray[np.int8],
        washout: int = 0,
    ) -> NDArray[np.float64]:
        """Drive the reservoir with a spike-encoded time series and collect states.

        Parameters
        ----------
        spike_matrix : ndarray of shape (T, N_sam)
            Each row is the spike sequence for one time step.
        washout : int
            Number of initial time steps to discard (reservoir warm-up).

        Returns
        -------
        X : ndarray of shape (N_res, T − washout)
            State collection matrix (Eq. 10), each column is x(t).
        """
        T = spike_matrix.shape[0]
        X_all = np.zeros((self.N_res, T), dtype=np.float64)
        x = np.zeros(self.N_res, dtype=np.float64)  # x(0) = 0

        for t in range(T):
            f_spike = self.compute_spike_current(spike_matrix[t])
            x = self.update_state(f_spike, x)
            X_all[:, t] = x

        # Discard washout period
        return X_all[:, washout:]
