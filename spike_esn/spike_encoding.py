"""
Spike Input Layer — Poisson-distribution-based spike encoding.

Implements Section III-A of the paper:
  1. Compute the average spike interval h_κ(t) from normalized input  (Eq. 2)
  2. Sample spike intervals κ_l(t) from a Poisson distribution        (Eq. 3-5)
  3. Generate binary spike sequences s_i(t)                            (Eq. 6-7)

The spike input layer converts a scalar input u(t) into a binary vector
of length N_sam, where 1s indicate spike activations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SpikeEncoder:
    """Poisson-distribution spike encoder for time series data.

    Parameters
    ----------
    N_sam : int
        Spike sampling times — length of the output spike sequence.
        Higher values project inputs into a higher temporal dimension,
        improving feature extraction at the cost of computation.
    N_int : int or None
        Number of spike intervals. If None, it is adaptively determined
        from the Poisson sampling process.
    """

    def __init__(self, N_sam: int = 100, N_int: int | None = None) -> None:
        self.N_sam = N_sam
        self.N_int = N_int

    # ------------------------------------------------------------------
    # Eq. 2 — average spike interval
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_avg_interval(
        u: float, U_max: float, U_min: float, N_sam: int
    ) -> float:
        """Compute h_κ(t) = N_sam × (U_max − u(t)) / (U_max − U_min).

        When u(t) is large (close to U_max), h_κ is small → high spike
        frequency.  When u(t) is small, h_κ is large → low spike frequency.
        """
        denom = U_max - U_min
        if denom == 0:
            # Constant signal: return a neutral interval
            return N_sam / 2.0
        return N_sam * (U_max - u) / denom

    # ------------------------------------------------------------------
    # Eq. 3-6 — generate a single spike sequence from one scalar input
    # ------------------------------------------------------------------
    def encode_scalar(
        self,
        u: float,
        U_max: float,
        U_min: float,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
    ) -> NDArray[np.int8]:
        """Encode a single scalar value into a spike sequence of length N_sam.

        Parameters
        ----------
        u : float
            The (normalised) input value at time *t*.
        U_max, U_min : float
            Global max / min of the input time series (for normalisation).
        rng : numpy Generator, optional
            Random number generator for reproducibility.
        deterministic : bool
            If True, use a seed derived from 'u' to make the encoding 
            consistent for the same input value.
        """
        if deterministic:
            # Create a stable seed from the value of u
            # Using a fixed precision to avoid float noise
            seed = int(abs(hash(round(float(u), 8))) % (2**32))
            local_rng = np.random.default_rng(seed)
        elif rng is None:
            local_rng = np.random.default_rng()
        else:
            local_rng = rng

        N_sam = self.N_sam
        # Eq. 2 — average interval
        h_kappa = self._compute_avg_interval(u, U_max, U_min, N_sam)
        h_kappa = max(h_kappa, 1.0)

        intervals: list[int] = []
        cumsum = 0
        while cumsum < N_sam:
            kappa = int(local_rng.poisson(lam=h_kappa))
            kappa = max(kappa, 1)
            if cumsum + kappa > N_sam:
                break
            intervals.append(kappa)
            cumsum += kappa

        # If a fixed N_int was requested, truncate or extend
        if self.N_int is not None and len(intervals) > self.N_int:
            intervals = intervals[: self.N_int]

        # Eq. 6 — build spike sequence from intervals
        spike_seq = np.zeros(N_sam, dtype=np.int8)
        pos = 0
        for kappa in intervals:
            pos += kappa
            if pos <= N_sam:
                spike_seq[pos - 1] = 1  # 0-indexed → pos-1

        return spike_seq

    # ------------------------------------------------------------------
    # Batch helper — encode an entire time series
    # ------------------------------------------------------------------
    def encode_series(
        self,
        u_series: NDArray[np.floating],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.int8]:
        """Encode a 1-D time series into a matrix of spike sequences.

        Parameters
        ----------
        u_series : ndarray of shape (T,)
            Input time series.

        Returns
        -------
        spikes : ndarray of shape (T, N_sam)
            Each row is the spike sequence for the corresponding time step.
        """
        if rng is None:
            rng = np.random.default_rng()

        T = len(u_series)
        U_max = float(np.max(u_series))
        U_min = float(np.min(u_series))

        spikes = np.zeros((T, self.N_sam), dtype=np.int8)
        for t in range(T):
            spikes[t] = self.encode_scalar(u_series[t], U_max, U_min, rng)
        return spikes
