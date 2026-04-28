"""
Spike Echo State Network (Spike-ESN) — full model.

Implements Algorithm 1: Spike Input Layer + Spike Reservoir + Ridge Regression.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .spike_encoding import SpikeEncoder
from .reservoir import SpikeReservoir


class SpikeESN:
    """Brain-Inspired Spike Echo State Network for time series prediction.

    Parameters
    ----------
    N_res : int  — Number of reservoir neurons (default: 100).
    N_sam : int  — Spike sampling times (default: 100).
    rho   : float — Spectral radius (default: 0.9).
    eta   : float — Reservoir sparsity (default: 0.1).
    mu    : float — Ridge regularisation (default: 1e-8).
    psi   : float — Synaptic time constant (default: 5000).
    input_scaling : float — W_in scaling (default: 0.8).
    seed  : int or None — Random seed.
    """

    def __init__(self, N_res=100, N_sam=100, rho=0.9, eta=0.1,
                 mu=1e-8, psi=5000.0, input_scaling=0.8, seed=None):
        self.N_res = N_res
        self.N_sam = N_sam
        self.rho = rho
        self.eta = eta
        self.mu = mu
        self.psi = psi
        self.input_scaling = input_scaling
        self.seed = seed

        self.encoder = SpikeEncoder(N_sam=N_sam)
        self.reservoir = SpikeReservoir(
            N_res=N_res, N_sam=N_sam, rho=rho, eta=eta,
            psi=psi, input_scaling=input_scaling, seed=seed,
        )
        self.W_out: NDArray[np.float64] | None = None
        self._train_states: NDArray[np.float64] | None = None

    def fit(self, u, y, washout=200):
        """Train the Spike-ESN (Algorithm 1).

        Parameters
        ----------
        u : ndarray (T,) — Input time series.
        y : ndarray (T,) or (T-washout,) — Target output.
        washout : int — Initial steps to discard (default: 200).
        """
        T = len(u)
        rng = np.random.default_rng(self.seed)

        # Steps 1-7: Spike-encode entire input
        spike_matrix = self.encoder.encode_series(u, rng=rng)

        # Steps 10-13: Drive reservoir and collect states (Eq. 10)
        X = self.reservoir.harvest_states(spike_matrix, washout=washout)
        self._train_states = X

        # Align target
        T_eff = X.shape[1]
        if len(y) == T:
            y_target = y[washout:washout + T_eff].reshape(1, -1)
        elif len(y) == T_eff:
            y_target = y.reshape(1, -1)
        else:
            raise ValueError(
                f"Target length {len(y)} doesn't match T={T} or T_eff={T_eff}")

        # Step 14: Ridge regression (Eq. 13)
        # W_out = y · X^T · (X·X^T + μ·I)^{-1}
        XXT = X @ X.T
        reg = self.mu * np.eye(self.N_res)
        self.W_out = y_target @ X.T @ np.linalg.inv(XXT + reg)
        return self

    def predict(self, u, washout=0):
        """Predict using trained model.

        Parameters
        ----------
        u : ndarray (T,) — Input time series.
        washout : int — Steps to discard (default: 0).

        Returns
        -------
        y_hat : ndarray (T-washout,) — Predicted output.
        """
        if self.W_out is None:
            raise RuntimeError("Call fit() first.")

        rng = np.random.default_rng(self.seed)
        spike_matrix = self.encoder.encode_series(u, rng=rng)
        X = self.reservoir.harvest_states(spike_matrix, washout=washout)
        return (self.W_out @ X).flatten()

    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Square Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return float("inf")
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

    def get_state_matrix(self):
        """Return state collection matrix X from last fit()."""
        return self._train_states

    def get_output_weights(self):
        """Return trained W_out."""
        return self.W_out

    def __repr__(self):
        return (f"SpikeESN(N_res={self.N_res}, N_sam={self.N_sam}, "
                f"rho={self.rho}, eta={self.eta}, mu={self.mu}, psi={self.psi})")
