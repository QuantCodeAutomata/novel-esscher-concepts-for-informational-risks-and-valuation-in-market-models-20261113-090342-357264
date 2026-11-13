"""G-martingale transform T(M) (eq. 2.6) and the G-martingale N^G (eq. 2.7)
used to build the Esscher density Z^G of Theorem 3.3.

All quantities are stopped at tau (every process below is evaluated on
t in [0, tau], consistent with "Z^G = (Z^G)^tau" in Theorem 3.3), since the
degenerate third term of (2.6) -- the F-predictable projection correction on
{Gtilde=0<G_-} -- is not needed once G is kept strictly positive (see
src/random_time.py, G_FLOOR): with G_FLOOR>0 the event {Gtilde=0<G_-} is
empty on every simulated path, so that term is omitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model import SimulatedPaths
from src.random_time import RandomTimePaths


def _stop_at_tau(path: np.ndarray, tau_index: np.ndarray) -> np.ndarray:
    """Freeze each row of ``path`` at its own tau_index (process stopped at tau)."""
    n_sim, n_steps_plus_1 = path.shape
    stopped = path.copy()
    idx = np.arange(n_steps_plus_1)[None, :]
    mask = idx > tau_index[:, None]
    # broadcast the value at tau_index into every later column
    tau_val = path[np.arange(n_sim), tau_index][:, None]
    stopped = np.where(mask, tau_val, path)
    return stopped


@dataclass
class GMartingaleTransforms:
    """T(W), T(N^F) and N^G, shape (n_sim, n_steps+1), stopped at tau."""

    T_W: np.ndarray
    T_N_F: np.ndarray
    N_G: np.ndarray


def compute_transforms(
    paths: SimulatedPaths,
    rt: RandomTimePaths,
    alpha: float,
    beta: float,
) -> GMartingaleTransforms:
    """Build T(W), T(N^F) (eq. 2.6) and N^G (eq. 2.7), stopped at tau.

    [W, m]_t = alpha * int_0^t G_{s-} ds           (continuous covariation)
    [N^F, m]_t = beta * sum_{s<=t} G_{s-} dN_s      (pure-jump covariation)
    T(M)_t = M_{t and tau} - (Gtilde_{s-}^{-1} 1_{[0,tau]} . [M,m])_{t and tau}
    N^G_t = D_t - Gtilde_t^{-1} 1_{t<=tau} . D^{o,F}_t
    """
    n_sim, n_steps_plus_1 = paths.W.shape
    tau_index = rt.tau_index
    G_minus = np.concatenate([rt.G[:, :1], rt.G[:, :-1]], axis=1)  # G_{t-}
    G_tilde_minus = np.concatenate([rt.G_tilde[:, :1], rt.G_tilde[:, :-1]], axis=1)  # Gtilde_{t-}

    # "Gtilde^{-1} 1_{[0,tau]} . [M,m]" in (2.6) is the stochastic (Stieltjes)
    # integral of the predictable process Gtilde_{s-}^{-1} against d[M,m]_s,
    # i.e. int_0^t Gtilde_{s-}^{-1} d[M,m]_s -- NOT the cumulative covariation
    # divided by its terminal Gtilde value. Build it increment-by-increment.
    dcov_W_m = alpha * G_minus[:, :-1] * paths.dt
    dcov_NF_m = beta * G_minus[:, :-1] * paths.dN
    inv_G_tilde_minus = 1.0 / G_tilde_minus[:, :-1]

    corr_W_m = np.cumsum(dcov_W_m * inv_G_tilde_minus, axis=1)
    corr_W_m = np.concatenate([np.zeros((n_sim, 1)), corr_W_m], axis=1)

    corr_NF_m = np.cumsum(dcov_NF_m * inv_G_tilde_minus, axis=1)
    corr_NF_m = np.concatenate([np.zeros((n_sim, 1)), corr_NF_m], axis=1)

    W_stopped = _stop_at_tau(paths.W, tau_index)
    N_F_stopped = _stop_at_tau(paths.N_F, tau_index)
    corr_W_m_stopped = _stop_at_tau(corr_W_m, tau_index)
    corr_NF_m_stopped = _stop_at_tau(corr_NF_m, tau_index)

    T_W = W_stopped - corr_W_m_stopped
    T_N_F = N_F_stopped - corr_NF_m_stopped

    # N^G_t = D_t - Gtilde_t^{-1} 1_{[0,tau]}(t) . D^{o,F}_t   (eq. 2.7)
    idx = np.arange(n_steps_plus_1)[None, :]
    before_tau = idx <= tau_index[:, None]
    integrand = np.where(before_tau, 1.0 / rt.G_tilde, 0.0)
    dD_oF = np.diff(rt.D_oF, axis=1, prepend=rt.D_oF[:, :1])
    N_G = rt.default_indicator - np.cumsum(integrand * dD_oF, axis=1)

    return GMartingaleTransforms(T_W=T_W, T_N_F=T_N_F, N_G=N_G)
