"""Market model (S, F): geometric jump-diffusion with Brownian motion and a
non-homogeneous Poisson process, equation (2.1) of the paper.

S^(0)_t = exp(int_0^t r_s ds),
S_t = S0 exp(X_t),  X = b t + sigma W + gamma . N^F,
N^F = N - int lambda ds  (F-compensated Poisson process),
btilde = b + sigma^2/2 + lambda (e^gamma - 1 - gamma),  gammatilde = e^gamma - 1.

All model parameters (b, sigma, gamma, lambda) are taken constant in time,
which trivially satisfies Assumption 2.1 (local boundedness, sigma,lambda>0,
gamma>-1). Simulation uses an Euler-Maruyama scheme for the continuous part
and direct Bernoulli thinning (one jump per step at most, valid since the
time step is chosen so that lambda*dt << 1) for the Poisson part.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelParams:
    """Constant parameters of the jump-diffusion market model (2.1)."""

    b: float
    sigma: float
    gamma: float
    lam: float
    r: float = 0.0

    def __post_init__(self) -> None:
        assert self.sigma > 0, "Assumption 2.1 requires sigma > 0"
        assert self.lam > 0, "Assumption 2.1 requires lambda > 0"
        assert self.gamma > -1, "Assumption 2.1 requires gamma > -1"

    @property
    def gamma_tilde(self) -> float:
        """gammatilde = e^gamma - 1, equation (2.1)."""
        return float(np.expm1(self.gamma))

    @property
    def b_tilde(self) -> float:
        """btilde = b + sigma^2/2 + lambda (e^gamma - 1 - gamma), equation (2.1)."""
        return self.b + 0.5 * self.sigma**2 + self.lam * (self.gamma_tilde - self.gamma)

    def zeta(self, mode: str) -> float:
        """Unified notation zeta in {gamma, gammatilde} of Remark 3.2."""
        if mode == "gamma":
            return self.gamma
        if mode == "gamma_tilde":
            return self.gamma_tilde
        raise ValueError(f"unknown zeta mode: {mode}")


@dataclass
class SimulatedPaths:
    """Discretized paths shared by all computations.

    Shapes are (n_sim, n_steps+1) unless stated otherwise.
    """

    t_grid: np.ndarray          # (n_steps+1,) time grid 0=t0<...<tM=T
    dt: float
    dW: np.ndarray              # (n_sim, n_steps) Brownian increments
    dN: np.ndarray              # (n_sim, n_steps) Poisson jump counts in {0,1}
    W: np.ndarray               # (n_sim, n_steps+1) Brownian motion path
    N: np.ndarray               # (n_sim, n_steps+1) Poisson counting process
    N_F: np.ndarray             # (n_sim, n_steps+1) compensated Poisson process
    X: np.ndarray               # (n_sim, n_steps+1) log-price process X
    S: np.ndarray               # (n_sim, n_steps+1) risky asset price S


def simulate_paths(
    params: ModelParams,
    t_horizon: float,
    n_steps: int,
    n_sim: int,
    s0: float,
    seed: int,
) -> SimulatedPaths:
    """Simulate (W, N, N^F, X, S) on an equally spaced grid on [0, T].

    Euler-Maruyama discretization for the continuous part; the Poisson part
    is simulated by direct Bernoulli thinning with per-step jump probability
    lambda*dt (valid since dt is chosen so that lambda*dt << 1).
    """
    assert n_steps > 0 and n_sim > 0 and t_horizon > 0
    rng = np.random.default_rng(seed)
    dt = t_horizon / n_steps
    assert params.lam * dt < 0.2, "time step too coarse for Bernoulli-thinned Poisson jumps"

    t_grid = np.linspace(0.0, t_horizon, n_steps + 1)
    dW = rng.normal(scale=np.sqrt(dt), size=(n_sim, n_steps))
    dN = rng.binomial(1, params.lam * dt, size=(n_sim, n_steps)).astype(float)

    W = np.concatenate([np.zeros((n_sim, 1)), np.cumsum(dW, axis=1)], axis=1)
    N = np.concatenate([np.zeros((n_sim, 1)), np.cumsum(dN, axis=1)], axis=1)
    N_F = N - params.lam * t_grid[None, :]

    # X = b.t + sigma.W + gamma.N^F uses the COMPENSATED Poisson process N^F
    # = N - int(lambda ds) (equation 2.1); dN_F = dN - lambda*dt per step.
    dN_F = dN - params.lam * dt
    dX = params.b * dt + params.sigma * dW + params.gamma * dN_F
    X = np.concatenate([np.zeros((n_sim, 1)), np.cumsum(dX, axis=1)], axis=1)
    S = s0 * np.exp(X)

    return SimulatedPaths(t_grid=t_grid, dt=dt, dW=dW, dN=dN, W=W, N=N, N_F=N_F, X=X, S=S)
