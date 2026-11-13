"""Unit tests for src/model.py: the jump-diffusion market model (S, F),
equation (2.1). Synthetic inputs only, small Monte Carlo sizes."""
from __future__ import annotations

import numpy as np
import pytest

from src.model import ModelParams, simulate_paths


def test_model_params_rejects_assumption_2_1_violations():
    with pytest.raises(AssertionError):
        ModelParams(b=0.0, sigma=-0.1, gamma=0.0, lam=1.0)  # sigma must be > 0
    with pytest.raises(AssertionError):
        ModelParams(b=0.0, sigma=0.2, gamma=0.0, lam=-1.0)  # lambda must be > 0
    with pytest.raises(AssertionError):
        ModelParams(b=0.0, sigma=0.2, gamma=-1.5, lam=1.0)  # gamma must be > -1


def test_gamma_tilde_and_b_tilde_formulas():
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0, r=0.0)
    expected_gamma_tilde = np.expm1(-0.1)
    assert np.isclose(mp.gamma_tilde, expected_gamma_tilde)
    expected_b_tilde = 0.05 + 0.5 * 0.2**2 + 3.0 * (expected_gamma_tilde - (-0.1))
    assert np.isclose(mp.b_tilde, expected_b_tilde)


def test_zeta_mode_selects_gamma_or_gamma_tilde():
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0)
    assert mp.zeta("gamma") == mp.gamma
    assert mp.zeta("gamma_tilde") == mp.gamma_tilde
    with pytest.raises(ValueError):
        mp.zeta("unknown")


def test_simulate_paths_shapes_and_finiteness():
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0)
    n_sim, n_steps = 200, 50
    paths = simulate_paths(mp, t_horizon=1.0, n_steps=n_steps, n_sim=n_sim, s0=100.0, seed=0)

    assert paths.W.shape == (n_sim, n_steps + 1)
    assert paths.S.shape == (n_sim, n_steps + 1)
    assert paths.t_grid.shape == (n_steps + 1,)
    assert np.isclose(paths.t_grid[0], 0.0) and np.isclose(paths.t_grid[-1], 1.0)
    for name in ("W", "N", "N_F", "X", "S"):
        arr = getattr(paths, name)
        assert np.all(np.isfinite(arr)), f"{name} has NaN/Inf"
    assert np.all(paths.S > 0.0), "S = S0 exp(X) must stay strictly positive"
    # W starts at 0 and N starts at 0 on every path (definition of the processes).
    assert np.allclose(paths.W[:, 0], 0.0)
    assert np.allclose(paths.N[:, 0], 0.0)


def test_N_F_is_compensated_poisson_process():
    """N^F = N - lambda t must have zero mean increments on average (law of
    large numbers check with a large path count), consistent with (2.1)."""
    mp = ModelParams(b=0.0, sigma=0.2, gamma=-0.1, lam=3.0)
    n_sim, n_steps = 20_000, 50
    paths = simulate_paths(mp, t_horizon=1.0, n_steps=n_steps, n_sim=n_sim, s0=100.0, seed=1)
    mean_N_F_T = float(paths.N_F[:, -1].mean())
    se = float(paths.N_F[:, -1].std() / np.sqrt(n_sim))
    assert abs(mean_N_F_T) < 5 * se


def test_simulate_paths_rejects_too_coarse_grid():
    mp = ModelParams(b=0.0, sigma=0.2, gamma=0.0, lam=1000.0)
    with pytest.raises(AssertionError):
        simulate_paths(mp, t_horizon=1.0, n_steps=2, n_sim=10, s0=100.0, seed=0)
