"""Unit tests for src/transforms.py: the G-martingale transform T(M) (2.6)
and the G-martingale N^G (2.7). Synthetic inputs only."""
from __future__ import annotations

import numpy as np

from src.model import ModelParams, simulate_paths
from src.random_time import RandomTimeParams, simulate_random_time
from src.transforms import compute_transforms


def _setup(n_sim=500, n_steps=60, seed=0, alpha=0.02, beta=0.01, eta=0.05):
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0)
    paths = simulate_paths(mp, t_horizon=1.0, n_steps=n_steps, n_sim=n_sim, s0=100.0, seed=seed)
    rt_params = RandomTimeParams(alpha=alpha, beta=beta, eta=eta, hazard=0.15)
    rt_paths = simulate_random_time(rt_params, paths, seed=seed + 1, lam=mp.lam)
    return mp, paths, rt_params, rt_paths


def test_transforms_shapes_and_finiteness():
    mp, paths, rt_params, rt_paths = _setup()
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)

    n_sim, n_steps_plus1 = paths.W.shape
    assert transforms.T_W.shape == (n_sim, n_steps_plus1)
    assert transforms.T_N_F.shape == (n_sim, n_steps_plus1)
    assert transforms.N_G.shape == (n_sim, n_steps_plus1)
    for name in ("T_W", "T_N_F", "N_G"):
        arr = getattr(transforms, name)
        assert np.all(np.isfinite(arr)), f"{name} has NaN/Inf"


def test_transforms_start_at_zero():
    mp, paths, rt_params, rt_paths = _setup()
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)
    assert np.allclose(transforms.T_W[:, 0], 0.0)
    assert np.allclose(transforms.T_N_F[:, 0], 0.0)
    assert np.allclose(transforms.N_G[:, 0], 0.0)


def test_N_G_is_zero_or_one_valued_component_plus_bounded_correction():
    """N^G_t = D_t - Gtilde_t^{-1} 1_{t<=tau} . D^{o,F}_t, equation (2.7): the
    correction term is a predictable-compensator subtraction, so N^G should
    stay within a bounded range around the {0,1}-valued D for small
    horizons/intensities (sanity check, not an exact bound from the paper)."""
    mp, paths, rt_params, rt_paths = _setup(eta=0.0)
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)
    assert np.all(np.abs(transforms.N_G) < 5.0)


def test_alpha_beta_zero_reduces_T_to_identity_up_to_stopping():
    """When alpha=beta=0 the covariation [W,m] and [N^F,m] used in (2.6) are
    identically zero, so T(W)=W^tau and T(N^F)=(N^F)^tau exactly."""
    n_sim, n_steps = 300, 40
    mp, paths, rt_params, rt_paths = _setup(n_sim=n_sim, n_steps=n_steps, alpha=0.0, beta=0.0, eta=0.0)
    transforms = compute_transforms(paths, rt_paths, alpha=0.0, beta=0.0)

    tau_index = rt_paths.tau_index
    idx = np.arange(n_steps + 1)[None, :]
    mask = idx > tau_index[:, None]
    tau_val_W = paths.W[np.arange(n_sim), tau_index][:, None]
    W_stopped = np.where(mask, tau_val_W, paths.W)
    assert np.allclose(transforms.T_W, W_stopped)
