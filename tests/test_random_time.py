"""Unit tests for src/random_time.py: the random horizon tau parametrized
by (alpha, beta, eta, D^{p,F}) of Lemma 2.4. Synthetic inputs only."""
from __future__ import annotations

import numpy as np
import pytest

from src.model import ModelParams, simulate_paths
from src.random_time import RandomTimeParams, simulate_random_time


def _paths(n_sim=500, n_steps=60, seed=0):
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0)
    return mp, simulate_paths(mp, t_horizon=1.0, n_steps=n_steps, n_sim=n_sim, s0=100.0, seed=seed)


def test_random_time_params_rejects_lemma_2_4_violations():
    with pytest.raises(AssertionError):
        RandomTimeParams(alpha=0.0, beta=-1.5, eta=0.0, hazard=0.1)  # beta > -1
    with pytest.raises(AssertionError):
        RandomTimeParams(alpha=0.0, beta=0.0, eta=-0.1, hazard=0.1)  # eta >= 0


def test_G_starts_at_one_and_stays_in_bounds():
    mp, paths = _paths()
    rt_params = RandomTimeParams(alpha=0.02, beta=0.01, eta=0.05, hazard=0.15)
    rt_paths = simulate_random_time(rt_params, paths, seed=1, lam=mp.lam)

    assert np.allclose(rt_paths.G[:, 0], 1.0), "G_0 = P(tau > 0) = 1 a.s."
    assert np.all(rt_paths.G > 0.0) and np.all(rt_paths.G <= 1.0 + 1e-8)
    assert np.all(rt_paths.G_tilde > 0.0) and np.all(rt_paths.G_tilde <= 1.0 + 1e-8)
    assert 0.0 <= rt_paths.clip_fraction <= 1.0


def test_D_pF_is_continuous_nondecreasing():
    """D^{p,F} is modelled as a continuous increasing process with constant
    rate h + eta*lam (module docstring, eq. 2.10); check monotonicity and
    that no jump exceeds one Euler step of that rate."""
    mp, paths = _paths()
    rt_params = RandomTimeParams(alpha=0.0, beta=0.0, eta=0.2, hazard=0.15)
    rt_paths = simulate_random_time(rt_params, paths, seed=2, lam=mp.lam)

    diffs = np.diff(rt_paths.D_pF, axis=1)
    assert np.all(diffs >= -1e-12), "D^{p,F} must be nondecreasing"
    expected_rate = rt_params.hazard + rt_params.eta * mp.lam
    assert np.allclose(diffs, expected_rate * paths.dt, atol=1e-10)


def test_default_indicator_is_zero_one_and_nondecreasing():
    mp, paths = _paths(n_sim=1000)
    rt_params = RandomTimeParams(alpha=0.02, beta=0.01, eta=0.1, hazard=0.3)
    rt_paths = simulate_random_time(rt_params, paths, seed=3, lam=mp.lam)

    D = rt_paths.default_indicator
    assert set(np.unique(D)).issubset({0.0, 1.0})
    assert np.all(np.diff(D, axis=1) >= 0.0), "default indicator must be nondecreasing"
    # some but not all paths should default over the horizon at this hazard level
    frac_defaulted = D[:, -1].mean()
    assert 0.0 < frac_defaulted < 1.0


def test_eta_zero_recovers_no_jump_in_G_tilde_minus_G():
    """Under eta=0 (Assumption 4.7, tau avoids F-stopping times), Gtilde=G
    identically since the only jump mechanism in Gtilde (the eta-linked jump
    at a Poisson arrival) is switched off."""
    mp, paths = _paths()
    rt_params = RandomTimeParams(alpha=0.02, beta=0.01, eta=0.0, hazard=0.15)
    rt_paths = simulate_random_time(rt_params, paths, seed=4, lam=mp.lam)
    assert np.allclose(rt_paths.G, rt_paths.G_tilde)
