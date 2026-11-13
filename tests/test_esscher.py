"""Unit tests for src/esscher.py: the Esscher density Z^G of order-two with
degree-of-freedom phi, Theorem 3.3 / equations (3.5), (3.7), (3.8). Synthetic
inputs only, small Monte Carlo sizes."""
from __future__ import annotations

import numpy as np
import pytest

from src.model import ModelParams, simulate_paths
from src.random_time import RandomTimeParams, simulate_random_time
from src.transforms import compute_transforms
from src.esscher import (
    H_functional,
    martingale_condition,
    solve_theta,
    compute_esscher_density,
    compute_F_esscher_density,
)


def test_H_functional_reduces_to_one_plus_beta_when_eta_zero():
    """Equation (3.5): H(phi) = (eta(e^phi-1)+1+beta)e^{-phi eta/(1+beta)};
    at eta=0 this collapses to 1+beta for every phi (Corollary 3.5a)."""
    for phi in (-1.0, 0.0, 0.5, 2.0):
        assert np.isclose(H_functional(phi, eta=0.0, beta=0.3), 1.3)


def test_H_functional_immersion_closed_form():
    """Corollary 3.5(b), immersion (beta=0): H(phi) = e^{-eta phi}(1+eta(e^phi-1))."""
    eta, phi = 0.2, 0.7
    expected = np.exp(-eta * phi) * (1 + eta * np.expm1(phi))
    assert np.isclose(H_functional(phi, eta=eta, beta=0.0), expected)


def test_martingale_condition_root_matches_solve_theta():
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0)
    theta = solve_theta(mp, alpha=0.02, H_phi=1.01, zeta=mp.gamma_tilde, psi=0.02)
    residual = martingale_condition(
        theta, mp.b_tilde, 0.02, mp.sigma, mp.lam, mp.gamma_tilde, 1.01, mp.gamma_tilde, 0.02
    )
    assert abs(residual) < 1e-8


def _setup(n_sim=20_000, n_steps=80, seed=0):
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0)
    paths = simulate_paths(mp, t_horizon=1.0, n_steps=n_steps, n_sim=n_sim, s0=100.0, seed=seed)
    rt_params = RandomTimeParams(alpha=0.02, beta=0.01, eta=0.05, hazard=0.15)
    rt_paths = simulate_random_time(rt_params, paths, seed=seed + 1, lam=mp.lam)
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)
    return mp, rt_params, paths, rt_paths, transforms


def test_esscher_density_is_martingale_in_expectation():
    """E[Z^G_T] approx 1, the normalization check of Theorem 3.3 / exp_1
    expected_outcomes #1."""
    mp, rt_params, paths, rt_paths, transforms = _setup()
    density = compute_esscher_density(mp, rt_params, rt_paths, transforms, phi=0.25, psi=0.02, zeta_mode="gamma_tilde")
    Z_T = density.Z_G[:, -1]
    se = Z_T.std() / np.sqrt(Z_T.size)
    assert abs(Z_T.mean() - 1.0) < 5 * se
    assert abs(density.martingale_residual) < 1e-8


def test_esscher_density_times_S_preserves_S0_in_expectation():
    """E[Z^G_T . S^tau_T] approx S0, the local-martingale check of Theorem
    3.3 / exp_1 expected_outcomes #2."""
    mp, rt_params, paths, rt_paths, transforms = _setup()
    density = compute_esscher_density(mp, rt_params, rt_paths, transforms, phi=0.25, psi=0.02, zeta_mode="gamma_tilde")
    Z_T = density.Z_G[:, -1]
    n_steps = paths.S.shape[1] - 1
    tau_idx = np.minimum(rt_paths.tau_index, n_steps)
    S_tau = paths.S[np.arange(paths.S.shape[0]), tau_idx]
    val = Z_T * S_tau
    se = val.std() / np.sqrt(val.size)
    assert abs(val.mean() - 100.0) < 5 * se


def test_independent_case_theta_matches_F_only_esscher():
    """Corollary 3.5(c): when tau is independent of F (alpha=beta=eta=0),
    theta(psi,phi) must equal the plain F-Esscher theta_F (exp_1 expected
    outcome #4, factorization)."""
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0)
    n_sim, n_steps = 5000, 60
    paths = simulate_paths(mp, t_horizon=1.0, n_steps=n_steps, n_sim=n_sim, s0=100.0, seed=7)
    rt_params = RandomTimeParams(alpha=0.0, beta=0.0, eta=0.0, hazard=0.15)
    rt_paths = simulate_random_time(rt_params, paths, seed=8, lam=mp.lam)
    transforms = compute_transforms(paths, rt_paths, alpha=0.0, beta=0.0)

    density = compute_esscher_density(mp, rt_params, rt_paths, transforms, phi=0.25, psi=0.02, zeta_mode="gamma_tilde")
    _, theta_F = compute_F_esscher_density(mp, paths, psi=0.02, zeta_mode="gamma_tilde")
    assert abs(density.theta - theta_F) < 1e-8


def test_zeta_mode_affects_theta_value():
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0)
    theta_gamma = solve_theta(mp, alpha=0.02, H_phi=1.0, zeta=mp.zeta("gamma"), psi=0.0)
    theta_gt = solve_theta(mp, alpha=0.02, H_phi=1.0, zeta=mp.zeta("gamma_tilde"), psi=0.0)
    assert not np.isclose(theta_gamma, theta_gt)
