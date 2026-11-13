"""Pure-synthetic tests for src/bsde.py (constrained-BSDE Esscher price
bounds, Section 4). No market data is used anywhere in this file."""
from __future__ import annotations

import numpy as np

from src.model import ModelParams, simulate_paths
from src.random_time import RandomTimeParams, simulate_random_time
from src.transforms import compute_transforms
from src import bsde


def _tiny_setup(n_sim: int, n_steps: int, seed: int = 0):
    mp = ModelParams(b=0.05, sigma=0.2, gamma=-0.1, lam=3.0, r=0.0)
    rt_params = RandomTimeParams(alpha=0.02, beta=0.01, eta=0.0, hazard=0.15)
    paths = simulate_paths(mp, t_horizon=1.0, n_steps=n_steps, n_sim=n_sim, s0=100.0, seed=seed)
    rt_paths = simulate_random_time(rt_params, paths, seed=seed + 1)
    return mp, rt_params, paths, rt_paths


def test_theta_of_psi_returns_finite_real_root():
    mp, rt_params, _, _ = _tiny_setup(n_sim=50, n_steps=20)
    theta = bsde.theta_of_psi(mp, rt_params, psi=0.0, zeta_mode="gamma_tilde")
    assert np.isfinite(theta)
    assert isinstance(theta, float)

    ref = bsde.reference_measure(mp, rt_params, zeta_mode="gamma_tilde")
    assert np.isfinite(ref["theta_0"])
    assert ref["H_phi"] == 1.0 + rt_params.beta
    assert ref["zeta_mode"] == "gamma_tilde"


def test_reference_dynamics_shapes_and_finite():
    mp, rt_params, paths, rt_paths = _tiny_setup(n_sim=200, n_steps=20)
    ref = bsde.reference_measure(mp, rt_params, zeta_mode="gamma_tilde")
    dyn = bsde.reference_dynamics(paths, rt_paths, mp, ref["theta_0"])

    n_sim, n_steps_plus1 = paths.W.shape
    n_steps = n_steps_plus1 - 1

    assert dyn["W0"].shape == (n_sim, n_steps_plus1)
    assert dyn["N_tilde0"].shape == (n_sim, n_steps_plus1)
    assert dyn["a"].shape == (n_sim, n_steps)
    for key in ("W0", "N_tilde0", "a"):
        assert np.all(np.isfinite(dyn[key])), f"{key} has NaN/Inf"
    assert np.all(dyn["a"] > 0) and np.all(dyn["a"] <= 1.0 + 1e-8)


def test_solve_truncated_bsde_runs_end_to_end_with_trivial_claim():
    mp, rt_params, paths, rt_paths = _tiny_setup(n_sim=300, n_steps=20)
    ref = bsde.reference_measure(mp, rt_params, zeta_mode="gamma_tilde")
    dyn = bsde.reference_dynamics(paths, rt_paths, mp, ref["theta_0"])
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)
    dyn["dN_G"] = np.diff(transforms.N_G, axis=1)

    n_sim, n_steps_plus1 = paths.S.shape
    xi = np.zeros(n_sim)

    result = bsde.solve_truncated_bsde(xi, paths, rt_paths, ref, dyn, mp, rt_params, n_trunc=2.0, regression_degree=2)

    assert result["Y_path"].shape == (n_sim, n_steps_plus1)
    assert result["Y0"].shape == (n_sim,)
    assert result["Z"].shape == (n_sim, n_steps_plus1 - 1)
    assert result["U"].shape == (n_sim, n_steps_plus1 - 1)
    assert result["V"].shape == (n_sim, n_steps_plus1 - 1)
    for key in ("Y_path", "Z", "U", "V"):
        assert np.all(np.isfinite(result[key])), f"{key} has NaN/Inf"
    # terminal condition xi=0 -> Y0 should stay close to 0 (driver is a
    # correction term scaled by dt, no explosive growth expected here).
    assert np.all(np.abs(result["Y0"]) < 10.0)


def test_monotonicity_direction_of_truncated_upper_bound():
    mp, rt_params, paths, rt_paths = _tiny_setup(n_sim=300, n_steps=20)
    xi = np.maximum(paths.S[:, -1] - 100.0, 0.0)

    bounds = bsde.pricing_bounds(
        xi, paths, rt_paths, mp, rt_params, zeta_mode="gamma_tilde",
        truncation_levels=(1.0, 2.0), regression_degree=2,
    )
    tol = 1e-2
    assert bounds["Y0_up"][1] >= bounds["Y0_up"][0] - tol
    assert bounds["Y0_inf"][1] <= bounds["Y0_inf"][0] + tol


def test_check_constraints_returns_fractions_in_unit_interval():
    mp, rt_params, paths, rt_paths = _tiny_setup(n_sim=300, n_steps=20)
    ref = bsde.reference_measure(mp, rt_params, zeta_mode="gamma_tilde")
    dyn = bsde.reference_dynamics(paths, rt_paths, mp, ref["theta_0"])
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)
    dyn["dN_G"] = np.diff(transforms.N_G, axis=1)

    xi = np.zeros(paths.S.shape[0])
    result = bsde.solve_truncated_bsde(xi, paths, rt_paths, ref, dyn, mp, rt_params, n_trunc=2.0, regression_degree=2)

    diag = bsde.check_constraints(
        result["Z"], result["U"], result["V"], mp.sigma, mp.gamma_tilde, rt_params.beta, bound="upper"
    )
    for key in ("frac_w_ok", "frac_v_ok"):
        assert 0.0 <= diag[key] <= 1.0
    assert np.isfinite(diag["mean_w"]) and np.isfinite(diag["mean_v"])

    diag_lower = bsde.check_constraints(
        result["Z"], result["U"], result["V"], mp.sigma, mp.gamma_tilde, rt_params.beta, bound="lower"
    )
    for key in ("frac_w_ok", "frac_v_ok"):
        assert 0.0 <= diag_lower[key] <= 1.0
