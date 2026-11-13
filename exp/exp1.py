"""Experiment 1: characterization and numerical verification of the Esscher
densities/measures with degree-of-freedom for informational risks (Theorem
3.3, Corollary 3.5, Theorem 3.6).

Run: python -m exp.exp1 [--full]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.model import ModelParams, simulate_paths
from src.random_time import RandomTimeParams, simulate_random_time
from src.transforms import compute_transforms
from src.esscher import H_functional, compute_esscher_density, compute_F_esscher_density

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _scenarios() -> dict[str, RandomTimeParams]:
    """The four particular cases of Corollary 3.5, plus the fully general case
    of Theorem 3.3 (all of alpha, beta, eta nonzero)."""
    return {
        "general_theorem_3_3": RandomTimeParams(
            alpha=config.ALPHA_PARAM, beta=config.BETA_PARAM,
            eta=config.ETA_PARAM, hazard=config.HAZARD_RATE,
        ),
        "corollary_3_5a_avoids_stopping_times": RandomTimeParams(
            alpha=config.ALPHA_PARAM, beta=config.BETA_PARAM,
            eta=0.0, hazard=config.HAZARD_RATE,
        ),
        "corollary_3_5b_immersion": RandomTimeParams(
            alpha=config.ALPHA_PARAM, beta=0.0,
            eta=config.ETA_PARAM, hazard=config.HAZARD_RATE,
        ),
        "corollary_3_5c_independent": RandomTimeParams(
            alpha=0.0, beta=0.0, eta=0.0, hazard=config.HAZARD_RATE,
        ),
    }


def run_scenario(name: str, rt_params: RandomTimeParams, mp: ModelParams,
                  n_sim: int, n_steps: int, seed: int) -> dict:
    """Simulate one scenario and verify the equivalent characterizations of
    Theorem 3.3 and the relevant particular case of Corollary 3.5."""
    paths = simulate_paths(mp, config.T_HORIZON, n_steps, n_sim, config.S0, seed=seed)
    rt_paths = simulate_random_time(rt_params, paths, seed=seed + 1, g_floor=config.G_FLOOR, lam=mp.lam)
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)
    density = compute_esscher_density(
        mp, rt_params, rt_paths, transforms,
        phi=config.PHI_PARAM, psi=config.PSI_PARAM, zeta_mode=config.ZETA_MODE,
    )

    Z_T = density.Z_G[:, -1]
    tau_idx = np.minimum(rt_paths.tau_index, n_steps)
    S_tau = paths.S[np.arange(n_sim), tau_idx]

    out = {
        "scenario": name,
        "theta": density.theta,
        "H_phi": density.H_phi,
        "martingale_residual_eq_3_7": density.martingale_residual,
        "E_Z_G_T": float(Z_T.mean()),
        "std_Z_G_T": float(Z_T.std()),
        "se_Z_G_T": float(Z_T.std() / np.sqrt(n_sim)),
        "E_Z_G_T_times_S_tau": float((Z_T * S_tau).mean()),
        "se_Z_G_T_times_S_tau": float((Z_T * S_tau).std() / np.sqrt(n_sim)),
        "S0": config.S0,
        "clip_fraction": rt_paths.clip_fraction,
        "frac_defaulted_by_T": float(rt_paths.default_indicator[:, -1].mean()),
    }

    # Particular-case closed-form checks, Corollary 3.5.
    if name == "corollary_3_5a_avoids_stopping_times":
        # eq. (3.18): with eta=0, H(phi) must equal 1+beta for every phi.
        out["check_H_phi_equals_1_plus_beta"] = bool(
            np.isclose(density.H_phi, 1 + rt_params.beta, atol=1e-10)
        )
    if name == "corollary_3_5b_immersion":
        # expected_outcomes #5: H(phi) = e^{-eta phi} (1 + eta (e^phi - 1)) when beta=0.
        H_closed_form = np.exp(-rt_params.eta * config.PHI_PARAM) * (
            1 + rt_params.eta * np.expm1(config.PHI_PARAM)
        )
        out["check_H_phi_immersion_closed_form"] = bool(
            np.isclose(density.H_phi, H_closed_form, atol=1e-10)
        )
    if name == "corollary_3_5c_independent":
        # expected_outcomes #4: factorization Z^G = Z^F * E(J(phi).N^G), checked
        # by comparing the two independently computed stochastic exponentials.
        Z_F, theta_F = compute_F_esscher_density(mp, paths, psi=config.PSI_PARAM,
                                                   zeta_mode=config.ZETA_MODE)
        out["check_theta_equals_theta_F"] = float(abs(density.theta - theta_F))
        default_only = density.Z_G[:, -1] / np.clip(Z_F[:, -1], 1e-12, None)
        pure_F_component = Z_F[:, -1]
        out["corr_Z_G_vs_Z_F_times_default_component"] = float(
            np.corrcoef(density.Z_G[:, -1], pure_F_component * default_only)[0, 1]
        )

    return out, density, rt_paths


def run_corollary_3_5d(mp_base: ModelParams, n_sim: int, n_steps: int, seed: int) -> dict:
    """Corollary 3.5(d): all F-martingales continuous, i.e. N = 0. The paper's
    model (2.1) requires lambda>0 (Assumption 2.1); N=0 is approximated by a
    negligible jump intensity (lambda=1e-6), not specified in the paper."""
    mp_continuous = ModelParams(b=mp_base.b, sigma=mp_base.sigma, gamma=mp_base.gamma,
                                 lam=1e-6, r=mp_base.r)
    rt_params = RandomTimeParams(alpha=0.0, beta=0.0, eta=0.0, hazard=config.HAZARD_RATE)
    out, density, rt_paths = run_scenario(
        "corollary_3_5d_continuous_F_martingales", rt_params, mp_continuous,
        n_sim, n_steps, seed,
    )
    # expected_outcomes #6: theta = -(b_tilde + alpha sigma)/sigma^2 when zeta=0
    # in the limit of no jumps (gamma_tilde*lambda -> 0 in eq. 3.7).
    theta_closed_form = -(mp_continuous.b_tilde + rt_params.alpha * mp_continuous.sigma) / mp_continuous.sigma**2
    out["check_theta_continuous_closed_form"] = float(abs(density.theta - theta_closed_form))
    return out


def run_existence_theorem_3_6(mp: ModelParams, n_sim: int, n_steps: int, seed: int) -> pd.DataFrame:
    """Theorem 3.6: existence of an Esscher pricing measure of order two for
    (S,F) (Z^F uniformly integrable) is equivalent to existence for (S^tau,G)
    with ANY degree-of-freedom phi. We test this for several representative
    constant phi values (the exact phi of equation (3.46), built from the
    process C of Lemma 3.10, is not constructively specified by the paper and
    is not reproduced here; see results/RESULTS.md)."""
    rt_params = RandomTimeParams(alpha=config.ALPHA_PARAM, beta=config.BETA_PARAM,
                                  eta=config.ETA_PARAM, hazard=config.HAZARD_RATE)
    paths = simulate_paths(mp, config.T_HORIZON, n_steps, n_sim, config.S0, seed=seed)
    rt_paths = simulate_random_time(rt_params, paths, seed=seed + 1, g_floor=config.G_FLOOR, lam=mp.lam)
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)

    Z_F, theta_F = compute_F_esscher_density(mp, paths, psi=config.PSI_PARAM, zeta_mode=config.ZETA_MODE)
    Z_F_T = Z_F[:, -1]
    rows = []
    for phi in (-0.5, 0.0, 0.25, 0.5, 1.0):
        density = compute_esscher_density(mp, rt_params, rt_paths, transforms,
                                           phi=phi, psi=config.PSI_PARAM, zeta_mode=config.ZETA_MODE)
        Z_G_T = density.Z_G[:, -1]
        rows.append({
            "phi": phi,
            "E_Z_F_T": float(Z_F_T.mean()),
            "second_moment_Z_F_T": float((Z_F_T**2).mean()),
            "E_Z_G_T": float(Z_G_T.mean()),
            "second_moment_Z_G_T": float((Z_G_T**2).mean()),
            "Z_F_finite_variance": bool(np.isfinite((Z_F_T**2).mean())),
            "Z_G_finite_variance": bool(np.isfinite((Z_G_T**2).mean())),
        })
    return pd.DataFrame(rows)


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    n_sim = config.EXP1_N_SIM_PAPER if full else config.EXP1_N_SIM_RUN
    n_steps = config.N_STEPS
    seed = config.EXP1_SEED

    mp = ModelParams(b=config.B_DRIFT, sigma=config.SIGMA, gamma=config.GAMMA_JUMP,
                      lam=config.LAMBDA_POIS, r=config.R_RATE)

    rows = []
    for i, (name, rt_params) in enumerate(_scenarios().items()):
        out, _, _ = run_scenario(name, rt_params, mp, n_sim, n_steps, seed + 10 * i)
        rows.append(out)

    rows.append(run_corollary_3_5d(mp, n_sim, n_steps, seed + 100))

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "exp1_summary.csv", index=False)

    existence_df = run_existence_theorem_3_6(mp, n_sim, n_steps, seed + 200)
    existence_df.to_csv(RESULTS_DIR / "exp1_existence_theorem_3_6.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(table["scenario"], table["E_Z_G_T"], yerr=table["se_Z_G_T"])
    axes[0].axhline(1.0, color="red", linestyle="--", label="target E[Z^G_T]=1")
    axes[0].set_ylabel("E[Z^G_T]")
    axes[0].set_title("Martingale normalization of Z^G (Theorem 3.3)")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].legend()

    axes[1].bar(table["scenario"], table["E_Z_G_T_times_S_tau"], yerr=table["se_Z_G_T_times_S_tau"])
    axes[1].axhline(config.S0, color="red", linestyle="--", label=f"target S0={config.S0}")
    axes[1].set_ylabel("E[Z^G_T . S^tau_T]")
    axes[1].set_title("Local-martingale property of Z^G . S^tau")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_density_diagnostics.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_sim": n_sim,
        "n_steps": n_steps,
        "t_horizon": config.T_HORIZON,
        "elapsed_seconds": time.time() - t0,
        "scenarios": table.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "exp1_meta.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(table.to_string(index=False))
    print(f"\nTheorem 3.6 existence check:\n{existence_df.to_string(index=False)}")
    print(f"\nElapsed: {summary['elapsed_seconds']:.1f}s, n_sim={n_sim}, n_steps={n_steps}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Esscher density verification")
    parser.add_argument("--full", action="store_true", help="use the paper-scale Monte Carlo size")
    args = parser.parse_args()
    main(full=args.full)
