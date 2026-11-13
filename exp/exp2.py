"""Experiment 2: constrained-BSDE (CBSDE) Esscher price bounds for a
vulnerable claim, Section 4 of the paper, under Assumption 4.7 (tau avoids
F-stopping times: eta=0) and r=0 (Assumption 4.1 discounted setting).

Simulates the market model and the random horizon, builds a vulnerable
European call payoff, computes the truncated upper/lower price bounds
Y_up^(n) / Y_inf^(n) for a family of truncation levels n (Theorem 4.9),
checks their monotone convergence (Theorem 4.10) and the CBSDE constraint
set (4.40), and writes the results to `results/`.
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
from src.model import ModelParams, SimulatedPaths, simulate_paths
from src.random_time import RandomTimeParams, RandomTimePaths, simulate_random_time
import src.transforms  # noqa: F401  (imported per spec; used indirectly via src.bsde)
import src.esscher  # noqa: F401  (imported per spec; used indirectly via src.bsde)
import src.bsde as bsde

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_claim(
    paths: SimulatedPaths, rt_paths: RandomTimePaths, strike: float, recovery_fraction: float
) -> np.ndarray:
    """Vulnerable European call payoff xi in L^0(F_{T and tau}): if the name
    survives to T, pay (S_T - K)^+; otherwise pay a flat recovery fraction
    of the strike at default. The precise payoff family is "not specified
    in the paper beyond xi in L^0(F_{T and tau})"; this flat-recovery
    vulnerable call is a representative, documented choice."""
    n_sim, n_steps_plus1 = paths.S.shape
    n_steps = n_steps_plus1 - 1
    stop_idx = np.minimum(rt_paths.tau_index, n_steps)
    S_stop = paths.S[np.arange(n_sim), stop_idx]
    survived = rt_paths.default_indicator[:, -1] < 0.5
    xi = np.where(survived, np.maximum(S_stop - strike, 0.0), recovery_fraction * strike)
    return xi


def main(full: bool = False) -> dict:
    start = time.time()
    try:
        mp = ModelParams(b=config.B_DRIFT, sigma=config.SIGMA, gamma=config.GAMMA_JUMP, lam=config.LAMBDA_POIS, r=0.0)
        rt_params = RandomTimeParams(
            alpha=config.ALPHA_PARAM, beta=config.BETA_PARAM, eta=0.0, hazard=config.HAZARD_RATE
        )

        n_sim = config.EXP2_N_SIM_PAPER if full else config.EXP2_N_SIM_RUN
        n_steps = config.EXP2_N_STEPS

        paths = simulate_paths(mp, config.T_HORIZON, n_steps, n_sim, config.S0, seed=config.EXP2_SEED)
        rt_paths = simulate_random_time(rt_params, paths, seed=config.EXP2_SEED + 1, lam=mp.lam)

        xi = build_claim(paths, rt_paths, config.EXP2_STRIKE, config.EXP2_RECOVERY_FRACTION)

        bounds = bsde.pricing_bounds(
            xi, paths, rt_paths, mp, rt_params, config.ZETA_MODE,
            config.EXP2_TRUNCATION_LEVELS, config.EXP2_REGRESSION_DEGREE,
        )

        diag_upper = bsde.check_constraints(
            bounds["Z_upper_final"], bounds["U_upper_final"], bounds["V_upper_final"],
            mp.sigma, mp.gamma_tilde, rt_params.beta, bound="upper",
        )
        diag_lower = bsde.check_constraints(
            bounds["Z_lower_final"], bounds["U_lower_final"], bounds["V_lower_final"],
            mp.sigma, mp.gamma_tilde, rt_params.beta, bound="lower",
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "truncation_level": bounds["truncation_levels"],
            "Y0_up": bounds["Y0_up"],
            "Y0_inf": bounds["Y0_inf"],
        })
        df.to_csv(RESULTS_DIR / "exp2_pricing_bounds.csv", index=False)

        elapsed = time.time() - start
        summary = {
            "n_sim": n_sim,
            "n_steps": n_steps,
            "clip_fraction": rt_paths.clip_fraction,
            "final_interval": bounds["final_interval"],
            "upper_monotone": bounds["upper_monotone"],
            "lower_monotone": bounds["lower_monotone"],
            "constraint_diagnostics_upper": diag_upper,
            "constraint_diagnostics_lower": diag_lower,
            "elapsed_seconds": elapsed,
        }
        with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bounds["truncation_levels"], bounds["Y0_up"], marker="o", label="Y0_up(n)")
        ax.plot(bounds["truncation_levels"], bounds["Y0_inf"], marker="s", label="Y0_inf(n)")
        ax.set_xscale("log")
        ax.set_xlabel("truncation level n (log scale)")
        ax.set_ylabel("price bound at t=0")
        ax.set_title("Esscher CBSDE price bounds: convergence in truncation level n")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "exp2_convergence.png", dpi=150)
        plt.close(fig)

        print("exp_2: constrained-BSDE Esscher price bounds")
        print(f"  n_sim={n_sim}, n_steps={n_steps}, clip_fraction={rt_paths.clip_fraction:.4f}")
        print(f"  truncation levels: {bounds['truncation_levels']}")
        print(f"  Y0_up: {bounds['Y0_up']}")
        print(f"  Y0_inf: {bounds['Y0_inf']}")
        print(f"  final interval [Y0_inf, Y0_up] = {bounds['final_interval']}")
        print(f"  upper_monotone={bounds['upper_monotone']}, lower_monotone={bounds['lower_monotone']}")
        print(f"  elapsed_seconds={elapsed:.2f}")

        return summary
    except Exception as exc:
        print(f"exp_2 FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp_2: constrained-BSDE Esscher price bounds")
    parser.add_argument("--full", action="store_true", help="run at paper scale (EXP2_N_SIM_PAPER)")
    args = parser.parse_args()
    main(full=args.full)
