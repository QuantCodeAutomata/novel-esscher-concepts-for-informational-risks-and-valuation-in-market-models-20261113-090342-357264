"""Constrained BSDE (CBSDE) for Esscher price bounds of a vulnerable claim,
Section 4 of the paper, under Assumption 4.7 (tau avoids F-stopping times:
G = Gtilde, eta = 0, Delta D^{o,F} = 0) and r = 0 (Assumption 4.1 discounted
setting).

Implements, in this order: the reference (pivot) measure Q^(0) of equation
(4.37) (Theorem 4.9 / Corollary 4.8a), the reference dynamics (W^(0),
N-tilde^(0)) of equation (4.5), the truncated driver kappa_n of equation
(4.87), a least-squares Monte Carlo (LSMC) backward solver for the truncated
BSDE, the upper/lower price bounds Y_up / Y_inf of Theorem 4.10 (increasing/
decreasing limit in n of the truncated solutions), and a diagnostic check of
the constraint set defining the CBSDE (4.40).

Deviations from the paper (flagged explicitly where they occur below): the
paper states existence/representation theorems constructively via BMO
martingale theory but does not prescribe a numerical scheme; the LSMC
backward induction, the pooled (one coefficient per time step, not per path)
OLS estimator of (Z, U, V), and the tilted reference dynamics are simplified,
documented numerical proxies "not specified constructively by the paper".
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.model import ModelParams, SimulatedPaths
from src.random_time import RandomTimeParams, RandomTimePaths
from src.transforms import compute_transforms
from src.esscher import solve_theta


# ---------------------------------------------------------------------------
# (a) Reference (pivot) measure Q^(0), equation (4.37)
# ---------------------------------------------------------------------------
def reference_measure(mp: ModelParams, rt_params: RandomTimeParams, zeta_mode: str) -> Dict[str, float]:
    """theta_0 = theta(0, 0), equation (4.37): under Assumption 4.7 (eta=0),
    H(phi) = 1+beta identically in phi (Corollary 4.8a), so the reference
    measure is pinned down by solving (3.7) at psi=0 with H_phi = 1+beta.
    `zeta_mode` is added explicitly (beyond the minimal sketch) since it is
    required to evaluate ``mp.zeta(zeta_mode)`` -- ZETA_MODE is a project
    configuration choice (Remark 3.2), not a paper constant.
    """
    H_phi = 1.0 + rt_params.beta
    zeta = mp.zeta(zeta_mode)
    theta_0 = solve_theta(mp, rt_params.alpha, H_phi=H_phi, zeta=zeta, psi=0.0)
    return {"theta_0": theta_0, "H_phi": H_phi, "zeta_mode": zeta_mode}


def theta_of_psi(mp: ModelParams, rt_params: RandomTimeParams, psi: float, zeta_mode: str) -> float:
    """theta(psi) solving (3.7) at H_phi = 1+beta; under Assumption 4.7
    (Corollary 4.8a) theta depends on psi only, not on phi."""
    H_phi = 1.0 + rt_params.beta
    zeta = mp.zeta(zeta_mode)
    return solve_theta(mp, rt_params.alpha, H_phi=H_phi, zeta=zeta, psi=psi)


# ---------------------------------------------------------------------------
# (c) Reference dynamics W^(0), N-tilde^(0), equation (4.5), and a_t = dt/dA_t
# ---------------------------------------------------------------------------
def reference_dynamics(
    paths: SimulatedPaths, rt_paths: RandomTimePaths, mp: ModelParams, theta_0: float
) -> Dict[str, np.ndarray]:
    """Build the reference-measure Brownian motion W^(0) and the reference
    compensated jump process N-tilde^(0) (equation 4.5), plus the
    time-change ratio a_t = dt/dA_t with A_t = t + D^{p,F}_t (paragraph
    right after eq. 4.5: dA_t aggregates the "clock" of continuous time t
    and of the predictable default compensator).

    W^(0)_t = W_t - theta_0*sigma*t is the standard Girsanov shift of the
    Esscher tilt by theta_0 (eq. 4.5, continuous part).

    (simplified LSMC proxy, not specified constructively by the paper):
    N-tilde^(0), the Q^(0)-compensated driving Poisson process, should in
    general retilt via H(phi); since this function only receives theta_0
    (not rt_params/zeta_mode), we hard-code zeta=mp.gamma_tilde (Remark
    3.2's exponential transform, the project default ZETA_MODE) to retilt
    the intensity lambda -> lambda*exp(theta_0*gamma_tilde); the eta=0,
    H(phi)=1+beta normalization of Assumption 4.7 was already absorbed into
    theta_0 at calibration in `reference_measure` and is not reapplied here.
    """
    t_grid = paths.t_grid
    dt = paths.dt

    W0 = paths.W - theta_0 * mp.sigma * t_grid[None, :]

    zeta = mp.gamma_tilde
    lam_ref = mp.lam * np.exp(theta_0 * zeta)
    N_tilde0 = paths.N - lam_ref * t_grid[None, :]

    dD_pF = np.diff(rt_paths.D_pF, axis=1)
    denom = np.clip(dt + dD_pF, 1e-8, None)
    a = dt / denom

    return {"W0": W0, "N_tilde0": N_tilde0, "a": a}


# ---------------------------------------------------------------------------
# (d) Truncated driver kappa_n, equation (4.87)
# ---------------------------------------------------------------------------
def driver_upper(
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    delta_theta: Tuple[float, float],
    sigma: float,
    gamma_tilde: float,
    beta: float,
    G_minus: np.ndarray,
    n: float,
) -> np.ndarray:
    """Truncated driver kappa_n of equation (4.87).

    delta_theta = (theta(-n) - theta(0), theta(0) - theta(n)), precomputed
    by the caller (via `theta_of_psi`) to avoid repeated root-finding inside
    the Monte Carlo backward loop.
    """
    dtheta_minus, dtheta_plus = delta_theta
    w = gamma_tilde * z - sigma / (1.0 + beta) * u

    term1 = dtheta_minus * sigma * np.maximum(w, 0.0) * a / gamma_tilde
    term2 = dtheta_plus * sigma * np.minimum(w, 0.0) * a / gamma_tilde
    term3 = (np.exp(n) - 1.0) * np.maximum(v, 0.0) / G_minus * (1.0 - a)
    term4 = (1.0 - np.exp(-n)) * np.minimum(v, 0.0) / G_minus * (1.0 - a)
    return term1 + term2 + term3 + term4


# ---------------------------------------------------------------------------
# (e) LSMC backward solver for the truncated BSDE
# ---------------------------------------------------------------------------
def _poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Centered/scaled monomial features x, x^2, ..., x^degree of a 1-D
    array (degree >= 0, no constant column -- the caller adds its own
    intercept). Centering/scaling keeps the regression well conditioned
    (simplified LSMC proxy: the paper does not specify a basis)."""
    x = np.asarray(x, dtype=float)
    mu, sd = x.mean(), x.std()
    sd = sd if sd > 1e-12 else 1.0
    xs = (x - mu) / sd
    cols = [xs**k for k in range(1, degree + 1)]
    return np.column_stack(cols) if cols else np.zeros((x.shape[0], 0))


def _ridge_lstsq(design: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Standardized, Tikhonov-regularized least squares.

    (simplified LSMC proxy, not specified constructively by the paper): some
    design columns (e.g. dN_G_k, since defaults are rare events) have
    near-zero within-step variance, making plain OLS ill-conditioned: the
    fitted coefficient is numerically unidentified (huge, noisy), which then
    explodes through backward induction and the 1/G_minus, 1/gamma_tilde
    scalings of driver (4.87). A small ridge penalty on the standardized
    design (columns rescaled to unit std, coefficients un-scaled afterwards)
    is a standard numerical stabilization, leaving well-identified columns
    essentially untouched while bounding rare-event columns' coefficients.
    """
    n, p = design.shape
    col_std = design.std(axis=0)
    col_std[col_std < 1e-12] = 1.0
    col_std[0] = 1.0  # keep the intercept column (all ones) unscaled
    design_s = design / col_std[None, :]

    XtX = design_s.T @ design_s
    Xty = design_s.T @ y
    penalty = ridge * np.trace(XtX) / max(p, 1)
    coeffs_s = np.linalg.solve(XtX + penalty * np.eye(p), Xty)
    return coeffs_s / col_std


def solve_truncated_bsde(
    xi: np.ndarray,
    paths: SimulatedPaths,
    rt_paths: RandomTimePaths,
    ref: Dict[str, float],
    dyn: Dict[str, np.ndarray],
    mp: ModelParams,
    rt_params: RandomTimeParams,
    n_trunc: float,
    regression_degree: int = 2,
    driver_clip: float | None = None,
) -> Dict[str, np.ndarray]:
    """Backward induction LSMC solver for the truncated CBSDE with driver
    kappa_n (Theorem 4.9), stopped at each path's own tau.

    (simplified LSMC proxy, not specified constructively by the paper): at
    each step the martingale-representation triplet (Z, U, V) is estimated
    as a single *pooled* OLS coefficient (one number per time step, shared
    by all paths) of Y_{k+1} on [1, poly(S_k), poly(G_k), dW0_k,
    dN_tilde0_k, dN_G_k]; Z_k, U_k, V_k are the raw coefficients on dW0_k,
    dN_tilde0_k, dN_G_k (no further dt rescaling: since
    Y_{k+1} - E[Y_{k+1} | F_k] = Z_k dW0_k + U_k dNtilde0_k + V_k dN^G_k
    is the one-step martingale-representation identity underlying the
    BSDE of Theorem 4.9, and dW0_k, dN_tilde0_k are the raw, non-
    standardized increments, the OLS slope on dW0_k already estimates
    Z_k directly, exactly as the slopes on dN_tilde0_k, dN_G_k estimate
    U_k, V_k without rescaling). The polynomial-in-state features (degree
    `regression_degree`) only improve the conditional-expectation fit of
    Y_{k+1} (reduce omitted-variable bias); they do not enter Z, U, V
    themselves. Standard LSMC simplification of the BMO martingale
    representation of Theorem 4.9.

    `driver_clip` (not specified constructively by the paper, which gives
    no numerical scheme for (4.87)): a single pooled-regression pass gives
    a noisy estimate of (Z, U, V); as n grows, the truncation penalty
    kappa_n scales this noise by (e^n - 1) (jump branch) and by the
    theta(-n)/theta(n) spread (continuous branch), so an unclipped
    explicit Euler step can diverge with n instead of converging as in
    Theorem 4.9(c). Clipping the driver's per-step contribution to the
    CFL-type bound (range of xi) / T_horizon -- the rate at which the
    bounded terminal payoff can possibly move -- stabilizes the explicit
    scheme without altering kappa_n at states where the constraint is not
    (numerically) binding.
    """
    n_sim, n_steps_plus1 = paths.S.shape
    n_steps = n_steps_plus1 - 1
    dt = paths.dt

    zeta_mode = ref["zeta_mode"]
    theta_0 = ref["theta_0"]
    theta_minus_n = theta_of_psi(mp, rt_params, -n_trunc, zeta_mode)
    theta_plus_n = theta_of_psi(mp, rt_params, n_trunc, zeta_mode)
    delta_theta = (theta_minus_n - theta_0, theta_0 - theta_plus_n)

    W0 = dyn["W0"]
    N_tilde0 = dyn["N_tilde0"]
    a_arr = dyn["a"]
    dN_G = dyn["dN_G"]
    dW0 = np.diff(W0, axis=1)
    dN_tilde0 = np.diff(N_tilde0, axis=1)

    G_minus = np.concatenate([rt_paths.G[:, :1], rt_paths.G[:, :-1]], axis=1)
    tau_index = rt_paths.tau_index

    Y_path = np.zeros((n_sim, n_steps_plus1))
    Z = np.zeros((n_sim, n_steps))
    U = np.zeros((n_sim, n_steps))
    V = np.zeros((n_sim, n_steps))

    Y_path[:, n_steps] = xi

    for k in range(n_steps - 1, -1, -1):
        active = k < tau_index
        Y_next = Y_path[:, k + 1]

        dW0_k = dW0[:, k]
        dN_tilde0_k = dN_tilde0[:, k]
        dN_G_k = dN_G[:, k]

        state_feats = np.column_stack([
            _poly_features(paths.S[:, k], regression_degree),
            _poly_features(rt_paths.G[:, k], regression_degree),
        ])
        design = np.column_stack([np.ones(n_sim), state_feats, dW0_k, dN_tilde0_k, dN_G_k])

        idx_active = np.where(active)[0]
        if idx_active.size >= design.shape[1] + 1:
            coeffs = _ridge_lstsq(design[idx_active], Y_next[idx_active])
        else:
            coeffs = np.zeros(design.shape[1])
        z_k, u_k, v_k = coeffs[-3], coeffs[-2], coeffs[-1]
        Z[:, k], U[:, k], V[:, k] = z_k, u_k, v_k

        kappa = driver_upper(
            np.full(n_sim, z_k), np.full(n_sim, u_k), np.full(n_sim, v_k),
            a_arr[:, k], delta_theta, mp.sigma, mp.gamma_tilde, rt_params.beta,
            G_minus[:, k], n_trunc,
        )
        if driver_clip is not None:
            kappa = np.clip(kappa, -driver_clip, driver_clip)

        Y_k_active = Y_next + kappa * dt
        Y_path[:, k] = np.where(active, Y_k_active, Y_next)

    return {"Y0": Y_path[:, 0].copy(), "Y_path": Y_path, "Z": Z, "U": U, "V": V, "n_trunc": n_trunc}


# ---------------------------------------------------------------------------
# (f) Upper/lower price bounds, Theorem 4.10
# ---------------------------------------------------------------------------
def pricing_bounds(
    xi: np.ndarray,
    paths: SimulatedPaths,
    rt_paths: RandomTimePaths,
    mp: ModelParams,
    rt_params: RandomTimeParams,
    zeta_mode: str,
    truncation_levels: Tuple[float, ...],
    regression_degree: int,
) -> Dict[str, object]:
    """Truncated upper bound Y_up^(n) and lower bound Y_inf^(n) for every n
    in `truncation_levels` (Theorem 4.9), using Y_inf(xi) = -Y_up(-xi)
    (Theorem 4.10) to reuse the upper solver for the lower bound."""
    ref = reference_measure(mp, rt_params, zeta_mode)
    dyn = reference_dynamics(paths, rt_paths, mp, ref["theta_0"])
    transforms = compute_transforms(paths, rt_paths, alpha=rt_params.alpha, beta=rt_params.beta)
    dyn["dN_G"] = np.diff(transforms.N_G, axis=1)

    # CFL-type stabilization of the explicit LSMC Euler scheme (not a
    # numerical scheme given by the paper for (4.87)): the per-step driver
    # contribution to Y cannot exceed the rate at which the (bounded)
    # terminal payoff range could possibly move.
    T_horizon = paths.dt * (paths.S.shape[1] - 1)
    driver_clip = float(np.ptp(xi)) / T_horizon

    Y0_up = []
    Y0_inf = []
    last_upper = None
    last_lower = None
    for n in truncation_levels:
        res_up = solve_truncated_bsde(xi, paths, rt_paths, ref, dyn, mp, rt_params, n, regression_degree, driver_clip)
        res_lo = solve_truncated_bsde(-xi, paths, rt_paths, ref, dyn, mp, rt_params, n, regression_degree, driver_clip)

        Y0_up.append(float(np.mean(res_up["Y0"])))
        Y0_inf.append(float(np.mean(-res_lo["Y0"])))
        last_upper = res_up
        last_lower = res_lo

    Y0_up_arr = np.array(Y0_up)
    Y0_inf_arr = np.array(Y0_inf)
    tol = 1e-6
    upper_monotone = bool(np.all(np.diff(Y0_up_arr) >= -tol))
    lower_monotone = bool(np.all(np.diff(Y0_inf_arr) <= tol))

    return {
        "truncation_levels": list(truncation_levels),
        "Y0_up": Y0_up,
        "Y0_inf": Y0_inf,
        "upper_monotone": upper_monotone,
        "lower_monotone": lower_monotone,
        "final_interval": (Y0_inf[-1], Y0_up[-1]),
        "Z_upper_final": last_upper["Z"],
        "U_upper_final": last_upper["U"],
        "V_upper_final": last_upper["V"],
        # Y_inf = -Y_up(-xi): its representation triplet is minus that of
        # the auxiliary -xi solve (same linearity argument as Theorem 4.10).
        "Z_lower_final": -last_lower["Z"],
        "U_lower_final": -last_lower["U"],
        "V_lower_final": -last_lower["V"],
    }


# ---------------------------------------------------------------------------
# (g) Constraint diagnostic for the CBSDE (4.40)
# ---------------------------------------------------------------------------
def check_constraints(
    Z: np.ndarray, U: np.ndarray, V: np.ndarray, sigma: float, gamma_tilde: float, beta: float, bound: str
) -> Dict[str, float]:
    """Fraction of (path, step) pairs satisfying the CBSDE (4.40) constraint
    set: upper bound w = gamma_tilde*Z - sigma/(1+beta)*U >= 0 and V <= 0;
    lower bound the reversed signs.

    (simplified LSMC proxy, not specified constructively by the paper):
    `Z`, `U`, `V` are pooled per-time-step coefficients broadcast across all
    paths by `solve_truncated_bsde`, so no further per-path before-tau
    masking is applied here (it would not change the result); tau-stopping
    was already enforced when freezing Y past tau and restricting the
    regression to active paths.
    """
    w = gamma_tilde * Z - sigma / (1.0 + beta) * U
    if bound == "upper":
        w_ok = w >= -1e-6
        v_ok = V <= 1e-6
    elif bound == "lower":
        w_ok = w <= 1e-6
        v_ok = V >= -1e-6
    else:
        raise ValueError(f"unknown bound: {bound}")

    return {
        "frac_w_ok": float(np.mean(w_ok)),
        "frac_v_ok": float(np.mean(v_ok)),
        "mean_w": float(np.mean(w)),
        "mean_v": float(np.mean(V)),
    }
