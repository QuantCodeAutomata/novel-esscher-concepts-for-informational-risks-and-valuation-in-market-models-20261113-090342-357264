"""Esscher martingale density Z^G of order-two with degree-of-freedom phi for
the stopped model (S^tau, G), Theorem 3.3 and Corollary 3.5.

Implements, in this order, equation (3.5) functionals H(phi), J(phi);
the martingale condition (3.7) solved for theta; and the density (3.8).

# Custom implementation (paper Theorem 3.3, equations 3.5, 3.7, 3.8): no
# library provides this novel two-parameter Esscher-with-degree-of-freedom
# construction for progressively enlarged filtrations; pyportfolioopt and
# skfolio cover portfolio-level Esscher/exponential tilting only for static
# return distributions, not this filtration-enlargement martingale-density
# theorem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from src.model import ModelParams
from src.random_time import RandomTimeParams, RandomTimePaths
from src.transforms import GMartingaleTransforms


def H_functional(phi: float, eta: float, beta: float) -> float:
    """H(phi) = (eta(e^phi-1)+1+beta) exp(-phi eta/(1+beta)), equation (3.5)."""
    return (eta * np.expm1(phi) + 1 + beta) * np.exp(-phi * eta / (1 + beta))


def martingale_condition(
    theta: float,
    b_tilde: float,
    alpha: float,
    sigma: float,
    lam: float,
    gamma_tilde: float,
    H_phi: float,
    zeta: float,
    psi: float,
) -> float:
    """Right-hand side of equation (3.7); its root in theta is theta(psi,phi)."""
    return b_tilde + alpha * sigma + theta * sigma**2 + lam * gamma_tilde * (
        H_phi * np.exp(theta * zeta + psi * zeta**2) - 1.0
    )


def solve_theta(
    mp: ModelParams, alpha: float, H_phi: float, zeta: float, psi: float,
    bracket: tuple[float, float] = (-50.0, 50.0),
) -> float:
    """Unique root theta(psi,phi) of the martingale condition (3.7)."""
    lo, hi = bracket
    f = lambda th: martingale_condition(
        th, mp.b_tilde, alpha, mp.sigma, mp.lam, mp.gamma_tilde, H_phi, zeta, psi
    )
    f_lo, f_hi = f(lo), f(hi)
    assert f_lo * f_hi < 0, (
        "martingale condition (3.7) has no sign change on the search bracket; "
        f"f({lo})={f_lo}, f({hi})={f_hi}"
    )
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)


@dataclass
class EsscherDensity:
    """Realized Esscher density Z^G on the simulated grid, shape (n_sim, n_steps+1)."""

    Z_G: np.ndarray
    theta: float
    H_phi: float
    J_phi: np.ndarray       # (n_sim, n_steps+1) process J(phi)_t, equation (3.5)
    martingale_residual: float  # |LHS-RHS| of (3.7) at the solved theta, should be ~0


def compute_esscher_density(
    mp: ModelParams,
    rt_params: RandomTimeParams,
    rt_paths: RandomTimePaths,
    transforms: GMartingaleTransforms,
    phi: float,
    psi: float,
    zeta_mode: str,
) -> EsscherDensity:
    """Build Z^G via equation (3.8) for given degree-of-freedom phi and
    second-order parameter psi, after solving theta from (3.7)."""
    zeta = mp.zeta(zeta_mode)
    H_phi = H_functional(phi, rt_params.eta, rt_params.beta)
    theta = solve_theta(mp, rt_params.alpha, H_phi, zeta, psi)
    residual = martingale_condition(
        theta, mp.b_tilde, rt_params.alpha, mp.sigma, mp.lam, mp.gamma_tilde,
        H_phi, zeta, psi,
    )

    G, G_tilde, D_oF = rt_paths.G, rt_paths.G_tilde, rt_paths.D_oF
    dD_oF = np.diff(D_oF, axis=1, prepend=D_oF[:, :1])
    # J(phi)_t = Gtilde (e^phi-1) / (G + e^phi * DeltaD^{o,F}), equation (3.5),
    # evaluated with the pre-jump G, Gtilde at each step (predictable integrand).
    J_phi = G_tilde * np.expm1(phi) / (G + np.exp(phi) * dD_oF)

    coeff_N_F = H_phi * np.exp(theta * zeta + psi * zeta**2) - 1.0 - rt_params.beta
    dT_W = np.diff(transforms.T_W, axis=1)
    dT_N_F = np.diff(transforms.T_N_F, axis=1)
    dN_G = np.diff(transforms.N_G, axis=1)

    dX = theta * mp.sigma * dT_W + coeff_N_F * dT_N_F
    # J(phi)_t (eq. 3.5) is an optional process built from Gtilde_t and the
    # jump Delta D^{o,F}_t landing AT t; the increment dN^G_k = N^G_k - N^G_{k-1}
    # lands at the SAME index k, so the stochastic integral J(phi).N^G pairs
    # J_phi[:,1:] (indices 1..n_steps) with dN_G (landings 1..n_steps), not the
    # pre-jump J_phi[:,:-1] (which would misalign J with the jump of the
    # previous step).
    dY = J_phi[:, 1:] * dN_G

    E_X = np.cumprod(np.concatenate([np.ones((dX.shape[0], 1)), 1 + dX], axis=1), axis=1)
    E_Y = np.cumprod(np.concatenate([np.ones((dY.shape[0], 1)), 1 + dY], axis=1), axis=1)
    Z_G = E_X * E_Y

    return EsscherDensity(Z_G=Z_G, theta=theta, H_phi=H_phi, J_phi=J_phi,
                           martingale_residual=float(residual))


def compute_F_esscher_density(
    mp: ModelParams, paths, psi: float, zeta_mode: str,
) -> tuple[np.ndarray, float]:
    """Order-two Esscher density Z^F for the un-enlarged model (S,F) alone
    (no random horizon), used as the reference existence object of
    Theorem 3.6: Z^F = E(theta_F sigma . W + (e^{theta_F zeta+psi zeta^2}-1).N^F).

    This is the degenerate case H(phi)=1 (eta=beta=0, i.e. no G-enlargement)
    and alpha=0 of the martingale condition (3.7), consistent with the
    classical F-Esscher transform of Remark 3.2.
    """
    zeta = mp.zeta(zeta_mode)
    theta_F = solve_theta(mp, alpha=0.0, H_phi=1.0, zeta=zeta, psi=psi)
    dW = paths.dW
    dN_F = np.diff(paths.N_F, axis=1)
    coeff = np.exp(theta_F * zeta + psi * zeta**2) - 1.0
    dX = theta_F * mp.sigma * dW + coeff * dN_F
    Z_F = np.cumprod(np.concatenate([np.ones((dX.shape[0], 1)), 1 + dX], axis=1), axis=1)
    return Z_F, theta_F
