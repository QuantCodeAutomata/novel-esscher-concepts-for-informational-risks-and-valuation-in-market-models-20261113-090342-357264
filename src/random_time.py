"""Random horizon tau, not necessarily an F-stopping time, parametrized
through F by the quadruple (alpha, beta, eta, D^{p,F}) of Lemma 2.4.

Rather than constructing a pathwise random time tau from an auxiliary
uniform/exponential variable and then deriving (alpha, beta, eta) from it
(which is possible only in special constructions and was left open by the
paper -- see data_requirements of exp_1: "Random time parametrization: the
quadruple (alpha, beta, eta, D^{p,F})"), we take the quadruple as the primitive
input, exactly as the paper's Lemma 2.4 represents it, and construct the
Azema supermartingales G, Gtilde and the default indicator consistently:

  m := G + D^{o,F}                                                 (2.4)
  m = m0 + G_- alpha . W + G_- beta . N^F,  -1 < beta <= G_-^{-1}   (2.9)
  D^{o,F} - D^{p,F} = G_- eta . N^F,  0 <= eta <= 1+beta            (2.10)

D^{p,F} is modelled as a continuous increasing process with constant rate
h + eta*lambda (a predictable compensator with no jumps -- a case compatible
with every scenario of Corollary 3.5, since the atoms of D^{o,F} are
controlled only by eta, not by D^{p,F}). Because D^{o,F}-D^{p,F}=G_- eta . N^F
in (2.10) is built from the compensated, zero-mean martingale N^F=N-lambda.t,
D^{p,F} alone must already carry the full average instantaneous default rate
of this mechanism, namely h+eta*lambda (the eta-linked jumps of N arrive at
rate lambda and each triggers default with probability eta, contributing an
average rate eta*lambda in addition to the continuous hazard h); omitting the
eta*lambda term from D^{p,F} would leave D^{o,F} with a systematic negative
bias of that size, which would break the martingale property of Z^G whenever
eta>0.

The realized default indicator D_t=1_{t>=tau} is drawn by the classical Cox
(threshold) construction of a random time from a given Azema supermartingale
G (Jeanblanc-Yor-Chesney, "Mathematical Methods for Financial Markets", and
El Karoui-Jeanblanc-Jiao on random times): draw U ~ Uniform(0,1) independent
of F (one draw per simulated path) and set tau := inf{t : G_t <= U}. Since U
is independent of F_t and uniform, P(tau>t|F_t) = P(U<G_t|F_t) = G_t exactly,
so D is pathwise consistent with the G already built from Lemma 2.4 -- unlike
an independent hazard-rate draw, which would make D^{o,F} (the F-optional
projection of the REALIZED D) differ from the abstractly constructed G/D^{o,F}
whenever alpha or beta is nonzero, breaking the martingale property of N^G
(2.7) by a fixed amount that would not vanish as the time step shrinks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model import ModelParams, SimulatedPaths


@dataclass(frozen=True)
class RandomTimeParams:
    """Constant quadruple (alpha, beta, eta, hazard) of Lemma 2.4."""

    alpha: float
    beta: float
    eta: float
    hazard: float

    def __post_init__(self) -> None:
        assert -1 < self.beta, "Lemma 2.4 requires beta > -1"
        assert 0.0 <= self.eta, "Lemma 2.4 requires eta >= 0"
        assert self.hazard >= 0.0


@dataclass
class RandomTimePaths:
    """Discretized paths of the random-horizon objects, shape (n_sim, n_steps+1)
    unless stated otherwise."""

    G: np.ndarray            # G_t = P(tau > t | F_t), Azema supermartingale
    G_tilde: np.ndarray      # Gtilde_t = P(tau >= t | F_t)
    D_oF: np.ndarray         # D^{o,F}_t, F-optional projection of D_t=1{t>=tau}
    D_pF: np.ndarray         # D^{p,F}_t, F-predictable compensator (continuous)
    default_indicator: np.ndarray  # D_t = 1{t>=tau} actually realized on each path
    tau_index: np.ndarray    # (n_sim,) index of the default step, n_steps if no default
    clip_fraction: float     # fraction of (path, step) entries where the G floor was applied


def simulate_random_time(
    rt_params: RandomTimeParams,
    paths: SimulatedPaths,
    seed: int,
    g_floor: float = 1e-4,
    lam: float = 0.0,
) -> RandomTimePaths:
    """Build (G, Gtilde, D^{o,F}, D^{p,F}) and a pathwise realization of the
    default indicator D=1{t>=tau} consistent with Lemma 2.4.

    ``lam`` is the intensity of the driving Poisson process N of the market
    model (2.1); it is needed to give D^{p,F} its correct average rate
    h + eta*lam (see module docstring). Passing the default lam=0.0 recovers
    the pure-hazard case and is only valid when eta=0.
    """
    n_sim, n_steps_plus_1 = paths.W.shape
    n_steps = n_steps_plus_1 - 1
    dt = paths.dt
    rng = np.random.default_rng(seed)

    G = np.ones((n_sim, n_steps_plus_1))
    G_tilde = np.ones((n_sim, n_steps_plus_1))
    D_oF = np.zeros((n_sim, n_steps_plus_1))
    D_pF = np.zeros((n_sim, n_steps_plus_1))
    m = np.ones(n_sim)  # m0 = G0 = 1 (tau > 0 a.s.)
    n_clipped = [0]
    n_total = [0]

    alpha, beta, eta, hazard = rt_params.alpha, rt_params.beta, rt_params.eta, rt_params.hazard
    for k in range(n_steps):
        G_prev = G[:, k]
        dW_k = paths.dW[:, k]
        dN_k = paths.dN[:, k]
        dN_F_k = paths.N_F[:, k + 1] - paths.N_F[:, k]

        # D^{p,F} carries the full average default rate h + eta*lam, since
        # D^{o,F}-D^{p,F}=G_- eta . N^F (2.10) is driven by the zero-mean
        # martingale N^F and so contributes no drift of its own.
        dD_pF = (hazard + eta * lam) * dt
        D_pF[:, k + 1] = D_pF[:, k] + dD_pF

        dD_oF_minus_pF = G_prev * eta * dN_F_k
        D_oF[:, k + 1] = D_oF[:, k] + dD_pF + dD_oF_minus_pF

        dm = G_prev * alpha * dW_k + G_prev * beta * dN_F_k
        m = m + dm
        G_new = m - D_oF[:, k + 1]
        # Numerical safeguard: the Euler discretization of the BMO martingale
        # m can occasionally push G outside (0,1]; clip and record how often.
        clipped = (G_new < g_floor) | (G_new > 1.0)
        n_clipped[0] += int(np.sum(clipped))
        n_total[0] += G_new.size
        G_new = np.clip(G_new, g_floor, 1.0)
        G[:, k + 1] = G_new
        # Gtilde = G + jump of D^{o,F} at this step (the eta-driven jump only;
        # the continuous hazard part never jumps).
        G_tilde[:, k + 1] = np.clip(G_new + G_prev * eta * dN_k, g_floor, 1.0)

    # Cox (threshold) construction of tau from the already built G: draw one
    # U ~ Uniform(0,1) per path, independent of F, and set
    # tau := inf{t : G_t <= U}, so that P(tau>t|F_t) = G_t exactly (module
    # docstring). This makes the realized default indicator D pathwise
    # consistent with G/D^{o,F}, which an independent hazard-rate draw is not
    # whenever alpha or beta is nonzero.
    u = rng.uniform(size=n_sim)
    crossed = G <= u[:, None]
    crossed[:, 0] = False  # tau > 0 a.s. (G_0 = 1 > U a.s.)
    has_crossed = crossed.any(axis=1)
    # n_steps+1 (strictly beyond the grid) is the "no default on [0,T]" sentinel,
    # so it never collides with "defaulted exactly at the last grid point" (n_steps)
    # in the ">=" comparison below.
    tau_index = np.where(has_crossed, crossed.argmax(axis=1), n_steps + 1)
    default_indicator = (np.arange(n_steps_plus_1)[None, :] >= tau_index[:, None]).astype(float)
    tau_index = np.minimum(tau_index, n_steps)

    return RandomTimePaths(
        G=G, G_tilde=G_tilde, D_oF=D_oF, D_pF=D_pF,
        default_indicator=default_indicator, tau_index=tau_index,
        clip_fraction=n_clipped[0] / max(n_total[0], 1),
    )
