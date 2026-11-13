"""Model and simulation parameters for the Esscher-with-degree-of-freedom
replication (Novel Esscher concepts for informational risks and valuation in
market models under random horizon).

No market data is used: the paper is purely theoretical and its experiments
are numerical verifications of theorems via Monte Carlo simulation of a
jump-diffusion model and of a random horizon parametrized as in Lemma 2.4.
Every variable below is a module-level constant with a comment on its
meaning, unit and source (paper section/equation, or "not specified in the
paper" when the paper leaves the value open and a representative value had
to be chosen, as explicitly allowed by the data_requirements of exp_1/exp_2).
"""

# ---------------------------------------------------------------------------
# Shared jump-diffusion market model (S, F), equation (2.1)
# ---------------------------------------------------------------------------
S0 = 100.0          # initial risky-asset price, in price units (not specified in the paper)
B_DRIFT = 0.05       # drift b of X=b t + sigma W + gamma N^F, annualized (not specified in the paper)
SIGMA = 0.20         # Brownian volatility sigma > 0, annualized (Assumption 2.1: sigma locally bounded and positive)
GAMMA_JUMP = -0.10   # jump size parameter gamma > -1 of the compound-Poisson part (Assumption 2.1: gamma > -1)
LAMBDA_POIS = 3.0    # Poisson intensity lambda > 0, jumps per year (Assumption 2.1: lambda locally bounded and positive)
R_RATE = 0.0         # risk-free rate r; set to 0 for exp_1 money-market numeraire (Section 2) and required by exp_2 (Assumption 4.1 discounted setting, r=0)

T_HORIZON = 1.0      # time horizon T < infinity, in years (paper requires T < infinity, value not specified)
N_STEPS = 252        # number of time steps on [0,T] for the Euler-Maruyama/thinning discretization (not specified in the paper)

# ---------------------------------------------------------------------------
# Random horizon tau parametrization (alpha, beta, eta, D^{p,F}), Lemma 2.4
# ---------------------------------------------------------------------------
ALPHA_PARAM = 0.02    # F-predictable coefficient alpha in m = m0 + G_- alpha . W + G_- beta . N^F (eq. 2.9); not specified, chosen small and constant so the Euler discretization of the BMO martingale m keeps G=m-D^{o,F} in (0,1) with high probability
BETA_PARAM = 0.01     # F-predictable coefficient beta, must satisfy -1 < beta <= G_-^{-1} (eq. 2.9); not specified, chosen constant and small
ETA_PARAM = 0.05      # coefficient eta in D^{o,F}-D^{p,F}=G_- eta . N^F, 0 <= eta <= 1+beta (eq. 2.10); not specified, constant
HAZARD_RATE = 0.15    # constant hazard rate h driving the continuous predictable compensator D^{p,F}_t = h t (not specified in the paper)
G_FLOOR = 1e-4        # numerical floor applied to the simulated survival process G (and Gtilde) so that the Euler discretization of the BMO martingale m=G+D^{o,F} (Lemma 2.4) stays a valid survival probability on every path; the fraction of paths requiring this safeguard is logged by the experiment

# Esscher parameters (theta solved from the martingale condition; psi, phi free)
PSI_PARAM = 0.02      # second-order Esscher parameter psi (order-two refinement), not specified in the paper
PHI_PARAM = 0.25      # degree-of-freedom parameter phi in I_loc^{exp}(N^G,G), not specified in the paper
ZETA_MODE = "gamma_tilde"  # which zeta in {gamma, gamma_tilde} is used in the unified notation of Remark 3.2; "gamma" selects the linear transform, "gamma_tilde" the exponential transform

# ---------------------------------------------------------------------------
# exp_1: Monte Carlo verification settings
# ---------------------------------------------------------------------------
EXP1_N_SIM_PAPER = 200_000   # paper-scale number of Monte Carlo paths for the martingale checks (not specified in the paper; chosen for statistical accuracy)
EXP1_N_SIM_RUN = 40_000      # reduced path count actually executed, to keep the full experiment suite within the available computation time
EXP1_SEED = 12345            # random seed for reproducibility (not specified in the paper)

# ---------------------------------------------------------------------------
# exp_2: constrained-BSDE (vulnerable claim) settings, Section 4
# ---------------------------------------------------------------------------
EXP2_N_SIM_PAPER = 100_000   # paper-scale number of simulated paths for the least-squares Monte Carlo BSDE solver (not specified in the paper)
EXP2_N_SIM_RUN = 20_000      # reduced path count actually executed to respect the available computation time
EXP2_N_STEPS = 60            # number of backward time steps for the BSDE discretization (not specified in the paper; smaller than N_STEPS for tractability of the regression)
EXP2_SEED = 2718

EXP2_STRIKE = 100.0          # strike K of the vulnerable European call payoff (not specified in the paper: "specific payoff structures are referenced but not detailed")
EXP2_RECOVERY_FRACTION = 0.40  # recovery rate paid at default, fraction of K (not specified in the paper)
EXP2_TRUNCATION_LEVELS = (1, 2, 4, 8, 16, 32, 64)  # truncation levels n used in Theorem 4.9/4.10 to build Psi_n, Phi_n and verify Y^{(n)} monotone convergence
EXP2_REGRESSION_DEGREE = 2    # polynomial regression degree used in the least-squares Monte Carlo BSDE backward induction (not specified in the paper)
