# Novel Esscher concepts for informational risks and valuation in market models under random horizon

This repository reproduces two numerical experiments from the paper
*"Novel Esscher concepts for informational risks and valuation in market
models under random horizon"*.

No external market data is used: the paper is a theoretical study and its
results are verified numerically by Monte Carlo simulation of the jump-diffusion
market model (S, F) of equation (2.1) and of a random horizon τ parametrized
by the quadruple (α, β, η, D^{p,F}) of Lemma 2.4.

## Experiments

### exp_1 — Esscher densities of order two with degree-of-freedom (Section 3)

Implements and numerically verifies the order-two Esscher martingale density
Z^G with degree-of-freedom φ for the stopped model (Sᵗ, G) (Theorem 3.3), the
functionals H(φ), J(φ) and the martingale condition (3.7), the four particular
cases of Corollary 3.5 (τ avoids F-stopping times, immersion, independence,
continuous F-martingales), and the existence equivalence of Theorem 3.6.

Run:

```bash
python -m exp.exp1            # reduced Monte Carlo size (see Deviations)
python -m exp.exp1 --full     # paper-scale Monte Carlo size
```

Outputs: `results/exp1_summary.csv`, `results/exp1_existence_theorem_3_6.csv`,
`results/exp1_density_diagnostics.png`, `results/exp1_meta.json`.

### exp_2 — Constrained-BSDE Esscher price bounds for vulnerable claims (Section 4)

Implements the constrained BSDE (CBSDE) framework for the upper and lower
Esscher pricing bounds of a vulnerable claim, under Assumption 4.7 (τ avoids
F-stopping times, G = G̃) and Assumption 4.1 (r = 0). Solves the truncated
unconstrained BSDEs of Theorem 4.9 (upper bound) and Theorem 4.10 (lower
bound) by a least-squares Monte Carlo backward scheme, checks the monotone
convergence in the truncation level n, and checks the CBSDE constraint set
(4.40).

Run:

```bash
python -m exp.exp2            # reduced Monte Carlo size (see Deviations)
python -m exp.exp2 --full     # paper-scale Monte Carlo size
```

Outputs: `results/exp2_pricing_bounds.csv`, `results/exp2_convergence.png`,
`results/exp2_summary.json`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No `.env` file or API key is required: both experiments are purely synthetic.

## Tests

```bash
python -m pytest
```

All tests run on synthetic fixtures (small Monte Carlo samples); none depends
on external data or network access.

## Reproduce everything

```bash
bash reproduce.sh
```

## Repository layout

```
src/     model.py (jump-diffusion market model, eq. 2.1), random_time.py
         (random horizon τ, Lemma 2.4), transforms.py (G-martingale
         transform T(M) and N^G, eq. 2.6/2.7), esscher.py (Esscher density
         Z^G, Theorem 3.3), bsde.py (constrained BSDE price bounds,
         Section 4), config.py (all parameters)
exp/     exp1.py, exp2.py
tests/   unit tests for every src/ module
results/ generated tables, plots and RESULTS.md
data/    empty package; no market data or external synthetic-data
         generator is used (see results/RESULTS.md)
```

See `results/RESULTS.md` for the numerical results, their comparison with
the paper's expected outcomes, and all deviations from the paper.
