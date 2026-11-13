# Repository guide for coding agents

This repository reproduces two numerical experiments from the paper "Novel
Esscher concepts for informational risks and valuation in market models under
random horizon": the order-two Esscher martingale density with
degree-of-freedom for a stopped jump-diffusion model under a progressively
enlarged filtration (Section 3), and the constrained-BSDE Esscher price
bounds for vulnerable claims (Section 4). Both experiments are purely
numerical/Monte Carlo verifications of the paper's theorems; no market data
or API key is used anywhere in this repository.

## Layout

- `src/` — flat library modules, each docstring citing the paper section or
  equation it implements:
  - `model.py` — jump-diffusion market model (S, F), equation (2.1)
  - `random_time.py` — random horizon tau parametrized by (alpha, beta, eta,
    D^{p,F}), Lemma 2.4, and its Monte Carlo simulation
  - `transforms.py` — G-martingale transform T(M) and N^G, equations (2.6)-(2.7)
  - `esscher.py` — Esscher density Z^G of order two with degree-of-freedom,
    Theorem 3.3, Corollary 3.5, Theorem 3.6
  - `bsde.py` — constrained BSDE price bounds for vulnerable claims, Section 4
  - `config.py` — every experiment parameter as a module-level UPPER_CASE
    constant with a comment stating its meaning, unit and source
- `exp/` — `exp1.py` and `exp2.py`, run as `python -m exp.exp1` /
  `python -m exp.exp2` (add `--full` for the paper-scale Monte Carlo size)
- `tests/` — pytest unit tests for every `src/` module, run on small synthetic
  Monte Carlo samples
- `results/` — generated tables (`*.csv`), plots (`*.png`), run metadata
  (`*.json`) and `RESULTS.md` with the comparison against the paper and the
  "Deviations from the paper" / "Not reproduced" sections
- `data/` — empty package kept for layout consistency; no market data or
  synthetic-data generator is needed because both experiments simulate the
  model directly inside `src/`

## Commands

```bash
pip install -r requirements.txt
python -m pytest                 # all unit tests, synthetic fixtures only
python -m exp.exp1                # exp_1, reduced Monte Carlo size
python -m exp.exp2                # exp_2, reduced Monte Carlo size
python -m exp.exp1 --full         # exp_1, paper-scale Monte Carlo size
python -m exp.exp2 --full         # exp_2, paper-scale Monte Carlo size
bash reproduce.sh                 # tests + both experiments
```

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  constants; no classes or parameter dicts. A reduced and a paper-scale value
  are kept side by side (e.g. `EXP1_N_SIM_RUN` / `EXP1_N_SIM_PAPER`) and the
  `--full` flag of each experiment script switches to the paper-scale value.
- `data/` is reserved for market-data loading and synthetic-data generation
  code; this repository needs neither, since both experiments simulate the
  model of equation (2.1) directly through `src/model.py` and
  `src/random_time.py`.
- Running `python -m exp.exp1` / `python -m exp.exp2` overwrites the files
  they produce in `results/`. After a rerun, check the numbers and verdicts
  quoted in `results/RESULTS.md` against the regenerated files and update
  them by hand if they changed.
- No API key or `.env` file is needed anywhere in this repository.
