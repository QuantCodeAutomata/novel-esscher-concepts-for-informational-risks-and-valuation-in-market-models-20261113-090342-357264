#!/usr/bin/env bash
# Reproduce every test and experiment of this repository.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python -m pytest -q

python -m exp.exp1
python -m exp.exp2
