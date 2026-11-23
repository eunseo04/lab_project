#!/usr/bin/env bash
set -euo pipefail
ACD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ACD_ROOT"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "This lock was verified with Python 3.10; run setup with Python 3.10."'
python3 scripts/fetch_sources.py
python3 -m venv --without-pip .venv
python3 -m pip --python .venv install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
.venv/bin/python scripts/prepare_sources.py
.venv/bin/python -m pip check
./run.sh doctor
