#!/usr/bin/env bash
set -euo pipefail
RESEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RESEARCH_ROOT"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for the recorded environments"'
if [[ ! -x .venv/bin/python ]]; then
  ./setup.sh
fi
python3 scripts/fetch_research_sources.py
if [[ ! -d runtime/CybORG ]]; then
  .venv/bin/python scripts/prepare_sources.py
fi
if [[ ! -x .model-venv/bin/python ]]; then
  python3 -m venv --without-pip .model-venv
fi
python3 -m pip --python .model-venv install -r requirements-model.lock.txt
python3 -m pip --python .model-venv check
if [[ "${1:-}" == "--download" ]]; then
  .model-venv/bin/python scripts/download_research_models.py
fi
printf '%s\n' 'Ready. Start the local model with scripts/start_qwen.sh, then ./research.sh doctor.'
