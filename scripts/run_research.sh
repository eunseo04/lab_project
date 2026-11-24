#!/usr/bin/env bash
set -euo pipefail
PUBLICATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" ]]; then
  printf '%s\n' 'Usage: scripts/run_research.sh B0|B1|B2|B3|B4 train|evaluate [research CLI arguments]'
  exit 0
fi
if [[ $# -lt 2 ]]; then
  printf '%s\n' 'Usage: scripts/run_research.sh B0|B1|B2|B3|B4 train|evaluate [research CLI arguments]' >&2
  exit 2
fi
PUBLICATION_ARM="$1"
PUBLICATION_COMMAND="$2"
shift 2
case "$PUBLICATION_ARM" in B0|B1|B2|B3|B4) ;; *) printf '%s\n' 'Unknown comparison arm' >&2; exit 2 ;; esac
case "$PUBLICATION_COMMAND" in train|evaluate) ;; *) printf '%s\n' 'Use train or evaluate' >&2; exit 2 ;; esac
python3 "$PUBLICATION_ROOT/scripts/prepare_checkout.py" \
  --destination "$PUBLICATION_ROOT/.local-configs" >/dev/null
exec "$PUBLICATION_ROOT/research.sh" "$PUBLICATION_COMMAND" \
  --config "$PUBLICATION_ROOT/.local-configs/rollout-$PUBLICATION_ARM.yaml" "$@"
