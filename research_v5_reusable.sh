#!/usr/bin/env bash
set -euo pipefail
REUSABLE_RESEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/revisions/rollout-v5-reusable"
exec "$REUSABLE_RESEARCH_ROOT/research.sh" "$@"
