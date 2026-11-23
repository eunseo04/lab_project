#!/usr/bin/env bash
set -euo pipefail
V5_RESEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/revisions/rollout-v5-static-pool-fix"
exec "$V5_RESEARCH_ROOT/research.sh" "$@"
