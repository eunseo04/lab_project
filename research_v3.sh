#!/usr/bin/env bash
# Experimental connection-history revision. See docs/behavior_skill_optimization.md.
set -euo pipefail
V3_RESEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/revisions/behavior-v3-dev-20261121"
exec "$V3_RESEARCH_ROOT/research.sh" "$@"
