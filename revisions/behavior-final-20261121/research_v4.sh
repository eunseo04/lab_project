#!/usr/bin/env bash
# Repaired observation/evidence-lifecycle revision; see the behavior report.
set -euo pipefail
V4_RESEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/revisions/behavior-v4-observation-20261121"
exec "$V4_RESEARCH_ROOT/research.sh" "$@"
