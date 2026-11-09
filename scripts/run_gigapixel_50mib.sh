#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ODIE_SERIAL_PYTHON:-python}" \
  "$SCRIPT_DIR/run_baselines.py" \
  --manifest "$SCRIPT_DIR/../configs/baselines/gigapixel_50mib.example.json" \
  --vendor-root "${ODIE_VENDOR_ROOT:-$SCRIPT_DIR/../vendor}" "$@"
