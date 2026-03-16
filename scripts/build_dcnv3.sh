#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DCNV3_DIR="$ROOT_DIR/ultralytics/nn/modules/ops_dcnv3"

cd "$DCNV3_DIR"
python -m pip install -v .
