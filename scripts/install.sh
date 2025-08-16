#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
pip install --upgrade pip


# (optional) login to huggingface if the model requires acceptance
# HUGGINGFACE_TOKEN must be set in env if you do this non-interactively
if [[ "${HUGGINGFACE_TOKEN:-}" != "" ]]; then
  python - <<'PY'
import os, subprocess
tok = os.environ["HUGGINGFACE_TOKEN"]
subprocess.run(["python","-m","huggingface_hub.login","--token",tok], check=True)
PY
fi

