#!/usr/bin/env bash
set -euo pipefail
# DON'T source .venv here; we are using the active conda env.
echo "Python: $(which python)"; python -V
echo "Pip:    $(which pip)";    pip -V

python scripts/build_teacher_concepts_codellama.py \
  --model "${MODEL_NAME:-codellama/CodeLlama-7b-Instruct-hf}" \
  --outdir "${OUTDIR:-teacher_concepts_out}" \
  --max "${MAX_ITEMS:-0}"
