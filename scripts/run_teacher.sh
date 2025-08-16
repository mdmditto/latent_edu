#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
set -a
[ -f .env ] && source .env
set +a

python scripts/build_teacher_concepts_codellama.py \
  --model "${MODEL_NAME:-codellama/CodeLlama-7b-Instruct-hf}" \
  --outdir "${OUTDIR:-teacher_concepts_out}" \
  --max "${MAX_ITEMS:-0}"
