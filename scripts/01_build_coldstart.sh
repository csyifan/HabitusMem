#!/usr/bin/env bash
# Stage 0 data: teacher-written cold-start belief memories (799 personas x 2 cuts).
source "$(dirname "$0")/_env.sh"

: "${TEACHER_BASE_URL:?set TEACHER_BASE_URL (OpenAI-compatible)}"
: "${TEACHER_MODEL:?set TEACHER_MODEL}"

python -m habitusmem.data.build_coldstart \
  --data-root "$DATA_ROOT" \
  --benchmark-csv benchmark/text/train.csv \
  --split-file "$SPLIT" \
  --output-file "$WORK/coldstart/sft_coldstart.jsonl" \
  --base-url "$TEACHER_BASE_URL" \
  --api-key "${TEACHER_API_KEY:-}" \
  --model "$TEACHER_MODEL" \
  --disable-thinking \
  --workers "${TEACHER_WORKERS:-8}"
