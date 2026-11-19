#!/usr/bin/env bash
# Stage 1 data: direct query-conditioned examples (parent writer as teacher).
source "$(dirname "$0")/_env.sh"

CUDA_VISIBLE_DEVICES="$GPU" python -m habitusmem.data.build_direct \
  --data-root "$DATA_ROOT" \
  --benchmark-csv benchmark/text/train.csv \
  --split-file "$SPLIT" \
  --model "$MODEL" \
  --adapter "$WORK/adapters/coldstart" \
  --output-file "$WORK/direct/direct_sft.jsonl"
