#!/usr/bin/env bash
# Stage 1 training: parent writer (1 epoch, lr 2e-5, continue from cold start).
source "$(dirname "$0")/_env.sh"

torchrun --nproc_per_node="$NPROC" -m habitusmem.train_sft \
  --model "$MODEL" \
  --init-adapter "$WORK/adapters/coldstart" \
  --data-file "$WORK/direct/direct_sft.jsonl" \
  --output-dir "$WORK/adapters/parent" \
  --num-epochs 1 \
  --learning-rate 2e-5
