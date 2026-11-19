#!/usr/bin/env bash
# Stage 2 training: final writer (1 epoch, lr 2e-5, continue from parent).
source "$(dirname "$0")/_env.sh"

torchrun --nproc_per_node="$NPROC" -m habitusmem.train_sft \
  --model "$MODEL" \
  --init-adapter "$WORK/adapters/parent" \
  --data-file "$WORK/unified/unified_sft_action3.jsonl" \
  --output-dir "$WORK/adapters/habitusmem" \
  --num-epochs 1 \
  --learning-rate 2e-5
