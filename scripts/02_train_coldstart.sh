#!/usr/bin/env bash
# Stage 0 training: cold-start LoRA adapter (2 epochs, lr 5e-5).
source "$(dirname "$0")/_env.sh"

torchrun --nproc_per_node="$NPROC" -m habitusmem.train_sft \
  --model "$MODEL" \
  --data-file "$WORK/coldstart/sft_coldstart.jsonl" \
  --output-dir "$WORK/adapters/coldstart" \
  --num-epochs 2 \
  --learning-rate 5e-5
