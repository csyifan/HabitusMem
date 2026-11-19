#!/usr/bin/env bash
# Validate/merge the four-task shards and apply the (1,1,1,3) task weights.
source "$(dirname "$0")/_env.sh"

python -m habitusmem.data.finalize \
  --input-dir "$WORK/unified" \
  --output-file "$WORK/unified/unified_sft.jsonl" \
  --model "$MODEL" \
  --num-shards "$NPROC" \
  --cutoff-len 12288 \
  --expected-personas 799

python -m habitusmem.data.reweight \
  --input-file "$WORK/unified/unified_sft.jsonl" \
  --output-file "$WORK/unified/unified_sft_action3.jsonl" \
  --direct-copies 1 \
  --action-copies 3
