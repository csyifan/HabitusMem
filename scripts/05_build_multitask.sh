#!/usr/bin/env bash
# Stage 2 data: state-update and projection examples, sharded over NPROC GPUs.
source "$(dirname "$0")/_env.sh"

for i in $(seq 0 $((NPROC - 1))); do
  CUDA_VISIBLE_DEVICES="$i" python -m habitusmem.data.build_multitask \
    --data-root "$DATA_ROOT" \
    --benchmark-csv benchmark/text/train.csv \
    --split-file "$SPLIT" \
    --direct-data "$WORK/direct/direct_sft.jsonl" \
    --model "$MODEL" \
    --teacher-adapter "$WORK/adapters/parent" \
    --output-dir "$WORK/unified" \
    --num-shards "$NPROC" \
    --shard-id "$i" &
done

wait
echo "all $NPROC shards finished; run 06_finalize_reweight.sh next"
