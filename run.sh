#!/usr/bin/env bash
# One-stop PersonaMem-v2 evaluation.
#
# Fill in the paths and endpoints below, then run:
#
#   bash run.sh
#
# Every variable can also be overridden from the environment, e.g.
#   NPROC=4 SIZE=32k bash run.sh
#
# The script launches one writer process per GPU, evaluates a disjoint row
# shard, then merges the shards into one aggregate summary.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,12p' "${BASH_SOURCE[0]}"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================== USER CONFIG ==============================
# --- Data (downloaded PersonaMem-v2) -------------------------------------
# Expected layout:
#   $DATA_ROOT/benchmark/text/benchmark.csv
#   $DATA_ROOT/data/chat_history_32k/...
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/data/PersonaMem-v2}"

# --- Writer (base model + released LoRA adapter) --------------------------
# MODEL may be a local directory or a Hugging Face model id.
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
# ADAPTER must be the local adapter directory, or a HF repo id that contains
# the adapter files at its root. Download a subfolder first if needed.
ADAPTER="${ADAPTER:-$SCRIPT_DIR/weights/adapter_habitusmem}"

# --- Answering agent (OpenAI-compatible chat endpoint) --------------------
AGENT_BASE_URL="${AGENT_BASE_URL:-}"       # e.g. http://localhost:8000/v1
AGENT_MODEL="${AGENT_MODEL:-}"             # e.g. Qwen/Qwen3.5-397B-A17B
AGENT_API_KEY="${AGENT_API_KEY:-}"

# --- Judge (leave empty to reuse the agent endpoint/model) ----------------
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"

# --- Evaluation settings --------------------------------------------------
SIZE="${SIZE:-32k}"                        # 32k | 128k
EVAL_MODE="${EVAL_MODE:-both}"             # mcq | generative | both
NPROC="${NPROC:-8}"                        # writer processes / GPU shards
PARALLEL="${PARALLEL:-4}"                  # concurrent agent calls per shard
MAX_ITEMS="${MAX_ITEMS:-0}"                # >0 for a quick smoke test
WRITER_MODE="${WRITER_MODE:-query}"        # query | action
HISTORY_SELECT="${HISTORY_SELECT:-head_safe}"  # head_safe | head | recent
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-800}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"  # 1 for Qwen3.5-style endpoints

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/personamem}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"

# Optional dataset download when the CSV is missing.
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"        # 1 = run scripts/download_personamem.sh
# ========================================================================

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/${RUN_ID}_run.log"

info() { printf '[run.sh] %s\n' "$*" | tee -a "$LOG"; }
fail() {
  if [[ -n "${LOG:-}" ]]; then
    printf '[run.sh] ERROR: %s\n' "$*" | tee -a "$LOG" >&2
  else
    printf '[run.sh] ERROR: %s\n' "$*" >&2
  fi
  exit 1
}

mask() {
  if [[ -n "${1:-}" ]]; then printf '***'; else printf '(empty)'; fi
}

# --- Validate -------------------------------------------------------------
case "$SIZE" in 32k|128k) ;; *) fail "SIZE must be 32k or 128k"; esac
case "$EVAL_MODE" in mcq|generative|both) ;; *) fail "EVAL_MODE must be mcq, generative, or both"; esac
case "$WRITER_MODE" in query|action) ;; *) fail "WRITER_MODE must be query or action"; esac
case "$HISTORY_SELECT" in head|head_safe|recent) ;; *) fail "Invalid HISTORY_SELECT"; esac
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || fail "NPROC must be a positive integer"
[[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || fail "PARALLEL must be a positive integer"

if [[ ! -f "$DATA_ROOT/benchmark/text/benchmark.csv" ]]; then
  if [[ "$DOWNLOAD_DATA" == "1" ]]; then
    info "benchmark CSV missing; downloading PersonaMem-v2"
    SIZE="$SIZE" DATA_ROOT="$DATA_ROOT" bash "$SCRIPT_DIR/scripts/download_personamem.sh"
  else
    fail "missing $DATA_ROOT/benchmark/text/benchmark.csv (set DATA_ROOT, or DOWNLOAD_DATA=1)"
  fi
fi
[[ -d "$DATA_ROOT/data/chat_history_$SIZE" ]] || \
  fail "missing $DATA_ROOT/data/chat_history_$SIZE"

if [[ ! -e "$MODEL" ]]; then
  [[ "$MODEL" == */* ]] || fail "MODEL not found and does not look like a HF id: $MODEL"
  info "MODEL is not a local path; treating it as a Hugging Face id: $MODEL"
elif [[ -d "$MODEL" && ! -f "$MODEL/config.json" ]]; then
  fail "no config.json in MODEL directory: $MODEL"
fi
if [[ ! -e "$ADAPTER" ]]; then
  [[ "$ADAPTER" == */* ]] || fail "ADAPTER not found: $ADAPTER"
  info "ADAPTER is not a local path; treating it as a Hugging Face id: $ADAPTER"
elif [[ -d "$ADAPTER" && ! -f "$ADAPTER/adapter_config.json" ]]; then
  fail "no adapter_config.json in $ADAPTER; point ADAPTER at the folder that contains it"
fi

[[ -n "$AGENT_BASE_URL" ]] || fail "set AGENT_BASE_URL (e.g. http://localhost:8000/v1)"
[[ -n "$AGENT_MODEL" ]] || fail "set AGENT_MODEL"

if [[ -z "$JUDGE_BASE_URL" ]]; then
  JUDGE_BASE_URL="$AGENT_BASE_URL"
  JUDGE_API_KEY="${JUDGE_API_KEY:-$AGENT_API_KEY}"
  info "JUDGE_BASE_URL empty; reusing the agent endpoint"
fi
if [[ -z "$JUDGE_MODEL" ]]; then
  JUDGE_MODEL="$AGENT_MODEL"
  info "JUDGE_MODEL empty; reusing the agent model"
fi
if [[ "$EVAL_MODE" != "mcq" && -z "$JUDGE_MODEL" ]]; then
  fail "judge endpoint/model is required for $EVAL_MODE evaluation"
fi

GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  GPU_COUNT="${#_visible_gpus[@]}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  if ! GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; then
    GPU_COUNT=0
  fi
fi
if [[ "$GPU_COUNT" -gt 0 && "$NPROC" -gt "$GPU_COUNT" ]]; then
  info "NPROC=$NPROC exceeds $GPU_COUNT visible GPUs; using $GPU_COUNT"
  NPROC="$GPU_COUNT"
fi

info "data:        $DATA_ROOT"
info "model:       $MODEL"
info "adapter:     $ADAPTER"
info "agent:       $AGENT_MODEL @ $AGENT_BASE_URL (key $(mask "$AGENT_API_KEY"))"
info "judge:       $JUDGE_MODEL @ $JUDGE_BASE_URL (key $(mask "$JUDGE_API_KEY"))"
info "eval:        size=$SIZE mode=$EVAL_MODE writer_mode=$WRITER_MODE select=$HISTORY_SELECT"
info "runtime:     NPROC=$NPROC parallel=$PARALLEL max_items=$MAX_ITEMS"
info "output:      $OUTPUT_DIR (run id $RUN_ID)"
info "log:         $LOG"

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --size "$SIZE"
  --eval-mode "$EVAL_MODE"
  --writer-mode "$WRITER_MODE"
  --writer-model "$MODEL"
  --writer-adapter "$ADAPTER"
  --history-select "$HISTORY_SELECT"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --agent-base-url "$AGENT_BASE_URL"
  --agent-model "$AGENT_MODEL"
  --judge-base-url "$JUDGE_BASE_URL"
  --judge-model "$JUDGE_MODEL"
  --parallel "$PARALLEL"
  --shard-count "$NPROC"
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
  --run-timestamp "$RUN_ID"
)
if [[ -n "$AGENT_API_KEY" ]]; then
  COMMON_ARGS+=(--agent-api-key "$AGENT_API_KEY")
fi
if [[ -n "$JUDGE_API_KEY" ]]; then
  COMMON_ARGS+=(--judge-api-key "$JUDGE_API_KEY")
fi
if [[ "$DISABLE_THINKING" == "1" ]]; then
  COMMON_ARGS+=(--disable-thinking)
fi
if [[ "$MAX_ITEMS" -gt 0 ]]; then
  COMMON_ARGS+=(--max-items "$MAX_ITEMS")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  info "dry run; commands that would execute:"
  for i in $(seq 0 $((NPROC - 1))); do
    printf 'CUDA_VISIBLE_DEVICES=%s' "$i"
    printf ' %q' "$PYTHON" -m habitusmem.eval_personamem \
      "${COMMON_ARGS[@]}" --shard-index "$i"
    printf '\n'
  done
  exit 0
fi

# --- Run shards -----------------------------------------------------------
info "launching $NPROC shard(s)"
PIDS=()
for i in $(seq 0 $((NPROC - 1))); do
  SHARD_LOG="$OUTPUT_DIR/shard_${i}.log"
  info "shard $i -> $SHARD_LOG"
  CUDA_VISIBLE_DEVICES="$i" "$PYTHON" -m habitusmem.eval_personamem \
    "${COMMON_ARGS[@]}" --shard-index "$i" \
    >"$SHARD_LOG" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    info "shard $idx failed; tail of $OUTPUT_DIR/shard_${idx}.log:"
    tail -n 30 "$OUTPUT_DIR/shard_${idx}.log" >&2 || true
    FAILED=1
  fi
done
if [[ "$FAILED" -ne 0 ]]; then
  fail "one or more shards failed; inspect $OUTPUT_DIR/shard_*.log"
fi

# --- Aggregate ------------------------------------------------------------
info "aggregating shards"
"$PYTHON" -m habitusmem.aggregate_personamem \
  --input-dir "$OUTPUT_DIR" \
  --pattern "${RUN_ID}_writer_${SIZE}_*.jsonl" \
  --output "$OUTPUT_DIR/${RUN_ID}_aggregate.json"

info "done"
info "aggregate summary: $OUTPUT_DIR/${RUN_ID}_aggregate.json"
info "per-shard records: $OUTPUT_DIR/${RUN_ID}_writer_${SIZE}_s*of${NPROC}.jsonl"
