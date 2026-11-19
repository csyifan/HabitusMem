#!/usr/bin/env bash
# Download the text benchmark and chat histories.
# Set SIZE=128k to include the 128k histories instead of the 32k ones.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${DATA_ROOT:-data/PersonaMem-v2}"
SIZE="${SIZE:-32k}"
INCLUDE=("benchmark/text/*")
if [[ "$SIZE" == "128k" ]]; then
  INCLUDE+=("data/chat_history_128k/*")
else
  INCLUDE+=("data/chat_history_32k/*")
fi
if command -v hf >/dev/null 2>&1; then
  CLI=hf
else
  CLI=huggingface-cli
fi
"$CLI" download bowen-upenn/PersonaMem-v2 \
  --repo-type dataset --local-dir "$DATA_ROOT" \
  --include "${INCLUDE[@]}"
echo "dataset ready at $DATA_ROOT ($SIZE histories)"
