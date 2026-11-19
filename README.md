# HabitusMem

This repository is the official code release for *HabitusMem*, a memory writer for personalized LLM agents.

## News

* **2026.11**: Initial release of the three-stage training pipeline, the BeliefWriter inference backend, and the PersonaMem-v2 evaluation code.

## Installation

```bash
git clone <REPO_URL>
cd habitusmem
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10+ and a CUDA GPU are required. Training uses 80 GB A100/H100 GPUs; inference fits on 24 GB.

## Data

Download PersonaMem-v2 with `scripts/download_personamem.sh` and point `DATA_ROOT` at it. The script fetches 32k histories by default; `SIZE=128k` selects the 128k histories. Training uses the personas in `splits/personamem_v2_train.json`.

## Quick start

### One command

Start OpenAI-compatible servers for the agent and judge, fill in the `USER CONFIG` block at the top of `run.sh`, then run:

```bash
bash run.sh
```

`run.sh` evaluates one row shard per GPU and merges the results into `*_aggregate.json` under `results/personamem/`. For a quick smoke test:

```bash
NPROC=1 MAX_ITEMS=4 bash run.sh
```

Set `DISABLE_THINKING=0` if your endpoint does not accept `chat_template_kwargs`. `HISTORY_SELECT=head` uses the exact paper path.

### Manual run

```bash
python -m habitusmem.eval_personamem \
  --data-root /path/to/PersonaMem-v2 \
  --writer-model Qwen/Qwen3.5-9B --writer-adapter /path/to/adapter_habitusmem \
  --agent-base-url http://localhost:8000/v1 --agent-model Qwen/Qwen3.5-397B-A17B \
  --judge-base-url http://localhost:8001/v1 --judge-model Qwen/Qwen3.5-9B \
  --output-dir results/pm2
```

Results are written as a resumable JSONL plus a `*_summary.json`. Use `--run-timestamp` to resume and `--shard-count/--shard-index` to split rows across machines. Merge shards with:

```bash
python -m habitusmem.aggregate_personamem \
  --input-dir results/pm2 \
  --pattern '20270101_120000_writer_32k_*.jsonl'
```

## Training from scratch

Fill in the paths at the top of each script and run them in order:

```bash
export DATA_ROOT=/path/to/PersonaMem-v2
export MODEL=Qwen/Qwen3.5-9B
export WORK=$PWD/work
export NPROC=8

# Stage 0: cold-start data and training
export TEACHER_BASE_URL=https://your-teacher/v1
export TEACHER_MODEL=deepseek-v4-flash
export TEACHER_API_KEY=...
bash scripts/01_build_coldstart.sh
bash scripts/02_train_coldstart.sh

# Stage 1: direct writing and parent writer
bash scripts/03_build_direct.sh
bash scripts/04_train_parent.sh

# Stage 2: four-task mixture and final writer
bash scripts/05_build_multitask.sh
bash scripts/06_finalize_reweight.sh
bash scripts/07_train_final.sh
```

The build scripts in `scripts/` turn PersonaMem-v2 histories into cold-start, direct-writing, and four-task data; training hyperparameters are set in each script.

## Inference API

```python
from habitusmem import BeliefWriter
from habitusmem.data.personamem import load_messages

writer = BeliefWriter(
    model_path="Qwen/Qwen3.5-9B",
    adapter_path="runs/habitusmem",
    max_input_tokens=8500,
    max_new_tokens=800,
    history_select="head_safe",
)

messages = load_messages("data/PersonaMem-v2/data/chat_history_32k/....json")
memory = writer.memory(messages, query="Where should I eat tonight?", mode="query")
print(memory)
```

`mode` selects the suffix: `query` for question answering or `action` for agentic tasks. For tool use, append the shared execution guidance:

```python
from habitusmem import with_execution_guidance

context = writer.memory(messages, query=tool_request, mode="action")
context = with_execution_guidance(context)
```

History selectors are `head_safe` for normal use, `recent` for rolling agentic histories, and `head` only for exact reproduction of the paper path.

## Repository layout

```
habitusmem/
├── run.sh                     # one-command evaluation
├── habitusmem/
│   ├── writer.py              # BeliefWriter inference backend
│   ├── train_sft.py           # standalone HF + PEFT LoRA SFT
│   ├── eval_personamem.py     # MCQ + judge evaluation
│   ├── aggregate_personamem.py # merge sharded evaluation results
│   ├── modeling.py            # model loading / chat-template generation
│   ├── llm_client.py          # minimal OpenAI-compatible client
│   ├── prompts.py             # instruction, suffixes, schema
│   └── data/
│       ├── build_coldstart.py
│       ├── build_direct.py
│       ├── build_multitask.py
│       ├── finalize.py
│       ├── reweight.py
│       ├── beliefs.py         # parsing, validation, merge
│       └── personamem.py      # dataset loading
├── splits/                    # training-persona split
└── scripts/                   # data construction and training recipes
```


