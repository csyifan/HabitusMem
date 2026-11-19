"""PersonaMem-v2 evaluation: MCQ accuracy and narrow LLM-judge score.

For every benchmark row the writer builds a belief memory from the 32k chat
history and the current request; a frozen agent answers from that memory only.
The script is resumable and shardable, and it uses OpenAI-compatible endpoints
for the agent and judge, so the same code works with hosted APIs or a local
vLLM/SGLang deployment.

Example::

    python -m habitusmem.eval_personamem \
        --data-root data/PersonaMem-v2 \
        --writer-model Qwen/Qwen3.5-9B \
        --writer-adapter runs/habitusmem \
        --agent-base-url http://localhost:8000/v1 \
        --agent-model Qwen/Qwen3.5-397B-A17B \
        --judge-base-url http://localhost:8001/v1 \
        --judge-model Qwen/Qwen3.5-9B \
        --output-dir results/pm2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import llm_client
from .data.personamem import (
    load_benchmark_rows,
    load_messages,
    parse_list,
    parse_query,
    render_history,
)

SYSTEM_PREAMBLE = (
    "You are a helpful AI assistant. Below is what you remember about this user "
    "from your past conversations with them. Use it to give a response that is "
    "personalized to the user's preferences.\n\n"
)

RECALL_HINT = (
    " Please recall my related preferences from our conversation history to "
    "give personalized responses."
)

MCQ_INSTRUCTION = (
    "Please choose the best answer from the following options:\n\n"
    "{options}\n\n"
    "Think step by step about which answer best fits the user's query and "
    "conversation context. Provide your reasoning first, then give your final "
    "answer as 'Final Answer: [Letter]'"
)

ANSWER_PATTERNS = [
    r"\$\\boxed\{([A-Z])\}\$",
    r"\\boxed\{([A-Z])\}",
    r"Final Answer:\s*([A-Z])",
    r"final answer:\s*([A-Z])",
    r"Answer:\s*([A-Z])",
    r"answer:\s*([A-Z])",
    r"final answer is\s*\$?\\boxed\{([A-Z])\}\$?",
    r"final answer is\s*([A-Z])",
    r"the answer is\s*\$?\\boxed\{([A-Z])\}\$?",
    r"the answer is\s*([A-Z])",
    r"\b([A-Z])\.\s*$",
]

JUDGE_POSITIVE = """You are evaluating an AI assistant's personalization capabilities in open-ended conversations.

User's query:
{user_query}

Ground truth user preference: {preference}

Model's response to evaluate:
{model_response}

Task: Evaluate how well the model personalizes its response according to the user's ground-truth preference. We expect the model to demonstrate awareness of this preference and provide a response that is specifically aligned with or relevant to it, rather than giving a generic answer that could apply to anyone.

A well-personalized response should:
- Acknowledge or incorporate the specific preference mentioned
- Tailor recommendations, advice, or information to align with this preference
- Show that the model remembers and respects the user's stated preference

A poorly personalized response:
- Ignores the preference entirely and gives generic advice
- Explicitly states it doesn't remember or track preferences
- Provides suggestions that contradict the stated preference

Give a score from 0.0 (completely irrelevant to or contradicts the preference) to 1.0 (explicitly and thoughtfully personalized to the preference).

Provide your reasoning first, then give your final score as \\boxed{{score}} where score is a number between 0.0 and 1.0.
Your response:"""

JUDGE_NEGATIVE = """You are evaluating an AI assistant's personalization capabilities in open-ended conversations.

User's query:
{user_query}

Ground truth user preference: {preference}

Model's response to evaluate:
{model_response}

In this case, the user's ground-truth preference is a "do not personalize" or "avoid" instruction about a certain topic or attribute. The user does not want the assistant to tailor its response based on that specific preference, and instead expects a more general or neutral answer that does not rely on or mention that preference.

Task: Evaluate how well the model respects this preference. The model should avoid using or referencing the sensitive preference in a way that personalizes the response, while still providing a helpful and relevant answer.

Give a score from 0.0 (clearly personalizes using the forbidden preference or ignores the instruction) to 1.0 (fully respects the instruction to not personalize while remaining helpful).

Provide your reasoning first, then give your final score as \\boxed{{score}} where score is a number between 0.0 and 1.0.
Your response:"""


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def deterministic_seed(persona_id: str, query: str) -> int:
    key = f"{persona_id}_{query}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % (2 ** 32)


def create_mcq_options(
    correct_answer: str, incorrect_answers: List[str], seed: int
) -> Tuple[str, Dict[str, str]]:
    rng = random.Random(seed)
    options = [correct_answer] + list(incorrect_answers)
    rng.shuffle(options)
    mapping: Dict[str, str] = {}
    parts: List[str] = []
    for index, option in enumerate(options):
        letter = chr(65 + index)
        mapping[letter] = option
        parts.append(f"{letter}. {option}")
    return MCQ_INSTRUCTION.format(options="\n".join(parts)), mapping


def extract_final_answer(response: str) -> str:
    if not response:
        return ""
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ""


def correct_letter(correct_answer: str, mapping: Dict[str, str]) -> str:
    for letter, answer in mapping.items():
        if answer == correct_answer:
            return letter
    return "N/A"


def check_mcq_correctness(
    predicted: str, correct_answer: str, mapping: Dict[str, str]
) -> bool:
    if not predicted or not mapping:
        return False
    return mapping.get(predicted.upper(), "") == correct_answer


def extract_judge_decision(response: str) -> float:
    if not response:
        return 0.0
    for pattern in (
        r"\\boxed\{([0-9]*\.?[0-9]+)\}",
        r"\$\\boxed\{([0-9]*\.?[0-9]+)\}\$",
        r"\\boxed\s*\{([0-9]*\.?[0-9]+)\}",
    ):
        match = re.search(pattern, response)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                continue
    for pattern in (
        r"score[:\s]+([0-9]*\.?[0-9]+)",
        r"rating[:\s]+([0-9]*\.?[0-9]+)",
        r"([0-9]*\.[0-9]+)\s*/\s*1\.?0?",
    ):
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                continue
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class HistoryCache:
    def __init__(self, data_root: str, capacity: int = 4):
        self.data_root = data_root
        self.capacity = capacity
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._lock = threading.Lock()

    def load(self, link: str) -> List[Dict[str, str]]:
        path = link if os.path.isabs(link) else os.path.join(self.data_root, link)
        with self._lock:
            cached = self._store.get(path)
        if cached is not None:
            return cached
        messages = load_messages(path)
        with self._lock:
            if len(self._store) >= self.capacity:
                self._store.clear()
            self._store[path] = messages
        return messages


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--benchmark-file", default="benchmark/text/benchmark.csv")
    parser.add_argument("--size", default="32k", choices=["32k", "128k"])
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)

    parser.add_argument("--writer-mode", choices=["query", "action"],
                        default="query")
    parser.add_argument("--writer-model", default="")
    parser.add_argument("--writer-adapter", default="")
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--max-input-tokens", type=int, default=8500)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--history-select", default="head_safe",
                        choices=["head", "head_safe", "recent"])
    parser.add_argument("--max-input-chars", type=int, default=32000)
    parser.add_argument("--head-frac", type=float, default=0.15)
    parser.add_argument("--history-budget-tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--agent-base-url", default=os.environ.get("AGENT_BASE_URL", ""))
    parser.add_argument("--agent-api-key", default=os.environ.get("AGENT_API_KEY", ""))
    parser.add_argument("--agent-model", default=os.environ.get("AGENT_MODEL", ""))
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument("--agent-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send chat_template_kwargs.enable_thinking=false to Qwen3.5-style "
             "endpoints. The paper's agent and judge ran with thinking disabled.",
    )

    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", ""))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""))
    parser.add_argument("--judge-max-tokens", type=int, default=4096)

    parser.add_argument("--eval-mode", choices=["mcq", "generative", "both"],
                        default="both")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output-dir", default="results/personamem")
    parser.add_argument("--run-timestamp", default="")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if not args.agent_base_url or not args.agent_model:
        raise ValueError("--agent-base-url and --agent-model are required")
    if args.eval_mode in ("generative", "both") and (
        not args.judge_base_url or not args.judge_model
    ):
        raise ValueError("judge endpoint is required for generative evaluation")

    benchmark_path = (
        Path(args.benchmark_file)
        if os.path.isabs(args.benchmark_file)
        else Path(args.data_root) / args.benchmark_file
    )
    rows = load_benchmark_rows(benchmark_path)
    if args.max_items:
        rows = rows[: args.max_items]
    total_rows = len(rows)
    rows = rows[args.shard_index:: args.shard_count]
    print(
        f"shard {args.shard_index}/{args.shard_count}: "
        f"{len(rows)} of {total_rows} rows",
        flush=True,
    )

    if not args.writer_model:
        raise ValueError("--writer-model is required")
    from .writer import BeliefWriter

    writer = BeliefWriter(
        args.writer_model,
        args.writer_adapter,
        instruction=args.instruction_file,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        history_select=args.history_select,
        max_input_chars=args.max_input_chars,
        head_frac=args.head_frac,
        history_budget_tokens=args.history_budget_tokens,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = args.run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "writer"
    shard_tag = f"s{args.shard_index}of{args.shard_count}"
    output_path = output_dir / f"{timestamp}_{tag}_{args.size}_{shard_tag}.jsonl"

    done = set()
    if output_path.is_file() and not args.no_resume:
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("error"):
                done.add((str(record.get("persona_id")), record.get("query", "")))
    pending = [
        row for row in rows
        if (
            str(row.get("persona_id")),
            _query_of(row),
        ) not in done
    ]
    pending.sort(key=lambda row: row.get(f"chat_history_{args.size}_link", ""))
    print(f"pending={len(pending)} already_done={len(done)}", flush=True)

    cache = HistoryCache(args.data_root)
    write_lock = threading.Lock()

    def process(row: Dict[str, Any]) -> Dict[str, Any]:
        return evaluate_row(
            row, args, writer, cache
        )

    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(process, row) for row in pending]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                if index % 20 == 0:
                    print(f"progress {index}/{len(pending)}", flush=True)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = {
        "memory_type": "writer",
        "writer_mode": args.writer_mode,
        "size": args.size,
        "eval_mode": args.eval_mode,
        "agent_model": args.agent_model,
        "judge_model": args.judge_model,
        "writer_model": args.writer_model,
        "writer_adapter": args.writer_adapter,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "history_select": args.history_select,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    summary = summarize(records, meta)
    summary_path = (
        output_dir / f"{timestamp}_{tag}_{args.size}_{shard_tag}_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _query_of(row: Dict[str, Any]) -> str:
    return parse_query(row.get("user_query", ""))


def evaluate_row(row, args, writer, cache) -> Dict[str, Any]:
    persona_id = str(row.get("persona_id", ""))
    query = parse_query(row.get("user_query", ""))
    record: Dict[str, Any] = {
        "persona_id": persona_id,
        "size": args.size,
        "memory_type": "writer",
        "writer_mode": args.writer_mode,
        "query": query,
        "pref_type": row.get("pref_type"),
        "conversation_scenario": row.get("conversation_scenario"),
        "topic_preference": row.get("topic_preference"),
        "who": row.get("who"),
        "updated": row.get("updated"),
        "distance": row.get(f"distance_from_related_snippet_to_query_{args.size}"),
        "preference": row.get("preference"),
        "agent_model": args.agent_model,
        "judge_model": args.judge_model,
        "writer_model": args.writer_model,
        "writer_adapter": args.writer_adapter,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "history_select": args.history_select,
        "mcq_predicted": "",
        "mcq_correct_letter": "",
        "mcq_is_correct": None,
        "openended_response": "",
        "judge_score": None,
        "error": "",
    }
    try:
        link = row.get(f"chat_history_{args.size}_link", "")
        messages = cache.load(link)
        history = render_history(messages)

        memory = writer.memory(history, query=query, mode=args.writer_mode)
        record["memory_read"] = (
            memory if len(memory) <= 30000 else memory[:30000] + "...[TRUNCATED]"
        )
        record["memory_is_beliefs"] = memory.lstrip().startswith("## Beliefs")

        if args.eval_mode in ("mcq", "both"):
            incorrect = parse_list(row.get("incorrect_answers", ""))
            correct_answer = row.get("correct_answer", "")
            seed = deterministic_seed(persona_id, query)
            mcq_instruction, mapping = create_mcq_options(
                correct_answer, incorrect, seed
            )
            user_content = f"{query}{RECALL_HINT}\n\n{mcq_instruction}"
            response = agent_answer(memory, user_content, args, max_tokens=args.agent_max_tokens)
            predicted = extract_final_answer(response)
            record["mcq_predicted"] = predicted
            record["mcq_correct_letter"] = correct_letter(correct_answer, mapping)
            record["mcq_is_correct"] = check_mcq_correctness(
                predicted, correct_answer, mapping
            )
            record["mcq_response"] = response

        if args.eval_mode in ("generative", "both"):
            user_content = f"{query}{RECALL_HINT}"
            response = agent_answer(memory, user_content, args, max_tokens=args.agent_max_tokens)
            record["openended_response"] = response
            preference = row.get("preference", "") or ""
            template = (
                JUDGE_NEGATIVE
                if preference.lower().startswith("do not")
                else JUDGE_POSITIVE
            )
            prompt = template.format(
                user_query=query,
                preference=preference,
                model_response=response,
            )
            judge_text = llm_client.chat(
                args.judge_base_url,
                args.judge_api_key,
                args.judge_model,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=args.judge_max_tokens,
                extra_body=_extra_body(args),
            )
            record["judge_score"] = extract_judge_decision(judge_text)
            record["judge_response"] = judge_text

    except Exception as error:  # noqa: BLE001
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _extra_body(args) -> Optional[Dict]:
    if args.disable_thinking:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def agent_answer(memory: str, user_content: str, args, max_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PREAMBLE + (memory or "")},
        {"role": "user", "content": user_content},
    ]
    return llm_client.chat(
        args.agent_base_url,
        args.agent_api_key,
        args.agent_model,
        messages,
        temperature=args.agent_temperature,
        max_tokens=max_tokens,
        extra_body=_extra_body(args),
    )


def summarize(records: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    def metric(rows: List[Dict[str, Any]]):
        correct = sum(1 for row in rows if row.get("mcq_is_correct") is True)
        total = sum(
            1 for row in rows if isinstance(row.get("mcq_is_correct"), bool)
        )
        scores = [
            row["judge_score"]
            for row in rows
            if isinstance(row.get("judge_score"), (int, float))
        ]
        return {
            "num_rows": len(rows),
            "mcq_correct": correct,
            "mcq_total": total,
            "mcq_accuracy": (correct / total) if total else 0.0,
            "judge_total": len(scores),
            "avg_judge_score": (sum(scores) / len(scores)) if scores else 0.0,
        }

    breakdown: Dict[str, Dict[str, Any]] = {}
    for field in ("pref_type", "conversation_scenario", "who", "updated"):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            key = str(record.get(field))
            groups.setdefault(key, []).append(record)
        breakdown[field] = {key: metric(value) for key, value in sorted(groups.items())}

    errors = sum(1 for record in records if record.get("error"))
    valid_memories = sum(1 for record in records if record.get("memory_is_beliefs"))
    return {
        **meta,
        "errors": errors,
        "memory_valid": valid_memories,
        "memory_total": len(records),
        "overall": metric(records),
        "breakdown": breakdown,
    }


if __name__ == "__main__":
    raise SystemExit(main())
