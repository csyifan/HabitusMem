"""Stage 0: cold-start belief data written by a frozen teacher.

For every training persona and every cut position along the user-turn axis,
serialize the history prefix and ask a teacher model (any OpenAI-compatible
chat endpoint) for one schema-valid belief memory. The student writer then
learns the record format from these targets.

Reference setting: 799 personas x 2 cuts = 1,598 rows; teacher temperature 0.2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..llm_client import chat
from ..prompts import COLDSTART_TEACHER_PROMPT, ELISION, SCHEMA, load_instruction
from .beliefs import validate_memory
from .personamem import (
    find_link_column,
    group_by_persona,
    load_benchmark_rows,
    load_messages,
    load_split,
    persona_sort_key,
    render_history,
    resolve_path,
    user_turn_cut_index,
)


def serialize_history(messages, max_chars: int = 32000) -> str:
    text = render_history(messages)
    if len(text) <= max_chars:
        return text
    head = 8000
    tail = max_chars - head - 40
    return text[:head] + ELISION + text[-tail:]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--benchmark-csv",
        default="benchmark/text/train.csv",
        help="Relative to --data-root unless absolute.",
    )
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="", help="Or set TEACHER_API_KEY.")
    parser.add_argument("--model", required=True, help="Teacher model name.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--cut-fracs", default="0.35,0.55")
    parser.add_argument("--history-size", default="32k")
    parser.add_argument("--history-chars", type=int, default=32000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Ask hybrid-reasoning endpoints (Qwen3.5, ...) to skip thinking.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.api_key = args.api_key or os.environ.get("TEACHER_API_KEY", "")
    csv_path = resolve_path(args.data_root, args.benchmark_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    train = set(load_split(args.split_file))
    rows = load_benchmark_rows(csv_path, persona_ids=train)
    if not rows:
        raise ValueError(
            f"no rows for the {len(train)} split personas in {csv_path}; "
            "check --split-file and --benchmark-csv"
        )
    grouped = group_by_persona(rows)
    link_column = find_link_column(rows[0].keys(), args.history_size)
    instruction = load_instruction(args.instruction_file)

    tasks = []
    for persona_id in sorted(grouped, key=persona_sort_key):
        link = next(
            (row[link_column] for row in grouped[persona_id] if row.get(link_column)),
            "",
        )
        if not link:
            continue
        path = resolve_path(args.data_root, link)
        if not path.is_file():
            print(f"skip persona {persona_id}: missing {path}", file=sys.stderr)
            continue
        for frac in (float(x) for x in args.cut_fracs.split(",") if x.strip()):
            tasks.append((persona_id, path, frac))
    print(f"personas={len(grouped)} tasks={len(tasks)}", flush=True)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add((str(record["persona_id"]), float(record["cut_frac"])))
        tasks = [t for t in tasks if (t[0], t[2]) not in done]
        print(f"resume: {len(done)} done, {len(tasks)} pending", flush=True)

    write_lock = threading.Lock()
    history_cache: dict[Path, list[dict]] = {}
    history_lock = threading.Lock()

    def get_messages(path: Path):
        with history_lock:
            cached = history_cache.get(path)
        if cached is not None:
            return cached
        messages = load_messages(path)
        with history_lock:
            if len(history_cache) >= 16:
                history_cache.clear()
            history_cache[path] = messages
        return messages

    stats = {"ok": 0, "error": 0}
    extra_body = (
        {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
        if args.disable_thinking
        else None
    )

    def build(task):
        persona_id, path, frac = task
        messages = get_messages(path)
        if len(messages) < 16:
            raise ValueError("history has fewer than 16 messages")
        cut = user_turn_cut_index(messages, frac)
        history = serialize_history(messages[:cut], args.history_chars)
        prompt = COLDSTART_TEACHER_PROMPT.format(schema=SCHEMA, history=history)
        last_error = None
        for attempt in range(args.max_retries):
            try:
                memory = chat(
                    args.base_url,
                    args.api_key,
                    args.model,
                    [{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens * (2 ** attempt),
                    extra_body=extra_body,
                )
                memory = validate_memory(
                    memory,
                    allow_trailing_incomplete=True,
                    context=f"persona {persona_id} cut {frac}",
                )
                return {
                    "sample_type": "coldstart",
                    "persona_id": persona_id,
                    "cut_frac": frac,
                    # The instruction is embedded in `input`; an empty
                    # `instruction` keeps the trainer from prepending it twice.
                    "instruction": "",
                    "input": instruction + "\n\n" + history,
                    "output": memory,
                }
            except Exception as error:  # noqa: BLE001
                last_error = error
        raise RuntimeError(f"persona {persona_id} cut {frac}: {last_error}")

    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(build, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    record = future.result()
                except Exception as error:  # noqa: BLE001
                    stats["error"] += 1
                    print(f"FAILED {task[0]} cut={task[2]}: {error}", flush=True)
                    continue
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    stats["ok"] += 1
                if index % 50 == 0:
                    print(f"progress {index}/{len(tasks)} {stats}", flush=True)

    manifest = {
        "output_file": str(output_path),
        "rows": stats["ok"],
        "errors": stats["error"],
        "personas": len({t[0] for t in tasks}),
        "instruction_file": str(args.instruction_file or "bundled"),
        "teacher_model": args.model,
        "cut_fracs": args.cut_fracs,
        "history_chars": args.history_chars,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
