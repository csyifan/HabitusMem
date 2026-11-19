"""Stage 1: direct query-conditioned writing data.

For each training persona the parent (or cold-start) writer produces one static
belief memory from the full history. For every training request of that
persona, the annotated preference is inserted at the top of that memory and
paired with the request-conditioned prompt. These 18,529 rows teach the writer
to surface the belief a request depends on.

Requires one GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import modeling
from ..prompts import ELISION, QUERY_SUFFIX, load_instruction
from .personamem import (
    find_link_column,
    group_by_persona,
    load_benchmark_rows,
    load_messages,
    load_split,
    parse_query,
    persona_sort_key,
    render_history,
    resolve_path,
)


def fit_history(tokenizer, text: str, budget: int, head_frac: float) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= budget:
        return text
    head_n = int(budget * head_frac)
    tail_n = budget - head_n
    head_text = tokenizer.decode(ids[:head_n], skip_special_tokens=True)
    if tail_n <= 0:
        return head_text
    return head_text + ELISION + tokenizer.decode(
        ids[-tail_n:], skip_special_tokens=True
    )


def preference_record(row: dict) -> str:
    preference = (row.get("preference") or "").strip()
    if not preference:
        return ""
    if preference.lower().startswith("do not"):
        topic = (row.get("topic_preference") or "a personal topic").strip()
        return (
            f"- The user has asked to forget a preference regarding {topic}; "
            f"do not use or mention it | evidence: user instruction "
            f"| scope: always | status: retracted"
        )
    if row.get("updated") == "True":
        return (
            f"- [update] {preference} | evidence: latest statement "
            f"| scope: always | status: active "
            f"(supersedes the earlier preference on this topic)"
        )
    return f"- {preference} | evidence: user statement | scope: always | status: active"


def inject_preference(static_memory: str, record: str) -> str:
    if record in static_memory:
        return static_memory
    body = static_memory
    if body.lstrip().startswith("## Beliefs"):
        body = body.lstrip()[len("## Beliefs"):].lstrip("\n")
    return "## Beliefs\n" + record + "\n" + body


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--benchmark-csv", default="benchmark/text/train.csv")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--model", required=True, help="Writer base model path or hub id.")
    parser.add_argument("--adapter", default="", help="Optional LoRA adapter to load.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--history-size", default="32k")
    parser.add_argument("--hist-budget-tokens", type=int, default=10500)
    parser.add_argument("--head-frac", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=11400)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--max-personas", type=int, default=0, help="Debug limit.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    csv_path = resolve_path(args.data_root, args.benchmark_csv)
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

    personas = sorted(grouped, key=persona_sort_key)
    if args.max_personas:
        personas = personas[: args.max_personas]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(str(json.loads(line)["persona_id"]))
        personas = [p for p in personas if p not in done]
        print(f"resume: {len(done)} personas done, {len(personas)} pending")

    tokenizer = modeling.load_tokenizer(args.model)
    model = modeling.load_model(
        args.model, args.adapter, device=args.device
    )
    instruction_tokens = len(
        tokenizer(instruction, add_special_tokens=False).input_ids
    )
    history_budget = min(
        args.hist_budget_tokens, args.max_input_tokens - instruction_tokens - 64
    )
    if history_budget <= 0:
        raise ValueError(
            "history budget is not positive; increase --max-input-tokens or "
            "shorten the instruction"
        )
    print(
        f"history budget={history_budget} tokens "
        f"(requested {args.hist_budget_tokens}, instruction {instruction_tokens}, "
        f"max_input {args.max_input_tokens}); head_frac={args.head_frac}",
        flush=True,
    )

    stats = {"personas": 0, "rows": 0, "skipped_rows": 0}
    with output_path.open("a", encoding="utf-8") as handle:
        for index, persona_id in enumerate(personas, 1):
            persona_rows = grouped[persona_id]
            link = next(
                (row[link_column] for row in persona_rows if row.get(link_column)),
                "",
            )
            if not link:
                print(f"skip persona {persona_id}: no history link", flush=True)
                continue
            messages = load_messages(resolve_path(args.data_root, link))
            history = fit_history(
                tokenizer,
                render_history(messages),
                history_budget,
                args.head_frac,
            )
            static_memory = modeling.generate(
                model,
                tokenizer,
                instruction + "\n\n" + history,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                max_input_tokens=args.max_input_tokens,
            )
            if not static_memory.lstrip().startswith("## Beliefs"):
                raise RuntimeError(
                    f"persona {persona_id}: writer did not return a belief "
                    f"memory: {static_memory[:200]!r}"
                )

            written = 0
            for row in persona_rows:
                record = preference_record(row)
                if not record:
                    stats["skipped_rows"] += 1
                    continue
                query = parse_query(row.get("user_query", ""))
                sample = {
                    "sample_type": "direct",
                    "persona_id": persona_id,
                    "instruction": instruction,
                    "input": history + QUERY_SUFFIX.format(query=query),
                    "output": inject_preference(static_memory, record),
                }
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            stats["personas"] += 1
            stats["rows"] += written
            if index % 10 == 0 or index == len(personas):
                print(
                    f"personas {index}/{len(personas)} rows={stats['rows']}",
                    flush=True,
                )

    manifest = {
        "output_file": str(output_path),
        "personas": stats["personas"],
        "rows": stats["rows"],
        "skipped_rows": stats["skipped_rows"],
        "model": args.model,
        "adapter": args.adapter,
        "hist_budget_tokens": args.hist_budget_tokens,
        "head_frac": args.head_frac,
        "max_input_tokens": args.max_input_tokens,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
