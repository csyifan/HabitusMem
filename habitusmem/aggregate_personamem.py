"""Aggregate sharded PersonaMem-v2 evaluation records into one summary.

Reads every ``*.jsonl`` under ``--input-dir``, keeps the latest successful
record per (persona, query), and reports overall MCQ accuracy and average
judge score with the same breakdowns as the evaluation script.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from .eval_personamem import summarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--output", default="")
    parser.add_argument("--output-records", default="", help="Optional merged JSONL.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {args.pattern} in {input_dir}")

    best: "OrderedDict[tuple, dict]" = OrderedDict()
    total_lines = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            total_lines += 1
            key = (
                str(record.get("persona_id")),
                record.get("query", ""),
                record.get("size"),
                record.get("memory_type"),
            )
            previous = best.get(key)
            if previous is None or (
                previous.get("error") and not record.get("error")
            ):
                best[key] = record
            elif not record.get("error"):
                best[key] = record

    records = list(best.values())
    first = records[0] if records else {}
    meta = {
        key: first.get(key)
        for key in (
            "memory_type",
            "writer_mode",
            "size",
            "agent_model",
            "judge_model",
            "writer_model",
            "writer_adapter",
            "max_input_tokens",
            "max_new_tokens",
            "history_select",
        )
    }
    has_mcq = any(isinstance(r.get("mcq_is_correct"), bool) for r in records)
    has_generative = any(
        isinstance(r.get("judge_score"), (int, float)) for r in records
    )
    meta["eval_mode"] = (
        "both" if has_mcq and has_generative
        else "mcq" if has_mcq
        else "generative"
    )
    meta["input_files"] = len(paths)
    meta["input_lines"] = total_lines
    meta["deduplicated_records"] = len(records)
    summary = summarize(records, meta)

    output = Path(args.output) if args.output else input_dir / "aggregate_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.output_records:
        with open(args.output_records, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary["overall"], indent=2), flush=True)
    print(f"summary -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
