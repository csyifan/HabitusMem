"""Validate and merge the four-task shards into one training file.

Checks every row's schema, canonicalizes belief records, counts the exact chat
tokens the trainer will see, and rejects a file that exceeds ``--cutoff-len``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .. import modeling
from . import SAMPLE_TYPES
from .beliefs import canonicalize_output


def conversation_tokens(
    tokenizer,
    instruction: str,
    input_text: str,
    output: str,
) -> int:
    prompt = instruction + "\n\n" + input_text if instruction else input_text
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output},
    ]
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, enable_thinking=False
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", required=True, help="Tokenizer source.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=0,
        help="0 = glob every unified_sft_shard_*.jsonl in input-dir.",
    )
    parser.add_argument("--cutoff-len", type=int, default=12288)
    parser.add_argument("--expected-personas", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if args.num_shards > 0:
        paths = [
            input_dir / f"unified_sft_shard_{i:02d}.jsonl"
            for i in range(args.num_shards)
        ]
    else:
        paths = sorted(input_dir.glob("unified_sft_shard_*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no shard files under {input_dir}")
        print(f"glob mode: {len(paths)} shard files")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shards: {missing}")

    tokenizer = modeling.load_tokenizer(args.model)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    counts: Counter[str] = Counter()
    personas: set[str] = set()
    rows = 0
    max_tokens = 0
    max_row = None
    sanitized: Counter[str] = Counter()
    dropped: Counter[str] = Counter()

    with temp_path.open("w", encoding="utf-8") as output:
        for shard_id, path in enumerate(paths):
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    for field in ("instruction", "input", "output",
                                  "sample_type", "persona_id"):
                        if not isinstance(row.get(field), str) or not row[field].strip():
                            raise ValueError(
                                f"{path}:{line_number} invalid field {field}"
                            )
                    if not row["output"].lstrip().startswith("## Beliefs"):
                        raise ValueError(
                            f"{path}:{line_number} output lacks Beliefs header"
                        )
                    original = row["output"].strip()
                    row["output"], dropped_count = canonicalize_output(original)
                    if row["output"] != original:
                        sanitized[row["sample_type"]] += 1
                    dropped[row["sample_type"]] += dropped_count

                    token_count = conversation_tokens(
                        tokenizer,
                        row["instruction"],
                        row["input"],
                        row["output"],
                    )
                    if token_count > max_tokens:
                        max_tokens = token_count
                        max_row = {
                            "shard_id": shard_id,
                            "line_number": line_number,
                            "sample_type": row["sample_type"],
                            "persona_id": row["persona_id"],
                        }
                    counts[row["sample_type"]] += 1
                    personas.add(row["persona_id"])
                    rows += 1
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")

    unknown = set(counts) - set(SAMPLE_TYPES)
    if unknown:
        raise ValueError(f"unknown sample types: {sorted(unknown)}")
    if len(personas) != args.expected_personas:
        raise ValueError(
            f"personas={len(personas)}, expected={args.expected_personas}"
        )
    if args.expected_rows and rows != args.expected_rows:
        raise ValueError(f"rows={rows}, expected={args.expected_rows}")
    if max_tokens > args.cutoff_len:
        raise ValueError(
            f"max training tokens={max_tokens} exceeds cutoff={args.cutoff_len}; "
            f"row={max_row}"
        )

    temp_path.replace(output_path)
    manifest = {
        "rows": rows,
        "personas": len(personas),
        "sample_counts": dict(counts),
        "sanitized_rows": dict(sanitized),
        "dropped_malformed_beliefs": dict(dropped),
        "max_training_tokens": max_tokens,
        "max_training_row": max_row,
        "cutoff_len": args.cutoff_len,
        "output_file": str(output_path),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
