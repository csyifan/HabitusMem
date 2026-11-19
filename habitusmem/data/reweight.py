"""Apply task weights by repeating rows.

The final mixture up-weights action projection 3x, turning 3,196 examples into
9,588 rows and the 34,374-row file into 40,766 training rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import SAMPLE_TYPES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--direct-copies", type=int, default=1)
    parser.add_argument("--action-copies", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.direct_copies < 1 or args.action_copies < 1:
        raise ValueError("copy counts must be positive")

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    input_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()

    with input_path.open(encoding="utf-8") as source, temp_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            sample_type = row.get("sample_type")
            if sample_type not in SAMPLE_TYPES:
                raise ValueError(
                    f"{input_path}:{line_number} invalid sample_type={sample_type!r}"
                )
            input_counts[sample_type] += 1
            copies = (
                args.direct_copies
                if sample_type == "direct_replay"
                else args.action_copies
                if sample_type == "action_query_projection"
                else 1
            )
            for _ in range(copies):
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_counts[sample_type] += 1

    expected = {
        sample_type: count
        * (
            args.direct_copies
            if sample_type == "direct_replay"
            else args.action_copies
            if sample_type == "action_query_projection"
            else 1
        )
        for sample_type, count in input_counts.items()
    }
    if dict(output_counts) != expected:
        raise ValueError(
            f"output counts={dict(output_counts)}, expected={expected}"
        )

    temp_path.replace(output_path)
    manifest = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "direct_copies": args.direct_copies,
        "action_copies": args.action_copies,
        "input_counts": dict(input_counts),
        "output_counts": dict(output_counts),
        "output_rows": sum(output_counts.values()),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
