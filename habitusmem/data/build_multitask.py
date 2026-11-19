"""Stage 2: four-task training mixture (direct, state update, projections).

Inputs
------
* the direct-writing rows from :mod:`build_direct` (replayed as ``direct``),
* a parent writer (base model + adapter) used as the frozen teacher.

For each persona the teacher
  1. reads the history in fixed-size chunks and merges chunk memories into a
     running state, producing ``state_update`` examples, and
  2. projects the final state onto sampled requests, producing
     ``state_query_projection`` and ``action_query_projection`` examples.

Output shards are resumable per persona. Run :mod:`finalize` over all shards
and then :mod:`reweight` to obtain the final training file.

Requires one GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .. import modeling
from ..prompts import (
    EMPTY_MEMORY,
    QUERY_SUFFIX,
    TRAIN_ACTION_SUFFIX,
    UPDATE_DIRECTIVE,
    UPDATE_PREAMBLE,
    load_instruction,
)
from .beliefs import merge_memories, validate_memory
from .personamem import (
    find_link_column,
    group_by_persona,
    load_benchmark_rows,
    load_messages,
    load_split,
    parse_query,
    persona_sort_key,
    render_lines,
    resolve_path,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--benchmark-csv", default="benchmark/text/train.csv")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--direct-data", required=True)
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--model", required=True, help="Writer base model.")
    parser.add_argument("--teacher-adapter", required=True, help="Parent adapter.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-size", default="32k")
    parser.add_argument("--chunk-tokens", type=int, default=5000)
    parser.add_argument("--max-input-tokens", type=int, default=11400)
    parser.add_argument("--state-max-new-tokens", type=int, default=4800)
    parser.add_argument("--view-max-new-tokens", type=int, default=1800)
    parser.add_argument("--max-state-tokens", type=int, default=2400)
    parser.add_argument("--max-view-tokens", type=int, default=800)
    parser.add_argument("--direct-replay-copies", type=int, default=1)
    parser.add_argument("--projections-per-persona", type=int, default=8)
    parser.add_argument("--action-projections-per-persona", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--persona-list", default="", help="Optional JSON list.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


class Teacher:
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        heartbeat: Path,
        max_input_tokens: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.heartbeat = heartbeat
        self.max_input_tokens = max_input_tokens

    def beat(self) -> None:
        try:
            self.heartbeat.touch()
        except OSError:
            pass

    def raw(self, content: str, max_new_tokens: int) -> str:
        output = modeling.generate(
            self.model,
            self.tokenizer,
            content,
            max_new_tokens=max_new_tokens,
            device=self.device,
            max_input_tokens=self.max_input_tokens,
        )
        self.beat()
        return output

    def valid_memory(
        self,
        content: str,
        max_new_tokens: int,
        context: str,
        max_escalations: int = 2,
    ) -> str:
        """Generate a schema-valid memory, doubling the budget on truncation.

        Greedy decoding makes escalation exact: a larger budget reproduces the
        same prefix and continues it, so generations that already fit are
        unchanged.
        """
        last_error: Exception | None = None
        for escalation in range(max_escalations + 1):
            budget = max_new_tokens * (2 ** escalation)
            raw = self.raw(content, budget)
            try:
                return validate_memory(
                    raw,
                    allow_trailing_incomplete=True,
                    context=f"{context} (esc {escalation})",
                )
            except ValueError as error:
                last_error = error
            repair = (
                "Rewrite the draft memory below into the exact schema. Output "
                "ONLY ## Beliefs followed by bullets. Every bullet must contain "
                "literal fields `| evidence: ... | scope: ... | status: active/"
                "retracted/superseded`. Preserve all supported information and "
                "message references; remove malformed fragments.\n\n" + raw
            )
            try:
                return validate_memory(
                    self.raw(repair, budget),
                    allow_trailing_incomplete=True,
                    context=f"{context} (repair {escalation})",
                )
            except ValueError as error:
                last_error = error
        raise RuntimeError(f"{context}: exhausted budget escalation: {last_error}")


def pack_chunks(tokenizer, lines: list[str], token_budget: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        ids = tokenizer(line, add_special_tokens=False).input_ids
        if len(ids) > token_budget:
            if current:
                chunks.append(current)
                current, current_tokens = [], 0
            for start in range(0, len(ids), token_budget):
                chunks.append(
                    [
                        tokenizer.decode(
                            ids[start:start + token_budget],
                            skip_special_tokens=True,
                        )
                    ]
                )
            continue
        if current and current_tokens + len(ids) + 1 > token_budget:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += len(ids) + 1
    if current:
        chunks.append(current)
    return chunks


def query_from_input(text: str) -> str:
    marker = "Question: "
    index = text.rfind(marker)
    return text[index + len(marker):].strip() if index >= 0 else ""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, num-shards)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"unified_sft_shard_{args.shard_id:02d}.jsonl"
    heartbeat = output_dir / f"heartbeat_{args.shard_id:02d}.txt"
    heartbeat.touch()

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

    ordered_rows = [row for persona_id in sorted(grouped, key=persona_sort_key)
                    for row in grouped[persona_id]]
    direct_rows = load_jsonl(Path(args.direct_data))
    if len(direct_rows) != len(ordered_rows):
        raise ValueError(
            f"direct rows={len(direct_rows)} benchmark train rows={len(ordered_rows)}"
        )
    for direct, row in zip(direct_rows, ordered_rows):
        if normalize(query_from_input(direct["input"])) != normalize(
            parse_query(row.get("user_query", ""))
        ):
            raise ValueError(
                f"direct-data ordering mismatch at persona {row['persona_id']}"
            )
    direct_by_persona: dict[str, list[dict]] = defaultdict(list)
    for direct, row in zip(direct_rows, ordered_rows):
        direct_by_persona[row["persona_id"]].append(direct)

    tokenizer = modeling.load_tokenizer(args.model)
    model = modeling.load_model(args.model, args.teacher_adapter, device=args.device)
    teacher = Teacher(
        model, tokenizer, args.device, heartbeat, args.max_input_tokens
    )

    personas = sorted(grouped, key=persona_sort_key)
    if args.persona_list:
        wanted = {str(p) for p in json.loads(Path(args.persona_list).read_text())}
        personas = [p for p in personas if p in wanted]
    else:
        personas = [
            p for index, p in enumerate(personas)
            if index % args.num_shards == args.shard_id
        ]

    done = set()
    if output_path.is_file() and output_path.stat().st_size > 0:
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(str(json.loads(line)["persona_id"]))
    pending = [p for p in personas if p not in done]
    print(
        f"shard={args.shard_id} personas={len(personas)} done={len(done)} "
        f"pending={len(pending)}",
        flush=True,
    )

    stats: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    skipped_personas: set[str] = set()
    with output_path.open("a", encoding="utf-8") as handle:
        for index, persona_id in enumerate(pending, 1):
            persona_samples: list[dict] = []
            for _ in range(args.direct_replay_copies):
                for sample in direct_by_persona[persona_id]:
                    persona_samples.append(
                        {**sample, "sample_type": "direct_replay",
                         "persona_id": persona_id}
                    )
                    stats["direct_replay"] += 1

            row = grouped[persona_id][0]
            messages = load_messages(resolve_path(args.data_root, row[link_column]))
            lines = render_lines(messages)
            chunks = pack_chunks(tokenizer, lines, args.chunk_tokens)

            state = ""
            for chunk_index, chunk in enumerate(chunks):
                chunk_text = "\n".join(chunk)
                try:
                    chunk_memory = teacher.valid_memory(
                        instruction + "\n\n" + chunk_text,
                        args.state_max_new_tokens,
                        f"persona {persona_id} state chunk {chunk_index}",
                    )
                    target = validate_memory(
                        merge_memories(
                            tokenizer, state, chunk_memory, args.max_state_tokens
                        ),
                        context=f"persona {persona_id} state target {chunk_index}",
                    )
                except RuntimeError as error:
                    print(f"SKIP {error}", flush=True)
                    skipped["state_update"] += 1
                    skipped_personas.add(persona_id)
                    continue
                update_input = (
                    UPDATE_PREAMBLE.format(prev=state or EMPTY_MEMORY)
                    + chunk_text
                    + UPDATE_DIRECTIVE
                )
                persona_samples.append(
                    {
                        "instruction": instruction,
                        "input": update_input,
                        "output": target,
                        "sample_type": "state_update",
                        "persona_id": persona_id,
                        "chunk_index": chunk_index,
                    }
                )
                stats["state_update"] += 1
                state = target

            persona_direct = direct_by_persona[persona_id]

            def add_projections(
                count: int,
                suffix: str,
                sample_type: str,
                midpoint: bool,
            ) -> None:
                count = min(count, len(persona_direct))
                for i in range(count):
                    position = ((i + 0.5) if midpoint else i) / count
                    direct_index = min(
                        int(position * len(persona_direct)), len(persona_direct) - 1
                    )
                    query = query_from_input(persona_direct[direct_index]["input"])
                    input_text = state + suffix.format(query=query)
                    raw_view = teacher.valid_memory(
                        instruction + "\n\n" + input_text,
                        args.view_max_new_tokens,
                        f"persona {persona_id} {sample_type}",
                    )
                    target = validate_memory(
                        merge_memories(
                            tokenizer, "", raw_view, args.max_view_tokens
                        ),
                        context=f"persona {persona_id} {sample_type} target",
                    )
                    persona_samples.append(
                        {
                            "instruction": instruction,
                            "input": input_text,
                            "output": target,
                            "sample_type": sample_type,
                            "persona_id": persona_id,
                            "direct_index": direct_index,
                        }
                    )
                    stats[sample_type] += 1

            for sample_type, count, suffix, midpoint in (
                ("state_query_projection", args.projections_per_persona,
                 QUERY_SUFFIX, False),
                ("action_query_projection", args.action_projections_per_persona,
                 TRAIN_ACTION_SUFFIX, True),
            ):
                try:
                    add_projections(count, suffix, sample_type, midpoint)
                except RuntimeError as error:
                    print(f"SKIP {error}", flush=True)
                    skipped[sample_type] += 1
                    skipped_personas.add(persona_id)

            for sample in persona_samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            handle.flush()
            teacher.beat()
            if index % 5 == 0 or index == len(pending):
                print(
                    f"shard={args.shard_id} personas={index}/{len(pending)} "
                    f"samples={sum(stats.values())}",
                    flush=True,
                )

    manifest = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "personas": len(personas),
        "samples": sum(stats.values()),
        "sample_counts": dict(stats),
        "skipped_samples": dict(skipped),
        "personas_with_skips": sorted(skipped_personas, key=persona_sort_key),
        "direct_replay_copies": args.direct_replay_copies,
        "projections_per_persona": args.projections_per_persona,
        "action_projections_per_persona": args.action_projections_per_persona,
        "chunk_tokens": args.chunk_tokens,
        "max_input_tokens": args.max_input_tokens,
        "state_max_new_tokens": args.state_max_new_tokens,
        "view_max_new_tokens": args.view_max_new_tokens,
        "max_state_tokens": args.max_state_tokens,
        "max_view_tokens": args.max_view_tokens,
    }
    (output_dir / f"manifest_{args.shard_id:02d}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
