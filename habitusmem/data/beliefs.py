"""Belief-record parsing, validation, merging, and canonicalization.

A valid record has the form

    - <claim> | evidence: <source> | scope: <context> | status: <state>

The helpers here are shared by the data builders, the finalizer, and the
writer so that every stage agrees on what counts as a well-formed memory.
"""

from __future__ import annotations

import re
from typing import List

BELIEF_FIELDS = ("evidence:", "scope:", "status:")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_belief_line(line: str) -> str:
    """Rewrite ", evidenced by X | scope:" into " | evidence: X | scope:"."""
    if "evidence:" not in line.lower():
        match = re.search(
            r",?\s+evidenced by\s+(.+?)(?=\s*\|\s*scope:)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            claim = line[: match.start()].rstrip(" ,")
            evidence = match.group(1).strip()
            line = f"{claim} | evidence: {evidence}{line[match.end():]}"
    return line


def is_complete_belief(line: str) -> bool:
    lower = line.lower()
    return all(field in lower for field in BELIEF_FIELDS)


def belief_lines(memory: str) -> List[str]:
    """Extract belief bullets, joining wrapped continuation lines."""
    beliefs: List[str] = []
    for raw_line in memory.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            beliefs.append(line)
        elif beliefs and not is_complete_belief(beliefs[-1]):
            beliefs[-1] += " " + line
    return [normalize_belief_line(line) for line in beliefs]


def claim_key(line: str) -> str:
    claim = line.lstrip("- ").split("| evidence:", 1)[0]
    return _normalize(claim)


def retraction_topic(line: str) -> str:
    if "status: retracted" not in line.lower():
        return ""
    match = re.search(r"regarding\s+(.+?)(?:;|\||$)", line, flags=re.IGNORECASE)
    return _normalize(match.group(1)) if match else ""


def topic_matches_claim(topic: str, line: str) -> bool:
    if not topic:
        return False
    claim = claim_key(line)
    if topic in claim:
        return True
    words = [
        word
        for word in re.findall(r"[a-z0-9]{3,}", topic)
        if word not in {"the", "user", "preference"}
    ]
    return bool(words) and all(word in claim for word in words)


def validate_memory(
    memory: str,
    *,
    allow_trailing_incomplete: bool = False,
    context: str = "",
) -> str:
    """Return a normalized memory or raise ValueError.

    ``allow_trailing_incomplete`` drops a truncated final bullet; the data
    builders use it only after escalating the generation budget failed.
    """
    lines = belief_lines(memory)
    complete = [is_complete_belief(line) for line in lines]
    if (
        not memory.lstrip().startswith("## Beliefs")
        or not lines
        or not any(complete)
    ):
        raise ValueError(f"{context}: malformed memory: {memory[:200]!r}")
    if not all(complete):
        if not allow_trailing_incomplete:
            raise ValueError(f"{context}: malformed memory: {memory[:200]!r}")
        lines = [line for line, valid in zip(lines, complete) if valid]
    return "## Beliefs\n" + "\n".join(lines)


def merge_memories(tokenizer, previous: str, new: str, max_tokens: int) -> str:
    """Line-aware union of two memories, recent beliefs first.

    Beliefs retracted by ``new`` are dropped from ``previous``. The result is
    capped at ``max_tokens`` without cutting a line in half.
    """
    ordered: List[str] = []
    seen: set[str] = set()
    new_lines = belief_lines(new)
    retracted_topics = [topic for topic in map(retraction_topic, new_lines) if topic]
    for line in new_lines + belief_lines(previous):
        if (
            line not in new_lines
            and "status: active" in line.lower()
            and any(topic_matches_claim(topic, line) for topic in retracted_topics)
        ):
            continue
        key = claim_key(line)
        if not key or key in seen:
            continue
        candidate = "## Beliefs\n" + "\n".join(ordered + [line])
        if len(tokenizer(candidate, add_special_tokens=False).input_ids) > max_tokens:
            continue
        ordered.append(line)
        seen.add(key)
    if not ordered and new_lines:
        # A single over-budget bullet is rare but must not crash a run.
        ordered = [new_lines[0]]
    return "## Beliefs\n" + "\n".join(ordered)


def canonicalize_output(memory: str) -> tuple[str, int]:
    """Keep only complete belief bullets; return (memory, dropped_count)."""
    beliefs = belief_lines(memory)
    canonical = [line for line in beliefs if is_complete_belief(line)]
    if not canonical:
        raise ValueError("output contains no complete beliefs")
    return "## Beliefs\n" + "\n".join(canonical), len(beliefs) - len(canonical)
