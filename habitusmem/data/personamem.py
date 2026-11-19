"""PersonaMem-v2 loading utilities (text benchmark only).

Download the dataset from https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2
and point ``--data-root`` at the local directory. ``benchmark/text/train.csv``
and ``benchmark/text/benchmark.csv`` reference chat histories with paths relative
to that directory.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

csv.field_size_limit(10 ** 9)


def resolve_path(data_root: str | Path, link: str) -> Path:
    path = Path(link)
    return path if path.is_absolute() else Path(data_root) / path


def find_link_column(fieldnames: Iterable[str], size: str = "32k") -> str:
    for name in fieldnames or []:
        if "chat_history" in name and "link" in name and size in name:
            return name
    raise ValueError(f"no chat_history_{size}_link column in benchmark CSV")


def load_benchmark_rows(
    csv_path: str | Path,
    *,
    persona_ids: Optional[set[str]] = None,
    max_items: Optional[int] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if persona_ids is not None and row.get("persona_id") not in persona_ids:
                continue
            rows.append(row)
            if max_items is not None and len(rows) >= max_items:
                break
    return rows


def group_by_persona(
    rows: Iterable[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["persona_id"], []).append(row)
    return grouped


def persona_sort_key(persona_id: str):
    return (0, int(persona_id)) if str(persona_id).isdigit() else (1, str(persona_id))


def load_split(path: str | Path) -> List[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("train", [])
    return [str(item) for item in payload]


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    parts.append(str(part["text"]))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
            else:
                parts.append(str(part))
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("text"):
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def load_messages(path: str | Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        raw = data.get("chat_history")
        if raw is None:
            raw = data.get("conversations")
        if raw is None:
            raw = next((v for v in data.values() if isinstance(v, list)), [])
    else:
        raw = data
    messages = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        content = flatten_content(message.get("content", ""))
        if content:
            messages.append(
                {"role": message.get("role", "user"), "content": content}
            )
    return messages


def parse_query(raw: str) -> str:
    """``user_query`` cells are Python/JSON literals such as {"content": ...}."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return str(raw)
    if isinstance(value, dict):
        return str(value.get("content", "") or "")
    return str(value)


def parse_list(raw: str) -> List[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return [str(item) for item in value] if isinstance(value, list) else []


def render_lines(messages: List[Dict[str, str]]) -> List[str]:
    return [
        f"[msg {index}] {message['role']}: {message['content']}"
        for index, message in enumerate(messages)
    ]


def render_history(messages: List[Dict[str, str]]) -> str:
    return "\n".join(render_lines(messages))


def user_turn_cut_index(messages: List[Dict[str, str]], frac: float) -> int:
    """Index just after the user turn at ``frac`` along the user-turn axis."""
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_indices:
        return len(messages)
    position = min(int(frac * len(user_indices)), len(user_indices) - 1)
    return user_indices[position] + 1
