"""Belief writer: turn an interaction history into a belief memory.

The writer is a base model plus one LoRA adapter, shared across datasets. It
supports the three history selectors and two request suffixes used in the
paper. Generation is greedy and serialized, so a single instance can be called
from several evaluation threads safely.

Example::

    writer = BeliefWriter(
        model_path="Qwen/Qwen3.5-9B",
        adapter_path="path/to/adapter",
    )
    memory = writer.memory(messages, query="Where should I eat tonight?")
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

from . import modeling
from .data.personamem import render_history
from .prompts import (
    ACTION_SUFFIX,
    ELISION,
    QUERY_SUFFIX,
    load_instruction,
)


class BeliefWriter:
    def __init__(
        self,
        model_path: str,
        adapter_path: str = "",
        *,
        instruction: Optional[str] = None,
        max_input_tokens: int = 8500,
        max_new_tokens: int = 800,
        history_select: str = "head_safe",
        max_input_chars: int = 32000,
        head_frac: float = 0.15,
        history_budget_tokens: int = 0,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        cache_size: int = 128,
    ):
        if history_select not in ("head", "head_safe", "recent"):
            raise ValueError(f"unknown history_select={history_select!r}")
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.history_select = history_select
        self.max_input_chars = int(max_input_chars)
        self.head_frac = float(head_frac)
        self.history_budget_tokens = int(history_budget_tokens)
        self.device = device
        self.instruction = load_instruction(instruction)
        self.tokenizer = modeling.load_tokenizer(model_path)
        self.model = modeling.load_model(
            model_path,
            adapter_path,
            device=device,
            attn_implementation=attn_implementation,
        )
        self._lock = threading.Lock()
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[tuple, str]" = OrderedDict()

    # -- prompt construction -------------------------------------------------

    def _suffix(self, query: Optional[str], mode: str) -> str:
        if mode not in ("query", "action"):
            raise ValueError(f"unknown mode={mode!r}")
        if not query:
            return ""
        template = ACTION_SUFFIX if mode == "action" else QUERY_SUFFIX
        return template.format(query=query)

    def _fixed_prefix(self) -> str:
        return (self.instruction + "\n\n") if self.instruction else ""

    def build_content(
        self,
        history: str,
        query: Optional[str] = None,
        mode: str = "query",
    ) -> tuple[str, Optional[int]]:
        """Return (prompt_content, tail_cut).

        ``tail_cut`` is None except in the reproduction-only ``head`` mode.
        """
        fixed = self._fixed_prefix()
        suffix = self._suffix(query, mode)

        if self.history_select == "head":
            # Paper-exact PersonaMem path: character cap, then keep the last
            # max_input_tokens prompt tokens. Can drop the instruction.
            hist = history[: self.max_input_chars]
            return fixed + hist + suffix, self.max_input_tokens

        budget = self.max_input_tokens - 32 - modeling.count_tokens(
            self.tokenizer, fixed + suffix
        )
        if self.history_budget_tokens:
            budget = min(budget, self.history_budget_tokens)
        if budget <= 0:
            raise ValueError(
                "history token budget is not positive; increase "
                "--max-input-tokens or shorten the instruction"
            )
        ids = self.tokenizer(history, add_special_tokens=False).input_ids
        if len(ids) > budget:
            if self.history_select == "head_safe":
                history = self.tokenizer.decode(
                    ids[:budget], skip_special_tokens=True
                )
            else:  # recent: bootstrap prefix + most recent interactions
                head_n = int(budget * self.head_frac)
                tail_n = budget - head_n
                history = self.tokenizer.decode(
                    ids[:head_n], skip_special_tokens=True
                )
                if tail_n > 0:
                    history += ELISION + self.tokenizer.decode(
                        ids[-tail_n:], skip_special_tokens=True
                    )
        return fixed + history + suffix, None

    # -- generation ----------------------------------------------------------

    def memory(
        self,
        history: str | List[Dict[str, str]],
        query: Optional[str] = None,
        mode: str = "query",
    ) -> str:
        """Generate a belief memory for ``history`` and ``query``.

        ``history`` may be a rendered string or a list of ``{role, content}``
        messages. ``mode`` is ``query`` (question answering) or ``action``
        (agentic tasks).
        """
        if isinstance(history, list):
            history = render_history(history)
        key = (
            hashlib.sha1(history.encode("utf-8")).hexdigest(),
            query or "",
            mode,
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            content, tail_cut = self.build_content(history, query, mode)
            memory = modeling.generate(
                self.model,
                self.tokenizer,
                content,
                max_new_tokens=self.max_new_tokens,
                device=self.device,
                tail_cut=tail_cut,
            )
            if self.cache_size > 0:
                self._cache[key] = memory
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return memory
