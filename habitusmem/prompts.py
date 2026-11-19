"""Prompt templates for belief-memory construction.

The writer always sees

    instruction + "\\n\\n" + (history | state) + suffix

where ``suffix`` is empty for static writing, the query suffix for
question answering, or the action suffix for agentic tasks.
"""

from __future__ import annotations

from pathlib import Path

# Final writer instruction ("instr A"), used for both training and inference.
DEFAULT_INSTRUCTION = (
    "Based on the conversation so far, write a compact belief-state memory "
    "about the user. Use exactly this schema:\n\n"
    "## Beliefs\n"
    "- <claim about the user> | evidence: <msg#> | scope: <always | "
    "situational description> | status: <active | retracted | superseded>\n\n"
    "Capture stable dispositions and current goals separately; mark revoked "
    "or superseded beliefs. Output ONLY the memory, starting with ## Beliefs."
)

SCHEMA = """## Beliefs
- <claim about the user> | evidence: <msg#> | scope: <always | situational description> | status: <active | retracted | superseded>"""

# Prompt used by the frozen teacher to annotate cold-start memories.
COLDSTART_TEACHER_PROMPT = """You are the memory module of a long-term personal assistant.
Based on the conversation so far, write a compact belief-state memory about the user. Use exactly this schema:

{schema}

Capture stable dispositions and current goals separately; mark revoked or superseded beliefs. Output ONLY the memory, starting with ## Beliefs.

Conversation so far:
{history}"""

# Marker inserted when a history is truncated in the middle. Must match the
# string used when the training targets were built.
ELISION = "\n[... earlier dialogue elided ...]\n"

# --- Query-conditioned writing (question answering) ------------------------
QUERY_SUFFIX = (
    "\n\nThe user will next ask this question. Make sure "
    "the memory above captures every belief, preference, "
    "fact, and standing instruction (including things the "
    "user asked to forget) needed to answer it well:\n"
    "Question: {query}"
)

# --- Action-ready writing (agentic tasks) ----------------------------------
# Inference suffix used by the released adapter. Training targets for the
# action-projection task were built with TRAIN_ACTION_SUFFIX below; the two
# differ in wording, a known train/deployment mismatch documented in the paper.
ACTION_SUFFIX = (
    "\n\nThe user will next ask this question. Prepare an "
    "action-ready belief memory for answering and executing "
    "it. Put every relevant explicit constraint first, one "
    "belief per line: exact product or category, brand, "
    "variant/specification, quantity, address, time, distance "
    "or rating, and any update or retraction. Preserve exact "
    "names and values. Prefer the user's explicit historical "
    "choice over a generic or health-inferred substitute. "
    "Use only the history and current question; do not invent "
    "missing constraints. Output ONLY the same ## Beliefs "
    "schema.\nQuestion: {query}"
)

TRAIN_ACTION_SUFFIX = (
    "\n\nThe user will next ask this question. Prepare an "
    "action-ready belief memory using only the history and current "
    "question. Put every relevant explicit constraint first: "
    "exact product or category, brand, variant/specification, "
    "quantity, address, time, distance/rating, and any update or "
    "retraction. Preserve exact names and values, and prefer an "
    "explicit historical choice over an inferred substitute. "
    "Output ONLY the same ## Beliefs schema.\nQuestion: {query}"
)

# --- Agentic consumption guidance ------------------------------------------
# Appended to the memory for tool-use benchmarks (e.g. VitaBench 2.0). The
# paper applies the same guidance to every compared system, so it is a
# consumer-side instruction rather than part of the writer.
AGENT_EXECUTION_GUIDANCE = """## Execution Guidance
The memory above is evidence, not a response to repeat. For the current request:
1. Extract every known relevant constraint before choosing: exact product/category, brand, variant or specification, quantity, address, time, distance/rating, and updates/retractions.
2. Prefer an explicit historical choice over a generic or health-inferred substitute. Do not replace the user's chosen green tea with another "healthier" tea, for example.
3. If the user says "随便", "你看着办", or otherwise delegates the choice, choose the single best matching option and continue the required tool workflow; do not stop at a list of recommendations or ask for information already present in memory/profile.
4. Complete the transaction when the request asks to buy/book/order. Avoid adding extra products unless the user requested multiple items."""


def with_execution_guidance(memory: str) -> str:
    if not memory:
        return AGENT_EXECUTION_GUIDANCE
    return f"{memory}\n\n{AGENT_EXECUTION_GUIDANCE}"


# --- Rolling state updates --------------------------------------------------
UPDATE_PREAMBLE = ("\n\n## Current memory\n{prev}\n\n"
                   "## New interactions since the last update\n")
UPDATE_DIRECTIVE = ("\n\nRewrite the memory so that it preserves every "
                    "still-supported belief, adds newly supported beliefs, "
                    "and marks revoked or replaced beliefs as retracted or "
                    "superseded. Cite at most three recent msg numbers per "
                    "belief. Output ONLY the updated memory starting with "
                    "## Beliefs.")
EMPTY_MEMORY = "(empty - no memory yet)"


def load_instruction(path: str | Path | None = None) -> str:
    """Return the embedded instruction, or an override from a file or raw string."""
    if path is None or str(path) == "":
        return DEFAULT_INSTRUCTION
    raw = str(path)
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip()
    return raw.strip()
