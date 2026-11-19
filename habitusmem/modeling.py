"""Model loading and generation helpers shared by data construction, the
writer backend, and evaluation.

The writer base checkpoint may be a text or vision-language model, so the
loader mirrors the paper's pipeline: prefer ``AutoModelForImageTextToText``
and fall back to ``AutoModelForCausalLM``.
"""

from __future__ import annotations

from typing import List, Optional

import torch


def _auto_model_class():
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_path: str,
    adapter_path: str = "",
    *,
    trainable_adapter: bool = False,
    device: str = "cuda",
    attn_implementation: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load a base model, optionally with a LoRA adapter.

    ``trainable_adapter=True`` keeps the adapter trainable and does not merge
    it; this is what the training script needs. Inference callers get a merged
    model, which is slightly faster and avoids PEFT dispatch overhead.
    """
    model_cls = _auto_model_class()
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=trainable_adapter
        )
        if not trainable_adapter:
            model = model.merge_and_unload()
    if device:
        model = model.to(device)
    return model


def chat_ids(tokenizer, content: str) -> List[int]:
    """Render one user turn with the tokenizer's chat template.

    ``enable_thinking=False`` is passed when the template supports it. Qwen3.5
    and similar hybrid-reasoning models otherwise emit a thinking block, which
    the belief schema does not expect.
    """
    messages = [{"role": "user", "content": content}]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def generate(
    model,
    tokenizer,
    content: str,
    *,
    max_new_tokens: int = 800,
    device: str = "cuda",
    max_input_tokens: Optional[int] = None,
    tail_cut: Optional[int] = None,
) -> str:
    """Greedy generation for one prompt, decoding only the new tokens.

    ``tail_cut`` reproduces the paper's PersonaMem ``head`` selector, which
    keeps the *last* ``tail_cut`` prompt tokens and can therefore drop the
    instruction. New code should use ``head_safe`` instead; the option exists
    only to reproduce the reported run exactly.
    """
    ids = torch.tensor([chat_ids(tokenizer, content)], device=device)
    if max_input_tokens is not None and ids.shape[1] > max_input_tokens:
        raise ValueError(
            f"prompt is {ids.shape[1]} tokens, over max_input_tokens="
            f"{max_input_tokens}; lower the history budget instead of cutting "
            "the prompt, which would drop the instruction."
        )
    if tail_cut is not None and ids.shape[1] > tail_cut:
        ids = ids[:, -tail_cut:]
    with torch.no_grad():
        output = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][ids.shape[1]:], skip_special_tokens=True
    ).strip()
