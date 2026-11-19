"""LoRA SFT for the HabitusMem writer.

Reads the alpaca-style JSONL produced by the data pipeline and trains only on
response tokens, matching the paper's recipe. Launch with ``torchrun`` for
multi-GPU training; the effective batch size is
``per_device_batch * grad_accum * world_size``.

Examples
--------
Cold start (base model, 2 epochs, lr 5e-5)::

    torchrun --nproc_per_node=8 -m habitusmem.train_sft \
        --model Qwen/Qwen3.5-9B \
        --data-file data/sft_coldstart.jsonl \
        --output-dir runs/coldstart \
        --num-epochs 2 --learning-rate 5e-5

Final writer (continue from the parent adapter, 1 epoch, lr 2e-5)::

    torchrun --nproc_per_node=8 -m habitusmem.train_sft \
        --model Qwen/Qwen3.5-9B \
        --init-adapter runs/parent \
        --data-file data/unified_sft_action3.jsonl \
        --output-dir runs/habitusmem
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


def apply_template(tokenizer, messages, add_generation_prompt: bool) -> List[int]:
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            enable_thinking=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


class AlpacaSFTDataset(Dataset):
    """Lazily tokenize CSV rows; mask everything before the answer.

    Tokenizing on the fly keeps memory bounded: the 40,766-row mixture would
    otherwise hold hundreds of millions of token ids in RAM.
    """

    def __init__(self, path: str | Path, tokenizer, cutoff_len: int):
        self.rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.truncated = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        row = self.rows[index]
        instruction = (row.get("instruction") or "").strip()
        input_text = row["input"]
        prompt = instruction + "\n\n" + input_text if instruction else input_text
        answer = row["output"]

        prompt_ids = apply_template(
            self.tokenizer, [{"role": "user", "content": prompt}], True
        )
        full_ids = apply_template(
            self.tokenizer,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            False,
        )
        # Chat templates already terminate the assistant turn with their
        # stop/eos token; appending another one would train a duplicate.
        prefix = 0
        for left, right in zip(prompt_ids, full_ids):
            if left != right:
                break
            prefix += 1
        labels = [-100] * prefix + list(full_ids[prefix:])

        if len(full_ids) > self.cutoff_len:
            if self.truncated == 0:
                print(
                    f"WARNING: row {index} is {len(full_ids)} tokens and will "
                    f"be truncated to {self.cutoff_len}; run finalize.py to "
                    "check the whole file before training.",
                    file=sys.stderr,
                    flush=True,
                )
            if all(label == -100 for label in labels[: self.cutoff_len]):
                # A right cut would remove the whole answer. Keep the tail so
                # the row still has a target.
                full_ids = full_ids[-self.cutoff_len:]
                labels = labels[-self.cutoff_len:]
            else:
                full_ids = full_ids[: self.cutoff_len]
                labels = labels[: self.cutoff_len]
            self.truncated += 1
        return {"input_ids": full_ids, "labels": labels}


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {
            "input_ids": torch.full(
                (len(features), max_len), self.pad_token_id, dtype=torch.long
            ),
            "labels": torch.full(
                (len(features), max_len), -100, dtype=torch.long
            ),
            "attention_mask": torch.zeros(
                (len(features), max_len), dtype=torch.long
            ),
        }
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            batch["input_ids"][index, :length] = torch.tensor(
                feature["input_ids"], dtype=torch.long
            )
            batch["labels"][index, :length] = torch.tensor(
                feature["labels"], dtype=torch.long
            )
            batch["attention_mask"][index, :length] = 1
        return batch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path or hub id.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--init-adapter",
        default="",
        help="Continue from this LoRA adapter. Empty = start a fresh adapter.",
    )
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--cutoff-len", type=int, default=12288)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help='"all-linear" (paper setting) or a comma-separated list.',
    )
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1000,
                        help="0 = only save the final adapter.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Trainer, TrainingArguments

    from . import modeling

    tokenizer = modeling.load_tokenizer(args.model)
    model = modeling.load_model(
        args.model,
        "",
        trainable_adapter=False,
        device="",
        attn_implementation=args.attn,
    )
    model.config.use_cache = False

    if args.init_adapter:
        model = PeftModel.from_pretrained(
            model, args.init_adapter, is_trainable=True
        )
    else:
        target_modules = (
            "all-linear"
            if args.target_modules == "all-linear"
            else [name.strip() for name in args.target_modules.split(",")]
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = AlpacaSFTDataset(args.data_file, tokenizer, args.cutoff_len)
    print(
        f"rows={len(dataset)} cutoff={args.cutoff_len} "
        f"world_size={os.environ.get('WORLD_SIZE', '1')}",
        flush=True,
    )

    bf16 = args.precision == "bf16"
    fp16 = args.precision == "fp16"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=bf16,
        fp16=fp16,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="no" if args.save_steps <= 0 else "steps",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=1,
        save_safetensors=True,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_workers,
        # The base checkpoint is a vision-language model and all inputs are
        # text, so vision LoRA parameters receive no gradient. DDP must allow
        # unused parameters for multi-GPU runs.
        ddp_find_unused_parameters=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint or None
    )
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved adapter to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
