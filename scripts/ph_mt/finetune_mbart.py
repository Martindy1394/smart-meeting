#!/usr/bin/env python3
"""LoRA fine-tune facebook/mbart-large-50 for Tagalog/Hiligaynon → English.

Uses ``tl_XX`` as the source language code (Hiligaynon has no mBART token).
LoRA keeps CPU/GPU training tractable; merge adapters before production load
or point Smart Meeting at the merged folder.

Requirements:
  pip install "transformers>=4.40" datasets accelerate peft sentencepiece protobuf torch

Example (GPU recommended):
  python scripts/ph_mt/finetune_mbart.py \\
    --train-jsonl scripts/ph_mt/prepared/train.jsonl \\
    --eval-jsonl scripts/ph_mt/prepared/eval.jsonl \\
    --output-dir models/mbart-ph-en-lora \\
    --num-train-epochs 1 \\
    --fp16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model-name",
        default="facebook/mbart-large-50-many-to-many-mmt",
    )
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--max-source-length", type=int, default=128)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-lora", action="store_true", help="Full fine-tune (needs GPU)")
    args = p.parse_args(argv)

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            DataCollatorForSeq2Seq,
            MBart50TokenizerFast,
            MBartForConditionalGeneration,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except Exception as exc:
        print(
            "Missing training deps. Install:\n"
            '  pip install "transformers>=4.40" datasets accelerate peft '
            "sentencepiece protobuf torch\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    train_rows = _load_jsonl(args.train_jsonl)
    if not train_rows:
        print("Training JSONL is empty.", file=sys.stderr)
        return 1

    print(f"Loading base model {args.model_name} …")
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_name)
    tokenizer.src_lang = "tl_XX"
    tokenizer.tgt_lang = "en_XX"
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)

    if not args.no_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            print(f"peft is required for LoRA training: {exc}", file=sys.stderr)
            return 2
        lora = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    def preprocess(batch: dict) -> dict:
        tokenizer.src_lang = batch.get("src_lang") or "tl_XX"
        model_inputs = tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            max_length=args.max_source_length,
            truncation=True,
        )
        # Cap label length separately when needed.
        if (
            isinstance(model_inputs.get("labels"), list)
            and len(model_inputs["labels"]) > args.max_target_length
        ):
            model_inputs["labels"] = model_inputs["labels"][: args.max_target_length]
        return model_inputs

    train_ds = Dataset.from_list(train_rows).map(preprocess)
    eval_ds = None
    if args.eval_jsonl and args.eval_jsonl.exists():
        eval_rows = _load_jsonl(args.eval_jsonl)
        if eval_rows:
            eval_ds = Dataset.from_list(eval_rows).map(preprocess)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=use_fp16,
        eval_strategy="no",
        save_steps=50,
        logging_steps=5,
        predict_with_generate=False,
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=2,
        warmup_steps=10,
    )

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    # transformers v5 renamed tokenizer → processing_class
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    meta = {
        "base_model": args.model_name,
        "lora": not args.no_lora,
        "train_rows": len(train_rows),
        "src_lang": "tl_XX",
        "tgt_lang": "en_XX",
        "note": (
            "Hiligaynon uses tl_XX proxy. Merge LoRA with merge_lora.py before "
            "setting MBART_PH_FINE_TUNED_MODEL unless loading adapters yourself."
        ),
    }
    (args.output_dir / "finetune_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"Saved checkpoint to {args.output_dir}")
    if not args.no_lora:
        print("Merge adapters:")
        print(
            f"  python scripts/ph_mt/merge_lora.py "
            f"--adapter-dir {args.output_dir} "
            f"--output-dir models/mbart-ph-en-merged"
        )
    print("Then in backend/.env:")
    print("  MBART_PH_FINE_TUNED_MODEL=models/mbart-ph-en-merged")
    print("  PH_TRANSLATE_BACKEND=mbart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
