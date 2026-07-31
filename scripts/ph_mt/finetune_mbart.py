#!/usr/bin/env python3
"""LoRA fine-tune facebook/mbart-large-50 for Tagalog or Hiligaynon → English.

Two modes (do not conflate):

  --lang tl
      Use existing ``tl_XX``. LoRA on q_proj/v_proj (r=16, alpha=32).

  --lang hil
      Add ``hil_XX`` to the tokenizer, resize embeddings, copy ``tl_XX``'s
      embedding into the new row, keep that embedding **fully trainable**
      (unfreeze ``model.shared`` + save ``shared_embeddings.pt``), LoRA the
      rest, and use a higher embedding LR.

Requirements:
  pip install "transformers>=4.40" datasets accelerate peft sentencepiece protobuf torch

Example:
  python scripts/ph_mt/finetune_mbart.py --lang tl \\
    --train-jsonl scripts/ph_mt/prepared/tl/train.jsonl \\
    --eval-jsonl scripts/ph_mt/prepared/tl/eval.jsonl \\
    --output-dir models/mbart-tl-en-lora --num-train-epochs 1

  python scripts/ph_mt/finetune_mbart.py --lang hil \\
    --train-jsonl scripts/ph_mt/prepared/hil/train.jsonl \\
    --eval-jsonl scripts/ph_mt/prepared/hil/eval.jsonl \\
    --output-dir models/mbart-hil-en-lora --embed-lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow sibling imports when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPT_DIR))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _add_hil_xx(tokenizer, model):
    """Register hil_XX, resize embeddings, init from tl_XX copy."""
    import torch

    from _hil_xx_tokenizer import register_hil_xx_lang_code

    hil_id = register_hil_xx_lang_code(tokenizer)
    print(f"hil_XX token id={hil_id}")

    old_n = model.get_input_embeddings().weight.shape[0]
    new_n = len(tokenizer)
    if new_n != old_n:
        model.resize_token_embeddings(new_n)
        print(f"Resized embeddings {old_n} → {new_n}")

    tl_id = tokenizer.lang_code_to_id.get("tl_XX")
    if tl_id is None:
        tl_id = tokenizer.convert_tokens_to_ids("tl_XX")
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[hil_id] = emb[tl_id].clone()
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.shape[0] > hil_id:
            out_emb.weight[hil_id] = out_emb.weight[tl_id].clone()
    print(f"Initialized hil_XX embedding from tl_XX (tl_id={tl_id})")
    return hil_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", choices=("tl", "hil"), required=True)
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
    p.add_argument(
        "--embed-lr",
        type=float,
        default=5e-4,
        help="Higher LR for new hil_XX embedding rows (hil mode only)",
    )
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
            "sentencepiece protobuf torch sacrebleu\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    train_rows = _load_jsonl(args.train_jsonl)
    if not train_rows:
        print("Training JSONL is empty.", file=sys.stderr)
        return 1

    default_src = "tl_XX" if args.lang == "tl" else "hil_XX"
    # Normalize rows to the intended source tag.
    for row in train_rows:
        row["src_lang"] = default_src
        row["tgt_lang"] = "en_XX"
        row["language"] = args.lang

    print(f"Loading base model {args.model_name} …")
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_name)
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)

    has_hil_xx = False
    if args.lang == "hil":
        _add_hil_xx(tokenizer, model)
        has_hil_xx = True

    tokenizer.src_lang = default_src
    tokenizer.tgt_lang = "en_XX"

    if not args.no_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            print(f"peft is required for LoRA training: {exc}", file=sys.stderr)
            return 2
        # LoRA only on attention projections. PEFT modules_to_save("embed_tokens")
        # breaks mBART's MBartScaledWordEmbedding / tied lm_head (shape error on
        # forward). For hil_XX we instead unfreeze ``model.shared`` after wrapping
        # and persist those weights beside the adapter (see save below).
        lora = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora)
        if args.lang == "hil":
            unfrozen = 0
            for name, param in model.named_parameters():
                # Tied input embedding lives on ``…model.shared.weight``.
                if name.endswith("model.shared.weight") or name.endswith(
                    ".shared.weight"
                ):
                    param.requires_grad = True
                    unfrozen += param.numel()
            print(
                f"Unfroze shared embedding for hil_XX "
                f"({unfrozen:,} params, full trainable — not LoRA)"
            )
        model.print_trainable_parameters()

    def preprocess(example: dict) -> dict:
        tokenizer.src_lang = example.get("src_lang") or default_src
        model_inputs = tokenizer(
            example["src"],
            text_target=example["tgt"],
            max_length=args.max_source_length,
            truncation=True,
            padding=False,
        )
        labels = model_inputs.get("labels")
        if isinstance(labels, list) and len(labels) > args.max_target_length:
            model_inputs["labels"] = labels[: args.max_target_length]
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": model_inputs["labels"],
        }

    train_ds = Dataset.from_list(train_rows).map(preprocess)
    keep = {"input_ids", "attention_mask", "labels"}
    drop = [c for c in train_ds.column_names if c not in keep]
    if drop:
        train_ds = train_ds.remove_columns(drop)

    eval_ds = None
    if args.eval_jsonl and args.eval_jsonl.exists():
        eval_rows = _load_jsonl(args.eval_jsonl)
        for row in eval_rows:
            row["src_lang"] = default_src
            row["tgt_lang"] = "en_XX"
        if eval_rows:
            eval_ds = Dataset.from_list(eval_rows).map(preprocess)
            drop_e = [c for c in eval_ds.column_names if c not in keep]
            if drop_e:
                eval_ds = eval_ds.remove_columns(drop_e)

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
        save_steps=100,
        logging_steps=5,
        predict_with_generate=False,
        report_to=[],
        remove_unused_columns=True,
        save_total_limit=2,
        warmup_steps=10,
        dataloader_pin_memory=False,
    )

    embed_lr = float(args.embed_lr)

    class _EmbedLRTrainer(Seq2SeqTrainer):
        """Higher LR for embedding / lm_head params (hil_XX cold-start)."""

        def create_optimizer(self):
            if args.lang != "hil" or args.no_lora:
                return super().create_optimizer()
            decay = set()
            no_decay = set()
            embed = set()
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                is_embed = any(
                    k in n for k in ("embed_tokens", "shared", "lm_head", "modules_to_save")
                )
                if is_embed:
                    embed.add(n)
                elif any(nd in n for nd in ("bias", "LayerNorm", "layer_norm")):
                    no_decay.add(n)
                else:
                    decay.add(n)
            named = dict(self.model.named_parameters())
            groups = []
            if decay:
                groups.append(
                    {
                        "params": [named[n] for n in sorted(decay)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    }
                )
            if no_decay:
                groups.append(
                    {
                        "params": [named[n] for n in sorted(no_decay)],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    }
                )
            if embed:
                groups.append(
                    {
                        "params": [named[n] for n in sorted(embed)],
                        "weight_decay": 0.0,
                        "lr": embed_lr,
                    }
                )
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, self.model
            )
            # Drop lr from kwargs — per-group lr wins.
            optimizer_kwargs = dict(optimizer_kwargs)
            optimizer_kwargs.pop("lr", None)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
            print(
                f"Optimizer groups: decay={len(decay)} nodecay={len(no_decay)} "
                f"embed={len(embed)} (embed_lr={embed_lr})"
            )
            return self.optimizer

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    try:
        trainer = _EmbedLRTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = _EmbedLRTrainer(**trainer_kwargs, tokenizer=tokenizer)

    trainer.train()
    trainer.save_model(str(args.output_dir))
    # Avoid KeyError on reload: MBart50TokenizerFast.__init__ looks up src_lang
    # in the stock lang_code_to_id map, which does not include hil_XX until we
    # re-register it. Save with a stock src_lang, plus a sidecar map.
    if args.lang == "hil":
        from _hil_xx_tokenizer import register_hil_xx_lang_code

        hil_id = register_hil_xx_lang_code(tokenizer)
        (args.output_dir / "extra_lang_codes.json").write_text(
            json.dumps({"hil_XX": hil_id}, indent=2), encoding="utf-8"
        )
        tokenizer.src_lang = "tl_XX"
        tokenizer.tgt_lang = "en_XX"
    tokenizer.save_pretrained(str(args.output_dir))

    # Persist resized + trained shared embedding for hil merges (LoRA adapter
    # alone does not store base embedding updates without modules_to_save).
    if args.lang == "hil":
        import torch

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        # PeftModel → MBartForConditionalGeneration → MBartModel.shared
        core = base.model if hasattr(base, "model") else base
        shared = getattr(core, "shared", None)
        if shared is None and hasattr(core, "model"):
            shared = getattr(core.model, "shared", None)
        if shared is None:
            print("WARNING: could not locate shared embedding to save", file=sys.stderr)
        else:
            emb_path = args.output_dir / "shared_embeddings.pt"
            torch.save(
                {
                    "shared.weight": shared.weight.detach().cpu().clone(),
                    "vocab_size": int(shared.weight.shape[0]),
                    "has_hil_xx": True,
                },
                emb_path,
            )
            print(f"Saved trainable shared embeddings → {emb_path}")

    meta = {
        "base_model": args.model_name,
        "lang": args.lang,
        "lora": not args.no_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "v_proj"],
        "train_rows": len(train_rows),
        "src_lang": default_src,
        "tgt_lang": "en_XX",
        "has_hil_xx": has_hil_xx,
        "embed_lr": embed_lr if args.lang == "hil" else None,
        "learning_rate": args.learning_rate,
        "note": (
            "Tagalog uses native tl_XX."
            if args.lang == "tl"
            else (
                "Hiligaynon uses new hil_XX (init from tl_XX). "
                "Merge with merge_lora.py before MBART_PH_FINE_TUNED_MODEL."
            )
        ),
    }
    (args.output_dir / "finetune_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"Saved checkpoint to {args.output_dir}")
    if not args.no_lora:
        merged = (
            "models/mbart-tl-en-merged"
            if args.lang == "tl"
            else "models/mbart-hil-en-merged"
        )
        print("Merge adapters:")
        print(
            f"  python scripts/ph_mt/merge_lora.py "
            f"--adapter-dir {args.output_dir} "
            f"--output-dir {merged}"
        )
    print("Then evaluate:")
    print(
        f"  python scripts/ph_mt/evaluate_mbart_checkpoint.py "
        f"--checkpoint <merged> --lang {args.lang} "
        f"--domain-jsonl scripts/ph_mt/fixtures/"
        f"{'tagalog' if args.lang == 'tl' else 'hiligaynon'}_en_sample.jsonl"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
