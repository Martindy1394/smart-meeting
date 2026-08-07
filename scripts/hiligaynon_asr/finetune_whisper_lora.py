#!/usr/bin/env python3
"""LoRA fine-tune Whisper for Hiligaynon (Smart Meeting ASR).

Adapted from the SmartScribe Hiligaynon LoRA guide for this repo:

* Reads the same JSONL format as ``finetune_whisper.py`` / ``prepare_whisper_pld.py``
  (``{"audio": "...", "text": "..."}``).
* Uses PEFT LoRA (r=64, alpha=128 by default) on attention + FFN projections.
* Does **not** force Whisper ``language=hil`` / ``hiligaynon`` — stock Whisper has
  no Hiligaynon token. Multilingual/auto decode matches runtime
  (``WHISPER_HILIGAYNON_FINAL_LANGUAGE_MODE=auto``).
* Saves LoRA adapters; merge with ``merge_whisper_lora.py`` before pointing
  ``WHISPER_HILIGAYNON_FINE_TUNED_MODEL`` at the checkpoint.

Requirements (GPU strongly recommended)::

  pip install "transformers>=4.40" datasets accelerate peft torch \\
              librosa soundfile evaluate jiwer
  # Optional 4-bit (CUDA + bitsandbytes):
  # pip install bitsandbytes

Example (PLD-cleaned JSONL)::

  python3 scripts/hiligaynon_asr/prepare_whisper_pld.py \\
    --pld-root ./data/PLD --language hil --out-dir ./data/pld_hiligaynon_clean

  python3 scripts/hiligaynon_asr/finetune_whisper_lora.py \\
    --train-jsonl ./data/pld_hiligaynon_clean/train.jsonl \\
    --eval-jsonl ./data/pld_hiligaynon_clean/dev.jsonl \\
    --output-dir ./models/whisper-medium-hil-lora \\
    --model-name openai/whisper-medium \\
    --fp16

  python3 scripts/hiligaynon_asr/merge_whisper_lora.py \\
    --adapter-dir ./models/whisper-medium-hil-lora \\
    --output-dir ./models/whisper-medium-hiligaynon

  # backend/.env
  # WHISPER_HILIGAYNON_FINE_TUNED_MODEL=/abs/path/models/whisper-medium-hiligaynon
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        audio = obj.get("audio") or obj.get("path") or obj.get("file_name")
        text = obj.get("text") or obj.get("transcription") or obj.get("transcript")
        if not audio or text is None:
            continue
        rows.append({"audio": str(audio), "text": str(text)})
    return rows


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad Mel features + labels; mask label pads with -100."""

    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        bos = self.processor.tokenizer.bos_token_id
        if bos is not None and (labels[:, 0] == bos).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--test-jsonl", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model-name",
        default="openai/whisper-medium",
        help="Base Whisper (medium recommended; large-v3 needs more VRAM)",
    )
    p.add_argument(
        "--language",
        default=None,
        help=(
            "Optional Whisper language code for the processor. "
            "Leave unset for Hiligaynon (no hil token — matches runtime auto-detect). "
            "Do not pass tl for Hiligaynon."
        ),
    )
    p.add_argument("--num-train-epochs", type=float, default=10.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="q_proj,v_proj,out_proj,fc1,fc2",
        help="Comma-separated LoRA target module names",
    )
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="bitsandbytes 4-bit load (CUDA only)",
    )
    p.add_argument(
        "--eval-strategy",
        default="epoch",
        choices=("no", "steps", "epoch"),
    )
    args = p.parse_args(argv)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except Exception as exc:
        print(
            "Missing training deps. Install:\n"
            '  pip install "transformers>=4.40" datasets accelerate peft torch '
            "librosa soundfile evaluate jiwer\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        import librosa
    except Exception as exc:
        print(f"librosa is required to load audio: {exc}", file=sys.stderr)
        return 2

    train_rows = _load_jsonl(args.train_jsonl)
    if not train_rows:
        print("Training JSONL is empty.", file=sys.stderr)
        return 1

    processor_kwargs: dict[str, str] = {"task": "transcribe"}
    if args.language:
        # Never use tl for Hiligaynon; never invent hil (unsupported by Whisper).
        lang = args.language.strip().lower()
        if lang in {"hil", "hiligaynon", "ilonggo"}:
            print(
                "WARNING: Whisper has no hiligaynon/hil token — ignoring "
                f"--language {args.language!r} (using multilingual/auto).",
                file=sys.stderr,
            )
        else:
            processor_kwargs["language"] = lang

    print(f"Loading processor + base model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name, **processor_kwargs)

    model_kwargs: dict = {}
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            print("--load-in-4bit requires CUDA.", file=sys.stderr)
            return 1
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            print(f"bitsandbytes/BitsAndBytesConfig unavailable: {exc}", file=sys.stderr)
            return 2
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name, **model_kwargs
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if "language" in processor_kwargs:
        model.generation_config.language = processor_kwargs["language"]
    model.generation_config.task = "transcribe"
    # Let the fine-tune learn decode behavior; runtime uses auto-detect for hil.
    model.generation_config.forced_decoder_ids = None

    targets = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    sampling_rate = 16000

    def to_features(batch: dict) -> dict:
        speech, _ = librosa.load(batch["audio"], sr=sampling_rate, mono=True)
        feats = processor.feature_extractor(
            speech, sampling_rate=sampling_rate, return_tensors=None
        )
        labels = processor.tokenizer(batch["text"], return_tensors=None).input_ids
        return {
            "input_features": feats["input_features"][0],
            "labels": labels,
        }

    print(f"Mapping {len(train_rows)} train rows …")
    train_ds = Dataset.from_list(train_rows).map(to_features)
    eval_ds = None
    if args.eval_jsonl and args.eval_jsonl.exists():
        eval_rows = _load_jsonl(args.eval_jsonl)
        if eval_rows:
            print(f"Mapping {len(eval_rows)} eval rows …")
            eval_ds = Dataset.from_list(eval_rows).map(to_features)

    compute_metrics = None
    if eval_ds is not None:
        try:
            import evaluate

            wer_metric = evaluate.load("wer")

            def _compute_metrics(pred):
                pred_ids = pred.predictions
                label_ids = pred.label_ids
                label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
                pred_str = processor.tokenizer.batch_decode(
                    pred_ids, skip_special_tokens=True
                )
                label_str = processor.tokenizer.batch_decode(
                    label_ids, skip_special_tokens=True
                )
                wer = wer_metric.compute(predictions=pred_str, references=label_str)
                return {"wer": round(float(wer), 4)}

            compute_metrics = _compute_metrics
        except Exception as exc:
            print(f"WER metric unavailable ({exc}); training without compute_metrics.")

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    eval_strategy = args.eval_strategy if eval_ds is not None else "no"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_train_batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=use_fp16,
        gradient_checkpointing=True,
        eval_strategy=eval_strategy,
        save_strategy="epoch" if eval_strategy == "epoch" else "steps",
        save_steps=200,
        eval_steps=200 if eval_strategy == "steps" else None,
        logging_steps=25,
        predict_with_generate=bool(compute_metrics),
        generation_max_length=225,
        load_best_model_at_end=bool(compute_metrics and eval_strategy != "no"),
        metric_for_best_model="wer" if compute_metrics else None,
        greater_is_better=False if compute_metrics else None,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    try:
        trainer = Seq2SeqTrainer(
            **trainer_kwargs, processing_class=processor.feature_extractor
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            **trainer_kwargs, tokenizer=processor.feature_extractor
        )

    print(f"Starting LoRA fine-tune → {args.output_dir}")
    trainer.train()
    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    meta = {
        "base_model": args.model_name,
        "method": "lora",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": targets,
        "train_rows": len(train_rows),
        "language_forced": processor_kwargs.get("language"),
        "note": (
            "Hiligaynon: no Whisper hil token — train multilingual/auto. "
            "Merge adapters before WHISPER_HILIGAYNON_FINE_TUNED_MODEL."
        ),
    }
    (args.output_dir / "finetune_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if args.test_jsonl and args.test_jsonl.exists() and compute_metrics:
        test_rows = _load_jsonl(args.test_jsonl)
        if test_rows:
            test_ds = Dataset.from_list(test_rows).map(to_features)
            print("Evaluating on --test-jsonl …")
            results = trainer.evaluate(test_ds)
            print(f"Test metrics: {results}")
            (args.output_dir / "test_metrics.json").write_text(
                json.dumps({k: float(v) if hasattr(v, "item") else v
                            for k, v in results.items()}, indent=2),
                encoding="utf-8",
            )

    print(f"\nLoRA adapter saved to {args.output_dir}")
    print("Merge for Smart Meeting:")
    print(
        f"  python3 scripts/hiligaynon_asr/merge_whisper_lora.py "
        f"--adapter-dir {args.output_dir} "
        f"--output-dir models/whisper-medium-hiligaynon"
    )
    print("Then in backend/.env:")
    print("  WHISPER_HILIGAYNON_FINE_TUNED_MODEL=/abs/path/models/whisper-medium-hiligaynon")
    print("  WHISPER_FINAL_BACKEND=auto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
