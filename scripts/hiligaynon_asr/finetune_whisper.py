#!/usr/bin/env python3
"""Fine-tune OpenAI Whisper on a Hiligaynon JSONL dataset (Hugging Face).

This is the practical implementation of the “fine-tune whisper-medium on a
low-resource language” suggestion. Training happens here; Smart Meeting only
*loads* the resulting checkpoint via WHISPER_HILIGAYNON_FINE_TUNED_MODEL.

Requirements (install in a GPU/CPU training env):
  pip install "transformers>=4.40" datasets accelerate torch librosa soundfile

Example:
  python scripts/hiligaynon_asr/finetune_whisper.py \\
    --train-jsonl ./hil-train.jsonl \\
    --output-dir ./models/whisper-medium-hiligaynon \\
    --model-name openai/whisper-medium \\
    --num-train-epochs 3

SpeechBrain is also a valid trainer; this script uses Transformers because it
exports a checkpoint Smart Meeting can load with ``transformers`` pipelines
without an extra runtime dependency.
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
    p.add_argument("--model-name", default="openai/whisper-medium")
    p.add_argument(
        "--language",
        default=None,
        help=(
            "Optional Whisper language code for the processor. "
            "Omit for multilingual/auto (recommended for Hiligaynon / PLD). "
            "Do not pass tl for Hiligaynon."
        ),
    )
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    args = p.parse_args(argv)

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except Exception as exc:
        print(
            "Missing training deps. Install:\n"
            '  pip install "transformers>=4.40" datasets accelerate torch librosa soundfile\n'
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

    processor_kwargs = {"task": "transcribe"}
    if args.language:
        processor_kwargs["language"] = args.language
    processor = WhisperProcessor.from_pretrained(args.model_name, **processor_kwargs)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if args.language:
        model.generation_config.language = args.language
    model.generation_config.task = "transcribe"

    sampling_rate = 16000

    def to_features(batch: dict) -> dict:
        audio_path = batch["audio"]
        text = batch["text"]
        speech, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
        feats = processor.feature_extractor(
            speech, sampling_rate=sampling_rate, return_tensors=None
        )
        labels = processor.tokenizer(text, return_tensors=None).input_ids
        return {
            "input_features": feats["input_features"][0],
            "labels": labels,
        }

    train_ds = Dataset.from_list(train_rows).map(to_features)
    eval_ds = None
    if args.eval_jsonl and args.eval_jsonl.exists():
        eval_ds = Dataset.from_list(_load_jsonl(args.eval_jsonl)).map(to_features)

    def collate(features: list[dict]):
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        eval_strategy="steps" if eval_ds is not None else "no",
        save_steps=200,
        eval_steps=200 if eval_ds is not None else None,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    print(f"Saved fine-tuned Whisper checkpoint to {args.output_dir}")
    print("Point Smart Meeting at it:")
    print(f"  WHISPER_HILIGAYNON_FINE_TUNED_MODEL={args.output_dir.resolve()}")
    print("  WHISPER_FINAL_BACKEND=auto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
