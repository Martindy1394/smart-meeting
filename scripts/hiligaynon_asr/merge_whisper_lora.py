#!/usr/bin/env python3
"""Merge a Whisper LoRA adapter into a full checkpoint for Smart Meeting.

Smart Meeting loads Hiligaynon fine-tunes via transformers
(``WHISPER_HILIGAYNON_FINE_TUNED_MODEL``). Merged folders work with both the
HF pipeline path and ``export_ct2.sh`` for live captions.

Example::

  python3 scripts/hiligaynon_asr/merge_whisper_lora.py \\
    --adapter-dir ./models/whisper-medium-hil-lora \\
    --output-dir ./models/whisper-medium-hiligaynon
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--base-model",
        default=None,
        help="Override base model (default: read from adapter finetune_meta.json)",
    )
    args = p.parse_args(argv)

    try:
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except Exception as exc:
        print(f"Missing deps (transformers/peft): {exc}", file=sys.stderr)
        return 2

    meta = {}
    meta_path = args.adapter_dir / "finetune_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base_id = (
        (args.base_model or "").strip()
        or meta.get("base_model")
        or "openai/whisper-medium"
    )

    print(f"Loading base {base_id}")
    base = WhisperForConditionalGeneration.from_pretrained(base_id)
    print(f"Loading adapter {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    merged = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir))
    # Prefer processor saved with the adapter (same tokenizer/feature extractor).
    try:
        proc = WhisperProcessor.from_pretrained(str(args.adapter_dir))
    except Exception:
        proc = WhisperProcessor.from_pretrained(base_id)
    proc.save_pretrained(str(args.output_dir))

    out_meta = {
        "base_model": base_id,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "merged": True,
        "method": "lora",
        "for": "WHISPER_HILIGAYNON_FINE_TUNED_MODEL",
    }
    (args.output_dir / "finetune_meta.json").write_text(
        json.dumps(out_meta, indent=2), encoding="utf-8"
    )
    print(f"Merged Whisper checkpoint → {args.output_dir}")
    print(f"Set WHISPER_HILIGAYNON_FINE_TUNED_MODEL={args.output_dir.resolve()}")
    print("Optional live CT2:")
    print(
        f"  ./scripts/hiligaynon_asr/export_ct2.sh "
        f"{args.output_dir} {args.output_dir}-ct2"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
