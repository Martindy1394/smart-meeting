#!/usr/bin/env python3
"""Merge a LoRA adapter into a full mBART checkpoint for Smart Meeting."""
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
        default="facebook/mbart-large-50-many-to-many-mmt",
    )
    args = p.parse_args(argv)

    try:
        from peft import PeftModel
        from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
    except Exception as exc:
        print(f"Missing deps (transformers/peft): {exc}", file=sys.stderr)
        return 2

    print(f"Loading base {args.base_model}")
    base = MBartForConditionalGeneration.from_pretrained(args.base_model)
    print(f"Loading adapter {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    merged = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir))
    tok = MBart50TokenizerFast.from_pretrained(str(args.adapter_dir))
    tok.save_pretrained(str(args.output_dir))

    meta = {
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "merged": True,
    }
    (args.output_dir / "merge_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Merged model saved to {args.output_dir}")
    print(f"Set MBART_PH_FINE_TUNED_MODEL={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
