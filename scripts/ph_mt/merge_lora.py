#!/usr/bin/env python3
"""Merge a LoRA adapter into a full mBART checkpoint for Smart Meeting.

Handles Hiligaynon ``hil_XX`` vocab extension:
  1. Resize base embeddings to match the adapter tokenizer
  2. Load LoRA and merge
  3. Overlay ``shared_embeddings.pt`` when present (trained hil_XX row)
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
        default="facebook/mbart-large-50-many-to-many-mmt",
    )
    args = p.parse_args(argv)

    try:
        import torch
        from peft import PeftModel
        from transformers import MBartForConditionalGeneration
    except Exception as exc:
        print(f"Missing deps (transformers/peft): {exc}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _hil_xx_tokenizer import load_mbart_tokenizer

    tok = load_mbart_tokenizer(args.adapter_dir)
    if "hil_XX" in getattr(tok, "lang_code_to_id", {}):
        print(f"Tokenizer has hil_XX id={tok.lang_code_to_id['hil_XX']}")
    print(f"Loading base {args.base_model}")
    base = MBartForConditionalGeneration.from_pretrained(args.base_model)

    base_n = base.get_input_embeddings().weight.shape[0]
    tok_n = len(tok)
    if tok_n != base_n:
        print(f"Resizing base embeddings {base_n} → {tok_n} (hil_XX extension)")
        base.resize_token_embeddings(tok_n)

    emb_path = args.adapter_dir / "shared_embeddings.pt"
    if emb_path.exists():
        blob = torch.load(emb_path, map_location="cpu", weights_only=True)
        weight = blob["shared.weight"]
        if weight.shape[0] != base.get_input_embeddings().weight.shape[0]:
            print(
                f"Embedding file vocab {weight.shape[0]} != model "
                f"{base.get_input_embeddings().weight.shape[0]}",
                file=sys.stderr,
            )
            return 1
        with torch.no_grad():
            base.model.shared.weight.copy_(weight)
            # Keep tied views consistent when present.
            if hasattr(base.model.encoder, "embed_tokens"):
                base.model.encoder.embed_tokens.weight = base.model.shared.weight
            if hasattr(base.model.decoder, "embed_tokens"):
                base.model.decoder.embed_tokens.weight = base.model.shared.weight
            if base.get_output_embeddings() is not None:
                base.get_output_embeddings().weight = base.model.shared.weight
        print(f"Loaded trained shared embeddings from {emb_path}")

    print(f"Loading adapter {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    merged = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir))
    tok.save_pretrained(str(args.output_dir))

    meta_in = {}
    meta_path = args.adapter_dir / "finetune_meta.json"
    if meta_path.exists():
        meta_in = json.loads(meta_path.read_text(encoding="utf-8"))

    meta = {
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "merged": True,
        "lang": meta_in.get("lang"),
        "has_hil_xx": bool(
            meta_in.get("has_hil_xx")
            or "hil_XX" in getattr(tok, "lang_code_to_id", {})
            or emb_path.exists()
        ),
        "src_lang": meta_in.get("src_lang"),
        "shared_embeddings_applied": emb_path.exists(),
    }
    (args.output_dir / "finetune_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (args.output_dir / "merge_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Merged model saved to {args.output_dir}")
    print(f"Set MBART_PH_FINE_TUNED_MODEL={args.output_dir.resolve()}")
    if meta["has_hil_xx"]:
        print("Checkpoint includes hil_XX — languages.py will map hil → hil_XX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
