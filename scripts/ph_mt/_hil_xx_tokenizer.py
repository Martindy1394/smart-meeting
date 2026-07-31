"""Helpers to register ``hil_XX`` on an mBART-50 tokenizer after reload."""
from __future__ import annotations

import json
from pathlib import Path


def register_hil_xx_lang_code(tokenizer) -> int:
    """Ensure ``hil_XX`` is in ``lang_code_to_id`` (token must already exist)."""
    hil_id = tokenizer.convert_tokens_to_ids("hil_XX")
    unk = getattr(tokenizer, "unk_token_id", None)
    if hil_id == unk:
        existing = list(tokenizer.additional_special_tokens or [])
        if "hil_XX" not in existing:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": existing + ["hil_XX"]}
            )
        hil_id = tokenizer.convert_tokens_to_ids("hil_XX")
    if hil_id == unk:
        raise RuntimeError("Failed to add hil_XX — still UNK")
    tokenizer.lang_code_to_id["hil_XX"] = hil_id
    if hasattr(tokenizer, "id_to_lang_code"):
        tokenizer.id_to_lang_code[hil_id] = "hil_XX"
    if hasattr(tokenizer, "fairseq_tokens_to_ids"):
        tokenizer.fairseq_tokens_to_ids["hil_XX"] = hil_id
    if hasattr(tokenizer, "fairseq_ids_to_tokens"):
        tokenizer.fairseq_ids_to_tokens[hil_id] = "hil_XX"
    return hil_id


def load_mbart_tokenizer(path: str | Path):
    """Load MBart50 tokenizer and restore hil_XX lang-code mapping if present."""
    from transformers import MBart50TokenizerFast

    path = Path(path)
    # Stock init fails if tokenizer_config src_lang is hil_XX (not in built-in map).
    # Patch config on disk if needed, then load.
    cfg_path = path / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        dirty = False
        for key in ("src_lang", "tgt_lang", "_src_lang"):
            if cfg.get(key) == "hil_XX":
                cfg[key] = "tl_XX" if key != "tgt_lang" else "en_XX"
                dirty = True
        if dirty:
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    tok = MBart50TokenizerFast.from_pretrained(str(path))
    extra = path / "extra_lang_codes.json"
    has_hil = (
        extra.exists()
        or "hil_XX" in (tok.additional_special_tokens or [])
        or tok.convert_tokens_to_ids("hil_XX") != tok.unk_token_id
    )
    if has_hil:
        hil_id = register_hil_xx_lang_code(tok)
        if not extra.exists():
            extra.write_text(json.dumps({"hil_XX": hil_id}, indent=2), encoding="utf-8")
    return tok
