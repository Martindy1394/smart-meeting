#!/usr/bin/env python3
"""Phase 1 audit: mBART/NLLB PH translation path on fixtures + real meetings."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.database import SessionLocal
from app.languages import LANGUAGES, mbart_code
from app.models import Meeting
from app.services import google_translate, lang_router, llm


def _token_f1(hyp: str, ref: str) -> float:
    h = {t.lower() for t in hyp.split() if len(t) > 2}
    r = {t.lower() for t in ref.split() if len(t) > 2}
    if not h or not r:
        return 0.0
    inter = len(h & r)
    if not inter:
        return 0.0
    prec = inter / len(h)
    rec = inter / len(r)
    return 2 * prec * rec / (prec + rec)


def main() -> int:
    has_ph = bool((settings.mbart_ph_finetuned_model or "").strip())
    backend = (settings.ph_translate_backend or "auto").strip().lower()
    prefer_mbart = backend == "mbart" or (backend == "auto" and has_ph)
    use_nllb = backend != "mbart"

    print("=== CONFIG ===")
    print("ph_translate_backend=", settings.ph_translate_backend)
    print("mbart_ph_finetuned=", repr(settings.mbart_ph_finetuned_model))
    print("prefer_mbart=", prefer_mbart, "use_nllb=", use_nllb)
    print("google_configured=", google_translate.is_configured())

    print("\n=== mbart_code resolution ===")
    for code in [
        "en",
        "tl",
        "id",
        "hil",
        "fil",
        "tagalog",
        "hiligaynon",
        "unknown_xx",
        "ceb",
        "auto",
    ]:
        resolved = mbart_code(code)
        print(
            f"  {code!r:12} -> {resolved!r:8} "
            f"in_table={code in LANGUAGES} "
            f"would_silent_en_XX={resolved is None}"
        )

    fixtures: list[tuple[str, str, str]] = []
    for name, lang in (
        ("tagalog_en_sample.jsonl", "tl"),
        ("hiligaynon_en_sample.jsonl", "hil"),
    ):
        path = REPO / "scripts/ph_mt/fixtures" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            fixtures.append((lang, obj["source"], obj["reference"]))

    print("\n=== FIXTURE classify + route attempts ===")
    for expect, src, _ref in fixtures:
        decision = lang_router.classify_line(src)
        attempts = llm._route_attempts_for_line(
            decision.language,
            prefer_mbart=prefer_mbart,
            has_ph_mbart=has_ph,
            use_nllb=use_nllb,
        )
        flag = "OK" if decision.language == expect else "MISMATCH"
        print(
            f"[{flag}] expect={expect} got={decision.language} "
            f"conf={decision.confidence:.2f} unc={decision.uncertain} "
            f"reason={decision.reason}"
        )
        print(f"  src: {src}")
        print(f"  scores={decision.scores}")
        print(f"  attempts={attempts}")
        mbart_tags = [
            (eng, src_code, mbart_code(src_code))
            for eng, src_code in attempts
            if eng == "mbart"
        ]
        print(f"  mbart_resolved={mbart_tags}")

    print("\n=== REAL meeting units ===")
    units: list[tuple[str, str]] = []
    db = SessionLocal()
    try:
        for meeting in db.query(Meeting).filter(Meeting.final_transcript.isnot(None)):
            for unit in llm._segment_idea_units(meeting.final_transcript or ""):
                if len(unit.split()) >= 3:
                    units.append((meeting.title or "?", unit))
    finally:
        db.close()

    counts: Counter[str] = Counter()
    uncertain_n = 0
    for title, unit in units:
        decision = lang_router.classify_line(unit)
        counts[decision.language] += 1
        uncertain_n += int(decision.uncertain)
        print(
            f"  {title[:8]:8} {decision.language:7} "
            f"c={decision.confidence:.2f} u={int(decision.uncertain)} "
            f"| {unit[:100]}"
        )
    print(
        "counts",
        dict(counts),
        "uncertain",
        uncertain_n,
        "/",
        len(units),
    )

    print("\n=== LIVE translate() on fixtures ===")
    buckets = {"ok": 0, "wrong": 0, "kept_source": 0}
    for expect, src, ref in fixtures:
        result = llm.translate(src, "en", source_language=expect)
        text = (result.text or "").strip()
        garbage = llm._is_garbage_english_translation(src, text)
        latin = llm._looks_like_latin_script(text)
        kept = (not text) or text == src
        f1 = _token_f1(text, ref)
        if kept:
            bucket = "kept_source"
        elif f1 < 0.15 or garbage:
            bucket = "wrong"
        else:
            bucket = "ok"
        buckets[bucket] += 1
        print(
            f"  [{bucket}] engine={result.engine} f1={f1:.2f} "
            f"garbage={garbage} latin={latin} routes={result.route_counts}"
        )
        print(f"    src: {src}")
        print(f"    hyp: {text[:180]}")
        print(f"    ref: {ref}")

    print("\nSUMMARY buckets:", buckets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
