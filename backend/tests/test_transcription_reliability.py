"""Unit tests for transcription reliability helpers (no Whisper weights required)."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

# Allow `python -m pytest` / direct run from repo root or backend/.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.transcription import (  # noqa: E402
    LanguageDetection,
    _ModelCache,
    _final_decode_language,
    _final_language_mode,
    _forced_language,
    _language_detection_from_info,
    auto_hf_candidates,
    effective_asr_language,
    final_faster_model_id,
    hiligaynon_hf_candidates,
    hiligaynon_model_id,
    initial_prompt,
    is_auto_language,
    is_hiligaynon_language,
    is_philippine_language,
    is_tagalog_language,
    live_model_id,
    merge_live_caption,
    resolve_final_backend,
    set_model_cache,
    tagalog_hf_candidates,
)


def test_merge_live_caption_appends_novel_overlap():
    prev = "good morning everyone welcome to the board meeting"
    prev_window = "welcome to the board meeting today we will"
    cur_window = "board meeting today we will discuss the budget"
    merged = merge_live_caption(prev, cur_window, previous_window=prev_window)
    assert merged.startswith(prev)
    assert "discuss the budget" in merged
    assert len(merged.split()) >= len(prev.split())


def test_merge_live_caption_never_shrinks():
    prev = "one two three four five six seven eight"
    merged = merge_live_caption(prev, "one two three", previous_window=prev)
    assert len(merged.split()) >= len(prev.split())


def test_model_cache_lru_eviction():
    cache = _ModelCache(max_models=2)
    set_model_cache(cache)
    try:
        with cache._lock:
            cache._fw_models["a"] = object()
            cache._fw_locks["a"] = threading.Lock()
            cache._touch("fw", "a")
            cache._fw_models["b"] = object()
            cache._fw_locks["b"] = threading.Lock()
            cache._touch("fw", "b")
            # Touch a so b becomes the LRU victim when c is inserted.
            cache._touch("fw", "a")
            cache._fw_models["c"] = object()
            cache._fw_locks["c"] = threading.Lock()
            cache._touch("fw", "c")
            assert "b" not in cache._fw_models
            assert "a" in cache._fw_models
            assert "c" in cache._fw_models
            assert cache.stats()["max_models"] == 2
    finally:
        set_model_cache(None)


def test_per_model_locks_are_distinct():
    cache = _ModelCache(max_models=2)
    lock_a = cache.fw_infer_lock("small")
    lock_b = cache.fw_infer_lock("medium")
    assert lock_a is not lock_b
    assert cache.fw_infer_lock("small") is lock_a


def test_hiligaynon_maps_to_whisper_tl():
    assert is_philippine_language("hil")
    assert is_philippine_language("Hiligaynon")
    assert is_philippine_language("tl")
    assert not is_philippine_language("en")
    # Whisper has no native hil token — decode uses configured PH code.
    assert _forced_language("hil") == "tl"
    assert _forced_language("fil") == "tl"
    assert _forced_language("en") == "en"


def test_auto_language_resolves_to_hiligaynon_default():
    assert is_auto_language("auto")
    assert is_auto_language("detect")
    assert is_auto_language(None)  # unset still counts as auto
    assert is_philippine_language("auto")
    # Product default: auto → Hiligaynon bias (no manual Spoken language step).
    assert effective_asr_language("auto") == "hil"
    assert effective_asr_language(None) == "hil"
    assert effective_asr_language("") == "hil"
    assert effective_asr_language("en") == "en"
    assert is_hiligaynon_language(effective_asr_language("auto"))
    assert _final_language_mode("auto") == "prefer_forced"
    assert _final_decode_language("auto") == "tl"
    prompt = initial_prompt("auto")
    assert prompt
    assert "Hiligaynon" in prompt or "Ilonggo" in prompt


def test_tagalog_uses_native_tl_and_prefer_forced():
    assert is_tagalog_language("tl")
    assert is_tagalog_language("Tagalog")
    assert is_tagalog_language("fil")
    assert is_tagalog_language("filipino")
    assert not is_tagalog_language("hil")
    assert _forced_language("tl") == "tl"
    assert _forced_language("tagalog") == "tl"
    assert _final_language_mode("tl") == "prefer_forced"
    assert _final_decode_language("tl") == "tl"
    prompt = initial_prompt("tl")
    assert prompt
    assert "Tagalog" in prompt or "Filipino" in prompt


def test_auto_final_backend_prefers_hf_for_hiligaynon(monkeypatch=None):
    # Hiligaynon prefers HF PH dialect model (Visayan-aware) then FW fallback.
    from app.config import settings

    prev = settings.whisper_final_backend
    prev_live = settings.whisper_live_hiligaynon_model
    prev_ft = settings.whisper_hiligaynon_fine_tuned_model
    prev_ph = settings.whisper_hiligaynon_model
    try:
        settings.whisper_final_backend = "auto"
        settings.whisper_hiligaynon_fine_tuned_model = ""
        settings.whisper_hiligaynon_model = "rbcurzon/whisper-medium-ph"
        assert resolve_final_backend("hil") == "huggingface"
        assert resolve_final_backend("en") == "faster-whisper"
        assert hiligaynon_hf_candidates() == ["rbcurzon/whisper-medium-ph"]
        assert hiligaynon_model_id() == "rbcurzon/whisper-medium-ph"

        settings.whisper_hiligaynon_fine_tuned_model = "/models/my-hil-ft"
        assert resolve_final_backend("hil") == "huggingface"
        assert hiligaynon_hf_candidates() == [
            "/models/my-hil-ft",
            "rbcurzon/whisper-medium-ph",
        ]
        assert hiligaynon_model_id() == "/models/my-hil-ft"

        settings.whisper_live_hiligaynon_model = "/models/hil-ct2"
        assert live_model_id("hil") == "/models/hil-ct2"
        assert live_model_id("en") == settings.whisper_live_model
        # Final FW always uses configured final model, not live CT2.
        assert final_faster_model_id("hil") == settings.whisper_final_model
    finally:
        settings.whisper_final_backend = prev
        settings.whisper_live_hiligaynon_model = prev_live
        settings.whisper_hiligaynon_fine_tuned_model = prev_ft
        settings.whisper_hiligaynon_model = prev_ph


def test_auto_final_backend_prefers_hf_for_tagalog():
    from app.config import settings

    prev = settings.whisper_final_backend
    prev_ft = settings.whisper_tagalog_fine_tuned_model
    prev_tl = settings.whisper_tagalog_model
    prev_ph = settings.whisper_hiligaynon_model
    prev_live = settings.whisper_live_tagalog_model
    try:
        settings.whisper_final_backend = "auto"
        settings.whisper_tagalog_fine_tuned_model = ""
        settings.whisper_tagalog_model = "LWobole/whisper-small-tagalog"
        settings.whisper_hiligaynon_model = "rbcurzon/whisper-medium-ph"
        assert resolve_final_backend("tl") == "huggingface"
        assert resolve_final_backend("fil") == "huggingface"
        assert tagalog_hf_candidates() == [
            "LWobole/whisper-small-tagalog",
            "rbcurzon/whisper-medium-ph",
        ]

        settings.whisper_tagalog_fine_tuned_model = "/models/my-tl-ft"
        assert tagalog_hf_candidates() == [
            "/models/my-tl-ft",
            "LWobole/whisper-small-tagalog",
            "rbcurzon/whisper-medium-ph",
        ]

        settings.whisper_live_tagalog_model = "/models/tl-ct2"
        assert live_model_id("tl") == "/models/tl-ct2"
        assert live_model_id("hil") != "/models/tl-ct2"
        assert final_faster_model_id("tl") == settings.whisper_final_model
    finally:
        settings.whisper_final_backend = prev
        settings.whisper_tagalog_fine_tuned_model = prev_ft
        settings.whisper_tagalog_model = prev_tl
        settings.whisper_hiligaynon_model = prev_ph
        settings.whisper_live_tagalog_model = prev_live


def test_visayan_markers_boost_quality_and_strip_prompt_echo():
    from app.services.transcription import (
        Segment,
        _candidate_quality_score,
        _strip_initial_prompt_echo,
    )

    hil = [
        Segment(
            text=(
                "Buwas naman nakapoy na ko mangita sang kostum. "
                "Wala gid ko kabalo. Wala gid problema."
            ),
            start=0.0,
            end=20.0,
        )
    ]
    tl_ish = [
        Segment(
            text=(
                "Bukas naman pagod na ako maghanap ng costume. "
                "Wala akong alam. Wala problema."
            ),
            start=0.0,
            end=20.0,
        )
    ]
    assert _candidate_quality_score(hil, 20.0) > _candidate_quality_score(tl_ish, 20.0)

    echoed = "Sang Nga Mga Kostyo Buwas naman nakapoy na ko"
    # Old long prompt words should not wipe real speech tokens we keep.
    cleaned = _strip_initial_prompt_echo(
        echoed,
        "Board meeting in Hiligaynon Ilonggo and English. Maayong aga. Indi. Kita. Sang. Nga.",
    )
    assert "nakapoy" in cleaned.lower()
    assert "Buwas" in cleaned or "buwas" in cleaned.lower()


def test_auto_meeting_uses_combined_ph_hf_candidates():
    from app.config import settings

    prev = settings.whisper_final_backend
    prev_hil_ft = settings.whisper_hiligaynon_fine_tuned_model
    prev_tl_ft = settings.whisper_tagalog_fine_tuned_model
    prev_tl = settings.whisper_tagalog_model
    prev_ph = settings.whisper_hiligaynon_model
    prev_live_hil = settings.whisper_live_hiligaynon_model
    try:
        settings.whisper_final_backend = "auto"
        settings.whisper_hiligaynon_fine_tuned_model = ""
        settings.whisper_tagalog_fine_tuned_model = ""
        settings.whisper_tagalog_model = "LWobole/whisper-small-tagalog"
        settings.whisper_hiligaynon_model = "rbcurzon/whisper-medium-ph"
        # auto → Hiligaynon default; uses PH dialect HF candidates.
        assert resolve_final_backend("auto") == "huggingface"
        # auto_hf_candidates still lists medium-PH for Tagalog/legacy callers.
        assert auto_hf_candidates() == [
            "rbcurzon/whisper-medium-ph",
        ]
        settings.whisper_live_hiligaynon_model = "/models/hil-ct2"
        assert live_model_id("auto") == "/models/hil-ct2"
    finally:
        settings.whisper_final_backend = prev
        settings.whisper_hiligaynon_fine_tuned_model = prev_hil_ft
        settings.whisper_tagalog_fine_tuned_model = prev_tl_ft
        settings.whisper_tagalog_model = prev_tl
        settings.whisper_hiligaynon_model = prev_ph
        settings.whisper_live_hiligaynon_model = prev_live_hil


def test_amplify_for_asr_boosts_quiet_audio():
    from app.services.audio import amplify_for_asr

    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(16000).astype(np.float32) * 0.01)
    # One loud click should not prevent boosting the quiet body.
    quiet[100] = 0.6
    boosted = amplify_for_asr(quiet, target_rms=0.1, max_gain=20.0)
    quiet_rms = float(np.sqrt(np.mean(np.square(quiet))))
    boost_rms = float(np.sqrt(np.mean(np.square(boosted))))
    assert boost_rms > quiet_rms * 2
    assert float(np.max(np.abs(boosted))) <= 1.0


def test_ellipsis_spam_is_junk():
    from app.services.transcription import _is_junk_transcript

    spam = (
        "magtest?... iyas o?... sa usa sa...... dito sa may...... "
        "dundi research hub ang...... asa piyang?... dili sir diri na...... "
        "ah dili lang nga building?... iyes magdiriqo?... dundi sir diri na?... "
        "o dundi lang nga building?... iyon magdiri diri......"
    )
    assert _is_junk_transcript(spam)


def test_candidate_quality_score_prefers_diverse_coverage():
    from app.services.transcription import Segment, _candidate_quality_score

    weak = [
        Segment(text="ang ang ang ang ang ang ang ang", start=0.0, end=10.0),
    ]
    strong = [
        Segment(
            text="Good morning. Today we discuss the budget and hiring plan.",
            start=0.0,
            end=10.0,
        ),
    ]
    # Same rough duration; diverse text should score higher than repetition.
    assert _candidate_quality_score(strong, 10.0) > _candidate_quality_score(weak, 10.0)


def test_collapse_phrase_and_sentence_loops():
    from app.services.transcription import _collapse_hallucinations, _is_junk_transcript

    looped = (
        "Sa kanil ang kanil ang kanil ang kanil ang kanil ang kanil "
        "ang kanil ang kanil ang kanil ang kanil"
    )
    collapsed = _collapse_hallucinations(looped)
    assert collapsed.lower().count("kanil") <= 2

    sentences = "I don't know why I'm so confused. " * 8
    collapsed2 = _collapse_hallucinations(sentences.strip())
    assert collapsed2.lower().count("confused") <= 2

    assert _is_junk_transcript(looped) or len(collapsed.split()) < 8


def test_language_detection_from_whisper_auto():
    class _Info:
        language = "tl"
        language_probability = 0.92

    det = _language_detection_from_info(_Info(), decode_language=None)
    assert det.language == "tl"
    assert det.confidence == 0.92
    assert det.detected_by == "whisper"
    assert det.as_dict() == {
        "language": "tl",
        "confidence": 0.92,
        "detected_by": "whisper",
    }


def test_language_detection_forced_fallback():
    class _Info:
        language = "tl"
        language_probability = 0.41

    det = _language_detection_from_info(_Info(), decode_language="tl")
    assert det.language == "tl"
    assert det.confidence == 0.41
    assert det.detected_by == "forced_fallback"


def test_language_detection_clamps_confidence():
    class _Info:
        language = "en"
        language_probability = 1.7

    det = _language_detection_from_info(_Info(), decode_language=None)
    assert det.confidence == 1.0
    assert isinstance(LanguageDetection(language="en").as_dict()["detected_by"], str)


if __name__ == "__main__":
    test_merge_live_caption_appends_novel_overlap()
    test_merge_live_caption_never_shrinks()
    test_model_cache_lru_eviction()
    test_per_model_locks_are_distinct()
    test_hiligaynon_maps_to_whisper_tl()
    test_auto_language_detects_and_uses_ph_models()
    test_tagalog_uses_native_tl_and_prefer_forced()
    test_auto_final_backend_prefers_hf_for_hiligaynon()
    test_auto_final_backend_prefers_hf_for_tagalog()
    test_auto_meeting_uses_combined_ph_hf_candidates()
    test_candidate_quality_score_prefers_diverse_coverage()
    test_collapse_phrase_and_sentence_loops()
    test_language_detection_from_whisper_auto()
    test_language_detection_forced_fallback()
    test_language_detection_clamps_confidence()
    print("all_unit_tests_passed")
