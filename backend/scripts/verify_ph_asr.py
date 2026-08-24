"""Verify Tagalog/Hiligaynon Whisper call-site wiring (live + final).

Logs resolved language, prompt, and model for each path. Uses the saved
meeting WAV as speech audio (format already validated as 16 kHz mono PCM).
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np

from app.services import audio, asr, transcription


def _window(samples: np.ndarray, start_s: float = 5.0, dur_s: float = 10.0):
    sr = 16000
    a = int(start_s * sr)
    b = int((start_s + dur_s) * sr)
    return samples[a:b]


def _report(label: str, result: asr.ASRResult, *, decode_hint: str) -> None:
    print(f"=== {label} ===")
    print(f"  decode_hint={decode_hint!r}")
    print(f"  engine={result.engine}")
    print(f"  language={result.language!r} conf={result.language_confidence}")
    print(f"  detected_by={result.language_detected_by!r}")
    print(f"  text={result.text[:180]!r}")
    print(f"  segments={len(result.segments)}")
    if result.segments:
        s0 = result.segments[0]
        print(
            f"  seg0 no_speech_prob={s0.no_speech_prob} "
            f"avg_logprob={s0.avg_logprob} low_conf={s0.low_confidence}"
        )


def main() -> int:
    wav = BACKEND / "data/audio/a4ce6fdc-f16e-44ae-a013-b727a3306562.wav"
    if not wav.exists():
        print("SKIP: no sample wav")
        return 0
    samples = audio.load_audio_float32(str(wav))
    assert samples.ndim == 1
    print(f"audio samples={samples.size} dur={samples.size/16000:.2f}s rms={float(np.sqrt(np.mean(samples**2))):.4f}")

    # --- Call-site wiring assertions (no model needed for these) ---
    assert transcription.whisper_language_arg("fil") == "tl"
    assert transcription.whisper_language_arg("hil") is None
    assert transcription.initial_prompt("tl")
    assert transcription.initial_prompt("hil")
    assert "Tagalog" in (transcription.initial_prompt("tl") or "")
    assert "Hiligaynon" in (transcription.initial_prompt("hil") or "")
    assert transcription._final_decode_language("tl") == "tl"
    assert transcription._final_decode_language("fil") == "tl"
    assert transcription._final_decode_language("hil") is None
    print("wiring_ok: fil→tl, hil→None, prompts present for live+final bias")

    win = _window(samples)

    # Tagalog — must force tl on live + final
    live_tl = asr.transcribe_pcm(win, "tl", live=True)
    _report("LIVE Tagalog (language=tl)", live_tl, decode_hint="tl")
    final_tl = asr.transcribe_pcm(samples, "tl", live=False)
    _report("FINAL Tagalog (language=tl)", final_tl, decode_hint="tl")

    # Hiligaynon — auto-detect + prompt; consistent session bias (no tl force)
    live_hil = asr.transcribe_pcm(win, "hil", live=True)
    _report("LIVE Hiligaynon (language=hil → auto-detect)", live_hil, decode_hint="None")
    final_hil = asr.transcribe_pcm(samples, "hil", live=False)
    _report("FINAL Hiligaynon (language=hil → auto-detect)", final_hil, decode_hint="None")

    # fil alias must behave like tl (never pass fil into Whisper)
    live_fil = asr.transcribe_pcm(win, "fil", live=True)
    _report("LIVE fil alias (must act as tl)", live_fil, decode_hint="tl")

    print("VERIFY_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
