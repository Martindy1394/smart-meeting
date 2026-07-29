# Hiligaynon language forcing (live + final)

## Design intent

**Never force Whisper’s Tagalog (`tl`) token for Hiligaynon or `auto→hil`
meetings.** Hiligaynon has no Whisper-native language id; forcing `tl` caused
bad Ilonggo transcripts historically.

## Runtime truth (matches `transcription.py`)

| Path | Function | Hiligaynon / `auto`→hil result |
|---|---|---|
| Forced-code helper | `_forced_language` | **`None`** (only `en` or Tagalog/`fil` → `tl`) |
| Final decode | `_final_decode_language` | **`None`** (always auto-detect for Hiligaynon) |
| Final mode | `_final_language_mode` | `WHISPER_HILIGAYNON_FINAL_LANGUAGE_MODE` default **`auto`** |
| Live decode | `transcribe_live` | `primary_lang = None` for hil/auto; prompt = Hiligaynon initial prompt |
| Live WS | `ws/transcription.py` | `effective_asr_language(meeting.language)` → default hil bias; calls `transcribe_live` |
| Optional RNN-T | `rnnt.should_use_rnnt_live` | **False** for Hiligaynon (Tagalog-only) |

Tagalog / Filipino meetings may still force `tl` when
`WHISPER_TAGALOG_FINAL_LANGUAGE_MODE=prefer_forced`.

## What *is* applied for Hiligaynon

1. Meeting language `auto` resolves ASR bias via `WHISPER_DEFAULT_LANGUAGE=hil`.
2. Hiligaynon **initial prompt** (live + final when configured).
3. Final HF candidate order: custom fine-tune → `rbcurzon/whisper-medium-ph` →
   faster-whisper `medium`.
4. Whisper **language auto-detect** (`language=None`), never `language="tl"`.

## Regression checks

```bash
python3 backend/tests/test_transcription_reliability.py
# asserts: _forced_language("hil"|"auto") is None
```

`/api/health` → `pipeline.whisper.hiligaynon_forced_language` is always `null`
when the detailed or pipeline map is present.
