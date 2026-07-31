# RNN-T live captions (optional)

Smart Meeting keeps **Whisper** for the final / re-transcribe pass. Live
captions can optionally use a **FastConformer hybrid RNN-T** (NeMo) for lower
latency on **Tagalog** meetings.

## Why RNN-T here

| Path | Backend | Role |
|------|---------|------|
| Live WebSocket captions (Tagalog) | RNN-T (optional) → Whisper fallback | Low-latency streaming |
| Live captions (Hiligaynon / auto) | Whisper + Hiligaynon prompt | Ilonggo words |
| Stop / re-transcribe | Whisper + PH-medium | Best offline accuracy |

RNN-T does **not** replace Hiligaynon fine-tuning. There is still no public
Hiligaynon RNNT. The Tagalog checkpoint
(`NCSpeech/stt_tl_fastconformer_hybrid_large`) is only used when the meeting
language resolves to Tagalog — not for Hiligaynon-biased `auto` meetings.

## Enable

```bash
cd backend
pip install -r requirements-rnnt.txt
```

In `.env`:

```bash
# auto = RNNT for Tagalog when NeMo is installed; Hiligaynon stays on Whisper
WHISPER_LIVE_BACKEND=auto
RNNT_LIVE_MODEL=NCSpeech/stt_tl_fastconformer_hybrid_large
# Force Whisper live only:
# WHISPER_LIVE_BACKEND=whisper
```

Restart the API. Check `/api/health` (debug) for `rnnt_live` status.

## Hiligaynon next step

Fine-tune the Tagalog FastConformer-Hybrid on labeled Ilonggo board audio
(NeMo RNNT fine-tune), then point `RNNT_LIVE_MODEL` at your `.nemo` file and
extend `should_use_rnnt_live` for Hiligaynon. Keep Whisper final +
`WHISPER_HILIGAYNON_FINE_TUNED_MODEL` for minutes quality.
