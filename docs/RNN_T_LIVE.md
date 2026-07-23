# RNN-T live captions (optional)

Smart Meeting keeps **Whisper** for the final / re-transcribe pass. Live
captions can optionally use a **FastConformer hybrid RNN-T** (NeMo) for lower
latency on Philippine meetings.

## Why RNN-T here

| Path | Backend | Role |
|------|---------|------|
| Live WebSocket captions | RNN-T (optional) → Whisper fallback | Low-latency streaming |
| Stop / re-transcribe | Whisper + PH-medium | Best offline accuracy |

RNN-T does **not** replace Hiligaynon fine-tuning. There is still no public
Hiligaynon RNNT. The default checkpoint is Tagalog/Filipino
(`NCSpeech/stt_tl_fastconformer_hybrid_large`) and is used for Hiligaynon-biased
`auto` meetings as the closest PH streaming model.

## Enable

```bash
cd backend
pip install -r requirements-rnnt.txt
```

In `.env`:

```bash
# auto = RNNT for PH meetings when NeMo is installed, else Whisper
WHISPER_LIVE_BACKEND=auto
RNNT_LIVE_MODEL=NCSpeech/stt_tl_fastconformer_hybrid_large
# Force Whisper live only:
# WHISPER_LIVE_BACKEND=whisper
```

Restart the API. Check `/api/health` (debug) for `rnnt_live` status.

## Hiligaynon next step

Fine-tune the Tagalog FastConformer-Hybrid on labeled Ilonggo board audio
(NeMo RNNT fine-tune), then point `RNNT_LIVE_MODEL` at your `.nemo` file.
Keep Whisper final + `WHISPER_HILIGAYNON_FINE_TUNED_MODEL` for minutes quality.
