# Model accuracy review — ASR re-check

**Date:** 2026-08-07 07:09 UTC
**Scope:** Whisper ASR (final multi-trial + live smoke) via production `asr.transcribe_file`.
**Artifacts:** `data/model_accuracy_review/report.json` → `asr`

---

## Whisper ASR

### Runtime

| Setting | Value |
|---|---|
| Final backend | `auto` |
| Tagalog HF candidates | `LWobole/whisper-small-tagalog, rbcurzon/whisper-medium-ph` |
| Hiligaynon HF candidates | `rbcurzon/whisper-medium-ph` |
| Hiligaynon fine-tune | `(empty)` |
| `whisper_language_arg(hil)` | `None` (auto) |

### Final pass — median of 5 trials (clean TTS)

| Clip | Lang | Median hyp | Med WER | Med F1 | Trial pass | Clip pass |
|---|---|---|---|---|---|---|
| `en_meeting.wav` | en | The board approved the quarterly budget after a careful review | **0.000** | **1.000** | 100% | ✓ |
| `en_action.wav` | en | Maria will send the report on Friday before the board meeting | **0.000** | **1.000** | 100% | ✓ |
| `tl_greeting.wav` | tl | Magandang umaga sa lahat kailangan natin nating gawin ang budget | **0.222** | **0.842** | 80% | ✓ |

**Bar:** median WER ≤ 0.25 or median token-F1 ≥ 0.75 (5 trials)
**Verdict: PASS** (mean median WER 0.074, mean median F1 0.947)
**Unstable clips** (trial variance): `tl_greeting.wav`

### Trial detail

**`en_meeting.wav`** best WER 0.000 · worst WER 0.200
- ✓ t1: WER 0.200 — `Approved the quarterly budget after a careful review`
- ✓ t2: WER 0.000 — `The Board approved the quarterly budget after a careful review`
- ✓ t3: WER 0.000 — `The board approved the quarterly budget after a careful review`
- ✓ t4: WER 0.000 — `The board approved the quarterly budget after a careful review`
- ✓ t5: WER 0.000 — `The board approved the quarterly budget after a careful review`

**`en_action.wav`** best WER 0.000 · worst WER 0.273
- ✓ t1: WER 0.000 — `Maria will send the report on Friday before the board meeting`
- ✓ t2: WER 0.091 — `Mariaa will send the report on Friday before the board meeting`
- ✓ t3: WER 0.273 — `Maria will send the report on Friday before!"`
- ✓ t4: WER 0.000 — `Maria will send the report on Friday before the board meeting`
- ✓ t5: WER 0.000 — `Maria will send the report on Friday before the board meeting`

**`tl_greeting.wav`** best WER 0.111 · worst WER 0.333
- ✓ t1: WER 0.222 — `Magandang umaga sa lahat kailangan natin nating gawin ang budget`
- ✓ t2: WER 0.222 — `Ang umaga sa lahat, kailangan nating aprobahan�도 ang budget`
- ✓ t3: WER 0.111 — `magandang umaga sa lahat kailangan nating apropahan ang budget`
- ✗ t4: WER 0.333 — `Magandang umaga sa cái lang nating aprobahan ang budget`
- ✓ t5: WER 0.111 — `Magandang umaga sa lahat, kailangan nating aprobahan ang budget`

**`hil_meeting.wav`** best WER 0.182 · worst WER 0.636
- ✗ t1: WER 0.364 — `Mayong aga sa tanan, kinahanglin Nathan aprobahan ang budget sang board`
- ✗ t2: WER 0.636 — `Mayong aga sa tanan Kinahang lan natin aprobahan ang bang board`
- ✗ t3: WER 0.455 — `Mayong aga sa tanan, kinahang lanit at aprobahanan ang budget sang board`
- ✓ t4: WER 0.182 — `mayong aga sa tanan kinahanglan niten aprubahan ang budget sang board`
- ✗ t5: WER 0.636 — `Mayo đó, aga sa tanan... Kinahanglin ni, tinaprobahan ang budget`

### Hiligaynon path (acoustic proxy)

gTTS has no Hiligaynon voice; Filipino TTS speaking Hiligaynon text.

**Routing:** `whisper_language_arg('hil') is None` → **PASS**

### Live smoke (`en_action.wav`)

| Engine | WER | F1 | Pass |
|---|---|---|---|
| whisper | 0.000 | 1.000 | ✓ |

### Accuracy assessment

**English works accurately** (final + live): median WER 0 on both clips; rare short truncations still within bar.

**Tagalog meets the median bar but is not stable:** 4/5 trials pass, 1/5 injects wrong-script tokens (`cái`, CJK). Same WAV can score WER 0.11 or 0.33 — do not treat Tagalog final as reliably deterministic yet.

**Hiligaynon routing works; accuracy not proven:** auto-detect (never force `tl`) PASS. Proxy TTS median WER 0.455 / 20% trial pass — not a native Ilonggo measurement. Need PLD/board labels + preferably a merged LoRA fine-tune in `WHISPER_HILIGAYNON_FINE_TUNED_MODEL`.

