# Speaker identification (introductions + attendance)

This document is the implementation plan for **named speaker identification**
during speech introductions, mapped onto Smart Meeting as it exists today.

Whisper **does not diarize**. The product already clusters talkers into anonymous
**Voice 1…N** (`live_speakers.py`). Named identity is a second stage that binds a
cluster to the attendance roster when someone introduces themselves.

## What ships in this pass (Phase 1)

| Capability | Status |
|---|---|
| Whisper transcription | Existing live `small` + final `medium` / PH HF (not Large-v3 by default) |
| Voice clustering | Existing spectral fingerprint + accuracy rank |
| Introduction keywords (EN / TL / Hil) | **Ships** — `speaker_id.extract_introduction` |
| Cross-check vs attendees + presiding officer | **Ships** — fuzzy roster match, officer not double-counted |
| Guest (temporary) enrollment | **Ships** — intro name not on the roster, marked guest |
| Correction memory | **Ships** — `POST /api/meetings/{id}/speakers/correct` |
| Attendance report | **Ships** — `GET /api/meetings/{id}/attendance` + export section |
| VAD before ASR | Existing webrtcvad / energy gate |
| PyAnnote / SpeechBrain embeddings | **Optional overlay** (`SPEAKER_ID_BACKEND=pyannote`) |
| Calendar pre-populate | **Not built** — roster is Meeting details; `meeting_date` is the schedule hook |
| Sub-2s neural ID on CPU | **Not claimed** — intro matching is millisecond-scale; pyannote is not |

## Target architecture

```
16 kHz PCM
    │
    ├─ VAD (skip silence)                  [existing]
    ├─ Whisper ASR (live window / final)   [existing]
    ├─ Voice cluster (pitch+mel+RMS)       [existing]
    └─ Speaker ID
           ├─ heuristic: intro phrases → roster / guest / corrections
           └─ optional: pyannote diarization timestamps (GPU + HF token)
    │
    └─ Attendance report (present / absent / guests / confidence / timestamps)
```

### Model selection (rationale)

**ASR — do not switch the default product models to Large-v3 on CPU.**  
Live captions already miss a 2s budget on CPU with `faster-whisper` `small`.
Large-v3 (~1.5B) is the right **offline / GPU** choice when speaker-ID quality
matters more than latency:

- Set `WHISPER_SPEAKER_ID_MODEL=large-v3` on a CUDA box if you add a dedicated
  ID decode later.
- Keep `WHISPER_LIVE_MODEL=small` and `WHISPER_FINAL_MODEL=medium` (or PH
  `rbcurzon/whisper-medium-ph`) for the thesis Windows CPU path.
- Faster-whisper CTranslate2 Large-v3 on GPU float16 is the practical way to
  run Large-v3 without the full PyTorch Whisper stack.

**Diarization / embeddings**

| Model | Role | Why |
|---|---|---|
| Current spectral clusters | Always-on Voice N | No extra deps, live-capable |
| `pyannote/speaker-diarization-3.1` | Overlap-aware turns | SOTA neural diarization; **gated HF model + token** |
| SpeechBrain `spkrec-ecapa-voxceleb` | Embeddings for re-ID | Strong ECAPA-TDNN; needs enrollment audio, not just names |
| Whisper Large-v3 | Better intro transcripts | Fewer mangled names (“Maria” vs “Marya”) feeding the matcher |

SpeechBrain embeddings only help after you **enroll** each attendee (10–20s of
clean speech). A name list is not an embedding gallery. Phase 1 therefore uses
**content** (introductions) plus the roster, which matches board-meeting
practice (“Ako si …”).

### Preprocessing

1. 16 kHz mono PCM (existing capture).
2. VAD: skip Whisper on silence (existing).
3. Optional noise: keep AGC as-is; do **not** enable aggressive final VAD
   (`whisper_final_vad_filter` defaults off) — it dropped real PH speech.
4. Strip `Voice N:` / timestamps before matching (existing `stripTranscriptMeta`).
5. Normalize names: titles (`Atty.`, `Dr.`), casefold, last-name fallback.

### Confidence thresholds (recommendations)

| Score | Action |
|---|---|
| ≥ **0.75** (`DISPLAY_THRESHOLD`) | Show the roster name on the chip and in exports |
| 0.50–0.74 | Keep Voice N; count as candidate / uncertain |
| Intro + exact roster | ~0.90–1.00 |
| Intro + last-name / fuzzy (≥ 0.82) | ~0.82–0.92 |
| Intro not on roster (guest) | cap **0.68** (needs a human if it is a real member) |
| Operator correction | **0.95** and stored as an alias |
| Low Whisper confidence | multiply identity by **0.7** |

Escalate to the operator (stay on Voice N) when score < 0.75 or two roster
names score within 0.05 of each other.

### Performance metrics (how to evaluate)

Run on a labeled board-meeting set (same talkers in Meeting details):

- **Identification accuracy** — share of Voice clusters whose displayed name
  matches the true talker, among clusters that had an introduction.
- **Attendance F1** — present/absent vs a human roll call.
- **Guest precision** — introductions not on the roster that really are guests.
- **DER** (diarization error rate) — only when pyannote is enabled; not defined
  for Voice-N spectral clusters.
- **Latency** — time from end of “Ako si X” to a named chip. Heuristic should
  stay **< 50 ms** after ASR; the ASR window itself is 2.5s warmup / 10s live.
- **Missed intro rate** — true introductions Whisper garbled so the regex never
  fires (this is why Large-v3 helps on GPU).

### Code layout

- `backend/app/services/speaker_id.py` — intros, roster match, attendance
- `backend/app/services/speaker_memory.py` — correction aliases
- `backend/app/services/live_speakers.py` — Voice clusters (unchanged algorithm)
- `POST /api/meetings/{id}/speakers/correct` — human-in-the-loop learning
- `GET /api/meetings/{id}/attendance` — roll-call JSON

## Phase 2 / 3 (not in this PR)

1. **Pyannote overlay** — `SPEAKER_ID_BACKEND=pyannote`, `HUGGINGFACE_TOKEN`,
   GPU. Align diarization turns to Whisper segments by midpoint. Mitigate:
   fall back to heuristic if import or token fails.
2. **SpeechBrain enrollment** — UI to record 10s per attendee; store ECAPA
   vectors under `data/speaker_memory/`; cosine match with threshold ~0.25
   (tune on your mics). Mitigate: never auto-rename below DISPLAY_THRESHOLD.
3. **Adaptive embeddings** — when a correction is accepted, average the new
   window embedding into that person’s gallery (capped N=8 utterances) so
   illness / emotion shifts are absorbed slowly.
4. **Overlap** — pyannote overlapping speech; until then, overlapping PCM
   stays one Voice cluster (failure mode: two people → one name).
5. **Calendar** — ICS/Google Calendar to pre-fill attendees from `meeting_date`.
   Out of scope until OAuth is an accepted product dependency.
6. **< 2s live neural ID** — only realistic on GPU with a streaming embedding
   model and VAD-triggered 1s crops, **not** full Large-v3 per window.

## Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Whisper mangles the name | Fuzzy match + correction alias; optional Large-v3 on GPU |
| No introduction spoken | Stay on Voice N; attendance marks them absent until a correction |
| Chair listed as attendee **and** officer | Roster dedupes (Phase 1) |
| Two Marias | Last name required; else leave Voice N (uncertain) |
| Overlapping speech | Phase 2 pyannote; Phase 1 cannot split |
| Noise / illness | VAD skip; correction memory; do not auto-update embeddings from low-confidence audio |
| Pyannote gated model 403 | Keep heuristic backend; log and continue |
| CPU timeout | Do not run Large-v3 or pyannote on the live path |
| Privacy | Embeddings/aliases stay on disk (`data/speaker_memory/`); no cloud speaker API |

## Configuration

```
SPEAKER_ID_BACKEND=heuristic   # or pyannote
WHISPER_SPEAKER_ID_MODEL=      # e.g. large-v3 on GPU only
HUGGINGFACE_TOKEN=             # required for pyannote 3.1
```
