# Smart Meeting — product overview

## Vision

The system is a speech-to-text platform that transcribes in-person meetings from
audio recordings, with specialized speech recognition engines for **Tagalog** and
**Hiligaynon**, including support for regional dialects, colloquialisms, and
common code-switching patterns. It incorporates noise filtering and speaker
diarization to enhance transcript clarity, and features an integrated translation
module that converts the transcribed text into fluent English for broader
accessibility. The final output includes both the full verbatim transcript in the
original language and its English equivalent, along with a structured summary
presented in bulleted or numbered format to highlight key decisions, action
items, and discussion points. Additional functionalities include timestamped
text, searchable keywords, and multi-format export options (PDF, DOCX), while
privacy is ensured through optional offline processing and end-to-end encryption,
making the system suitable for confidential business, legal, medical, academic,
and government settings. Overall, it streamlines meeting documentation, bridges
language gaps, and reduces manual note-taking efforts without sacrificing
accuracy or security.

## How that maps to Smart Meeting today

| Claim | Status in this repo |
|---|---|
| In-person / recorded meeting ASR | **Ships** — live WebSocket PCM + final WAV pass (Whisper) |
| Tagalog + Hiligaynon recognition | **Ships (biased)** — Hiligaynon-first defaults; Tagalog `tl` + optional RNN-T live; PH dialect HF models; PLD fine-tune path for Hiligaynon |
| Dialects / colloquialisms / code-switching | **Partial** — PH Whisper + prompts; code-switch aware language modes; quality improves with PLD / Tagalog fine-tunes |
| Noise filtering | **Partial** — energy gates, AGC/loudness handling, optional VAD on final pass |
| Speaker diarization | **Roadmap** — not implemented yet (segments are timed, not speaker-labeled) |
| English translation | **Ships** — NLLB (PH→EN default) + mBART many-to-many |
| Verbatim original + English + structured minutes | **Ships** — transcript, English translation, BART bullets/numbered (Discussion / Decisions / Action items) |
| Timestamped text | **Ships** — `TranscriptSegment` start/end; included in exports when present |
| Searchable keywords | **Ships** — history search over title, transcript, summary, and translation |
| Export PDF / DOCX | **Ships** — meeting export API + UI (`txt` / `docx` / `pdf`) |
| Offline processing | **Partial** — models run locally when installed (`requirements-ml.txt`); no cloud LLM required |
| End-to-end encryption | **Roadmap** — auth is short-lived JWT + refresh revocation; audio encryption-at-rest via `DATA_ENCRYPTION_KEY` (see [`ENCRYPTION_AT_REST.md`](ENCRYPTION_AT_REST.md)). True client E2E is not built yet |
| Confidential sectors | **Goal** — architecture aims at private deployments; sector certifications are out of scope of this codebase |

Related hardening docs: [`HILIGAYNON_LANGUAGE_FORCING.md`](HILIGAYNON_LANGUAGE_FORCING.md),
[`MT_TAG_BENCHMARK.md`](MT_TAG_BENCHMARK.md), [`ENCRYPTION_AT_REST.md`](ENCRYPTION_AT_REST.md).

Data-flow requirements view: [`DFD.md`](DFD.md).

## Primary outputs

1. **Original-language transcript** (verbatim, refined after stop)
2. **English translation** (accessibility + summarization input)
3. **Structured summary** (bulleted or numbered minutes)

Exports bundle those three (plus timestamps when available) as `.txt`, `.docx`,
or `.pdf` via `GET /api/meetings/{id}/export?format=…`.
