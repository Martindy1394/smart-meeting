# Case study: ACCO meeting English translation (2026-08-07)

**Meeting:** `b6edbb02-5e66-4f31-a416-f723ebaf8b1a` (title ACCO)  
**Audio:** `backend/data/audio/b6edbb02-5e66-4f31-a416-f723ebaf8b1a.wav` (~44.5s, 16 kHz mono)

## What the user saw

```
3 lines flagged for language review (Hiligaynon / Tagalog ambiguity).
Language review  <garbled PH lyric ASR>
Language review  Don't need to pretend.
Language review  Don't need to assume Everything.
Love that unconditionally laid down my beautiful garden …
```

## Root cause (audio + pipeline)

1. **Audio is sung / lyric Tagalog + English**, not clean board speech. Energy is continuous vocal (not sparse meeting turns).
2. **Whisper ASR mishears the lyric** into nonsense PH forms  
   (`andanan`, `nararang tamanan`, `pagkatikaw`, `mahanan`, …) while the English half is mostly recoverable.
3. **lang_router** marks the PH block `hil`/`tl` ambiguous → Language review.
4. **NLLB/mBART then invent fluent English nonsense** (“garden / beautiful / born”) from that ASR salad — the bad “English translation” the user quoted.
5. Short English lines were also flagged for review (`default_english`).

Re-ASR with `tl` / `hil` / `auto` produced the same salad; forced-`en` was slightly closer on a few words but still unusable for lyrics.

## Fix implemented

In `llm.py`:

- Detect **garbled PH ASR** (high unknown-token ratio) → keep `[untranslated: …]` instead of hallucinating English.
- Split PH→EN glued tails (`…ako and perhaps…`).
- Do not Language-review clear English passthrough lines.
- Treat fluent “garden…” hallucinations from garbled sources as garbage MT.

After fix (same transcript):

```
[untranslated: Pag-ibig na walang andanan … isang tao lang ako.]
Perhaps. We have to of course admit our mistakes. …
Don't need to pretend. Don't need to assume Everything. …
```

No garden hallucination; English tail preserved.

## Remaining (ASR / product)

- True lyric accuracy needs better Tagalog ASR (fine-tune / lyric-aware decode), not MT.
- Configure Google Translate for real Hiligaynon prose.
- Optional UI: show “source kept — ASR confidence low” instead of only Language review.
