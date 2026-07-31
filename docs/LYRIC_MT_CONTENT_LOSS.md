# Lyric / song translation content-loss fix

## Issue 1 — Diagnostic (where lines were lost)

Reproduction: ADCO sung Tagalog-heavy transcript.

| Stage | Lyric markers (`Buhus pa ulan`, `Hatid mo may bagyo`, `Damdamin…`, `humihiyaw`) |
|---|---|
| After normalize / hallucination collapse | **Present** |
| Pre-`_dedupe_units` | **Present** (embedded in one 53-word unit) |
| Post-`_dedupe_units` | **Present** (dedupe only dropped empty `Mga m.`) |
| Final English (pre-fix) | **Absent** after “bear witness” |

**Conclusion:** `_dedupe_units()` was **not** the loss site for this recording.
The verse stayed as one mega-unit; loss happened in
`_translate_unit_with_context()` when **English preamble context** was prepended
to the Tagalog verse. NLLB then truncated/degraded the verse and span trim kept
that short output.

Dedupe is still hardened (Issue 2) so punctuated lyric lines that *would*
segment separately are not collapsed by thematic word overlap.

## Before / after (ADCO)

**Before**
```
… This is a To honor m Swear. To obstacles and flowers May you never again,
to my side you're gone, I'll just remember to bear witness.
```
(no rain / storm / heart / love)

**After**
```
… To the obstacles and flowers Can you stop, to my side I'll leave, I'll only
witness Rain, my world will fall You know there's a storm to bring this my
heart's desire I'll flourish, my love will cry with joy, every time you rain
and kiss you.
```

Log: `Context-window skipped (cross-language context …)`.

## Issue 2 — Dedup

- Require ≥ **3** shared content words before ratio check
- Compare only the previous **2** kept units (adjacent ASR loops)
- Meeting ASR repetition fixtures still collapse

## Issue 3 — Kept-source marking

Failed MT units are wrapped as:

```text
[untranslated: <source clause>]
```

| Consumer | Behavior |
|---|---|
| UI English panel | Marker visible |
| `summarize_to_english` / BART | Spans **stripped** (`strip_untranslated_spans`) |
| `assess_translation_faithfulness` | Each span → `section: Untranslated` |

## Issue 4 — Garbage detector

Added vocabulary-free checks (kept salad list):

- Stray single-letter tokens (except `a` / `I`) → catches `To honor m Swear`
- Source/target length ratio outside ~0.35×–3.5×

## Issue 5 — Mass coverage

After PH→EN routing, compare token mass (`len>2`) of PH source units vs
translated English. If coverage &lt; **50%**, log a warning and append
`section: Coverage` to `review_lines`.

Pre-fix ADCO lyric mass ≈ **41%** (would trip). Post-fix does not.
