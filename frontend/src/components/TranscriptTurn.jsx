import {
  mapVoiceSlot,
  stripTranscriptMeta,
  voiceLabelForSlot,
} from "../lib/voiceLabels.js";

function speakerTone(index) {
  const n = Number(index);
  if (!Number.isFinite(n) || n < 1) return "speaker-tone-1";
  return `speaker-tone-${((n - 1) % 3) + 1}`;
}

function voiceAccuracyTitle(index) {
  const n = Number(index);
  if (n === 1) return "Highest transcription accuracy";
  if (n === 2) return "Second-highest transcription accuracy";
  return "Lower transcription accuracy";
}

/**
 * One transcript row: Voice N lives on the chip; `.transcript-words` is
 * words-only (no timestamps, labels, or status badges).
 */
export default function TranscriptTurn({
  segment,
  voiceSlots = 1,
  live = false,
}) {
  const slots = Math.max(1, Number(voiceSlots) || 1);
  const idx = mapVoiceSlot(segment?.speaker_index, slots);
  const name = String(segment?.speaker_name || "").trim();
  const namedConf = Number(segment?.speaker_confidence) || 0;
  const label =
    name && namedConf >= 0.75 ? name : voiceLabelForSlot(idx, slots);
  const titleBits = [voiceAccuracyTitle(idx)];
  if (name) {
    titleBits.push(
      `${name} (${Math.round(namedConf * 100)}% ${segment?.speaker_id_method || "id"})`
    );
  }
  const words = stripTranscriptMeta(segment?.text || "");
  if (!words) return null;
  const low = Boolean(segment?.low_confidence);

  return (
    <div className={live ? "transcript-turn is-live" : "transcript-turn"}>
      <span
        className={`speaker-chip ${speakerTone(idx)}`}
        title={titleBits.join(" · ")}
      >
        {label}
      </span>
      <span
        className={
          low ? "transcript-words transcript-seg-low caption-low-confidence" : "transcript-words"
        }
      >
        {words}
      </span>
      {low ? (
        <span className="caption-low-confidence-badge" title="ASR low confidence">
          Low confidence
        </span>
      ) : null}
    </div>
  );
}
