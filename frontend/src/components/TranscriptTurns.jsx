import { resolveTranscriptTurns } from "../lib/voiceLabels.js";
import TranscriptTurn from "./TranscriptTurn.jsx";

export default function TranscriptTurns({
  segments,
  fallbackText,
  keyword = "",
  voiceSlots = 1,
}) {
  try {
    const slots = Math.max(1, Number(voiceSlots) || 1);
    const { turns, emptyMessage } = resolveTranscriptTurns({
      segments,
      fallbackText,
      keyword,
      voiceSlots: slots,
    });
    if (!turns.length) {
      if (emptyMessage) {
        return <span className="transcript-find-empty">{emptyMessage}</span>;
      }
      return null;
    }
    return (
      <div className="transcript-turns">
        {turns.map((seg, i) => (
          <TranscriptTurn
            key={seg.id || `${seg.seq || i}-${seg.speaker_index || i}`}
            segment={seg}
            voiceSlots={slots}
          />
        ))}
      </div>
    );
  } catch (err) {
    console.error("TranscriptTurns failed", err);
    return (
      <span className="transcript-find-empty">Could not render transcript.</span>
    );
  }
}
