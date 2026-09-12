/** Highest Voice N we will render. Extra Whisper clusters stay labeled, never loop. */
export const MAX_VOICE_INDEX = 32;

function asNameList(value) {
  if (Array.isArray(value)) {
    return value.map((n) => String(n || "").trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(/[,;\n]+/)
      .map((n) => n.trim())
      .filter(Boolean);
  }
  return [];
}

/** Safe attendee list: never throws when meeting or attendees is null. */
export function listAttendees(meeting) {
  return asNameList(meeting?.attendees);
}

/** attendees.length + (presiding_officer ? 1 : 0), at least 1. */
export function registeredSpeakerCount(meeting) {
  const attendees = listAttendees(meeting);
  const officer = String(meeting?.presiding_officer || "").trim();
  // `meeting?.attendees?.length ?? 0` is safe while loading; parsed names win if present.
  const attendeeCount = attendees.length || (meeting?.attendees?.length ?? 0);
  return Math.max(
    1,
    Math.min(MAX_VOICE_INDEX, attendeeCount + (officer ? 1 : 0))
  );
}

/**
 * Map a Whisper speaker index to a display slot.
 * Extra voices (index > registered count) keep Voice N instead of indexing
 * into attendees or spinning while waiting for a matching name.
 */
export function mapVoiceSlot(index, _registeredSlots = 1) {
  const i = Number(index);
  if (!Number.isFinite(i) || i < 1) return 1;
  return Math.min(Math.floor(i), MAX_VOICE_INDEX);
}

export function voiceLabelForSlot(index, registeredSlots = 1) {
  return `Voice ${mapVoiceSlot(index, registeredSlots)}`;
}

/** Strip Voice N prefixes and bracket timestamps so the word box is text-only. */
export function stripTranscriptMeta(text) {
  let body = String(text || "").trim();
  body = body.replace(/^(Voice\s+\d+)\s*:\s*/i, "");
  body = body.replace(
    /^\[(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?(?:\s*[–\-—]\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?)?\]\s*/,
    ""
  );
  body = body.replace(/^\[\d+(?:\.\d+)?\s*[–\-—]\s*\d+(?:\.\d+)?\]\s*/, "");
  return body.trim();
}

export function segmentHaystack(seg) {
  return `${seg?.speaker_label || ""} ${seg?.text || ""}`.toLowerCase();
}

function parseFallbackLines(text, slots) {
  const lines = String(text || "")
    .split(/\n+/)
    .map((ln) => ln.trim())
    .filter(Boolean);
  return lines.map((ln, i) => {
    const m = ln.match(/^(Voice\s+\d+)\s*:\s*(.*)$/i);
    if (m) {
      const idx = Number((m[1].match(/\d+/) || ["1"])[0]);
      return {
        id: `line-${i}`,
        speaker_label: voiceLabelForSlot(idx, slots),
        speaker_index: mapVoiceSlot(idx, slots),
        text: stripTranscriptMeta(m[2]),
      };
    }
    return {
      id: `line-${i}`,
      speaker_label: voiceLabelForSlot(1, slots),
      speaker_index: 1,
      text: stripTranscriptMeta(ln),
    };
  });
}

export function groupByVoice(segments, voiceSlots = 1) {
  const out = [];
  const slots = Math.max(1, Number(voiceSlots) || 1);
  const list = Array.isArray(segments) ? segments : [];
  for (const seg of list) {
    const raw = stripTranscriptMeta(seg?.text || "");
    if (!raw) continue;
    let idx = Number(seg?.speaker_index) || 0;
    let label = String(seg?.speaker_label || "").trim();
    if (!idx && label) {
      idx = Number((label.match(/\d+/) || ["1"])[0]) || 1;
    }
    idx = mapVoiceSlot(idx || 1, slots);
    label = voiceLabelForSlot(idx, slots);
    const last = out[out.length - 1];
    if (last && last.speaker_index === idx) {
      last.text = `${last.text} ${raw}`.trim();
      continue;
    }
    out.push({
      ...seg,
      speaker_label: label,
      speaker_index: idx,
      text: raw,
    });
  }
  return out;
}

/**
 * One-pass turn list. Do not recurse during React render — empty + fallback
 * used to re-enter TranscriptTurns and could blow the stack / look hung.
 */
export function resolveTranscriptTurns({
  segments,
  fallbackText = "",
  keyword = "",
  voiceSlots = 1,
} = {}) {
  const kw = String(keyword || "").trim().toLowerCase();
  const slots = Math.max(1, Number(voiceSlots) || 1);
  try {
    const segs = Array.isArray(segments)
      ? segments.filter((s) => String(s?.text || "").trim())
      : [];
    const matchKw = (list) =>
      kw ? list.filter((s) => segmentHaystack(s).includes(kw)) : list;
    let filtered = groupByVoice(matchKw(segs), slots);
    if (filtered.length) {
      return { turns: filtered, emptyMessage: null };
    }
    const text = String(fallbackText || "").trim();
    if (!text) {
      if (kw && segs.length) {
        return {
          turns: [],
          emptyMessage: `No transcript lines match “${keyword}”.`,
        };
      }
      return { turns: [], emptyMessage: null };
    }
    if (kw && !text.toLowerCase().includes(kw)) {
      return {
        turns: [],
        emptyMessage: `No transcript lines match “${keyword}”.`,
      };
    }
    const parsed = parseFallbackLines(text, slots);
    filtered = groupByVoice(matchKw(parsed), slots);
    if (filtered.length) {
      return { turns: filtered, emptyMessage: null };
    }
    if (kw) {
      return {
        turns: [],
        emptyMessage: `No transcript lines match “${keyword}”.`,
      };
    }
    return { turns: [], emptyMessage: null };
  } catch (err) {
    console.error("resolveTranscriptTurns failed", err);
    return { turns: [], emptyMessage: "Could not render transcript." };
  }
}
