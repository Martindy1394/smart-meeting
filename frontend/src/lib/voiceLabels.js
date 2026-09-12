/** Highest Voice N we will render. Extra Whisper clusters stay labeled, never loop. */
export const MAX_VOICE_INDEX = 32;

/** Pre-slot-cap default (live_max_voices) used when participant data is missing. */
export const LEGACY_MAX_VOICES = 3;

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

/**
 * attendees.length + (presiding_officer ? 1 : 0).
 * When attendees are still loading/null, revert to the old Voice 1–3 default.
 */
export function registeredSpeakerCount(meeting) {
  try {
    if (meeting == null || meeting.attendees == null) {
      return LEGACY_MAX_VOICES;
    }
    const attendees = listAttendees(meeting);
    const officer = String(meeting?.presiding_officer || "").trim();
    const attendeeCount = attendees.length || (meeting?.attendees?.length ?? 0);
    return Math.max(
      1,
      Math.min(MAX_VOICE_INDEX, attendeeCount + (officer ? 1 : 0))
    );
  } catch (err) {
    console.error("registeredSpeakerCount failed; reverting to Voice 1–3", err);
    return LEGACY_MAX_VOICES;
  }
}

/**
 * Map a Whisper speaker index onto registered Voice 1…N slots.
 * Extra clusters clamp to the last participant slot (never invent Voice N+1
 * and never look up attendee names).
 */
export function mapVoiceSlot(index, registeredSlots = 1) {
  const slots = Math.max(
    1,
    Math.min(MAX_VOICE_INDEX, Number(registeredSlots) || 1)
  );
  const i = Number(index);
  if (!Number.isFinite(i) || i < 1) return 1;
  return Math.min(Math.floor(i), slots);
}

export function voiceLabelForSlot(index, registeredSlots = 1) {
  return `Voice ${mapVoiceSlot(index, registeredSlots)}`;
}

/** Strip Voice N prefixes and bracket timestamps so the word box is text-only. */
export function stripTranscriptMeta(text) {
  const original = String(text || "").trim();
  try {
    let body = original;
    let prev = "";
    while (body && body !== prev) {
      prev = body;
      body = body.replace(/^(Voice\s+\d+)\s*:\s*/i, "");
      body = body.replace(
        /^\[(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?(?:\s*[–\-—]\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?)?\]\s*/,
        ""
      );
      body = body.replace(/^\[\d+(?:\.\d+)?\s*[–\-—]\s*\d+(?:\.\d+)?\]\s*/, "");
      body = body.replace(/^\[(?:start|end)?_?time[^\]]*\]\s*/i, "");
      body = body.trim();
    }
    // Revert isolation if it would hide the utterance.
    return body || original;
  } catch (err) {
    console.error("stripTranscriptMeta failed; reverting to raw text", err);
    return original;
  }
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

/** Pre-voice-slot grouping: keep Whisper labels, do not remap onto attendee count. */
export function groupByVoiceLegacy(segments) {
  const out = [];
  const list = Array.isArray(segments) ? segments : [];
  for (const seg of list) {
    const raw = String(seg?.text || "").trim();
    if (!raw) continue;
    let label = String(seg?.speaker_label || "").trim();
    let idx = Number(seg?.speaker_index) || 0;
    if (!label) {
      label = "Voice 1";
      idx = 1;
    }
    if (!idx) {
      idx = Number((label.match(/\d+/) || ["1"])[0]) || 1;
    }
    let body = stripTranscriptMeta(raw);
    if (label && body.toLowerCase().startsWith(label.toLowerCase() + ":")) {
      body = body.slice(label.length + 1).trim() || body;
    }
    const last = out[out.length - 1];
    if (last && last.speaker_label === label) {
      last.text = `${last.text} ${body}`.trim();
      continue;
    }
    out.push({
      ...seg,
      speaker_label: label,
      speaker_index: idx,
      text: body,
    });
  }
  return out;
}

function groupByVoiceMapped(segments, voiceSlots = 1) {
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

export function groupByVoice(segments, voiceSlots = 1) {
  const list = Array.isArray(segments) ? segments : [];
  try {
    const mapped = groupByVoiceMapped(list, voiceSlots);
    if (mapped.length || !list.some((s) => String(s?.text || "").trim())) {
      return mapped;
    }
    console.warn("Voice mapping produced no turns; reverting to legacy grouping");
    return groupByVoiceLegacy(list);
  } catch (err) {
    console.error("Voice mapping failed; reverting to legacy grouping", err);
    return groupByVoiceLegacy(list);
  }
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
  const segs = Array.isArray(segments)
    ? segments.filter((s) => String(s?.text || "").trim())
    : [];
  const matchKw = (list) =>
    kw ? list.filter((s) => segmentHaystack(s).includes(kw)) : list;
  try {
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
    const legacy = groupByVoiceLegacy(matchKw(parsed.length ? parsed : segs));
    if (legacy.length) {
      return { turns: legacy, emptyMessage: null };
    }
    if (kw) {
      return {
        turns: [],
        emptyMessage: `No transcript lines match “${keyword}”.`,
      };
    }
    return { turns: [], emptyMessage: null };
  } catch (err) {
    console.error("resolveTranscriptTurns failed; reverting to legacy turns", err);
    try {
      const text = String(fallbackText || "").trim();
      const parsed = text ? parseFallbackLines(text, LEGACY_MAX_VOICES) : segs;
      const legacy = groupByVoiceLegacy(matchKw(parsed));
      if (legacy.length) {
        return { turns: legacy, emptyMessage: null };
      }
    } catch (inner) {
      console.error("Legacy transcript fallback failed", inner);
    }
    return { turns: [], emptyMessage: "Could not render transcript." };
  }
}
