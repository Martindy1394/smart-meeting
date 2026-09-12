import assert from "node:assert/strict";
import test from "node:test";
import {
  LEGACY_MAX_VOICES,
  groupByVoice,
  groupByVoiceLegacy,
  listAttendees,
  mapVoiceSlot,
  registeredSpeakerCount,
  resolveTranscriptTurns,
  stripTranscriptMeta,
  voiceLabelForSlot,
} from "./voiceLabels.js";

test("optional attendees / officer never throw", () => {
  assert.deepEqual(listAttendees(null), []);
  assert.deepEqual(listAttendees(undefined), []);
  assert.deepEqual(listAttendees({}), []);
  assert.deepEqual(listAttendees({ attendees: null }), []);
  assert.equal(registeredSpeakerCount(null), LEGACY_MAX_VOICES);
  assert.equal(
    registeredSpeakerCount({ attendees: null, presiding_officer: null }),
    LEGACY_MAX_VOICES
  );
  assert.equal(registeredSpeakerCount({ attendees: [] }), 1);
  assert.equal(
    registeredSpeakerCount({ attendees: ["Ada", "Bob"], presiding_officer: "Chair" }),
    3
  );
  assert.equal(registeredSpeakerCount({ attendees: "Ada, Bob" }), 2);
});

test("extra Whisper voices map onto registered participant slots", () => {
  assert.equal(mapVoiceSlot(9, 2), 2);
  assert.equal(voiceLabelForSlot(9, 2), "Voice 2");
  assert.equal(mapVoiceSlot(99, 1), 1);
  assert.equal(mapVoiceSlot(3, 3), 3);
  const grouped = groupByVoice(
    [
      { speaker_index: 1, text: "hello" },
      { speaker_index: 7, text: "board" },
    ],
    2
  );
  assert.equal(grouped.length, 2);
  assert.equal(grouped[1].speaker_label, "Voice 2");
});

test("word isolation strips labels and timestamps from the word box", () => {
  assert.equal(stripTranscriptMeta("Voice 1: hello board"), "hello board");
  assert.equal(stripTranscriptMeta("[00:01:02] hello"), "hello");
  assert.equal(stripTranscriptMeta("Voice 2: [1.0 – 2.0] motion carries"), "motion carries");
  assert.equal(stripTranscriptMeta("Voice 1:"), "Voice 1:");
});

test("legacy grouping is used when mapped grouping would drop text", () => {
  const legacy = groupByVoiceLegacy([
    { speaker_label: "Chair", speaker_index: 0, text: "Chair: the meeting is open" },
  ]);
  assert.equal(legacy[0].text, "the meeting is open");
  assert.equal(legacy[0].speaker_label, "Chair");
});

test("resolveTranscriptTurns is one-pass (no recursive fallback)", () => {
  const a = resolveTranscriptTurns({
    segments: [],
    fallbackText: "Voice 1: hello\nVoice 4: extra talker",
    voiceSlots: 2,
  });
  assert.equal(a.turns.length, 2);
  assert.equal(a.turns[1].speaker_label, "Voice 2");
  const b = resolveTranscriptTurns({
    segments: [{ text: "  " }],
    fallbackText: "",
    keyword: "zzz",
  });
  assert.equal(b.turns.length, 0);
});
