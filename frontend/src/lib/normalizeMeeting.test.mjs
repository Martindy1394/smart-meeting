import assert from "node:assert/strict";
import test from "node:test";
import { normalizeMeeting, normalizeMeetingList } from "./normalizeMeeting.js";

test("normalizeMeeting fills null attendees and officer", () => {
  assert.equal(normalizeMeeting(null), null);
  const m = normalizeMeeting({
    id: "abc",
    attendees: null,
    presiding_officer: null,
    segments: null,
    summary: null,
  });
  assert.deepEqual(m.attendees, []);
  assert.equal(m.presiding_officer, "");
  assert.deepEqual(m.segments, []);
  assert.equal(m.summary, "");
  assert.deepEqual(normalizeMeetingList(null), []);
  assert.equal(normalizeMeetingList([m, {}]).length, 1);
});
