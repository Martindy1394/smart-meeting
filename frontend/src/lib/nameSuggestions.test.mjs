import assert from "node:assert/strict";
import test from "node:test";
import {
  rankNameSuggestions,
  soundex,
} from "./nameSuggestions.js";

test("soundex groups similar last names", () => {
  assert.equal(soundex("Santos"), soundex("Santos"));
  assert.equal(soundex("Smith").length, 4);
});

test("ranks officers by query, recency, and chair count", () => {
  const ranked = rankNameSuggestions({
    query: "mar",
    role: "officer",
    recents: ["Maria Santos"],
    directory: {
      people: [
        {
          name: "Maria Santos",
          officer_count: 4,
          attendee_count: 1,
          sources: ["officer"],
          last_title: "Board huddle",
        },
        {
          name: "Mario Cruz",
          officer_count: 1,
          attendee_count: 0,
          sources: ["officer"],
        },
        { name: "Ana Reyes", officer_count: 2, attendee_count: 0, sources: ["officer"] },
      ],
    },
  });
  assert.equal(ranked[0].name, "Maria Santos");
  assert.ok(ranked[0].recent);
  assert.ok(ranked.some((r) => r.name === "Mario Cruz"));
  assert.ok(!ranked.some((r) => r.name === "Ana Reyes"));
});

test("session identified guests are flagged and not skipped", () => {
  const ranked = rankNameSuggestions({
    query: "",
    role: "attendee",
    exclude: ["Ada"],
    session: [{ name: "Guest Speaker", guest: true, confidence: 0.66 }],
    directory: { attendees: ["Ada", "Bob"] },
  });
  assert.ok(!ranked.some((r) => r.name === "Ada"));
  const guest = ranked.find((r) => r.name === "Guest Speaker");
  assert.ok(guest?.guest);
});

test("nickname maria matches mary", () => {
  const ranked = rankNameSuggestions({
    query: "mary",
    directory: { attendees: ["Maria Santos"] },
  });
  assert.equal(ranked[0].name, "Maria Santos");
});
