# Name suggestion dropdowns (attendees & presiding officer)

Smart Meeting already stores a **per-user name directory** from past meetings
(`GET /api/meetings/suggestions`). This note is the architecture for the
searchable comboboxes on Meeting details, and how they connect to speaker ID.

## What ships

| Source | Used now |
|---|---|
| Past presiding officers | Yes — ranked by how often they chaired |
| Past attendees | Yes — ranked by attendance count |
| Speech-identified names (`attendance`) | Yes — “Identified” / “Guest” badges |
| Recent local picks | Yes — `localStorage` `smart-meeting:recent-names` |
| Fuzzy / nickname / Soundex | Yes — client-side `nameSuggestions.js` |
| Keyboard (↑↓ Enter Esc) + ARIA combobox | Yes — `NameSuggestField.jsx` |
| Calendar invited list | **Not connected** (no Google/ICS OAuth in this app) |
| Org / HR employee directory | **Not connected** |
| ML meeting-type ranking | **Not used** — counts + recency + session confidence |

## Data

`GET /api/meetings/suggestions` (JWT):

```json
{
  "presiding_officers": ["Maria Santos"],
  "attendees": ["Ada", "Bob"],
  "people": [
    {
      "name": "Maria Santos",
      "officer_count": 4,
      "attendee_count": 2,
      "identified_count": 1,
      "last_seen": "2026-09-01T08:00:00Z",
      "last_title": "Board huddle",
      "sources": ["officer", "attendee", "identified"]
    }
  ]
}
```

The string lists stay for older clients. `people` is the ranked payload.

Session overlay is **not** a second request: `meeting.attendance` from
`GET /api/meetings/{id}` (speaker-ID report) is merged in the browser.

## Client ranking (no extra ML service)

Score ≈ query match (prefix > contains > nickname > Soundex last name)
+ recent pick bonus
+ this-session identified/guest bonus (uses voice-ID confidence)
+ officer_count (officer field) or attendee_count (attendee field)

Exclude names already in the attendee chip list.

## Accessibility / mobile

- `role="combobox"` + `listbox` / `option` + `aria-activedescendant`
- List max-height `min(240px, 50vh)`, rows ≥ ~40px for touch
- Native `<datalist>` removed (it fought the custom list on mobile)

## Later integration points

1. Calendar: map invitees into `people` with `sources: ["calendar"]`.
2. Org directory: same shape, cache TTL ~15 min.
3. Bulk import: `POST /api/meetings/suggestions/import` `{ names: string[] }`.
4. Feedback: `POST /api/meetings/{id}/speakers/correct` already teaches aliases;
   dropdown recents are the lightweight UI loop.
