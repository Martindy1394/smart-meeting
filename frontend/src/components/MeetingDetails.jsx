import { useEffect, useState } from "react";
import { api } from "../api/client";

// Converts an ISO timestamp to the value expected by <input type="datetime-local">.
function toLocalInput(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const offset = d.getTimezoneOffset();
  return new Date(d.getTime() - offset * 60000).toISOString().slice(0, 16);
}

export default function MeetingDetails({ meeting, onUpdated }) {
  const [title, setTitle] = useState(meeting.title || "");
  const [venue, setVenue] = useState(meeting.venue || "");
  const [dateTime, setDateTime] = useState(toLocalInput(meeting.meeting_date));
  const [attendees, setAttendees] = useState(meeting.attendees || []);
  const [attendeeInput, setAttendeeInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    setTitle(meeting.title || "");
    setVenue(meeting.venue || "");
    setDateTime(toLocalInput(meeting.meeting_date));
    setAttendees(meeting.attendees || []);
    setAttendeeInput("");
    setError("");
    setSavedAt(0);
  }, [meeting.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function addAttendee() {
    const name = attendeeInput.trim();
    if (!name) return;
    setAttendees((prev) => (prev.includes(name) ? prev : [...prev, name]));
    setAttendeeInput("");
  }

  function onAttendeeKeyDown(e) {
    if (e.key === "Enter") {
      e.preventDefault();
      addAttendee();
    }
  }

  function removeAttendee(name) {
    setAttendees((prev) => prev.filter((n) => n !== name));
  }

  async function save() {
    setError("");
    // Include a name still sitting in the input box.
    const pending = attendeeInput.trim();
    const finalAttendees = pending
      ? Array.from(new Set([...attendees, pending]))
      : attendees;

    // All fields are required.
    if (!title.trim()) return setError("Title is required.");
    if (!venue.trim()) return setError("Venue is required.");
    if (!dateTime) return setError("Date & time is required.");
    if (finalAttendees.length === 0)
      return setError("Add at least one attendee.");

    setSaving(true);
    try {
      await api.updateMeeting(meeting.id, {
        title: title.trim(),
        venue: venue.trim(),
        meeting_date: dateTime,
        attendees: finalAttendees,
      });
      setAttendees(finalAttendees);
      setAttendeeInput("");
      setSavedAt(Date.now());
      if (onUpdated)
        onUpdated({
          ...meeting,
          title: title.trim(),
          venue: venue.trim(),
          meeting_date: dateTime,
          attendees: finalAttendees,
        });
    } catch (err) {
      setError(err.message || "Could not save details.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="details-card">
      <div className="details-head">
        <h3>Meeting details</h3>
        <div className="details-actions">
          {savedAt > 0 && !saving && <span className="saved-tag">Saved ✓</span>}
          {error && <span className="details-error">{error}</span>}
          <button className="btn secondary" onClick={save} disabled={saving}>
            {saving ? <span className="spinner" /> : "Save details"}
          </button>
        </div>
      </div>

      <div className="details-grid">
        <div className="field title-field">
          <label>
            Title <span className="req">*</span>
          </label>
          <input
            type="text"
            placeholder="Meeting title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </div>

        <div className="field">
          <label>
            Venue <span className="req">*</span>
          </label>
          <input
            type="text"
            placeholder="e.g. Conference Room A / Zoom"
            value={venue}
            onChange={(e) => setVenue(e.target.value)}
            required
          />
        </div>

        <div className="field">
          <label>
            Date &amp; time <span className="req">*</span>
          </label>
          <input
            type="datetime-local"
            value={dateTime}
            onChange={(e) => setDateTime(e.target.value)}
            required
          />
        </div>

        <div className="field attendees-field">
          <label>
            Attendees <span className="req">*</span>
          </label>
          <div className="chips">
            {attendees.map((name) => (
              <span className="chip" key={name}>
                {name}
                <button
                  type="button"
                  className="chip-x"
                  onClick={() => removeAttendee(name)}
                  aria-label={`Remove ${name}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
          <div className="attendee-input-row">
            <input
              type="text"
              placeholder="Type an attendee's name"
              value={attendeeInput}
              onChange={(e) => setAttendeeInput(e.target.value)}
              onKeyDown={onAttendeeKeyDown}
            />
            <button
              type="button"
              className="btn secondary"
              onClick={addAttendee}
              disabled={!attendeeInput.trim()}
            >
              + Add
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
