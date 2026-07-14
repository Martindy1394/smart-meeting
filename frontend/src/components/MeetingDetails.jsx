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
  const [venue, setVenue] = useState(meeting.venue || "");
  const [dateTime, setDateTime] = useState(toLocalInput(meeting.meeting_date));
  const [attendees, setAttendees] = useState(meeting.attendees || []);
  const [attendeeInput, setAttendeeInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    setVenue(meeting.venue || "");
    setDateTime(toLocalInput(meeting.meeting_date));
    setAttendees(meeting.attendees || []);
    setAttendeeInput("");
    setError("");
    setSavedAt(0);
  }, [meeting.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function addAttendees(raw) {
    const names = raw
      .split(",")
      .map((n) => n.trim())
      .filter(Boolean);
    if (names.length === 0) return;
    setAttendees((prev) => {
      const set = new Set(prev);
      names.forEach((n) => set.add(n));
      return Array.from(set);
    });
    setAttendeeInput("");
  }

  function onAttendeeKeyDown(e) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addAttendees(attendeeInput);
    }
  }

  function removeAttendee(name) {
    setAttendees((prev) => prev.filter((n) => n !== name));
  }

  async function save() {
    setSaving(true);
    setError("");
    try {
      const pending = attendeeInput.trim();
      const finalAttendees = pending
        ? Array.from(new Set([...attendees, pending]))
        : attendees;
      await api.updateMeeting(meeting.id, {
        venue: venue.trim(),
        meeting_date: dateTime ? dateTime : null,
        attendees: finalAttendees,
      });
      setAttendees(finalAttendees);
      setAttendeeInput("");
      setSavedAt(Date.now());
      if (onUpdated) onUpdated();
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
        <div className="field">
          <label>Venue</label>
          <input
            type="text"
            placeholder="e.g. Conference Room A / Zoom"
            value={venue}
            onChange={(e) => setVenue(e.target.value)}
          />
        </div>

        <div className="field">
          <label>Date &amp; time</label>
          <input
            type="datetime-local"
            value={dateTime}
            onChange={(e) => setDateTime(e.target.value)}
          />
        </div>

        <div className="field attendees-field">
          <label>Attendees</label>
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
          <input
            type="text"
            placeholder="Type a name and press Enter (comma-separated OK)"
            value={attendeeInput}
            onChange={(e) => setAttendeeInput(e.target.value)}
            onKeyDown={onAttendeeKeyDown}
            onBlur={() => attendeeInput.trim() && addAttendees(attendeeInput)}
          />
        </div>
      </div>
    </div>
  );
}
