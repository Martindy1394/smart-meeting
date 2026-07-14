import { forwardRef, useEffect, useImperativeHandle, useState } from "react";
import { api } from "../api/client";

// Converts an ISO timestamp to the value expected by <input type="datetime-local">.
function toLocalInput(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const offset = d.getTimezoneOffset();
  return new Date(d.getTime() - offset * 60000).toISOString().slice(0, 16);
}

function nowLocalInput() {
  return toLocalInput(new Date().toISOString());
}

function resolveAttendees(attendees, attendeeInput) {
  const pending = attendeeInput.trim();
  return pending
    ? Array.from(new Set([...attendees, pending]))
    : attendees;
}

const MeetingDetails = forwardRef(function MeetingDetails(
  { meeting, onUpdated, onValidityChange },
  ref
) {
  const [title, setTitle] = useState(meeting.title || "");
  const [venue, setVenue] = useState(meeting.venue || "");
  const [dateTime, setDateTime] = useState(
    toLocalInput(meeting.meeting_date) || nowLocalInput()
  );
  const [attendees, setAttendees] = useState(meeting.attendees || []);
  const [attendeeInput, setAttendeeInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    setTitle(meeting.title || "");
    setVenue(meeting.venue || "");
    setDateTime(toLocalInput(meeting.meeting_date) || nowLocalInput());
    setAttendees(meeting.attendees || []);
    setAttendeeInput("");
    setError("");
    setSavedAt(0);
  }, [meeting.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const isComplete = () => {
    const names = resolveAttendees(attendees, attendeeInput);
    return Boolean(
      title.trim() && venue.trim() && dateTime && names.length > 0
    );
  };

  useEffect(() => {
    if (onValidityChange) onValidityChange(isComplete());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [title, venue, dateTime, attendees, attendeeInput]);

  async function save({ silent = false } = {}) {
    setError("");
    const finalAttendees = resolveAttendees(attendees, attendeeInput);

    if (!title.trim()) {
      if (!silent) setError("Title is required.");
      return false;
    }
    if (!venue.trim()) {
      if (!silent) setError("Venue is required.");
      return false;
    }
    if (!dateTime) {
      if (!silent) setError("Date & time is required.");
      return false;
    }
    if (finalAttendees.length === 0) {
      if (!silent) setError("Add at least one attendee.");
      return false;
    }

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
      if (onUpdated) {
        onUpdated({
          ...meeting,
          title: title.trim(),
          venue: venue.trim(),
          meeting_date: dateTime,
          attendees: finalAttendees,
        });
      }
      return true;
    } catch (err) {
      if (!silent) setError(err.message || "Could not save details.");
      return false;
    } finally {
      setSaving(false);
    }
  }

  useImperativeHandle(ref, () => ({
    save,
    isComplete,
    isSaving: () => saving,
  }));

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

  return (
    <div className="details-card">
      <div className="details-head">
        <h3>Meeting details</h3>
        <div className="details-actions">
          {savedAt > 0 && !saving && <span className="saved-tag">Saved ✓</span>}
          {error && <span className="details-error">{error}</span>}
        </div>
      </div>

      <div className="details-grid">
        <div className="field title-field">
          <label>
            Title <span className="req">*</span>
          </label>
          <input
            type="text"
            placeholder="Enter meeting title"
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
});

export default MeetingDetails;
