import { forwardRef, useEffect, useImperativeHandle, useState } from "react";
import { api } from "../api/client";

function pad2(n) {
  return String(n).padStart(2, "0");
}

/** Format a Date as the value expected by <input type="datetime-local">. */
function formatLocalInput(d) {
  if (!(d instanceof Date) || Number.isNaN(d.getTime())) return "";
  return (
    `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}` +
    `T${pad2(d.getHours())}:${pad2(d.getMinutes())}`
  );
}

function nowLocalInput() {
  return formatLocalInput(new Date());
}

function toLocalInput(iso) {
  if (!iso) return "";
  return formatLocalInput(new Date(iso));
}

/** Convert datetime-local value to ISO-8601 for the API. */
function toIsoFromLocalInput(local) {
  if (!local) return null;
  const d = new Date(local);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString();
}

function isFreshMeeting(meeting) {
  return (
    meeting.status !== "finalized" &&
    !(meeting.final_transcript || "").trim() &&
    !(meeting.title || "").trim() &&
    !(meeting.venue || "").trim()
  );
}

function resolveAttendees(attendees, attendeeInput) {
  const pending = attendeeInput.trim();
  return pending
    ? Array.from(new Set([...attendees, pending]))
    : attendees;
}

function sameAttendees(a, b) {
  if (a.length !== b.length) return false;
  return a.every((name, i) => name === b[i]);
}

const MeetingDetails = forwardRef(function MeetingDetails(
  { meeting, onUpdated, onValidityChange, onDirtyChange },
  ref
) {
  const [title, setTitle] = useState(meeting.title || "");
  const [venue, setVenue] = useState(meeting.venue || "");
  const [dateTime, setDateTime] = useState(() =>
    isFreshMeeting(meeting)
      ? nowLocalInput()
      : toLocalInput(meeting.meeting_date) || nowLocalInput()
  );
  const [attendees, setAttendees] = useState(meeting.attendees || []);
  const [attendeeInput, setAttendeeInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    setTitle(meeting.title || "");
    setVenue(meeting.venue || "");
    // New meetings always open on the current local date & time.
    setDateTime(
      isFreshMeeting(meeting)
        ? nowLocalInput()
        : toLocalInput(meeting.meeting_date) || nowLocalInput()
    );
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

  const isDirty = () => {
    const currentAttendees = resolveAttendees(attendees, attendeeInput);
    const savedAttendees = Array.isArray(meeting.attendees)
      ? meeting.attendees
      : [];
    const savedDateTime = meeting.meeting_date
      ? toLocalInput(meeting.meeting_date)
      : "";
    return (
      title.trim() !== (meeting.title || "").trim() ||
      venue.trim() !== (meeting.venue || "").trim() ||
      dateTime !== savedDateTime ||
      !sameAttendees(currentAttendees, savedAttendees)
    );
  };

  useEffect(() => {
    if (onValidityChange) onValidityChange(isComplete());
    if (onDirtyChange) onDirtyChange(isDirty());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [title, venue, dateTime, attendees, attendeeInput, meeting]);

  function useCurrentDateTime() {
    const current = nowLocalInput();
    setDateTime(current);
    return current;
  }

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

    const meetingDateIso = toIsoFromLocalInput(dateTime);
    if (!meetingDateIso) {
      if (!silent) setError("Date & time is invalid.");
      return false;
    }

    setSaving(true);
    try {
      await api.updateMeeting(meeting.id, {
        title: title.trim(),
        venue: venue.trim(),
        meeting_date: meetingDateIso,
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
          meeting_date: meetingDateIso,
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
    isDirty,
    isSaving: () => saving,
    useCurrentDateTime,
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
          <div className="datetime-row">
            <input
              type="datetime-local"
              value={dateTime}
              onChange={(e) => setDateTime(e.target.value)}
              required
            />
            <button
              type="button"
              className="btn secondary"
              onClick={useCurrentDateTime}
              title="Set to the current date and time"
            >
              Now
            </button>
          </div>
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
