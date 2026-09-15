import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { api } from "../api/client";
import {
  loadRecentNames,
  rememberRecentName,
  sessionPeople,
} from "../lib/nameSuggestions.js";
import { listAttendees } from "../lib/voiceLabels.js";
import NameSuggestField from "./NameSuggestField.jsx";
import HistorySuggestField from "./HistorySuggestField.jsx";

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
    meeting?.status !== "finalized" &&
    !(meeting?.final_transcript || "").trim() &&
    !(meeting?.title || "").trim() &&
    !(meeting?.venue || "").trim()
  );
}

function resolveAttendees(attendees, attendeeInput) {
  const base = Array.isArray(attendees) ? attendees : listAttendees({ attendees });
  const pending = String(attendeeInput || "").trim();
  return pending ? Array.from(new Set([...base, pending])) : base;
}

function sameDateTimeLocal(a, b) {
  return String(a || "").slice(0, 16) === String(b || "").slice(0, 16);
}

function sameAttendees(a, b) {
  const left = Array.isArray(a) ? a : [];
  const right = Array.isArray(b) ? b : [];
  if (left.length !== right.length) return false;
  return left.every((name, i) => name === right[i]);
}

const AUTOSAVE_MS = 750;

const MeetingDetails = forwardRef(function MeetingDetails(
  { meeting, onUpdated, onValidityChange, onDirtyChange, onAutosaveStatus },
  ref
) {
  const [title, setTitle] = useState(meeting?.title || "");
  const [venue, setVenue] = useState(meeting?.venue || "");
  const [presidingOfficer, setPresidingOfficer] = useState(
    meeting?.presiding_officer || ""
  );
  const [dateTime, setDateTime] = useState(() =>
    isFreshMeeting(meeting)
      ? nowLocalInput()
      : toLocalInput(meeting?.meeting_date) || nowLocalInput()
  );
  const [attendees, setAttendees] = useState(() => listAttendees(meeting));
  const [attendeeInput, setAttendeeInput] = useState("");
  const [directory, setDirectory] = useState({
    presiding_officers: [],
    attendees: [],
    people: [],
    titles: [],
    venues: [],
  });
  const [recents, setRecents] = useState(() => loadRecentNames());
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(0);
  const [error, setError] = useState("");
  const autosaveTimer = useRef(null);
  const saveSeq = useRef(0);
  const skipAutosave = useRef(false);

  useEffect(() => {
    skipAutosave.current = true;
    setTitle(meeting?.title || "");
    setVenue(meeting?.venue || "");
    setPresidingOfficer(meeting?.presiding_officer || "");
    // New meetings always open on the current local date & time.
    setDateTime(
      isFreshMeeting(meeting)
        ? nowLocalInput()
        : toLocalInput(meeting?.meeting_date) || nowLocalInput()
    );
    setAttendees(listAttendees(meeting));
    setAttendeeInput("");
    setError("");
    setSavedAt(0);
    // Allow the next edit cycle to autosave after state settles.
    const t = setTimeout(() => {
      skipAutosave.current = false;
    }, 0);
    return () => clearTimeout(t);
  }, [meeting?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await api.meetingSuggestions();
        if (cancelled || !data) return;
        setDirectory({
          presiding_officers: Array.isArray(data.presiding_officers)
            ? data.presiding_officers
            : [],
          attendees: Array.isArray(data.attendees) ? data.attendees : [],
            people: Array.isArray(data.people) ? data.people : [],
            titles: Array.isArray(data.titles) ? data.titles : [],
            venues: Array.isArray(data.venues) ? data.venues : [],
        });
      } catch {
        /* suggestions are optional */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const isComplete = () => {
    const names = resolveAttendees(attendees, attendeeInput);
    return Boolean(
      title.trim() && venue.trim() && dateTime && names.length > 0
    );
  };

  const isDirty = () => {
    const currentAttendees = resolveAttendees(attendees, attendeeInput);
    const savedAttendees = listAttendees(meeting);
    const savedDateTime = meeting?.meeting_date
      ? toLocalInput(meeting.meeting_date)
      : "";
    return (
      title.trim() !== (meeting?.title || "").trim() ||
      venue.trim() !== (meeting?.venue || "").trim() ||
      presidingOfficer.trim() !== (meeting?.presiding_officer || "").trim() ||
      !sameDateTimeLocal(dateTime, savedDateTime) ||
      !sameAttendees(currentAttendees, savedAttendees)
    );
  };

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

    if (!meeting?.id) {
      if (!silent) setError("Meeting is still loading.");
      return false;
    }
    const seq = ++saveSeq.current;
    setSaving(true);
    try {
      // Spoken language is not user-selected — always auto (Hiligaynon-biased).
      await api.updateMeeting(meeting?.id, {
        title: title.trim(),
        venue: venue.trim(),
        presiding_officer: presidingOfficer.trim(),
        meeting_date: meetingDateIso,
        attendees: finalAttendees,
        language: "auto",
      });
      if (seq !== saveSeq.current) return false;
      setAttendees(finalAttendees);
      setAttendeeInput("");
      setSavedAt(Date.now());
      if (onUpdated) {
        onUpdated({
          ...meeting,
          title: title.trim(),
          venue: venue.trim(),
          presiding_officer: presidingOfficer.trim(),
          meeting_date: meetingDateIso,
          attendees: finalAttendees,
          language: "auto",
        });
      }
      return true;
    } catch (err) {
      console.error("Meeting details save failed", err);
      if (seq !== saveSeq.current) return false;
      setError(err.message || "Could not autosave details.");
      return false;
    } finally {
      if (seq === saveSeq.current) setSaving(false);
    }
  }

  const savedAttendeeKey = listAttendees(meeting).join("\n");
  const savedOfficer = meeting?.presiding_officer || "";
  const savedTitle = meeting?.title || "";
  const savedVenue = meeting?.venue || "";
  const savedDate = meeting?.meeting_date || "";

  useEffect(() => {
    try {
      if (onValidityChange) onValidityChange(isComplete());
      if (onDirtyChange) onDirtyChange(isDirty());
    } catch (err) {
      console.error("Meeting details validity sync failed", err);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [title, venue, presidingOfficer, dateTime, attendees, attendeeInput, savedTitle, savedVenue, savedOfficer, savedDate, savedAttendeeKey]);

  // Debounced autosave whenever required fields are complete and dirty.
  useEffect(() => {
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    if (skipAutosave.current) return undefined;
    try {
      if (!isDirty() || !isComplete() || saving) return undefined;
    } catch (err) {
      console.error("Meeting details dirty check failed", err);
      return undefined;
    }

    autosaveTimer.current = setTimeout(() => {
      void save({ silent: true });
    }, AUTOSAVE_MS);

    return () => {
      if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [title, venue, presidingOfficer, dateTime, attendees, attendeeInput, savedTitle, savedVenue, savedOfficer, savedDate, savedAttendeeKey, saving]);

  useEffect(() => {
    if (!onAutosaveStatus) return;
    onAutosaveStatus({
      saving,
      savedAt,
      error,
      dirty: isDirty(),
      ready: isComplete(),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [saving, savedAt, error, title, venue, presidingOfficer, dateTime, attendees, attendeeInput, savedTitle, savedVenue, savedOfficer, savedDate, savedAttendeeKey]);

  useEffect(() => {
    return () => {
      if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
      if (onAutosaveStatus) onAutosaveStatus(null);
    };
  }, [onAutosaveStatus]);

  useImperativeHandle(ref, () => ({
    save,
    isComplete,
    isDirty,
    isSaving: () => saving,
    useCurrentDateTime,
  }));

  function useCurrentDateTime() {
    const current = nowLocalInput();
    setDateTime(current);
    return current;
  }

  function addAttendee(nameOverride) {
    const name = String(nameOverride || attendeeInput).trim();
    if (!name) return;
    setAttendees((prev) => {
      const list = Array.isArray(prev) ? prev : [];
      return list.some((a) => a.toLowerCase() === name.toLowerCase())
        ? list
        : [...list, name];
    });
    rememberRecentName("attendee", name);
    setRecents(loadRecentNames());
    setAttendeeInput("");
  }

  function removeAttendee(name) {
    setAttendees((prev) => (Array.isArray(prev) ? prev : []).filter((n) => n !== name));
  }

  const sessionHits = sessionPeople(meeting);

  let statusLabel = "Autosave on";
  if (saving) statusLabel = "Saving…";
  else if (error) statusLabel = "Autosave failed";
  else if (savedAt > 0) statusLabel = "Saved";
  else if (isDirty() && !isComplete()) statusLabel = "Fill required fields";

  return (
    <div className="details-card">
      <div className="details-head">
        <h3>Meeting details</h3>
        <div className="details-actions">
          <span
            className={`saved-tag${saving ? " is-saving" : ""}${error ? " is-error" : ""}`}
            title="Changes save automatically"
          >
            {statusLabel}
          </span>
          {error && <span className="details-error">{error}</span>}
        </div>
      </div>

      <div className="details-grid">
        <div className="field title-field">
          <label htmlFor="meeting-title">
            Title <span className="req">*</span>
          </label>
          <HistorySuggestField
            id="meeting-title"
            placeholder="Enter meeting title"
            value={title}
            onChange={setTitle}
            options={directory.titles}
          />
        </div>

        <div className="field">
          <label htmlFor="meeting-venue">
            Venue <span className="req">*</span>
          </label>
          <HistorySuggestField
            id="meeting-venue"
            placeholder="e.g. Conference Room A / Zoom"
            value={venue}
            onChange={setVenue}
            options={directory.venues}
          />
        </div>

        <div className="field">
          <label htmlFor="presiding-officer">Presiding officer</label>
          <NameSuggestField
            id="presiding-officer"
            role="officer"
            placeholder="e.g. Chair / Dean / Presiding Officer"
            value={presidingOfficer}
            onChange={setPresidingOfficer}
            directory={directory}
            recents={recents.officers}
            session={sessionHits}
            onSelect={(name) => {
              setPresidingOfficer(name);
              rememberRecentName("officer", name);
              setRecents(loadRecentNames());
            }}
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
              {(Array.isArray(attendees) ? attendees : []).map((name) => (
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
              <NameSuggestField
                role="attendee"
                placeholder="Type an attendee's name"
                value={attendeeInput}
                onChange={setAttendeeInput}
                directory={directory}
                exclude={attendees}
                recents={recents.attendees}
                session={sessionHits}
                onSelect={(name) => addAttendee(name)}
                onEnterWithoutHighlight={() => addAttendee()}
              />
              <button
                type="button"
                className="btn secondary"
                onClick={() => addAttendee()}
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
