import { useState } from "react";

function fmtDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function fmtDuration(sec) {
  const s = Math.max(0, Math.round(Number(sec) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, "0")}:${r.toString().padStart(2, "0")}`;
  }
  return `${m}:${r.toString().padStart(2, "0")}`;
}

function statusLabel(status) {
  if (status === "finalized") return "Refined";
  if (status === "processing") return "Processing";
  if (status === "recording") return "Draft";
  if (status === "failed") return "Failed";
  return status || "Saved";
}

function Icon({ children }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

function IconButton({ label, onClick, tone = "default", children }) {
  return (
    <button
      type="button"
      className={`icon-btn ${tone === "danger" ? "danger" : ""}`}
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      {children}
    </button>
  );
}

function HistoryRow({ meeting, active, onSelect, onDelete }) {
  const hasTranscript = Boolean(
    meeting.has_transcript || meeting.status === "finalized"
  );
  const hasAudio = Boolean(meeting.has_audio);

  return (
    <div className={`history-row ${active ? "active" : ""}`}>
      <div className="history-row-main">
        <div className="history-row-title">
          {meeting.title || "Untitled meeting"}
        </div>
        <div className="history-row-meta">
          <span>{fmtDate(meeting.meeting_date || meeting.created_at)}</span>
          {meeting.duration_seconds > 0 ? (
            <span>{fmtDuration(meeting.duration_seconds)}</span>
          ) : null}
          {meeting.venue ? <span>{meeting.venue}</span> : null}
          <span
            className={`badge ${
              meeting.status === "finalized"
                ? "refined"
                : meeting.status === "processing"
                  ? "processing"
                  : ""
            }`}
          >
            {statusLabel(meeting.status)}
          </span>
          {hasAudio ? <span className="badge">Audio</span> : null}
          {hasTranscript ? <span className="badge">Transcribed</span> : null}
          {meeting.has_summary ? <span className="badge">Summary</span> : null}
          {meeting.has_translation ? (
            <span className="badge">Translation</span>
          ) : null}
        </div>
      </div>

      <div className="history-row-actions icon-actions">
        <IconButton label="Open meeting" onClick={() => onSelect(meeting.id)}>
          <Icon>
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
            <polyline points="15 3 21 3 21 9" />
            <line x1="10" y1="14" x2="21" y2="3" />
          </Icon>
        </IconButton>
        <IconButton
          label="Remove meeting"
          onClick={() => onDelete(meeting.id)}
          tone="danger"
        >
          <Icon>
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
            <path d="M10 11v6" />
            <path d="M14 11v6" />
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
          </Icon>
        </IconButton>
      </div>
    </div>
  );
}

export default function HistoryPanel({
  meetings,
  loading,
  search,
  onSearch,
  activeId,
  onSelect,
  onDelete,
  onCreate,
}) {
  const count = meetings.length;

  return (
    <div className="content history-panel">
      <div className="card history-card">
        <div className="card-head">
          <h3>History</h3>
          <span className="card-tag">Meetings &amp; recordings</span>
        </div>
        <div className="card-body">
          <p className="settings-intro">
            All saved meetings and recordings in one place. Open a record to
            copy, download, play audio, or re-transcribe.
          </p>

          <div className="history-toolbar">
            <input
              className="history-search"
              placeholder="Search meetings…"
              value={search}
              onChange={(e) => onSearch(e.target.value)}
            />
            <button type="button" className="btn" onClick={onCreate}>
              + New meeting
            </button>
          </div>

          <section
            className="saved-meetings-container"
            aria-label="Saved recorded meetings"
          >
            <div className="saved-meetings-container-head">
              <h4 className="saved-meetings-container-title">
                Saved recorded meetings
              </h4>
              <span className="saved-meetings-container-count">
                {loading ? "Loading…" : `${count} meeting${count === 1 ? "" : "s"}`}
              </span>
            </div>
            <div className="saved-meetings-container-body">
              {loading ? (
                <div className="center-spin">
                  <span className="spinner" /> Loading…
                </div>
              ) : count === 0 ? (
                <div className="history-list-empty">
                  No saved meetings yet. Create a new meeting to start recording.
                </div>
              ) : (
                <div className="history-list">
                  {meetings.map((m) => (
                    <HistoryRow
                      key={m.id}
                      meeting={m}
                      active={m.id === activeId}
                      onSelect={onSelect}
                      onDelete={onDelete}
                    />
                  ))}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
