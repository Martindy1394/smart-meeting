import { useState } from "react";
import { api } from "../api/client";

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

function statusLabel(status) {
  if (status === "finalized") return "Refined";
  if (status === "processing") return "Processing";
  if (status === "recording") return "Draft";
  if (status === "failed") return "Failed";
  return status || "Saved";
}

function safeFilename(title) {
  const base = (title || "transcript").trim() || "transcript";
  return base.replace(/[^\w\-]+/g, "_").replace(/_+/g, "_").slice(0, 80);
}

async function loadTranscript(meetingId) {
  const detail = await api.getMeeting(meetingId);
  const text = (detail.final_transcript || "").trim();
  if (!text) {
    throw new Error("No transcript is available for this meeting.");
  }
  return { text, title: detail.title || "transcript" };
}

function downloadTextFile(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".txt") ? filename : `${filename}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function HistoryRow({
  meeting,
  active,
  onSelect,
  onDelete,
}) {
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const hasTranscript = Boolean(meeting.has_transcript || meeting.status === "finalized");

  async function handleCopy() {
    setBusy("copy");
    setMessage("");
    try {
      const { text } = await loadTranscript(meeting.id);
      await navigator.clipboard.writeText(text);
      setMessage("Copied");
      setTimeout(() => setMessage(""), 1800);
    } catch (err) {
      setMessage(err.message || "Copy failed.");
    } finally {
      setBusy("");
    }
  }

  async function handleDownload() {
    setBusy("download");
    setMessage("");
    try {
      const { text, title } = await loadTranscript(meeting.id);
      downloadTextFile(`${safeFilename(title)}_transcript`, text);
      setMessage("Downloaded");
      setTimeout(() => setMessage(""), 1800);
    } catch (err) {
      setMessage(err.message || "Download failed.");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className={`history-row ${active ? "active" : ""}`}>
      <div className="history-row-main">
        <div className="history-row-title">
          {meeting.title || "Untitled meeting"}
        </div>
        <div className="history-row-meta">
          <span>{fmtDate(meeting.meeting_date || meeting.created_at)}</span>
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
          {meeting.has_audio ? <span className="badge">Audio</span> : null}
          {hasTranscript ? <span className="badge">Transcribed</span> : null}
          {meeting.has_summary ? <span className="badge">Summary</span> : null}
          {meeting.has_translation ? (
            <span className="badge">Translation</span>
          ) : null}
          {message ? <span className="card-tag">{message}</span> : null}
        </div>
      </div>
      <div className="history-row-actions">
        {hasTranscript && (
          <>
            <button
              type="button"
              className="btn secondary meeting-action-btn"
              onClick={handleCopy}
              disabled={Boolean(busy)}
              title="Copy transcript text"
            >
              {busy === "copy" ? <span className="spinner" /> : "Copy text"}
            </button>
            <button
              type="button"
              className="btn secondary meeting-action-btn"
              onClick={handleDownload}
              disabled={Boolean(busy)}
              title="Download transcript as a text file"
            >
              {busy === "download" ? <span className="spinner" /> : "Download"}
            </button>
          </>
        )}
        <button
          type="button"
          className="btn secondary meeting-action-btn"
          onClick={() => onSelect(meeting.id)}
        >
          Open
        </button>
        <button
          type="button"
          className="btn ghost meeting-action-btn meeting-remove-btn"
          onClick={() => onDelete(meeting.id)}
        >
          Remove
        </button>
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
          <span className="card-tag">Saved records</span>
        </div>
        <div className="card-body">
          <p className="settings-intro">
            All of your saved meeting records live here. Open one to review the
            transcript, or copy / download transcribed text.
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
