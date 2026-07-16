import { useEffect, useRef, useState } from "react";
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

const ICONS = {
  copy: (
    <Icon>
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </Icon>
  ),
  download: (
    <Icon>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </Icon>
  ),
  open: (
    <Icon>
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </Icon>
  ),
  play: (
    <Icon>
      <polygon points="6 3 20 12 6 21 6 3" />
    </Icon>
  ),
  audio: (
    <Icon>
      <path d="M9 18V5l12-2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="16" r="3" />
    </Icon>
  ),
  refresh: (
    <Icon>
      <polyline points="23 4 23 10 17 10" />
      <polyline points="1 20 1 14 7 14" />
      <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
    </Icon>
  ),
  trash: (
    <Icon>
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </Icon>
  ),
};

function IconButton({
  label,
  onClick,
  disabled,
  busy,
  tone = "default",
  children,
}) {
  return (
    <button
      type="button"
      className={`icon-btn ${tone === "danger" ? "danger" : ""}`}
      onClick={onClick}
      disabled={disabled || busy}
      title={label}
      aria-label={label}
    >
      {busy ? <span className="spinner" /> : children}
    </button>
  );
}

function HistoryRow({ meeting, active, onSelect, onDelete, onRefresh }) {
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [showPlayer, setShowPlayer] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loadingAudio, setLoadingAudio] = useState(false);
  const audioRef = useRef(null);
  const urlRef = useRef(null);

  const hasTranscript = Boolean(
    meeting.has_transcript || meeting.status === "finalized"
  );
  const hasAudio = Boolean(meeting.has_audio);

  useEffect(() => {
    return () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!showPlayer || !audioUrl) return;
    const el = audioRef.current;
    if (!el) return;
    el.play().catch(() => {});
  }, [showPlayer, audioUrl]);

  function flash(msg) {
    setMessage(msg);
    setTimeout(() => setMessage(""), 1800);
  }

  async function handleCopy() {
    setBusy("copy");
    setError("");
    try {
      const { text } = await loadTranscript(meeting.id);
      await navigator.clipboard.writeText(text);
      flash("Copied");
    } catch (err) {
      setError(err.message || "Copy failed.");
    } finally {
      setBusy("");
    }
  }

  async function handleDownloadTranscript() {
    setBusy("download-text");
    setError("");
    try {
      const { text, title } = await loadTranscript(meeting.id);
      downloadTextFile(`${safeFilename(title)}_transcript`, text);
      flash("Downloaded");
    } catch (err) {
      setError(err.message || "Download failed.");
    } finally {
      setBusy("");
    }
  }

  async function ensureAudio() {
    if (audioUrl) return audioUrl;
    setLoadingAudio(true);
    setError("");
    try {
      const url = await api.getMeetingAudioUrl(meeting.id);
      if (!url) {
        setError("Audio file not found.");
        return null;
      }
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      urlRef.current = url;
      setAudioUrl(url);
      return url;
    } catch (err) {
      setError(err.message || "Could not load audio.");
      return null;
    } finally {
      setLoadingAudio(false);
    }
  }

  async function handlePlay() {
    setShowPlayer(true);
    await ensureAudio();
  }

  async function handleDownloadAudio() {
    setBusy("download-audio");
    setError("");
    try {
      const name = (meeting.title || "recording").trim() || "recording";
      await api.downloadMeetingAudio(meeting.id, name);
      flash("Audio saved");
    } catch (err) {
      setError(err.message || "Audio download failed.");
    } finally {
      setBusy("");
    }
  }

  async function handleRetranscribe() {
    setBusy("retranscribe");
    setError("");
    try {
      await api.retranscribeMeeting(meeting.id);
      flash("Transcribed");
      if (onRefresh) onRefresh();
    } catch (err) {
      setError(err.message || "Transcription failed.");
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
          {message ? <span className="card-tag">{message}</span> : null}
        </div>

        {showPlayer && (
          <div className="recording-player">
            {loadingAudio && !audioUrl ? (
              <span className="card-tag">Loading audio…</span>
            ) : audioUrl ? (
              <audio
                ref={audioRef}
                className="meeting-audio-player"
                controls
                src={audioUrl}
                preload="metadata"
              />
            ) : null}
          </div>
        )}

        {error ? (
          <div className="error-banner" style={{ marginTop: 10 }}>
            {error}
          </div>
        ) : null}
      </div>

      <div className="history-row-actions icon-actions">
        {hasTranscript && (
          <>
            <IconButton
              label="Copy text"
              onClick={handleCopy}
              busy={busy === "copy"}
              disabled={Boolean(busy)}
            >
              {ICONS.copy}
            </IconButton>
            <IconButton
              label="Download transcript"
              onClick={handleDownloadTranscript}
              busy={busy === "download-text"}
              disabled={Boolean(busy)}
            >
              {ICONS.download}
            </IconButton>
          </>
        )}
        {hasAudio && (
          <>
            <IconButton
              label="Play audio"
              onClick={handlePlay}
              busy={loadingAudio}
              disabled={Boolean(busy)}
            >
              {ICONS.play}
            </IconButton>
            <IconButton
              label="Download audio"
              onClick={handleDownloadAudio}
              busy={busy === "download-audio"}
              disabled={Boolean(busy)}
            >
              {ICONS.audio}
            </IconButton>
            <IconButton
              label={hasTranscript ? "Re-transcribe" : "Transcribe"}
              onClick={handleRetranscribe}
              busy={busy === "retranscribe"}
              disabled={Boolean(busy)}
            >
              {ICONS.refresh}
            </IconButton>
          </>
        )}
        <IconButton
          label="Open meeting"
          onClick={() => onSelect(meeting.id)}
          disabled={Boolean(busy)}
        >
          {ICONS.open}
        </IconButton>
        <IconButton
          label="Remove meeting"
          onClick={() => onDelete(meeting.id)}
          disabled={Boolean(busy)}
          tone="danger"
        >
          {ICONS.trash}
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
  onRefresh,
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
            All saved meetings and recordings in one place. Use the icons to
            copy, download, play, open, or remove a record.
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
                      onRefresh={onRefresh}
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
