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
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, "0")}`;
}

function RecordingRow({
  meeting,
  onOpen,
  onDelete,
  onTranscribed,
}) {
  const [audioUrl, setAudioUrl] = useState(null);
  const [loadingAudio, setLoadingAudio] = useState(false);
  const [showPlayer, setShowPlayer] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState("");
  const audioRef = useRef(null);
  const urlRef = useRef(null);

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

  async function handleDownload() {
    setDownloading(true);
    setError("");
    try {
      const name = (meeting.title || "recording").trim() || "recording";
      await api.downloadMeetingAudio(meeting.id, name);
    } catch (err) {
      setError(err.message || "Download failed.");
    } finally {
      setDownloading(false);
    }
  }

  async function handleTranscribe() {
    setTranscribing(true);
    setError("");
    setPreview("");
    try {
      const detail = await api.retranscribeMeeting(meeting.id);
      setPreview(detail.final_transcript || "");
      if (onTranscribed) onTranscribed(detail);
    } catch (err) {
      setError(err.message || "Transcription failed.");
    } finally {
      setTranscribing(false);
    }
  }

  const hasTranscript = Boolean(meeting.has_summary || meeting.status === "finalized");

  return (
    <div className="recording-row">
      <div className="recording-row-main">
        <div className="history-row-title">{meeting.title || "Untitled meeting"}</div>
        <div className="history-row-meta">
          <span>{fmtDate(meeting.meeting_date || meeting.created_at)}</span>
          <span>{fmtDuration(meeting.duration_seconds)}</span>
          {meeting.venue ? <span>{meeting.venue}</span> : null}
          <span className="badge">WAV</span>
          {meeting.status === "finalized" && (
            <span className="badge refined">Transcribed</span>
          )}
          {meeting.status === "processing" && (
            <span className="badge processing">Processing</span>
          )}
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

        {preview ? (
          <div className="recording-transcript-preview">
            <div className="sidebar-section-label" style={{ padding: 0 }}>
              Latest transcript
            </div>
            <p>{preview.length > 360 ? `${preview.slice(0, 360)}…` : preview}</p>
          </div>
        ) : null}

        {error ? <div className="error-banner" style={{ marginTop: 10 }}>{error}</div> : null}
      </div>

      <div className="recording-row-actions">
        <button
          type="button"
          className="btn secondary meeting-action-btn"
          onClick={handlePlay}
          disabled={loadingAudio}
        >
          {loadingAudio ? <span className="spinner" /> : "Play"}
        </button>
        <button
          type="button"
          className="btn secondary meeting-action-btn"
          onClick={handleTranscribe}
          disabled={transcribing}
          title="Run Whisper ASR on this recording"
        >
          {transcribing ? (
            <span className="spinner" />
          ) : hasTranscript ? (
            "Re-transcribe"
          ) : (
            "Transcribe"
          )}
        </button>
        <button
          type="button"
          className="btn secondary meeting-action-btn"
          onClick={() => onOpen(meeting.id)}
        >
          Open
        </button>
        <button
          type="button"
          className="btn ghost meeting-action-btn"
          onClick={handleDownload}
          disabled={downloading}
        >
          {downloading ? <span className="spinner" /> : "Download"}
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

export default function RecordingsPanel({
  recordings,
  loading,
  search,
  onSearch,
  onOpen,
  onDelete,
  onRefresh,
}) {
  return (
    <div className="content history-panel recordings-panel">
      <div className="card history-card">
        <div className="card-head">
          <h3>Recordings</h3>
          <span className="card-tag">Saved audio</span>
        </div>
        <div className="card-body">
          <p className="settings-intro">
            Every saved meeting recording appears here. Play audio, download the
            WAV, run or re-run Whisper transcription, or open the full meeting.
          </p>

          <div className="history-toolbar">
            <input
              className="history-search"
              placeholder="Search recordings…"
              value={search}
              onChange={(e) => onSearch(e.target.value)}
            />
          </div>

          {loading ? (
            <div className="center-spin">
              <span className="spinner" /> Loading…
            </div>
          ) : recordings.length === 0 ? (
            <div className="placeholder" style={{ height: "auto", padding: 32 }}>
              No recordings yet. Stop a meeting recording to save audio here.
            </div>
          ) : (
            <div className="history-list">
              {recordings.map((m) => (
                <RecordingRow
                  key={m.id}
                  meeting={m}
                  onOpen={onOpen}
                  onDelete={onDelete}
                  onTranscribed={() => onRefresh && onRefresh()}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
