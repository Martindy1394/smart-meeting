import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { useRecorder } from "../hooks/useRecorder.js";
import ConfirmDialog from "./ConfirmDialog.jsx";
import MeetingDetails from "./MeetingDetails.jsx";

function fmtTime(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0));
  const h = Math.floor(total / 3600)
    .toString()
    .padStart(2, "0");
  const m = Math.floor((total % 3600) / 60)
    .toString()
    .padStart(2, "0");
  const s = (total % 60).toString().padStart(2, "0");
  // Always H:MM:SS so long board meetings read clearly from the first second.
  return `${h}:${m}:${s}`;
}

function MiniIcon({ children }) {
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

export default function MeetingRoom({
  meeting,
  onMeetingUpdated,
  onAutosaveStatus,
  historyView = false,
  onBack,
}) {
  const detailsRef = useRef(null);
  const [detailsReady, setDetailsReady] = useState(false);
  const [savingDetails, setSavingDetails] = useState(false);

  const [finalTranscript, setFinalTranscript] = useState(meeting.final_transcript || "");
  const [status, setStatus] = useState(meeting.status);

  // Summary state
  const [summaryFormat, setSummaryFormat] = useState(meeting.summary_format || "bullets");
  const [summary, setSummary] = useState(meeting.summary || "");
  const [summaryEngine, setSummaryEngine] = useState("");
  const [extractiveFallback, setExtractiveFallback] = useState(false);
  const [faithfulness, setFaithfulness] = useState(null);
  const [exportFormat, setExportFormat] = useState("pdf");
  const [summarizing, setSummarizing] = useState(false);
  const [summaryError, setSummaryError] = useState("");

  // Translation state — always auto-translates the transcript into English.
  const [translation, setTranslation] = useState(meeting.translation || "");
  const [translationLang, setTranslationLang] = useState(
    meeting.translation_language || "English"
  );
  const [translating, setTranslating] = useState(false);
  const [translateError, setTranslateError] = useState("");
  const autoTranslateRef = useRef("");
  const autoSummaryRef = useRef("");
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [asrBusy, setAsrBusy] = useState(false);
  const [asrError, setAsrError] = useState("");
  const [hasAudio, setHasAudio] = useState(Boolean(meeting.has_audio));
  const [copyState, setCopyState] = useState("");
  const [endConfirmOpen, setEndConfirmOpen] = useState(false);
  const audioUrlRef = useRef(null);

  const revokeAudioUrl = useCallback(() => {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    setAudioUrl(null);
  }, []);

  const loadAudio = useCallback(
    async (meetingId) => {
      setAudioLoading(true);
      try {
        const url = await api.getMeetingAudioUrl(meetingId);
        if (!url) {
          revokeAudioUrl();
          return;
        }
        if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = url;
        setAudioUrl(url);
      } catch {
        revokeAudioUrl();
      } finally {
        setAudioLoading(false);
      }
    },
    [revokeAudioUrl]
  );

  const saveDetails = useCallback(async ({ silent = false } = {}) => {
    if (!detailsRef.current) return false;
    setSavingDetails(true);
    try {
      return await detailsRef.current.save({ silent });
    } finally {
      setSavingDetails(false);
    }
  }, []);

  const summarizeFromTranscript = useCallback(
    async (format = summaryFormat, { forceRetranslate = false } = {}) => {
      setSummarizing(true);
      setSummaryError("");
      try {
        // Backend: full-transcript → English, then contextual BART summary.
        const res = await api.summarize({
          meeting_id: meeting.id,
          output_format: format,
          force_retranslate: Boolean(forceRetranslate),
        });
        setSummary(res.summary || "");
        setSummaryEngine(res.engine || "");
        setExtractiveFallback(Boolean(res.extractive_fallback));
        setFaithfulness(res.faithfulness || null);
        // Always refresh translation from this pass (clear if empty).
        setTranslation(res.translation || "");
        setTranslationLang(res.translation_language || "English");
        autoTranslateRef.current = (res.translation || "").trim()
          ? "synced"
          : "";
        autoSummaryRef.current = `${meeting.id}:${format}`;
        if (onMeetingUpdated) onMeetingUpdated();
      } catch (err) {
        setSummaryError(err.message || "Summarization failed.");
      } finally {
        setSummarizing(false);
      }
    },
    [meeting.id, onMeetingUpdated, summaryFormat]
  );

  const translateToEnglish = useCallback(
    async (transcriptText, { force = false } = {}) => {
      const text = (transcriptText || "").trim();
      if (!text) return;
      if (!force && autoTranslateRef.current === text) return;
      autoTranslateRef.current = text;
      setTranslating(true);
      setTranslateError("");
      try {
        const res = await api.translate({
          meeting_id: meeting.id,
          target_language: "en",
        });
        setTranslation(res.translation);
        setTranslationLang(res.language_name || "English");
        if (onMeetingUpdated) onMeetingUpdated();
      } catch (err) {
        autoTranslateRef.current = "";
        setTranslateError(err.message || "English translation failed.");
      } finally {
        setTranslating(false);
      }
    },
    [meeting.id, onMeetingUpdated]
  );

  const applyTranscriptResult = useCallback(
    async (detailOrPayload, { persistDetails = false } = {}) => {
      const text =
        detailOrPayload.final_transcript ||
        detailOrPayload.text ||
        "";
      setFinalTranscript(text);
      setStatus(detailOrPayload.status || "finalized");
      setHasAudio(Boolean(detailOrPayload.has_audio || text || audioUrl));
      setAsrError("");
      // New transcript invalidates prior AI outputs — clear UI immediately.
      setSummary("");
      setTranslation("");
      setSummaryError("");
      setTranslateError("");
      autoTranslateRef.current = "";
      autoSummaryRef.current = "";
      if (persistDetails) {
        await saveDetails({ silent: true });
      }
      if (onMeetingUpdated) onMeetingUpdated();
      await loadAudio(meeting.id);
      if (text.trim()) {
        // Force a fresh English translation of the new transcript, then summarize.
        await summarizeFromTranscript(summaryFormat, { forceRetranslate: true });
      }
    },
    [
      audioUrl,
      loadAudio,
      meeting.id,
      onMeetingUpdated,
      saveDetails,
      summarizeFromTranscript,
      summaryFormat,
    ]
  );

  const onFinalTranscript = useCallback(
    async (data) => {
      // Live recording stop → Whisper ASR full-accuracy pass finished.
      await applyTranscriptResult(
        { text: data.text || "", status: "finalized", has_audio: true },
        { persistDetails: true }
      );
    },
    [applyTranscriptResult]
  );

  const runWhisperAsr = useCallback(async () => {
    setAsrBusy(true);
    setAsrError("");
    setStatus("processing");
    // Clear stale AI panels while Whisper rewrites the transcript.
    setSummary("");
    setTranslation("");
    setSummaryError("");
    setTranslateError("");
    autoTranslateRef.current = "";
    autoSummaryRef.current = "";
    try {
      let detail = await api.retranscribeMeeting(meeting.id);
      // Whisper can take several minutes on CPU — poll instead of one long POST.
      const deadline = Date.now() + 30 * 60 * 1000;
      while (detail?.status === "processing" && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 2500));
        detail = await api.getMeeting(meeting.id);
      }
      if (!detail || detail.status === "processing") {
        throw new Error(
          "Transcription is still running. Wait a bit, then refresh the meeting."
        );
      }
      if (detail.status === "failed") {
        throw new Error("Whisper ASR failed. Try re-transcribe again.");
      }
      await applyTranscriptResult(detail);
    } catch (err) {
      setAsrError(err.message || "Whisper ASR failed.");
      setStatus(meeting.status || "finalized");
    } finally {
      setAsrBusy(false);
    }
  }, [applyTranscriptResult, meeting.id, meeting.status]);

  const recorder = useRecorder({ onFinalTranscript });

  // Reset local state when the selected meeting changes.
  useEffect(() => {
    setFinalTranscript(meeting.final_transcript || "");
    setStatus(meeting.status);
    setSummary(meeting.summary || "");
    setSummaryFormat(meeting.summary_format || "bullets");
    setTranslation(meeting.translation || "");
    setTranslationLang(meeting.translation_language || "English");
    setSummaryError("");
    setTranslateError("");
    setAsrError("");
    setSummaryEngine("");
    setHasAudio(Boolean(meeting.has_audio));
    autoTranslateRef.current = meeting.translation ? meeting.final_transcript || "" : "";
    autoSummaryRef.current = meeting.summary
      ? `${meeting.id}:${meeting.summary_format || "bullets"}`
      : "";
    revokeAudioUrl();
    if (meeting.has_audio || meeting.status === "finalized") {
      loadAudio(meeting.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id]);

  useEffect(() => {
    return () => revokeAudioUrl();
  }, [revokeAudioUrl]);

  // Auto-summarize finalized transcripts (also produces English translation).
  useEffect(() => {
    const text = (meeting.final_transcript || "").trim();
    if (!text) return;
    if (meeting.summary) return;
    if (autoSummaryRef.current.startsWith(`${meeting.id}:`)) return;
    summarizeFromTranscript(summaryFormat);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id, meeting.final_transcript, meeting.summary]);

  // If a summary already exists but English translation is missing, translate only.
  useEffect(() => {
    const text = (meeting.final_transcript || "").trim();
    if (!text) return;
    if (meeting.translation) return;
    if (!meeting.summary) return; // summarize pipeline will create both
    translateToEnglish(text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id, meeting.final_transcript, meeting.translation, meeting.summary]);

  // Preload the AudioWorklet so Start recording does not wait on the network.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        const ctx = new AudioCtx();
        await ctx.audioWorklet.addModule("/pcm-worklet.js");
        window.__smWorkletReady = true;
        if (!cancelled) await ctx.close();
      } catch {
        /* worklet will load on start */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function toggleRecord() {
    if (recorder.recording || recorder.status === "starting") {
      if (recorder.recording && recorder.status !== "starting") {
        setEndConfirmOpen(true);
      }
      return;
    }
    if (!detailsRef.current?.isComplete()) return;
    try {
      // Stamp meeting date/time to the current moment when recording starts.
      detailsRef.current.useCurrentDateTime?.();
      await recorder.start(meeting.id);
      setStatus("recording");
    } catch {
      /* handled inside hook via message */
    }
  }

  function confirmEndMeeting() {
    setEndConfirmOpen(false);
    recorder.stop();
  }

  const canStart =
    detailsReady &&
    recorder.status !== "finalizing" &&
    recorder.status !== "starting";
  const isRefined = status === "finalized" && finalTranscript;
  const hasTranscript = Boolean(finalTranscript);
  const hasEnglish = Boolean((translation || "").trim());
  const hasMinutes = Boolean((summary || "").trim());
  const showLive =
    recorder.recording ||
    recorder.status === "finalizing" ||
    recorder.status === "starting" ||
    recorder.status === "paused";
  const isStarting = recorder.status === "starting";
  const isPaused = Boolean(recorder.paused || recorder.status === "paused");
  // History detail: never show Start live transcription. New meetings keep it.
  const hideLiveUploadActions =
    historyView &&
    !recorder.recording &&
    recorder.status !== "starting" &&
    recorder.status !== "paused" &&
    recorder.status !== "finalizing";

  async function runSummarize() {
    if (!hasTranscript) {
      setSummaryError("Finalize a transcript first.");
      return;
    }
    await summarizeFromTranscript(summaryFormat);
  }

  async function copyTranscript() {
    const text = (finalTranscript || "").trim();
    if (!text) {
      setAsrError("No transcript is available to copy.");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCopyState("Copied");
      setTimeout(() => setCopyState(""), 1800);
    } catch {
      setCopyState("Copy failed");
      setTimeout(() => setCopyState(""), 1800);
    }
  }

  async function downloadTranscript(format = "txt") {
    const hasAny =
      Boolean((finalTranscript || "").trim()) ||
      Boolean((summary || "").trim()) ||
      Boolean((translation || "").trim());
    if (!hasAny) {
      setAsrError("Nothing to export yet — finalize a transcript first.");
      return;
    }
    try {
      const { blob, filename } = await api.exportMeeting(meeting.id, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setAsrError(err.message || "Export failed.");
    }
  }

  return (
    <div className={`content ${historyView ? "history-detail" : ""}`}>
      {historyView && (
        <div className="history-detail-toolbar">
          {onBack && (
            <button
              type="button"
              className="icon-btn back-icon-btn"
              onClick={onBack}
              title="Back to saved meetings"
              aria-label="Back to saved meetings"
            >
              <MiniIcon>
                <line x1="19" y1="12" x2="5" y2="12" />
                <polyline points="12 19 5 12 12 5" />
              </MiniIcon>
            </button>
          )}
          <div className="history-detail-heading">
            <span className="history-detail-kicker">Saved meeting</span>
            <h2 className="history-detail-title">
              {meeting.title || "Untitled meeting"}
            </h2>
          </div>
          <div className="history-detail-actions icon-actions">
            <button
              type="button"
              className="icon-btn"
              onClick={copyTranscript}
              disabled={!hasTranscript}
              title={copyState === "Copied" ? "Copied" : "Copy text"}
              aria-label={copyState === "Copied" ? "Copied" : "Copy text"}
            >
              {copyState === "Copied" ? (
                <span className="card-tag" style={{ margin: 0 }}>✓</span>
              ) : (
                <MiniIcon>
                  <rect x="9" y="9" width="13" height="13" rx="2" />
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </MiniIcon>
              )}
            </button>
            <div className="export-controls">
              <label className="sr-only" htmlFor="export-format-toolbar">
                Export format
              </label>
              <select
                id="export-format-toolbar"
                className="export-format-select"
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value)}
                aria-label="Export format"
              >
                <option value="pdf">PDF</option>
                <option value="docx">DOCX</option>
                <option value="txt">TXT</option>
              </select>
              <button
                type="button"
                className="icon-btn"
                onClick={() => downloadTranscript(exportFormat)}
                disabled={!hasTranscript && !hasMinutes && !hasEnglish}
                title={`Export as ${exportFormat.toUpperCase()}`}
                aria-label={`Export as ${exportFormat.toUpperCase()}`}
              >
                <MiniIcon>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </MiniIcon>
              </button>
            </div>
            {hasAudio && (
              <button
                type="button"
                className="icon-btn"
                onClick={runWhisperAsr}
                disabled={asrBusy}
                title={hasTranscript ? "Re-transcribe" : "Transcribe"}
                aria-label={hasTranscript ? "Re-transcribe" : "Transcribe"}
              >
                {asrBusy ? (
                  <span className="spinner" />
                ) : (
                  <MiniIcon>
                    <polyline points="23 4 23 10 17 10" />
                    <polyline points="1 20 1 14 7 14" />
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                  </MiniIcon>
                )}
              </button>
            )}
          </div>
        </div>
      )}

      {!recorder.transcriptionAvailable && recorder.message && (
        <div className="banner-warn">{recorder.message}</div>
      )}
      {recorder.status === "error" && recorder.message && (
        <div className="error-banner">{recorder.message}</div>
      )}

      <div className="top-row">
        <MeetingDetails
          ref={detailsRef}
          meeting={meeting}
          onUpdated={onMeetingUpdated}
          onValidityChange={setDetailsReady}
          onAutosaveStatus={onAutosaveStatus}
        />

        <div className="card transcript-card">
          <div className="card-head">
            <h3>
              Transcript
              {isRefined && <span className="badge refined">Refined</span>}
              {showLive && (
                <span className="badge live">
                  <span className="dot-live" /> Live
                </span>
              )}
            </h3>
            <div className="transcript-head-meta">
              {/* On saved-meeting pages, actions live in the top toolbar. */}
              {!historyView && (hasTranscript || hasAudio) && (
                <div className="transcript-actions">
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={copyTranscript}
                    disabled={!hasTranscript}
                    title={copyState === "Copied" ? "Copied" : "Copy text"}
                    aria-label={copyState === "Copied" ? "Copied" : "Copy text"}
                  >
                    {copyState === "Copied" ? (
                      <span className="card-tag" style={{ margin: 0 }}>✓</span>
                    ) : (
                      <MiniIcon>
                        <rect x="9" y="9" width="13" height="13" rx="2" />
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                      </MiniIcon>
                    )}
                  </button>
                  <div className="export-controls">
                    <label className="sr-only" htmlFor="export-format-card">
                      Export format
                    </label>
                    <select
                      id="export-format-card"
                      className="export-format-select"
                      value={exportFormat}
                      onChange={(e) => setExportFormat(e.target.value)}
                      aria-label="Export format"
                    >
                      <option value="pdf">PDF</option>
                      <option value="docx">DOCX</option>
                      <option value="txt">TXT</option>
                    </select>
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={() => downloadTranscript(exportFormat)}
                      disabled={!hasTranscript && !hasMinutes && !hasEnglish}
                      title={`Export as ${exportFormat.toUpperCase()}`}
                      aria-label={`Export as ${exportFormat.toUpperCase()}`}
                    >
                      <MiniIcon>
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                      </MiniIcon>
                    </button>
                  </div>
                  {hasAudio && !recorder.recording && (
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={runWhisperAsr}
                      disabled={asrBusy || recorder.status === "finalizing"}
                      title={
                        hasTranscript
                          ? "Re-transcribe"
                          : "Transcribe with Whisper"
                      }
                      aria-label={
                        hasTranscript
                          ? "Re-transcribe"
                          : "Transcribe with Whisper"
                      }
                    >
                      {asrBusy ? (
                        <span className="spinner" />
                      ) : (
                        <MiniIcon>
                          <polyline points="23 4 23 10 17 10" />
                          <polyline points="1 20 1 14 7 14" />
                          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                        </MiniIcon>
                      )}
                    </button>
                  )}
                </div>
              )}
              <span
                className="card-tag"
                title="Live captions: Whisper, or FastConformer RNN-T when NeMo is installed. Final/re-transcribe: Whisper with Hiligaynon bias."
              >
                ASR · auto · Hiligaynon bias
              </span>
            </div>
          </div>
          <div className="card-body">
            {asrError && <div className="error-banner">{asrError}</div>}
            {showLive && recorder.liveText ? (
              <span className="transcript-live">{recorder.liveText}</span>
            ) : hasTranscript ? (
              finalTranscript
            ) : showLive ? (
              <span className="transcript-live">
                {isStarting
                  ? "Starting meeting and microphone…"
                  : isPaused
                    ? recorder.message || "Paused"
                    : recorder.message || "Listening…"}
              </span>
            ) : asrBusy ? (
              <div className="center-spin">
                <span className="spinner" /> Running Whisper ASR on audio…
              </div>
            ) : (
              <div className="placeholder">
                Press <strong>Start a meeting</strong> for live Whisper captions.
                Ending a meeting runs the full-accuracy Whisper pass
                automatically. Meeting details autosave as you edit.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Live start for new meetings only. History detail uses toolbar actions. */}
      {!hideLiveUploadActions && (
        <div className="recorder-bar">
          <div className="record-control">
            <button
              className={`btn record ${recorder.recording || isStarting ? "active" : ""}`}
              onClick={toggleRecord}
              disabled={
                asrBusy ||
                isStarting ||
                (recorder.recording ? recorder.status === "finalizing" : !canStart)
              }
              title={
                !detailsReady && !recorder.recording
                  ? "Fill in all meeting details first"
                  : undefined
              }
            >
              {isStarting
                ? "Starting…"
                : recorder.recording
                  ? "■ End"
                  : "● Start a meeting"}
            </button>
            <div className="transport-row" role="group" aria-label="Pause and play">
              <button
                type="button"
                className={`btn secondary transport-btn ${
                  recorder.recording && !isPaused ? "transport-active" : ""
                }`}
                onClick={() => recorder.pause()}
                disabled={!recorder.recording || isStarting || isPaused || asrBusy}
                title="Pause live transcription"
                aria-label="Pause live transcription"
              >
                ❚❚ Pause
              </button>
              <button
                type="button"
                className={`btn secondary transport-btn ${
                  isPaused ? "transport-active" : ""
                }`}
                onClick={() => recorder.resume()}
                disabled={!recorder.recording || isStarting || !isPaused || asrBusy}
                title="Resume live transcription"
                aria-label="Play live transcription"
              >
                ▶ Play
              </button>
            </div>
            <div
              className={`record-duration ${
                recorder.recording && !isPaused
                  ? "active"
                  : isPaused
                    ? "paused"
                    : recorder.status === "finalizing"
                      ? "finalizing"
                      : ""
              }`}
              aria-live="polite"
              aria-label={`Meeting duration ${fmtTime(recorder.elapsed)}`}
            >
              {(recorder.recording ||
                recorder.status === "starting" ||
                recorder.status === "finalizing" ||
                recorder.elapsed > 0) && (
                <>
                  {recorder.recording && !isPaused && (
                    <span className="dot-live" />
                  )}
                  <span className="record-duration-label">
                    {recorder.status === "finalizing"
                      ? "Duration"
                      : isPaused
                        ? "Paused"
                        : "Time"}
                  </span>
                  <span className="record-duration-value">
                    {fmtTime(recorder.elapsed)}
                  </span>
                </>
              )}
              {!recorder.recording &&
                recorder.status !== "starting" &&
                recorder.status !== "finalizing" &&
                recorder.elapsed === 0 && (
                  <span className="record-duration-idle">00:00:00</span>
                )}
            </div>
          </div>

          {!detailsReady && !recorder.recording && (
            <span className="card-tag">
              Fill in title, venue, date &amp; time, and attendees first
            </span>
          )}
          {(recorder.status === "finalizing" || asrBusy) && (
            <span className="center-spin" style={{ padding: 0 }}>
              <span className="spinner" />{" "}
              {asrBusy
                ? "Whisper ASR processing audio…"
                : "Finalizing full-accuracy Whisper transcript…"}
            </span>
          )}
          {recorder.connectionState === "connecting" && (
            <span className="card-tag">connecting…</span>
          )}
        </div>
      )}

      <div className="recording-player-bar" aria-label="Meeting recording player">
        <div className="recording-player-meta">
          <span className="recording-player-label">Recording</span>
          <span className="card-tag">
            {audioLoading
              ? "Loading…"
              : audioUrl || hasAudio
                ? "Saved audio"
                : "No recording yet"}
          </span>
        </div>
        <div className="audio-player-wrap">
          {audioLoading && !audioUrl ? (
            <div className="audio-player-empty">
              <span className="audio-player-hint">Loading recording…</span>
            </div>
          ) : audioUrl ? (
            <audio
              className="meeting-audio-player"
              controls
              src={audioUrl}
              preload="metadata"
            >
              Your browser does not support audio playback.
            </audio>
          ) : (
            <div className="audio-player-empty">
              <span className="audio-player-hint">
                Finish a recording to play it here.
              </span>
            </div>
          )}
        </div>
      </div>

      <div
        className="ai-pipeline"
        aria-label="AI pipeline: Whisper transcript, mBART English, BART minutes"
      >
        <span className="ai-pipeline-label">AI pipeline</span>
        <ol className="ai-pipeline-steps">
          <li className={hasTranscript || showLive ? "done" : ""}>
            <span className="ai-pipeline-model">Whisper</span>
            <span className="ai-pipeline-role">
              {showLive && !isRefined ? "live captions" : "transcript"}
            </span>
          </li>
          <li className="ai-pipeline-sep" aria-hidden="true">
            →
          </li>
          <li className={hasEnglish ? "done" : ""}>
            <span className="ai-pipeline-model">mBART / NLLB</span>
            <span className="ai-pipeline-role">English</span>
          </li>
          <li className="ai-pipeline-sep" aria-hidden="true">
            →
          </li>
          <li className={hasMinutes ? "done" : ""}>
            <span className="ai-pipeline-model">BART</span>
            <span className="ai-pipeline-role">minutes</span>
          </li>
        </ol>
      </div>

      <div className="cards bottom-cards">
        {/* Summary card (BART) — from English translation of transcript */}
        <div className="card">
          <div className="card-head">
            <h3>Summary</h3>
            <span className="card-tag">
              {extractiveFallback
                ? "Extractive fallback · not full BART"
                : "BART · English minutes"}
              {summaryEngine ? ` · ${summaryEngine}` : ""}
            </span>
          </div>
          <div className="card-head" style={{ borderTop: "none", paddingTop: 0 }}>
            <div className="toggle-group">
              <button
                className={summaryFormat === "bullets" ? "active" : ""}
                onClick={() => setSummaryFormat("bullets")}
              >
                Bullet points
              </button>
              <button
                className={summaryFormat === "numbered" ? "active" : ""}
                onClick={() => setSummaryFormat("numbered")}
              >
                Numbered
              </button>
            </div>
            <button
              className="btn"
              onClick={runSummarize}
              disabled={!hasTranscript || summarizing}
            >
              {summarizing ? <span className="spinner" /> : "Summarize"}
            </button>
          </div>
          <div className="card-body">
            {extractiveFallback && summary && (
              <div className="banner-warn" role="status">
                Extractive summary fallback — install <code>requirements-ml.txt</code>{" "}
                for full BART minutes. Do not treat this as generative board minutes.
              </div>
            )}
            {faithfulness?.status === "warn" &&
              Array.isArray(faithfulness.untraced) &&
              faithfulness.untraced.length > 0 && (
                <div className="banner-warn faithfulness-warn" role="status">
                  <strong>Faithfulness check:</strong>{" "}
                  {faithfulness.untraced.length} Decision/Action line
                  {faithfulness.untraced.length === 1 ? "" : "s"} could not be
                  traced to the English source — review before using as a legal
                  or board record.
                  <ul className="faithfulness-list">
                    {faithfulness.untraced.slice(0, 8).map((item, idx) => (
                      <li key={`${item.section}-${idx}`}>
                        <span className="faithfulness-section">{item.section}</span>
                        {item.line}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            {summaryError && <div className="error-banner">{summaryError}</div>}
            {summarizing ? (
              <div className="center-spin">
                <span className="spinner" /> Summarizing transcript…
              </div>
            ) : summary ? (
              <div
                className={
                  faithfulness?.status === "warn"
                    ? "summary-text has-faithfulness-warn"
                    : "summary-text"
                }
              >
                {summary}
              </div>
            ) : (
              <div className="placeholder">
                {!hasTranscript
                  ? "Finalize a transcript first."
                  : "Summary runs automatically after transcription."}
              </div>
            )}
          </div>
        </div>

        {/* Translation card (mBART/NLLB) — auto English for BART */}
        <div className="card">
          <div className="card-head">
            <h3>English translation</h3>
            <span className="card-tag">
              mBART / NLLB · {translationLang || "English"}
            </span>
          </div>
          <div className="card-body">
            {translateError && <div className="error-banner">{translateError}</div>}
            {translating ? (
              <div className="center-spin">
                <span className="spinner" /> Translating to English…
              </div>
            ) : translation ? (
              translation
            ) : (
              <div className="placeholder">
                {hasTranscript
                  ? "English translation will appear here automatically."
                  : "Finalize a transcript to auto-translate into English."}
              </div>
            )}
          </div>
        </div>
      </div>

      <ConfirmDialog
        open={endConfirmOpen}
        tone="danger"
        title="End meeting?"
        message="Stop recording and run the full-accuracy Whisper transcript? You won’t be able to continue this live session afterward."
        confirmLabel="End meeting"
        cancelLabel="Keep recording"
        onCancel={() => setEndConfirmOpen(false)}
        onConfirm={confirmEndMeeting}
      />
    </div>
  );
}

export { MeetingRoom };
