import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { useRecorder } from "../hooks/useRecorder.js";
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

export default function MeetingRoom({ meeting, onMeetingUpdated, onSaveControls }) {
  const detailsRef = useRef(null);
  const [detailsReady, setDetailsReady] = useState(false);
  const [savingDetails, setSavingDetails] = useState(false);

  const [finalTranscript, setFinalTranscript] = useState(meeting.final_transcript || "");
  const [status, setStatus] = useState(meeting.status);

  // Summary state
  const [summaryFormat, setSummaryFormat] = useState(meeting.summary_format || "bullets");
  const [summary, setSummary] = useState(meeting.summary || "");
  const [summaryEngine, setSummaryEngine] = useState("");
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
  const audioUrlRef = useRef(null);
  const uploadInputRef = useRef(null);

  const revokeAudioUrl = useCallback(() => {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    setAudioUrl(null);
  }, []);

  const loadAudio = useCallback(async (meetingId) => {
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
  }, [revokeAudioUrl]);

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
    async (format = summaryFormat) => {
      setSummarizing(true);
      setSummaryError("");
      try {
        const res = await api.summarize({
          meeting_id: meeting.id,
          output_format: format,
        });
        setSummary(res.summary);
        setSummaryEngine(res.engine);
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
    async (transcriptText) => {
      const text = (transcriptText || "").trim();
      if (!text) return;
      if (autoTranslateRef.current === text) return;
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
      if (persistDetails) {
        await saveDetails({ silent: true });
      }
      if (onMeetingUpdated) onMeetingUpdated();
      await loadAudio(meeting.id);
      if (text.trim()) {
        await Promise.all([
          summarizeFromTranscript(summaryFormat),
          translateToEnglish(text),
        ]);
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
      translateToEnglish,
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
    try {
      const detail = await api.retranscribeMeeting(meeting.id);
      await applyTranscriptResult(detail);
    } catch (err) {
      setAsrError(err.message || "Whisper ASR failed.");
    } finally {
      setAsrBusy(false);
    }
  }, [applyTranscriptResult, meeting.id]);

  const uploadForWhisper = useCallback(
    async (file) => {
      if (!file) return;
      setAsrBusy(true);
      setAsrError("");
      try {
        const detail = await api.uploadMeetingAudio(meeting.id, file, {
          transcribe: true,
        });
        setHasAudio(true);
        await applyTranscriptResult(detail);
      } catch (err) {
        setAsrError(err.message || "Audio upload / Whisper ASR failed.");
      } finally {
        setAsrBusy(false);
        if (uploadInputRef.current) uploadInputRef.current.value = "";
      }
    },
    [applyTranscriptResult, meeting.id]
  );

  const recorder = useRecorder({ onFinalTranscript });

  // Publish save controls so the parent can place the Save button in the topbar.
  useEffect(() => {
    if (!onSaveControls) return;
    onSaveControls({
      save: () => saveDetails(),
      saving: savingDetails,
      disabled: recorder.recording || recorder.status === "finalizing",
    });
  }, [
    onSaveControls,
    saveDetails,
    savingDetails,
    recorder.recording,
    recorder.status,
  ]);

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
    // Load whenever the meeting may already have a saved WAV.
    if (meeting.has_audio || meeting.status === "finalized") {
      loadAudio(meeting.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id]);

  useEffect(() => {
    return () => revokeAudioUrl();
  }, [revokeAudioUrl]);

  // Auto-translate existing finalized transcripts that don't have English yet.
  useEffect(() => {
    const text = (meeting.final_transcript || "").trim();
    if (!text) return;
    if (meeting.translation) return;
    translateToEnglish(text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id, meeting.final_transcript, meeting.translation]);

  // Auto-summarize finalized transcripts that don't have a summary yet.
  useEffect(() => {
    const text = (meeting.final_transcript || "").trim();
    if (!text) return;
    if (meeting.summary) return;
    if (autoSummaryRef.current.startsWith(`${meeting.id}:`)) return;
    summarizeFromTranscript(summaryFormat);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id, meeting.final_transcript, meeting.summary]);

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
        recorder.stop();
      }
      return;
    }
    if (!detailsRef.current?.isComplete()) return;
    try {
      await recorder.start(meeting.id);
      setStatus("recording");
    } catch {
      /* handled inside hook via message */
    }
  }

  const canStart =
    detailsReady &&
    recorder.status !== "finalizing" &&
    recorder.status !== "starting";
  const isRefined = status === "finalized" && finalTranscript;
  const hasTranscript = Boolean(finalTranscript);
  const showLive =
    recorder.recording ||
    recorder.status === "finalizing" ||
    recorder.status === "starting";
  const isStarting = recorder.status === "starting";

  async function runSummarize() {
    if (!hasTranscript) {
      setSummaryError("Finalize a transcript first.");
      return;
    }
    await summarizeFromTranscript(summaryFormat);
  }

  async function copyTranscript() {
    const text = (finalTranscript || "").trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopyState("Copied");
      setTimeout(() => setCopyState(""), 1800);
    } catch {
      setCopyState("Copy failed");
      setTimeout(() => setCopyState(""), 1800);
    }
  }

  function downloadTranscript() {
    const text = (finalTranscript || "").trim();
    if (!text) return;
    const base = (meeting.title || "transcript").trim() || "transcript";
    const filename = `${base.replace(/[^\w\-]+/g, "_").replace(/_+/g, "_").slice(0, 80)}_transcript.txt`;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="content">
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
              {hasTranscript && (
                <div className="transcript-actions">
                  <button
                    type="button"
                    className="btn secondary meeting-action-btn"
                    onClick={copyTranscript}
                    title="Copy transcript text"
                  >
                    {copyState === "Copied" ? "Copied" : "Copy text"}
                  </button>
                  <button
                    type="button"
                    className="btn secondary meeting-action-btn"
                    onClick={downloadTranscript}
                    title="Download transcript as a text file"
                  >
                    Download
                  </button>
                </div>
              )}
              <span className="card-tag">Whisper ASR · {meeting.language}</span>
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
                  : recorder.message || "Listening…"}
              </span>
            ) : asrBusy ? (
              <div className="center-spin">
                <span className="spinner" /> Running Whisper ASR on audio…
              </div>
            ) : (
              <div className="placeholder">
                Press <strong>Start recording</strong> for live Whisper captions,
                or upload a WAV to transcribe with Whisper ASR. Ending a meeting
                runs the full-accuracy Whisper pass automatically.
              </div>
            )}
          </div>
        </div>
      </div>

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
                ? "■ End meeting"
                : "● Start recording"}
          </button>
          <div
            className={`record-duration ${
              recorder.recording || recorder.status === "starting"
                ? "active"
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
                {(recorder.recording || recorder.status === "starting") && (
                  <span className="dot-live" />
                )}
                <span className="record-duration-label">
                  {recorder.status === "finalizing" ? "Duration" : "Time"}
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

        {!recorder.recording && (
          <div
            className="audio-player-wrap"
            title={
              audioUrl
                ? "Meeting recording"
                : "Recording available after you end the meeting"
            }
          >
            {audioLoading ? (
              <span className="card-tag">Loading audio…</span>
            ) : audioUrl ? (
              <audio className="meeting-audio-player" controls src={audioUrl} preload="metadata">
                Your browser does not support audio playback.
              </audio>
            ) : (
              <div className="audio-player-empty">
                <span className="audio-player-label">Audio</span>
                <span className="audio-player-hint">No recording yet</span>
              </div>
            )}
          </div>
        )}

        {!recorder.recording && (
          <>
            <input
              ref={uploadInputRef}
              type="file"
              accept="audio/wav,audio/x-wav,.wav,.pcm,.raw"
              hidden
              onChange={(e) => uploadForWhisper(e.target.files?.[0])}
            />
            <button
              type="button"
              className="btn secondary"
              disabled={asrBusy || recorder.status === "finalizing" || !detailsReady}
              onClick={() => uploadInputRef.current?.click()}
              title="Upload WAV/PCM and run Whisper ASR"
            >
              {asrBusy ? <span className="spinner" /> : "Upload audio"}
            </button>
            {(hasAudio || audioUrl) && (
              <button
                type="button"
                className="btn secondary"
                disabled={asrBusy || recorder.status === "finalizing"}
                onClick={runWhisperAsr}
                title="Run Whisper ASR on the saved recording"
              >
                {asrBusy ? (
                  <span className="spinner" />
                ) : hasTranscript ? (
                  "Re-transcribe"
                ) : (
                  "Transcribe with Whisper"
                )}
              </button>
            )}
          </>
        )}

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

      <div className="cards bottom-cards">
        {/* Summary card (BART) — from finalized transcript */}
        <div className="card">
          <div className="card-head">
            <h3>Summary</h3>
            <span className="card-tag">
              BART · from transcript{summaryEngine ? ` · ${summaryEngine}` : ""}
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
            {summaryError && <div className="error-banner">{summaryError}</div>}
            {summarizing ? (
              <div className="center-spin">
                <span className="spinner" /> Summarizing transcript…
              </div>
            ) : summary ? (
              summary
            ) : (
              <div className="placeholder">
                {!hasTranscript
                  ? "Finalize a transcript first."
                  : "Summary runs automatically after transcription."}
              </div>
            )}
          </div>
        </div>

        {/* Translation card (mBART) — auto English */}
        <div className="card">
          <div className="card-head">
            <h3>English translation</h3>
            <span className="card-tag">
              mBART · {translationLang || "English"}
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
    </div>
  );
}

export { MeetingRoom };
