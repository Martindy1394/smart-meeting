import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { useRecorder } from "../hooks/useRecorder.js";
import MeetingDetails from "./MeetingDetails.jsx";

function fmtTime(sec) {
  const m = Math.floor(sec / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(sec % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
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

  // Translation state
  const [languages, setLanguages] = useState([]);
  const [targetLang, setTargetLang] = useState("es");
  const [translation, setTranslation] = useState(meeting.translation || "");
  const [translationLang, setTranslationLang] = useState(meeting.translation_language || "");
  const [translating, setTranslating] = useState(false);
  const [translateError, setTranslateError] = useState("");

  const saveDetails = useCallback(async ({ silent = false } = {}) => {
    if (!detailsRef.current) return false;
    setSavingDetails(true);
    try {
      return await detailsRef.current.save({ silent });
    } finally {
      setSavingDetails(false);
    }
  }, []);

  const onFinalTranscript = useCallback(
    async (data) => {
      setFinalTranscript(data.text || "");
      setStatus("finalized");
      // Automatically persist meeting details after recording finishes.
      await saveDetails({ silent: true });
      if (onMeetingUpdated) onMeetingUpdated();
    },
    [onMeetingUpdated, saveDetails]
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
    setTranslationLang(meeting.translation_language || "");
    setSummaryError("");
    setTranslateError("");
    setSummaryEngine("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id]);

  useEffect(() => {
    api
      .languages()
      .then(setLanguages)
      .catch(() => setLanguages([]));
  }, []);

  async function toggleRecord() {
    if (recorder.recording) {
      recorder.stop();
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

  const canStart = detailsReady && recorder.status !== "finalizing";

  async function runSummarize() {
    setSummarizing(true);
    setSummaryError("");
    try {
      const res = await api.summarize({
        meeting_id: meeting.id,
        output_format: summaryFormat,
      });
      setSummary(res.summary);
      setSummaryEngine(res.engine);
      if (onMeetingUpdated) onMeetingUpdated();
    } catch (err) {
      setSummaryError(err.message || "Summarization failed.");
    } finally {
      setSummarizing(false);
    }
  }

  async function runTranslate() {
    setTranslating(true);
    setTranslateError("");
    try {
      const res = await api.translate({
        meeting_id: meeting.id,
        target_language: targetLang,
      });
      setTranslation(res.translation);
      setTranslationLang(res.language_name);
      if (onMeetingUpdated) onMeetingUpdated();
    } catch (err) {
      setTranslateError(err.message || "Translation failed.");
    } finally {
      setTranslating(false);
    }
  }

  const isRefined = status === "finalized" && finalTranscript;
  const hasTranscript = Boolean(finalTranscript);
  const showLive = recorder.recording || recorder.status === "finalizing";

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
            <span className="card-tag">Whisper · {meeting.language}</span>
          </div>
          <div className="card-body">
            {showLive && recorder.liveText ? (
              <span className="transcript-live">{recorder.liveText}</span>
            ) : hasTranscript ? (
              finalTranscript
            ) : showLive ? (
              <span className="transcript-live">Listening…</span>
            ) : (
              <div className="placeholder">
                <div style={{ fontSize: 32 }}>🎙️</div>
                Press <strong>Start recording</strong> to begin live
                transcription. The final transcript is refined to full accuracy
                when you stop.
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="recorder-bar">
        <button
          className={`btn record ${recorder.recording ? "active" : ""}`}
          onClick={toggleRecord}
          disabled={recorder.recording ? recorder.status === "finalizing" : !canStart}
          title={
            !detailsReady && !recorder.recording
              ? "Fill in all meeting details first"
              : undefined
          }
        >
          {recorder.recording ? "■ Stop recording" : "● Start recording"}
        </button>
        {!detailsReady && !recorder.recording && (
          <span className="card-tag">
            Fill in title, venue, date &amp; time, and attendees first
          </span>
        )}
        {recorder.recording && (
          <>
            <span className="dot-live" />
            <span className="timer">{fmtTime(recorder.elapsed)}</span>
          </>
        )}
        {recorder.status === "finalizing" && (
          <span className="center-spin" style={{ padding: 0 }}>
            <span className="spinner" /> Finalizing full-accuracy transcript…
          </span>
        )}
        {recorder.connectionState === "connecting" && (
          <span className="card-tag">connecting…</span>
        )}
      </div>

      <div className="cards bottom-cards">
        {/* Summary card (BART) */}
        <div className="card">
          <div className="card-head">
            <h3>Summary</h3>
            <span className="card-tag">
              BART{summaryEngine ? ` · ${summaryEngine}` : ""}
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
            {summary ? (
              summary
            ) : (
              <div className="placeholder">
                {hasTranscript
                  ? "Choose a format and click Summarize."
                  : "Finalize a transcript first, then summarize."}
              </div>
            )}
          </div>
        </div>

        {/* Translation card (mBART) */}
        <div className="card">
          <div className="card-head">
            <h3>Translation</h3>
            <span className="card-tag">
              mBART{translationLang ? ` · ${translationLang}` : ""}
            </span>
          </div>
          <div className="card-head" style={{ borderTop: "none", paddingTop: 0 }}>
            <div className="controls-row">
              <select
                className="inline"
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
              >
                {languages.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.name}
                  </option>
                ))}
              </select>
              <button
                className="btn"
                onClick={runTranslate}
                disabled={!hasTranscript || translating}
              >
                {translating ? <span className="spinner" /> : "Translate"}
              </button>
            </div>
          </div>
          <div className="card-body">
            {translateError && <div className="error-banner">{translateError}</div>}
            {translation ? (
              translation
            ) : (
              <div className="placeholder">
                {hasTranscript
                  ? "Pick a language and click Translate."
                  : "Finalize a transcript first, then translate."}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export { MeetingRoom };
