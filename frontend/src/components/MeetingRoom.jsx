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

  // Translation state — always auto-translates the transcript into English.
  const [translation, setTranslation] = useState(meeting.translation || "");
  const [translationLang, setTranslationLang] = useState(
    meeting.translation_language || "English"
  );
  const [translating, setTranslating] = useState(false);
  const [translateError, setTranslateError] = useState("");
  const autoTranslateRef = useRef("");
  const autoSummaryRef = useRef("");

  const saveDetails = useCallback(async ({ silent = false } = {}) => {
    if (!detailsRef.current) return false;
    setSavingDetails(true);
    try {
      return await detailsRef.current.save({ silent });
    } finally {
      setSavingDetails(false);
    }
  }, []);

  const summarizeFromEnglish = useCallback(
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
        // BART summarizes from the English translation for accuracy.
        await summarizeFromEnglish(summaryFormat);
      } catch (err) {
        autoTranslateRef.current = "";
        setTranslateError(err.message || "English translation failed.");
      } finally {
        setTranslating(false);
      }
    },
    [meeting.id, onMeetingUpdated, summarizeFromEnglish, summaryFormat]
  );

  const onFinalTranscript = useCallback(
    async (data) => {
      const text = data.text || "";
      setFinalTranscript(text);
      setStatus("finalized");
      // Automatically persist meeting details after recording finishes.
      await saveDetails({ silent: true });
      if (onMeetingUpdated) onMeetingUpdated();
      // Automatically translate the finalized transcript into English.
      await translateToEnglish(text);
    },
    [onMeetingUpdated, saveDetails, translateToEnglish]
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
    setSummaryEngine("");
    autoTranslateRef.current = meeting.translation ? meeting.final_transcript || "" : "";
    autoSummaryRef.current = meeting.summary
      ? `${meeting.id}:${meeting.summary_format || "bullets"}`
      : "";
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id]);

  // Auto-translate existing finalized transcripts that don't have English yet.
  useEffect(() => {
    const text = (meeting.final_transcript || "").trim();
    if (!text) return;
    if (meeting.translation) return;
    translateToEnglish(text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id, meeting.final_transcript, meeting.translation]);

  // If English exists but summary does not, auto-summarize from the translation.
  useEffect(() => {
    if (!meeting.translation) return;
    if (meeting.summary) return;
    if (autoSummaryRef.current.startsWith(`${meeting.id}:`)) return;
    summarizeFromEnglish(summaryFormat);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meeting.id, meeting.translation, meeting.summary]);

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
  const hasEnglish = Boolean((translation || "").trim());

  async function runSummarize() {
    if (!hasEnglish) {
      setSummaryError("Wait for the English translation to finish first.");
      return;
    }
    await summarizeFromEnglish(summaryFormat);
  }

  const isRefined = status === "finalized" && finalTranscript;
  const hasTranscript = Boolean(finalTranscript);
  const showLive = recorder.recording || recorder.status === "finalizing";

  return (
    <div className="content">
      {!recorder.transcriptionAvailable && recorder.message && (
        <div className="banner-warn">{recorder.message}</div>
      )}
      {recorder.transcriptionAvailable && recorder.message && recorder.recording && (
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
            <span
              className="mic-meter"
              title={`Mic level ${(recorder.micLevel * 100).toFixed(1)}%`}
            >
              <span
                className="mic-meter-fill"
                style={{
                  width: `${Math.min(100, Math.max(4, recorder.micLevel * 400))}%`,
                }}
              />
            </span>
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
        {/* Summary card (BART) — from English translation */}
        <div className="card">
          <div className="card-head">
            <h3>Summary</h3>
            <span className="card-tag">
              BART · from English{summaryEngine ? ` · ${summaryEngine}` : ""}
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
              disabled={!hasEnglish || summarizing || translating}
            >
              {summarizing ? <span className="spinner" /> : "Summarize"}
            </button>
          </div>
          <div className="card-body">
            {summaryError && <div className="error-banner">{summaryError}</div>}
            {summarizing ? (
              <div className="center-spin">
                <span className="spinner" /> Summarizing English translation…
              </div>
            ) : summary ? (
              summary
            ) : (
              <div className="placeholder">
                {!hasTranscript
                  ? "Finalize a transcript first."
                  : translating || !hasEnglish
                    ? "Waiting for English translation, then summary runs automatically."
                    : "Choose a format and click Summarize."}
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
