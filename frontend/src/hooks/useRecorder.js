import { useCallback, useEffect, useRef, useState } from "react";
import { api, ensureFreshAccessToken, getToken } from "../api/client";

// How long to wait for WS final_transcript after sending {type:stop}
// before falling back to REST POST /meetings/{id}/stop.
const FINALIZE_WATCHDOG_MS = 90_000;
// Poll interval when REST reports finalize already in progress.
const FINALIZE_POLL_MS = 2_500;
const FINALIZE_POLL_MAX_MS = 45 * 60_000;

// Manages microphone capture (AudioWorklet -> 16 kHz PCM) and streams the audio
// to the backend over a WebSocket, exposing live caption + finalization state.
export function useRecorder({ onFinalTranscript } = {}) {
  const [recording, setRecording] = useState(false);
  const [paused, setPaused] = useState(false);
  const [status, setStatus] = useState("idle"); // idle|starting|recording|paused|finalizing|error
  const [liveText, setLiveText] = useState("");
  const [message, setMessage] = useState("");
  const [transcriptionAvailable, setTranscriptionAvailable] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [connectionState, setConnectionState] = useState("disconnected");

  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const workletNodeRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null);
  const liveSegmentsRef = useRef({});
  const timerRef = useRef(null);
  const startedAtRef = useRef(null);
  const pausedTotalMsRef = useRef(0);
  const pauseStartedAtRef = useRef(null);
  const pausedRef = useRef(false);
  const meetingIdRef = useRef(null);
  const reconnectRef = useRef({ attempts: 0, timer: null, active: false });
  const stoppingRef = useRef(false);
  const stopSentOverWsRef = useRef(false);
  const finalizeDoneRef = useRef(false);
  const finalizeWatchdogRef = useRef(null);
  const finalizeInFlightRef = useRef(false);
  const tokenRefreshRef = useRef(null);
  const authRetryRef = useRef(0);
  const keepaliveRef = useRef(null);
  const audioWatchRef = useRef(null);
  const onFinalRef = useRef(onFinalTranscript);
  const pendingPcmRef = useRef([]);
  const bytesSentRef = useRef(0);

  const computeElapsedSec = useCallback(() => {
    const started = startedAtRef.current;
    if (!started) return 0;
    let pausedMs = pausedTotalMsRef.current;
    if (pausedRef.current && pauseStartedAtRef.current) {
      pausedMs += Date.now() - pauseStartedAtRef.current;
    }
    return Math.max(0, Math.floor((Date.now() - started - pausedMs) / 1000));
  }, []);

  useEffect(() => {
    onFinalRef.current = onFinalTranscript;
  }, [onFinalTranscript]);

  const composeLive = useCallback(() => {
    const keys = Object.keys(liveSegmentsRef.current)
      .map(Number)
      .sort((a, b) => a - b);
    return keys.map((k) => liveSegmentsRef.current[k]).join(" ");
  }, []);

  const cleanupAudio = useCallback(() => {
    try {
      if (workletNodeRef.current) {
        workletNodeRef.current.port.onmessage = null;
        workletNodeRef.current.disconnect();
      }
      if (sourceRef.current) sourceRef.current.disconnect();
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    } catch {
      /* ignore */
    }
    workletNodeRef.current = null;
    sourceRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
  }, []);

  /** Flush remaining PCM from the worklet, then tear down capture. */
  const flushAndCleanupAudio = useCallback(() => {
    return new Promise((resolve) => {
      const node = workletNodeRef.current;
      if (!node) {
        cleanupAudio();
        resolve();
        return;
      }
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        cleanupAudio();
        resolve();
      };
      const prevHandler = node.port.onmessage;
      const timeout = setTimeout(finish, 300);
      node.port.onmessage = (event) => {
        if (event.data && event.data.type === "flushed") {
          clearTimeout(timeout);
          finish();
          return;
        }
        // Keep forwarding PCM while flush completes.
        if (prevHandler) prevHandler(event);
      };
      try {
        node.port.postMessage({ type: "flush" });
      } catch {
        clearTimeout(timeout);
        finish();
      }
    });
  }, [cleanupAudio]);

  const clearFinalizeWatchdog = useCallback(() => {
    if (finalizeWatchdogRef.current) {
      clearTimeout(finalizeWatchdogRef.current);
      finalizeWatchdogRef.current = null;
    }
  }, []);

  const clearTokenRefreshLoop = useCallback(() => {
    if (tokenRefreshRef.current) {
      clearInterval(tokenRefreshRef.current);
      tokenRefreshRef.current = null;
    }
  }, []);

  const markFinalized = useCallback(
    (payload) => {
      if (finalizeDoneRef.current) return;
      finalizeDoneRef.current = true;
      clearFinalizeWatchdog();
      clearTokenRefreshLoop();
      stoppingRef.current = true;
      reconnectRef.current.active = false;
      if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
      liveSegmentsRef.current = {};
      setLiveText("");
      setRecording(false);
      setStatus("idle");
      if (onFinalRef.current) {
        onFinalRef.current(
          payload?.type
            ? payload
            : {
                type: "final_transcript",
                text: payload?.text || "",
                segments: payload?.segments || [],
              }
        );
      }
    },
    [clearFinalizeWatchdog, clearTokenRefreshLoop]
  );

  /** REST finalize (+ poll when Whisper is already running). */
  const finalizeViaRest = useCallback(
    async (meetingId, { reason = "fallback" } = {}) => {
      if (!meetingId || finalizeDoneRef.current || finalizeInFlightRef.current) {
        return false;
      }
      finalizeInFlightRef.current = true;
      clearFinalizeWatchdog();
      setStatus("finalizing");
      setMessage(
        reason === "watchdog"
          ? "Socket stalled — finishing transcription via server…"
          : "Finishing transcription…"
      );
      try {
        await ensureFreshAccessToken({ minValiditySeconds: 60 });
        let result = await api.stopMeetingRecording(meetingId);
        const started = Date.now();
        while (
          result?.in_progress &&
          !finalizeDoneRef.current &&
          Date.now() - started < FINALIZE_POLL_MAX_MS
        ) {
          setMessage("Transcription in progress…");
          await new Promise((r) => setTimeout(r, FINALIZE_POLL_MS));
          const detail = await api.getMeeting(meetingId);
          if (detail?.status === "finalized") {
            markFinalized({
              text: detail.final_transcript || "",
              segments: detail.segments || [],
            });
            return true;
          }
          if (detail?.status === "failed") {
            setStatus("error");
            setMessage("Transcription failed. You can retry from History.");
            return false;
          }
          // Nudge again in case the first stop only observed in_progress.
          try {
            result = await api.stopMeetingRecording(meetingId);
          } catch {
            /* keep polling */
          }
        }
        if (finalizeDoneRef.current) return true;
        if (result?.ok) {
          markFinalized({
            text: result.text || result.meeting?.final_transcript || "",
            segments: result.segments || result.meeting?.segments || [],
          });
          return true;
        }
        setStatus("error");
        setMessage(result?.message || "Could not finalize recording.");
        return false;
      } catch (err) {
        if (!finalizeDoneRef.current) {
          setStatus("error");
          setMessage(err.message || "Could not finalize recording.");
        }
        return false;
      } finally {
        finalizeInFlightRef.current = false;
      }
    },
    [clearFinalizeWatchdog, markFinalized]
  );

  const armFinalizeWatchdog = useCallback(
    (meetingId) => {
      clearFinalizeWatchdog();
      finalizeWatchdogRef.current = setTimeout(() => {
        if (!finalizeDoneRef.current && meetingId) {
          finalizeViaRest(meetingId, { reason: "watchdog" });
        }
      }, FINALIZE_WATCHDOG_MS);
    },
    [clearFinalizeWatchdog, finalizeViaRest]
  );

  const startTokenRefreshLoop = useCallback(() => {
    clearTokenRefreshLoop();
    // Proactively refresh well before the ~30m access JWT expires so reconnects
    // and REST /stop always have a usable Bearer token. The live WS itself only
    // validates JWT at handshake — we refresh storage, not the open socket.
    tokenRefreshRef.current = setInterval(() => {
      if (!reconnectRef.current.active && !stoppingRef.current) return;
      ensureFreshAccessToken({ minValiditySeconds: 180 }).catch(() => {});
    }, 60_000);
  }, [clearTokenRefreshLoop]);

  const buildWsUrl = useCallback((meetingId, token) => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws/transcribe?token=${encodeURIComponent(
      token || ""
    )}&meeting_id=${encodeURIComponent(meetingId)}`;
  }, []);

  const openSocket = useCallback(
    async (meetingId) => {
      // Close a prior socket without triggering reconnect storms.
      const prev = wsRef.current;
      if (prev) {
        try {
          prev.onclose = null;
          prev.onerror = null;
          prev.onmessage = null;
          prev.close();
        } catch {
          /* ignore */
        }
      }

      // Fresh JWT before every connect/reconnect (long meetings > access TTL).
      const token = await ensureFreshAccessToken({ minValiditySeconds: 120 });
      if (!token) {
        setStatus("error");
        setMessage("Session expired — sign in again to continue recording.");
        reconnectRef.current.active = false;
        return;
      }

      const ws = new WebSocket(buildWsUrl(meetingId, token));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      setConnectionState("connecting");

      ws.onopen = () => {
        setConnectionState("connected");
        reconnectRef.current.attempts = 0;
        authRetryRef.current = 0;
        ws.send(JSON.stringify({ type: "start" }));
        // Flush any PCM buffered while the socket was down.
        const pending = pendingPcmRef.current;
        while (pending.length && ws.readyState === WebSocket.OPEN) {
          const chunk = pending.shift();
          ws.send(chunk);
          bytesSentRef.current += chunk.byteLength || chunk.length || 0;
        }
        // Client keepalive — keeps proxies from idling out 8h+ board meetings.
        if (keepaliveRef.current) clearInterval(keepaliveRef.current);
        keepaliveRef.current = setInterval(() => {
          const sock = wsRef.current;
          if (sock && sock.readyState === WebSocket.OPEN) {
            sock.send(JSON.stringify({ type: "ping", ts: Date.now() }));
          }
        }, 20000);
      };

      ws.onmessage = (evt) => {
        let data;
        try {
          data = JSON.parse(evt.data);
        } catch {
          return;
        }
        if (data.type === "ping") {
          // Answer server keepalive so the socket stays warm.
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "pong", ts: Date.now() }));
          }
          return;
        }
        if (data.type === "status") {
          setTranscriptionAvailable(data.transcription_available);
          // Keep the short startup copy until the first caption arrives.
          // Always surface explicit reconnect/resume status from the server.
          if (data.resumed && data.message) {
            setMessage(data.message);
          } else {
            setMessage((prev) =>
              prev === "Starting meeting and microphone…" ||
              prev === "Connecting live transcription…" ||
              prev === "Listening…"
                ? prev
                : data.message || prev
            );
          }
          // Resume captions restored from Redis after a reconnect.
          if (data.live_caption) {
            setLiveText((prev) => {
              const current = (prev || "").trim();
              const incoming = String(data.live_caption || "").trim();
              if (!incoming) return prev;
              if (!current) return incoming;
              // Prefer the longer buffer so a short client buffer cannot wipe
              // Redis-restored captions (or vice versa).
              return incoming.length >= current.length ? incoming : prev;
            });
          }
        } else if (data.type === "info" || data.type === "warning") {
          setMessage(data.message || "");
        } else if (data.type === "live_caption") {
          // Cumulative caption — never allow a shorter update to erase older words
          // (protects against aggressive overlap dedupe or WS reconnect resets).
          const incoming = (data.text || "").trim();
          if (!incoming) return;
          setLiveText((prev) => {
            const current = (prev || "").trim();
            if (!current) return incoming;
            if (incoming === current) return prev;
            if (incoming.startsWith(current)) return incoming;
            if (current.startsWith(incoming)) return prev;
            // Reconnect / reset case: server restarted from a short window.
            if (incoming.length < current.length * 0.8) {
              const curWords = current.toLowerCase().split(/\s+/);
              const inWords = incoming.toLowerCase().split(/\s+/);
              let overlap = 0;
              const max = Math.min(curWords.length, inWords.length, 12);
              for (let n = max; n >= 2; n -= 1) {
                if (curWords.slice(-n).join(" ") === inWords.slice(0, n).join(" ")) {
                  overlap = n;
                  break;
                }
              }
              const addition = incoming.split(/\s+/).slice(overlap).join(" ").trim();
              return addition ? `${current} ${addition}`.trim() : current;
            }
            return incoming.length >= current.length ? incoming : prev;
          });
          setMessage("");
          setStatus((s) => (s === "starting" ? "recording" : s));
        } else if (data.type === "live_segment") {
          // Legacy per-chunk segments (kept for compatibility / persistence).
          liveSegmentsRef.current[data.seq] = data.text;
          // Only fall back to composed chunks if we have not received live_caption yet.
          setLiveText((prev) => prev || composeLive());
        } else if (data.type === "finalizing") {
          // Server started finalize — never reconnect into a wiped session.
          stoppingRef.current = true;
          reconnectRef.current.active = false;
          if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
          setRecording(false);
          setStatus("finalizing");
          // Keep the watchdog armed until final_transcript arrives.
          if (!finalizeWatchdogRef.current && meetingIdRef.current) {
            armFinalizeWatchdog(meetingIdRef.current);
          }
        } else if (data.type === "final_transcript") {
          markFinalized(data);
        } else if (data.type === "error") {
          // Finished / fatal errors should not keep reconnecting.
          const msg = data.message || "";
          if (
            /already (finalized|processing|failed)/i.test(msg) ||
            /already being recorded/i.test(msg)
          ) {
            stoppingRef.current = true;
            reconnectRef.current.active = false;
            if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
            setRecording(false);
            if (/already finalized/i.test(msg) && meetingIdRef.current) {
              finalizeViaRest(meetingIdRef.current, { reason: "already-finalized" });
              return;
            }
            if (/already processing/i.test(msg) && meetingIdRef.current) {
              finalizeViaRest(meetingIdRef.current, { reason: "in-progress" });
              return;
            }
          }
          setStatus("error");
          setMessage(msg || "Transcription error.");
        }
      };

      ws.onclose = (evt) => {
        // Ignore close events from superseded sockets.
        if (wsRef.current && wsRef.current !== ws) return;
        setConnectionState("disconnected");
        if (keepaliveRef.current) {
          clearInterval(keepaliveRef.current);
          keepaliveRef.current = null;
        }

        const code = evt?.code || 0;

        // After stop: socket drop / timeout → REST finalize (prevents stuck Finalizing…).
        if (
          stoppingRef.current &&
          !finalizeDoneRef.current &&
          (stopSentOverWsRef.current || code === 1000 || code === 1001 || code === 1006)
        ) {
          finalizeViaRest(meetingId, { reason: "ws-closed-after-stop" });
          return;
        }

        // Auth failure: refresh once, then reconnect with a new token.
        if (code === 4401 && reconnectRef.current.active && !stoppingRef.current) {
          if (authRetryRef.current < 2) {
            authRetryRef.current += 1;
            setMessage("Session renewing — reconnecting live captions…");
            reconnectRef.current.timer = setTimeout(() => {
              openSocket(meetingId).catch(() => {});
            }, 400);
            return;
          }
          setStatus("error");
          setMessage("Session expired — sign in again to continue recording.");
          reconnectRef.current.active = false;
          return;
        }

        // Another tab holds the live lock — do not reconnect-loop.
        if (code === 4409) {
          reconnectRef.current.active = false;
          setStatus("error");
          setMessage(
            "This meeting is already being recorded in another tab or window."
          );
          return;
        }

        // Reconnect only if we are actively recording (unexpected drop).
        // Allow many retries — board meetings can run 8h+ and proxies flap.
        // Server keeps Redis PCM on drop (does not finalize) so captions resume.
        if (reconnectRef.current.active && !stoppingRef.current) {
          const attempts = ++reconnectRef.current.attempts;
          if (attempts <= 60) {
            const delay = Math.min(1000 * 2 ** Math.min(attempts - 1, 4), 15000);
            setMessage(`Connection lost — reconnecting (attempt ${attempts})…`);
            reconnectRef.current.timer = setTimeout(
              () => openSocket(meetingId).catch(() => {}),
              delay
            );
          } else {
            setStatus("error");
            setMessage("Connection lost. Could not reconnect.");
          }
        }
      };

      ws.onerror = () => {
        // onclose handles reconnection logic.
      };
    },
    [armFinalizeWatchdog, buildWsUrl, composeLive, finalizeViaRest, markFinalized]
  );

  const start = useCallback(
    async (meetingId) => {
      setLiveText("");
      liveSegmentsRef.current = {};
      meetingIdRef.current = meetingId;
      stoppingRef.current = false;
      stopSentOverWsRef.current = false;
      finalizeDoneRef.current = false;
      finalizeInFlightRef.current = false;
      authRetryRef.current = 0;
      clearFinalizeWatchdog();
      reconnectRef.current = { attempts: 0, timer: null, active: true };
      // Keep access JWT fresh for the whole live session (reconnects + REST stop).
      startTokenRefreshLoop();
      // Immediate UI feedback — do not wait for mic/worklet/WS to finish.
      setStatus("starting");
      setMessage("Starting meeting and microphone…");
      setRecording(true);
      setPaused(false);
      pausedRef.current = false;
      pausedTotalMsRef.current = 0;
      pauseStartedAtRef.current = null;
      startedAtRef.current = Date.now();
      setElapsed(0);
      pendingPcmRef.current = [];
      bytesSentRef.current = 0;
      if (timerRef.current) clearInterval(timerRef.current);
      // Wall-clock based so long board meetings stay accurate if the tab throttles.
      // Pause time is excluded from the displayed duration.
      timerRef.current = setInterval(() => {
        setElapsed(computeElapsedSec());
      }, 250);

      // Open the WebSocket in parallel with mic setup (biggest perceived win).
      // Access JWT is refreshed before connect and kept fresh during long meetings.
      openSocket(meetingId).catch(() => {});

      let stream;
      try {
        // echoCancellation OFF: wiring the worklet into the audio graph with AEC
        // on can make Chromium mute the mic after a few seconds ("transcription
        // stopped"). We also avoid connecting to audioCtx.destination below.
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            // Browser AGC raises the noise floor so quiet tails look like
            // speech and Whisper loops the last phrase after you stop talking.
            autoGainControl: false,
          },
        });
      } catch (err) {
        setRecording(false);
        setStatus("error");
        setMessage("Microphone access denied or unavailable.");
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        reconnectRef.current.active = false;
        if (wsRef.current) wsRef.current.close();
        throw err;
      }
      streamRef.current = stream;
      const track = stream.getAudioTracks()[0];
      if (track) {
        track.onended = () => {
          if (stoppingRef.current) return;
          setStatus("error");
          setMessage("Microphone stopped unexpectedly. Click Start recording to try again.");
          setRecording(false);
          reconnectRef.current.active = false;
          if (wsRef.current) wsRef.current.close();
        };
        track.onmute = () => {
          if (!stoppingRef.current) {
            setMessage("Microphone muted — check browser / system mic settings.");
          }
        };
        track.onunmute = () => {
          if (!stoppingRef.current) setMessage("Listening…");
        };
      }
      setMessage("Connecting live transcription…");

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 48000 });
      audioCtxRef.current = audioCtx;
      if (audioCtx.state === "suspended") {
        try {
          await audioCtx.resume();
        } catch {
          /* ignore */
        }
      }
      const resumeCtx = () => {
        const ctx = audioCtxRef.current;
        if (ctx && ctx.state === "suspended" && !stoppingRef.current) {
          ctx.resume().catch(() => {});
        }
      };
      audioCtx.addEventListener("statechange", resumeCtx);
      try {
        // Prefer a preloaded module; fall back to loading now.
        if (!window.__smWorkletReady) {
          await audioCtx.audioWorklet.addModule("/pcm-worklet.js");
          window.__smWorkletReady = true;
        } else {
          try {
            await audioCtx.audioWorklet.addModule("/pcm-worklet.js");
          } catch {
            /* already registered in this context */
          }
        }
      } catch (err) {
        setRecording(false);
        setStatus("error");
        setMessage("Failed to initialize audio processor.");
        cleanupAudio();
        reconnectRef.current.active = false;
        if (wsRef.current) wsRef.current.close();
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        throw err;
      }

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const node = new AudioWorkletNode(audioCtx, "pcm-worklet");
      workletNodeRef.current = node;
      // Buffer PCM while the socket is connecting / reconnecting (~30s).
      pendingPcmRef.current = [];
      node.port.onmessage = (event) => {
        if (event.data && event.data.type === "flushed") return;
        // Drop mic audio while paused — do not buffer silence into ASR.
        if (pausedRef.current || stoppingRef.current) return;
        const chunk = event.data;
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          const pending = pendingPcmRef.current;
          while (pending.length) {
            const p = pending.shift();
            ws.send(p);
            bytesSentRef.current += p.byteLength || p.length || 0;
          }
          ws.send(chunk);
          bytesSentRef.current += chunk.byteLength || chunk.length || 0;
        } else if (pendingPcmRef.current.length < 120) {
          // ~30s at 0.25s/chunk — covers reconnect backoff without dropping audio.
          pendingPcmRef.current.push(chunk);
        }
      };
      source.connect(node);
      // Keep the worklet running WITHOUT connecting to speakers.
      // Connecting a mic graph to destination + echoCancellation was muting
      // capture after a few seconds in Chromium (captions looked "stopped").
      const silent = audioCtx.createGain();
      silent.gain.value = 0;
      const sink = audioCtx.createMediaStreamDestination();
      node.connect(silent);
      silent.connect(sink);

      setStatus("recording");
      setMessage("Listening…");
      // Aggressively keep AudioContext alive (browsers suspend quiet contexts).
      if (audioWatchRef.current) clearInterval(audioWatchRef.current);
      audioWatchRef.current = setInterval(() => {
        resumeCtx();
        const t = streamRef.current?.getAudioTracks?.()[0];
        if (t && t.readyState === "ended" && !stoppingRef.current) {
          setStatus("error");
          setMessage("Microphone stopped unexpectedly. Click Start recording to try again.");
          setRecording(false);
          reconnectRef.current.active = false;
          if (wsRef.current) wsRef.current.close();
        }
      }, 1000);
    },
    [
      cleanupAudio,
      clearFinalizeWatchdog,
      computeElapsedSec,
      openSocket,
      startTokenRefreshLoop,
    ]
  );

  const pause = useCallback(() => {
    if (!recording || pausedRef.current || stoppingRef.current) return;
    if (status !== "recording" && status !== "starting") return;
    pausedRef.current = true;
    pauseStartedAtRef.current = Date.now();
    setPaused(true);
    setStatus("paused");
    setMessage("Paused — press Play to continue live transcription.");
    // Mute the hardware track so the OS/browser shows pause clearly.
    try {
      streamRef.current?.getAudioTracks?.().forEach((t) => {
        t.enabled = false;
      });
    } catch {
      /* ignore */
    }
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "pause", ts: Date.now() }));
      } catch {
        /* ignore */
      }
    }
  }, [recording, status]);

  const resume = useCallback(() => {
    if (!recording || !pausedRef.current || stoppingRef.current) return;
    if (pauseStartedAtRef.current) {
      pausedTotalMsRef.current += Date.now() - pauseStartedAtRef.current;
      pauseStartedAtRef.current = null;
    }
    pausedRef.current = false;
    setPaused(false);
    setStatus("recording");
    setMessage("Listening…");
    try {
      streamRef.current?.getAudioTracks?.().forEach((t) => {
        t.enabled = true;
      });
    } catch {
      /* ignore */
    }
    const ctx = audioCtxRef.current;
    if (ctx && ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "resume", ts: Date.now() }));
      } catch {
        /* ignore */
      }
    }
    setElapsed(computeElapsedSec());
  }, [computeElapsedSec, recording]);

  const stop = useCallback(async () => {
    stoppingRef.current = true;
    reconnectRef.current.active = false;
    if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
    // Stop proactive JWT refresh; finalize uses ensureFreshAccessToken on demand.
    clearTokenRefreshLoop();
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    // Freeze the final elapsed time for the UI while Whisper finalizes.
    if (pausedRef.current && pauseStartedAtRef.current) {
      pausedTotalMsRef.current += Date.now() - pauseStartedAtRef.current;
      pauseStartedAtRef.current = null;
    }
    pausedRef.current = false;
    setPaused(false);
    setElapsed(computeElapsedSec());
    if (keepaliveRef.current) {
      clearInterval(keepaliveRef.current);
      keepaliveRef.current = null;
    }
    if (audioWatchRef.current) {
      clearInterval(audioWatchRef.current);
      audioWatchRef.current = null;
    }
    setRecording(false);
    setStatus("finalizing");
    setMessage("Finishing transcription…");
    // Flush trailing PCM (~up to 0.25s) before tearing down the mic graph.
    await flushAndCleanupAudio();
    const meetingId = meetingIdRef.current;
    const ws = wsRef.current;
    stopSentOverWsRef.current = false;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        // Primary stop path: server finalizes from live PCM and replies
        // with {type:"final_transcript"} (or {type:"finalizing"} first).
        ws.send(JSON.stringify({ type: "stop" }));
        stopSentOverWsRef.current = true;
        // If the socket drops, times out, or never delivers final_transcript,
        // armFinalizeWatchdog / onclose → REST POST /meetings/{id}/stop.
        if (meetingId) armFinalizeWatchdog(meetingId);
      } catch {
        stopSentOverWsRef.current = false;
      }
    }
    // Immediate REST fallback when WS is already down or stop send failed.
    if (!stopSentOverWsRef.current && meetingId) {
      await finalizeViaRest(meetingId, { reason: "ws-unavailable" });
    }
  }, [
    armFinalizeWatchdog,
    clearTokenRefreshLoop,
    computeElapsedSec,
    finalizeViaRest,
    flushAndCleanupAudio,
  ]);

  useEffect(() => {
    // Best-effort finalize if the tab closes mid-recording (WS stop may be lost).
    // Proactive refresh during live keeps the access JWT usable for keepalive.
    const onUnload = () => {
      const meetingId = meetingIdRef.current;
      if (!meetingId) return;
      // Skip if we already finished; still fire if actively recording OR stopping
      // (stop may have been in-flight when the tab closed).
      if (!reconnectRef.current.active && !stoppingRef.current) return;
      if (finalizeDoneRef.current) return;
      try {
        const token = getToken();
        const url = `/api/meetings/${meetingId}/stop`;
        fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: "",
          keepalive: true,
        }).catch(() => {});
      } catch {
        /* ignore */
      }
    };
    window.addEventListener("pagehide", onUnload);
    return () => {
      window.removeEventListener("pagehide", onUnload);
      stoppingRef.current = true;
      reconnectRef.current.active = false;
      if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
      if (timerRef.current) clearInterval(timerRef.current);
      if (keepaliveRef.current) clearInterval(keepaliveRef.current);
      if (audioWatchRef.current) clearInterval(audioWatchRef.current);
      if (tokenRefreshRef.current) {
        clearInterval(tokenRefreshRef.current);
        tokenRefreshRef.current = null;
      }
      if (finalizeWatchdogRef.current) {
        clearTimeout(finalizeWatchdogRef.current);
        finalizeWatchdogRef.current = null;
      }
      cleanupAudio();
      if (wsRef.current) wsRef.current.close();
    };
  }, [cleanupAudio]);

  return {
    recording,
    paused,
    status,
    liveText,
    message,
    transcriptionAvailable,
    elapsed,
    connectionState,
    start,
    stop,
    pause,
    resume,
  };
}
