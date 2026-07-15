import { useCallback, useEffect, useRef, useState } from "react";
import { getToken } from "../api/client";

// Manages microphone capture (AudioWorklet -> 16 kHz PCM) and streams the audio
// to the backend over a WebSocket, exposing live caption + finalization state.
export function useRecorder({ onFinalTranscript } = {}) {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("idle"); // idle|starting|recording|finalizing|error
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
  const meetingIdRef = useRef(null);
  const reconnectRef = useRef({ attempts: 0, timer: null, active: false });
  const stoppingRef = useRef(false);
  const keepaliveRef = useRef(null);
  const audioWatchRef = useRef(null);
  const onFinalRef = useRef(onFinalTranscript);
  const pendingPcmRef = useRef([]);
  const bytesSentRef = useRef(0);

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

  const buildWsUrl = useCallback((meetingId) => {
    const token = getToken();
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws/transcribe?token=${encodeURIComponent(
      token
    )}&meeting_id=${encodeURIComponent(meetingId)}`;
  }, []);

  const openSocket = useCallback(
    (meetingId) => {
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

      const ws = new WebSocket(buildWsUrl(meetingId));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      setConnectionState("connecting");

      ws.onopen = () => {
        setConnectionState("connected");
        reconnectRef.current.attempts = 0;
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
          setMessage((prev) =>
            prev === "Starting meeting and microphone…" ||
            prev === "Connecting live transcription…" ||
            prev === "Listening…"
              ? prev
              : data.message || prev
          );
          // Resume captions restored from Redis after a reconnect.
          if (data.live_caption) {
            setLiveText((prev) => prev || data.live_caption);
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
        } else if (data.type === "final_transcript") {
          // Two-pass: clear live fragments, surface the finalized transcript.
          stoppingRef.current = true;
          reconnectRef.current.active = false;
          if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
          liveSegmentsRef.current = {};
          setLiveText("");
          setRecording(false);
          setStatus("idle");
          if (onFinalRef.current) onFinalRef.current(data);
        } else if (data.type === "error") {
          // Finished / fatal errors should not keep reconnecting.
          if (/already (finalized|processing|failed)/i.test(data.message || "")) {
            stoppingRef.current = true;
            reconnectRef.current.active = false;
            if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
            setRecording(false);
          }
          setStatus("error");
          setMessage(data.message || "Transcription error.");
        }
      };

      ws.onclose = () => {
        // Ignore close events from superseded sockets.
        if (wsRef.current && wsRef.current !== ws) return;
        setConnectionState("disconnected");
        if (keepaliveRef.current) {
          clearInterval(keepaliveRef.current);
          keepaliveRef.current = null;
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
              () => openSocket(meetingId),
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
    [buildWsUrl, composeLive]
  );

  const start = useCallback(
    async (meetingId) => {
      setLiveText("");
      liveSegmentsRef.current = {};
      meetingIdRef.current = meetingId;
      stoppingRef.current = false;
      reconnectRef.current = { attempts: 0, timer: null, active: true };
      // Immediate UI feedback — do not wait for mic/worklet/WS to finish.
      setStatus("starting");
      setMessage("Starting meeting and microphone…");
      setRecording(true);
      startedAtRef.current = Date.now();
      setElapsed(0);
      pendingPcmRef.current = [];
      bytesSentRef.current = 0;
      if (timerRef.current) clearInterval(timerRef.current);
      // Wall-clock based so long board meetings stay accurate if the tab throttles.
      timerRef.current = setInterval(() => {
        const started = startedAtRef.current;
        if (!started) return;
        setElapsed(Math.floor((Date.now() - started) / 1000));
      }, 250);

      // Open the WebSocket in parallel with mic setup (biggest perceived win).
      openSocket(meetingId);

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
            autoGainControl: true,
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
    [cleanupAudio, openSocket]
  );

  const stop = useCallback(async () => {
    stoppingRef.current = true;
    reconnectRef.current.active = false;
    if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    // Freeze the final elapsed time for the UI while Whisper finalizes.
    if (startedAtRef.current) {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }
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
    // Flush trailing PCM (~up to 0.25s) before tearing down the mic graph.
    await flushAndCleanupAudio();
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
  }, [flushAndCleanupAudio]);

  useEffect(() => {
    return () => {
      stoppingRef.current = true;
      reconnectRef.current.active = false;
      if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
      if (timerRef.current) clearInterval(timerRef.current);
      if (keepaliveRef.current) clearInterval(keepaliveRef.current);
      if (audioWatchRef.current) clearInterval(audioWatchRef.current);
      cleanupAudio();
      if (wsRef.current) wsRef.current.close();
    };
  }, [cleanupAudio]);

  return {
    recording,
    status,
    liveText,
    message,
    transcriptionAvailable,
    elapsed,
    connectionState,
    start,
    stop,
  };
}
