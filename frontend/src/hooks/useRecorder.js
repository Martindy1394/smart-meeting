import { useCallback, useEffect, useRef, useState } from "react";
import { getToken } from "../api/client";

// Manages microphone capture (AudioWorklet -> 16 kHz PCM) and streams the audio
// to the backend over a WebSocket, exposing live caption + finalization state.
export function useRecorder({ onFinalTranscript } = {}) {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("idle"); // idle|recording|finalizing|error
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
  const meetingIdRef = useRef(null);
  const reconnectRef = useRef({ attempts: 0, timer: null, active: false });
  const stoppingRef = useRef(false);

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
      const ws = new WebSocket(buildWsUrl(meetingId));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      setConnectionState("connecting");

      ws.onopen = () => {
        setConnectionState("connected");
        reconnectRef.current.attempts = 0;
        ws.send(JSON.stringify({ type: "start" }));
      };

      ws.onmessage = (evt) => {
        let data;
        try {
          data = JSON.parse(evt.data);
        } catch {
          return;
        }
        if (data.type === "status") {
          setTranscriptionAvailable(data.transcription_available);
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
        } else if (data.type === "live_segment") {
          // Legacy per-chunk segments (kept for compatibility / persistence).
          liveSegmentsRef.current[data.seq] = data.text;
          // Only fall back to composed chunks if we have not received live_caption yet.
          setLiveText((prev) => prev || composeLive());
        } else if (data.type === "finalizing") {
          setStatus("finalizing");
        } else if (data.type === "final_transcript") {
          // Two-pass: clear live fragments, surface the finalized transcript.
          liveSegmentsRef.current = {};
          setLiveText("");
          setStatus("idle");
          if (onFinalTranscript) onFinalTranscript(data);
        } else if (data.type === "error") {
          setStatus("error");
          setMessage(data.message || "Transcription error.");
        }
      };

      ws.onclose = () => {
        setConnectionState("disconnected");
        // Reconnect only if we are actively recording (unexpected drop).
        if (reconnectRef.current.active && !stoppingRef.current) {
          const attempts = ++reconnectRef.current.attempts;
          if (attempts <= 5) {
            const delay = Math.min(1000 * 2 ** (attempts - 1), 16000);
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
    [buildWsUrl, composeLive, onFinalTranscript]
  );

  const start = useCallback(
    async (meetingId) => {
      setMessage("");
      setLiveText("");
      liveSegmentsRef.current = {};
      meetingIdRef.current = meetingId;
      stoppingRef.current = false;
      reconnectRef.current = { attempts: 0, timer: null, active: true };

      // 1) Microphone + AudioWorklet.
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            // Browser noise suppression can smear consonants Whisper needs.
            noiseSuppression: false,
            autoGainControl: true,
          },
        });
      } catch (err) {
        setStatus("error");
        setMessage("Microphone access denied or unavailable.");
        throw err;
      }
      streamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioCtxRef.current = audioCtx;
      if (audioCtx.state === "suspended") {
        try {
          await audioCtx.resume();
        } catch {
          /* ignore */
        }
      }
      try {
        await audioCtx.audioWorklet.addModule("/pcm-worklet.js");
      } catch (err) {
        setStatus("error");
        setMessage("Failed to initialize audio processor.");
        cleanupAudio();
        throw err;
      }

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const node = new AudioWorkletNode(audioCtx, "pcm-worklet");
      workletNodeRef.current = node;
      node.port.onmessage = (event) => {
        if (event.data && event.data.type === "flushed") return;
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(event.data);
        }
      };
      source.connect(node);
      // Keep the worklet in the graph without audible playback (avoids AEC feedback).
      const silent = audioCtx.createGain();
      silent.gain.value = 0;
      node.connect(silent);
      silent.connect(audioCtx.destination);

      // 2) WebSocket stream.
      openSocket(meetingId);

      setRecording(true);
      setStatus("recording");
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
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
