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
  const [micLevel, setMicLevel] = useState(0);

  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const workletNodeRef = useRef(null);
  const silentGainRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null);
  const liveSegmentsRef = useRef({});
  const timerRef = useRef(null);
  const meetingIdRef = useRef(null);
  const reconnectRef = useRef({ attempts: 0, timer: null, active: false });
  const stoppingRef = useRef(false);
  const lowLevelSinceRef = useRef(0);

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
      if (silentGainRef.current) silentGainRef.current.disconnect();
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
    silentGainRef.current = null;
    sourceRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
  }, []);

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
        } else if (data.type === "warning") {
          setMessage(data.message || "");
        } else if (data.type === "live_segment") {
          liveSegmentsRef.current[data.seq] = data.text;
          setLiveText(composeLive());
          setMessage("");
        } else if (data.type === "finalizing") {
          setStatus("finalizing");
        } else if (data.type === "final_transcript") {
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
      setMicLevel(0);
      liveSegmentsRef.current = {};
      meetingIdRef.current = meetingId;
      stoppingRef.current = false;
      lowLevelSinceRef.current = 0;
      reconnectRef.current = { attempts: 0, timer: null, active: true };

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
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
      const audioCtx = new AudioCtx({ sampleRate: 48000 });
      audioCtxRef.current = audioCtx;
      if (audioCtx.state === "suspended") {
        try {
          await audioCtx.resume();
        } catch {
          /* ignore */
        }
      }

      try {
        // Cache-bust so worklet updates are always picked up.
        await audioCtx.audioWorklet.addModule(`/pcm-worklet.js?v=${Date.now()}`);
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

      // Keep the graph alive without playing mic audio through speakers.
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      silentGainRef.current = silentGain;

      node.port.onmessage = (event) => {
        const payload = event.data;
        if (payload && payload.type === "level") {
          setMicLevel(payload.rms || 0);
          if ((payload.rms || 0) < 0.002) {
            if (!lowLevelSinceRef.current) lowLevelSinceRef.current = Date.now();
            else if (Date.now() - lowLevelSinceRef.current > 2500) {
              setMessage(
                "Mic level is very low — check the correct input device and permissions."
              );
            }
          } else {
            lowLevelSinceRef.current = 0;
          }
          return;
        }
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN && payload) {
          ws.send(payload);
        }
      };

      source.connect(node);
      node.connect(silentGain);
      silentGain.connect(audioCtx.destination);

      openSocket(meetingId);

      setRecording(true);
      setStatus("recording");
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    },
    [cleanupAudio, openSocket]
  );

  const stop = useCallback(() => {
    stoppingRef.current = true;
    reconnectRef.current.active = false;
    if (reconnectRef.current.timer) clearTimeout(reconnectRef.current.timer);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setRecording(false);
    setStatus("finalizing");
    cleanupAudio();
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
  }, [cleanupAudio]);

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
    micLevel,
    start,
    stop,
  };
}
