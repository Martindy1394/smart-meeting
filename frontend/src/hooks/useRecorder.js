import { useCallback, useEffect, useRef, useState } from "react";
import { getToken } from "../api/client";

// Manages microphone capture (16 kHz PCM) and streams audio to the backend over
// a WebSocket for live captions + finalization.
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
  const processorRef = useRef(null);
  const silentGainRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null);
  const liveSegmentsRef = useRef({});
  const timerRef = useRef(null);
  const meetingIdRef = useRef(null);
  const reconnectRef = useRef({ attempts: 0, timer: null, active: false });
  const stoppingRef = useRef(false);
  const pcmBufferRef = useRef([]);
  const inputRateRef = useRef(48000);

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
      if (processorRef.current) {
        processorRef.current.onaudioprocess = null;
        processorRef.current.disconnect();
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
    processorRef.current = null;
    silentGainRef.current = null;
    sourceRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
    pcmBufferRef.current = [];
  }, []);

  const sendPcmFloat = useCallback((floatSamples) => {
    // Downsample to 16 kHz mono PCM16 and flush ~0.25s chunks.
    const ratio = inputRateRef.current / 16000;
    const outLen = Math.floor(floatSamples.length / ratio);
    const buf = pcmBufferRef.current;
    for (let i = 0; i < outLen; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(floatSamples.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      let count = 0;
      for (let j = start; j < end; j++) {
        sum += floatSamples[j];
        count++;
      }
      buf.push(count ? sum / count : 0);
    }
    const ws = wsRef.current;
    while (buf.length >= 4096) {
      const slice = buf.splice(0, 4096);
      const pcm = new Int16Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        const s = Math.max(-1, Math.min(1, slice[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pcm.buffer);
      }
    }
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
          if (data.message) setMessage(data.message);
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

      ws.onerror = () => {};
    },
    [buildWsUrl, composeLive, onFinalTranscript]
  );

  const startScriptProcessor = useCallback(
    (audioCtx, source) => {
      // Reliable fallback path (works even when AudioWorklet capture is silent).
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        let sum = 0;
        for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
        setMicLevel(Math.sqrt(sum / Math.max(1, input.length)));
        // Copy because the underlying buffer is reused.
        sendPcmFloat(new Float32Array(input));
      };
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      silentGainRef.current = silentGain;
      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(audioCtx.destination);
    },
    [sendPcmFloat]
  );

  const start = useCallback(
    async (meetingId) => {
      setMessage("");
      setLiveText("");
      setMicLevel(0);
      liveSegmentsRef.current = {};
      pcmBufferRef.current = [];
      meetingIdRef.current = meetingId;
      stoppingRef.current = false;
      reconnectRef.current = { attempts: 0, timer: null, active: true };

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: { ideal: 1 },
            echoCancellation: true,
            noiseSuppression: true,
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
      // Do NOT force sampleRate — a mismatch can yield silent MediaStreamSource output.
      const audioCtx = new AudioCtx();
      audioCtxRef.current = audioCtx;
      inputRateRef.current = audioCtx.sampleRate || 48000;
      if (audioCtx.state === "suspended") {
        try {
          await audioCtx.resume();
        } catch {
          /* ignore */
        }
      }

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      // Prefer ScriptProcessor for capture reliability (live captions depend on
      // real PCM arriving). AudioWorklet remains available as a secondary path
      // but ScriptProcessor has been more consistent across browsers here.
      startScriptProcessor(audioCtx, source);

      openSocket(meetingId);

      setRecording(true);
      setStatus("recording");
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    },
    [openSocket, startScriptProcessor]
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
    // Flush remaining PCM before tearing down audio.
    const buf = pcmBufferRef.current;
    const ws = wsRef.current;
    if (buf.length > 0 && ws && ws.readyState === WebSocket.OPEN) {
      const pcm = new Int16Array(buf.length);
      for (let i = 0; i < buf.length; i++) {
        const s = Math.max(-1, Math.min(1, buf[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      ws.send(pcm.buffer);
      pcmBufferRef.current = [];
    }
    cleanupAudio();
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
