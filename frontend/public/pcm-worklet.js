// AudioWorkletProcessor that downsamples the microphone input to 16 kHz mono
// and emits little-endian 16-bit PCM chunks (~ up to 4096 samples) to the main
// thread. Using an AudioWorklet (rather than ScriptProcessorNode) gives us raw
// PCM with no server-side WebM/Opus decoding and minimal latency.
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.inputSampleRate = sampleRate; // global provided by AudioWorkletGlobalScope
    this.ratio = this.inputSampleRate / this.targetSampleRate;
    this._buffer = [];
    // Fractional source index carried across process() callbacks so 48→16 kHz
    // (and 44.1→16 kHz) does not drop/duplicate samples at block edges.
    this._frac = 0;
    // Emit roughly every ~0.25s worth of 16k samples to keep chunks small.
    this._emitEvery = 4096;
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === "flush") {
        this._flush();
      }
    };
  }

  _emit(samples) {
    if (!samples.length) return;
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      // Soft knee before hard clip — hard clipping (peak=1.0) confused VAD
      // and caused Whisper to skip real speech at the start of windows.
      let s = samples[i];
      const a = Math.abs(s);
      if (a > 0.9) {
        const sign = s < 0 ? -1 : 1;
        s = sign * (0.9 + 0.1 * Math.tanh((a - 0.9) / 0.1));
      }
      s = Math.max(-1, Math.min(1, s));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
  }

  _flush() {
    if (this._buffer.length === 0) {
      this.port.postMessage({ type: "flushed" });
      return;
    }
    const slice = this._buffer.splice(0, this._buffer.length);
    this._emit(slice);
    this.port.postMessage({ type: "flushed" });
  }

  _mono(input) {
    const left = input[0];
    if (!left) return null;
    if (input.length < 2 || !input[1]) return left;
    const right = input[1];
    const n = Math.min(left.length, right.length);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = 0.5 * (left[i] + right[i]);
    }
    return out;
  }

  // Linear-interpolation resampler with a rolling fractional index.
  _downsample(input) {
    if (!input || !input.length) return new Float32Array(0);
    if (this.ratio <= 1) {
      this._frac = 0;
      return input;
    }
    const out = [];
    let pos = this._frac;
    const last = input.length - 1;
    while (pos <= last) {
      const i0 = Math.min(Math.floor(pos), last);
      const i1 = Math.min(i0 + 1, last);
      const frac = pos - i0;
      out.push(input[i0] * (1 - frac) + input[i1] * frac);
      pos += this.ratio;
    }
    this._frac = pos - input.length;
    if (!Number.isFinite(this._frac) || this._frac < 0) this._frac = 0;
    return out.length ? Float32Array.from(out) : new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const channel = this._mono(input);
    if (!channel) {
      return true;
    }
    const down = this._downsample(channel);
    for (let i = 0; i < down.length; i++) {
      this._buffer.push(down[i]);
    }
    while (this._buffer.length >= this._emitEvery) {
      const slice = this._buffer.splice(0, this._emitEvery);
      this._emit(slice);
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorkletProcessor);
