// AudioWorkletProcessor that downsamples the microphone input to 16 kHz mono
// and emits little-endian 16-bit PCM chunks (~ up to 4096 samples) to the main
// thread. Using an AudioWorklet (rather than MediaRecorder) gives us raw PCM
// with no server-side WebM/Opus decoding and minimal latency.
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.inputSampleRate = sampleRate; // global provided by AudioWorkletGlobalScope
    this.ratio = this.inputSampleRate / this.targetSampleRate;
    this._buffer = [];
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

  // Linear-interpolation resampler — more accurate than block averaging when
  // the hardware rate is not an integer multiple of 16 kHz (e.g. 44.1 kHz).
  _downsample(input) {
    if (this.ratio <= 1) {
      return input;
    }
    const outLength = Math.floor(input.length / this.ratio);
    if (outLength <= 0) return new Float32Array(0);
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const src = i * this.ratio;
      const i0 = Math.floor(src);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = src - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const channel = input[0];
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
