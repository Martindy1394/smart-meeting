// AudioWorkletProcessor that downsamples the microphone input to 16 kHz mono
// and emits little-endian 16-bit PCM chunks (~ up to 4096 samples) to the main
// thread. Using an AudioWorklet (rather than MediaRecorder) gives us raw PCM
// with no server-side WebM/Opus decoding and minimal latency.
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = 16000;
    this.inputSampleRate = sampleRate; // global provided by AudioWorkletGlobalScope
    this.ratio = this.inputSampleRate / this.targetSampleRate;
    this._buffer = [];
    // Emit roughly every ~0.25s worth of 16k samples to keep chunks small.
    this._emitEvery = 4096;
  }

  _downsample(input) {
    if (this.ratio <= 1) {
      return input;
    }
    const outLength = Math.floor(input.length / this.ratio);
    const out = new Float32Array(outLength);
    let pos = 0;
    for (let i = 0; i < outLength; i++) {
      const start = Math.floor(i * this.ratio);
      const end = Math.floor((i + 1) * this.ratio);
      let sum = 0;
      let count = 0;
      for (let j = start; j < end && j < input.length; j++) {
        sum += input[j];
        count++;
      }
      out[pos++] = count > 0 ? sum / count : 0;
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
      const pcm = new Int16Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        let s = Math.max(-1, Math.min(1, slice[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorkletProcessor);
