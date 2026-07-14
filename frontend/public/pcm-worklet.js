// AudioWorkletProcessor that downsamples the microphone input to 16 kHz mono
// and emits little-endian 16-bit PCM chunks to the main thread, plus a simple
// RMS level meter so the UI can warn when the mic is silent.
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.inputSampleRate = sampleRate;
    this.ratio = this.inputSampleRate / this.targetSampleRate;
    this._buffer = [];
    // Emit roughly every ~0.25s worth of 16k samples.
    this._emitEvery = 4096;
    this._levelCounter = 0;
  }

  _downsample(input) {
    if (this.ratio <= 1.01) {
      return input;
    }
    const outLength = Math.floor(input.length / this.ratio);
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const start = Math.floor(i * this.ratio);
      const end = Math.min(input.length, Math.floor((i + 1) * this.ratio));
      let sum = 0;
      let count = 0;
      for (let j = start; j < end; j++) {
        sum += input[j];
        count++;
      }
      out[i] = count > 0 ? sum / count : 0;
    }
    return out;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    // Prefer first channel; if stereo somehow arrives, average L/R.
    let channel = input[0];
    if (!channel) {
      return true;
    }
    if (input.length > 1 && input[1]) {
      const mixed = new Float32Array(channel.length);
      const right = input[1];
      for (let i = 0; i < channel.length; i++) {
        mixed[i] = (channel[i] + (right[i] || 0)) * 0.5;
      }
      channel = mixed;
    }

    // Level meter (~every ~100ms at 48k / 128 render quantum).
    this._levelCounter += 1;
    if (this._levelCounter % 8 === 0) {
      let sum = 0;
      for (let i = 0; i < channel.length; i++) sum += channel[i] * channel[i];
      const rms = Math.sqrt(sum / Math.max(1, channel.length));
      this.port.postMessage({ type: "level", rms });
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
