"""Lightweight live speaker labeling (Voice 1, Voice 2, Voice 3).

Does **not** replace Whisper. PCM windows are clustered with a short spectral
fingerprint so turns from the same talker stay grouped. Display labels are then
**ranked by ASR accuracy**: Voice 1 is the cluster with the highest Whisper
confidence, Voice 2 the next, and so on. Labels are per-meeting and are not
enrolled names.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

try:
    from ..config import settings
except Exception:  # pragma: no cover — unit tests without pydantic
    class _Fallback:
        live_max_voices = 3
        live_speaker_labels = True
        audio_sample_rate = 16000

    settings = _Fallback()

_MAX_VOICES_DEFAULT = 3
_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, "_VoiceSession"] = {}


def voice_label(index: int) -> str:
    """Human-readable anonymous label (Voice 1 …)."""
    n = max(1, int(index or 1))
    return f"Voice {n}"


def _max_voices() -> int:
    try:
        n = int(getattr(settings, "live_max_voices", _MAX_VOICES_DEFAULT) or _MAX_VOICES_DEFAULT)
    except (TypeError, ValueError):
        n = _MAX_VOICES_DEFAULT
    return max(1, min(8, n))


def _enabled() -> bool:
    return bool(getattr(settings, "live_speaker_labels", True))


def fingerprint_pcm16(
    pcm: bytes,
    *,
    sample_rate: int = 16000,
) -> tuple[float, ...]:
    """Compact voice fingerprint: pitch + 8 log-mel bands + RMS.

    Cheap enough for live 5–10s windows on CPU. Not a neural speaker embedder.
    """
    if not pcm or len(pcm) < 4:
        return ()
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size < 160:
        return ()
    samples /= 32768.0
    rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-9)
    # Autocorrelation pitch (Hz), speech-ish 80–400 Hz.
    hop = max(1, int(sample_rate / 800))
    dec = samples[::hop]
    if dec.size < 64:
        return ()
    dec = dec - float(np.mean(dec))
    min_lag = max(1, int(sample_rate / hop / 400))
    max_lag = min(len(dec) - 2, int(sample_rate / hop / 80))
    if max_lag <= min_lag + 2:
        f0 = 0.0
    else:
        best_lag = min_lag
        best = -1.0
        for lag in range(min_lag, max_lag):
            a = dec[:-lag]
            b = dec[lag:]
            denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-9)
            corr = float(np.dot(a, b) / denom)
            if corr > best:
                best = corr
                best_lag = lag
        f0 = (sample_rate / hop) / float(best_lag) if best > 0.25 else 0.0
    # 8-band log energy via rFFT.
    n = int(2 ** int(math.floor(math.log2(min(len(samples), 4096)))))
    spec = np.abs(np.fft.rfft(samples[:n] * np.hanning(n))) + 1e-9
    bands = np.array_split(spec, 8)
    log_e = [float(np.log(np.mean(b) + 1e-9)) for b in bands]
    f0n = math.log(f0 + 1.0)
    rmsn = math.log(rms + 1e-6)
    # Weight spectral shape more than (unreliable short-window) pitch.
    return tuple([f0n * 0.25, rmsn * 0.5, *log_e])


def fingerprint_float32(samples: np.ndarray, *, sample_rate: int = 16000) -> tuple[float, ...]:
    if samples is None or getattr(samples, "size", 0) < 160:
        return ()
    clipped = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).tobytes()
    return fingerprint_pcm16(pcm, sample_rate=sample_rate)


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 1e9
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def segment_accuracy(seg) -> float | None:
    """0–1 Whisper confidence for a segment (higher = more accurate)."""
    if isinstance(seg, dict):
        lp = seg.get("avg_logprob")
        nsp = seg.get("no_speech_prob")
        low = bool(seg.get("low_confidence", False))
    else:
        lp = getattr(seg, "avg_logprob", None)
        nsp = getattr(seg, "no_speech_prob", None)
        low = bool(getattr(seg, "low_confidence", False))
    parts: list[float] = []
    try:
        if lp is not None:
            # Typical Whisper avg_logprob is about -1.2 (poor) to 0 (strong).
            parts.append(max(0.0, min(1.0, (float(lp) + 1.2) / 1.2)))
    except (TypeError, ValueError):
        pass
    try:
        if nsp is not None:
            parts.append(max(0.0, min(1.0, 1.0 - float(nsp))))
    except (TypeError, ValueError):
        pass
    if not parts:
        return None
    score = sum(parts) / len(parts)
    if low:
        score *= 0.8
    return score


def asr_accuracy(result) -> float | None:
    """Mean segment accuracy for a live/final Whisper result."""
    segs = getattr(result, "segments", None) or []
    scores = [s for s in (segment_accuracy(seg) for seg in segs) if s is not None]
    if scores:
        return sum(scores) / len(scores)
    conf = getattr(result, "language_confidence", None)
    try:
        if conf is not None:
            return max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        return None
    return None


def rank_voice_ids(
    cluster_ids: list[int],
    scores: dict[int, float],
) -> dict[int, int]:
    """Map internal cluster ids → Voice N ranked by accuracy (Voice 1 = best)."""
    unique: list[int] = []
    seen: set[int] = set()
    for cid in cluster_ids:
        n = int(cid or 0)
        if n < 1 or n in seen:
            continue
        seen.add(n)
        unique.append(n)
    if not unique:
        return {}
    order = list(range(len(unique)))

    def sort_key(i: int) -> tuple:
        cid = unique[i]
        if cid in scores:
            return (0, -float(scores[cid]), i)
        return (1, i)

    ranked = [unique[i] for i in sorted(order, key=sort_key)]
    return {cid: pos + 1 for pos, cid in enumerate(ranked)}


@dataclass
class _VoiceSession:
    centroids: list[tuple[float, ...]] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    acc_counts: list[int] = field(default_factory=list)
    last_index: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock)

    def assign(self, feat: tuple[float, ...], *, threshold: float = 1.35) -> int:
        if not feat:
            return self.last_index
        with self.lock:
            if not self.centroids:
                self.centroids.append(feat)
                self.counts.append(1)
                self.accuracies.append(0.0)
                self.acc_counts.append(0)
                self.last_index = 1
                return 1
            dists = [_dist(feat, c) for c in self.centroids]
            best_i = int(min(range(len(dists)), key=lambda i: dists[i]))
            max_v = _max_voices()
            pitch_gap = abs(feat[0] - self.centroids[best_i][0]) if feat and self.centroids[best_i] else 0.0
            spec_gap = _dist(feat[2:], self.centroids[best_i][2:]) if feat and self.centroids[best_i] else 0.0
            new_voice = dists[best_i] > 0.85 or pitch_gap > 0.18
            if new_voice and len(self.centroids) < max_v:
                self.centroids.append(feat)
                self.counts.append(1)
                self.accuracies.append(0.0)
                self.acc_counts.append(0)
                self.last_index = len(self.centroids)
                return self.last_index
            # EMA update of matched centroid.
            old = self.centroids[best_i]
            n = self.counts[best_i]
            alpha = 1.0 / float(n + 1)
            self.centroids[best_i] = tuple(
                (1.0 - alpha) * o + alpha * f for o, f in zip(old, feat)
            )
            self.counts[best_i] = n + 1
            self.last_index = best_i + 1
            return self.last_index

    def observe(self, cluster_id: int, score: float | None) -> None:
        if score is None:
            return
        i = int(cluster_id) - 1
        with self.lock:
            if i < 0 or i >= len(self.centroids):
                return
            n = self.acc_counts[i]
            prev = self.accuracies[i]
            self.accuracies[i] = (prev * n + float(score)) / float(n + 1)
            self.acc_counts[i] = n + 1

    def ranked_index(self, cluster_id: int) -> int:
        cid = int(cluster_id or 0)
        if cid < 1:
            return 1
        with self.lock:
            ids = list(range(1, len(self.centroids) + 1))
            scores = {
                i + 1: self.accuracies[i]
                for i in range(len(self.centroids))
                if self.acc_counts[i]
            }
        if not scores:
            return cid
        mapping = rank_voice_ids(ids, scores)
        return int(mapping.get(cid, cid))


def _session(meeting_id: str) -> _VoiceSession:
    key = (meeting_id or "").strip() or "_default"
    with _SESSION_LOCK:
        sess = _SESSIONS.get(key)
        if sess is None:
            sess = _VoiceSession()
            _SESSIONS[key] = sess
        return sess


def reset_meeting(meeting_id: str) -> None:
    key = (meeting_id or "").strip()
    if not key:
        return
    with _SESSION_LOCK:
        _SESSIONS.pop(key, None)


def label_pcm(
    meeting_id: str,
    pcm: bytes,
    *,
    sample_rate: int | None = None,
) -> tuple[int, str]:
    """Return ``(cluster_index, 'Voice N')`` for a live PCM window.

    ``cluster_index`` is the stable talker group (order of first appearance).
    Call ``bind_asr_accuracy`` after Whisper so the displayed Voice N is ranked
    by transcription accuracy.
    """
    if not _enabled():
        return 1, voice_label(1)
    sr = int(sample_rate or getattr(settings, "audio_sample_rate", 16000) or 16000)
    feat = fingerprint_pcm16(pcm, sample_rate=sr)
    idx = _session(meeting_id).assign(feat)
    return idx, voice_label(idx)


def bind_asr_accuracy(
    meeting_id: str,
    cluster_index: int,
    result=None,
    *,
    score: float | None = None,
) -> tuple[int, str]:
    """Rank Voice N by running ASR accuracy; Voice 1 is the most accurate talker."""
    sess = _session(meeting_id)
    observed = score if score is not None else asr_accuracy(result)
    sess.observe(int(cluster_index or 0), observed)
    display = sess.ranked_index(int(cluster_index or 1))
    return display, voice_label(display)


def label_float32(
    meeting_id: str,
    samples: np.ndarray,
    *,
    sample_rate: int | None = None,
) -> tuple[int, str]:
    if not _enabled():
        return 1, voice_label(1)
    sr = int(sample_rate or getattr(settings, "audio_sample_rate", 16000) or 16000)
    feat = fingerprint_float32(samples, sample_rate=sr)
    idx = _session(meeting_id).assign(feat)
    return idx, voice_label(idx)


def _seg_text(seg) -> str:
    if isinstance(seg, dict):
        return (seg.get("text") or "").strip()
    return (getattr(seg, "text", None) or "").strip()


def _set_voice(seg, index: int, label: str) -> None:
    if isinstance(seg, dict):
        seg["speaker_index"] = int(index)
        seg["speaker_label"] = label
        return
    try:
        seg.speaker_index = int(index)
        seg.speaker_label = label
    except Exception:
        pass


def label_segments(
    meeting_id: str,
    segments: list,
    samples: np.ndarray | None = None,
    *,
    sample_rate: int | None = None,
) -> list:
    """Assign Voice 1…N from each segment's audio slice, ranked by ASR accuracy.

    Clustering groups the same talker; Voice 1 is the cluster with the highest
    Whisper confidence. Without confidence scores, labels keep first-seen order.
    """
    if not segments:
        return []
    reset_meeting(meeting_id)
    sr = int(sample_rate or getattr(settings, "audio_sample_rate", 16000) or 16000)
    from .segment_times import coerce_times

    n = int(getattr(samples, "size", 0) or 0)
    cluster_ids: list[int] = []
    for seg in segments:
        if not _seg_text(seg):
            cluster_ids.append(1)
            continue
        start, end = coerce_times(seg)
        if samples is not None and n > 0 and end > start:
            i0 = max(0, int(start * sr))
            i1 = min(n, max(i0 + 1, int(end * sr)))
            idx, _lab = label_float32(meeting_id, samples[i0:i1], sample_rate=sr)
        else:
            idx = 1
        cluster_ids.append(int(idx or 1))

    score_sum: dict[int, float] = {}
    score_n: dict[int, int] = {}
    for seg, cid in zip(segments, cluster_ids):
        sc = segment_accuracy(seg)
        if sc is None:
            continue
        score_sum[cid] = score_sum.get(cid, 0.0) + sc
        score_n[cid] = score_n.get(cid, 0) + 1
    scores = {cid: score_sum[cid] / score_n[cid] for cid in score_n}
    ranking = rank_voice_ids(cluster_ids, scores)
    for seg, cid in zip(segments, cluster_ids):
        display = int(ranking.get(cid, cid) or 1)
        _set_voice(seg, display, voice_label(display))
    return segments


def prefix_text(label: str, text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    lab = (label or "").strip()
    if not lab:
        return raw
    if raw.lower().startswith(lab.lower() + ":"):
        return raw
    return f"{lab}: {raw}"


def format_transcript(segments: list) -> str:
    """Join labeled segments, grouping consecutive same-voice lines."""
    lines: list[str] = []
    last_lab = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, last_lab
        if not buf:
            return
        body = " ".join(buf).strip()
        if last_lab:
            lines.append(f"{last_lab}: {body}")
        else:
            lines.append(body)
        buf = []

    for seg in segments or []:
        text = (getattr(seg, "text", None) or "").strip()
        if isinstance(seg, dict):
            text = (seg.get("text") or "").strip()
        if not text:
            continue
        lab = getattr(seg, "speaker_label", None)
        if isinstance(seg, dict):
            lab = seg.get("speaker_label") or lab
        lab = (lab or "").strip()
        if lab and text.lower().startswith(lab.lower() + ":"):
            text = text.split(":", 1)[1].strip()
        if lab != last_lab:
            flush()
            last_lab = lab
        buf.append(text)
    flush()
    return "\n".join(lines).strip()
