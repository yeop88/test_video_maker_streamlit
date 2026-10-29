"""나레이션 문장을 타임라인에 배치하고, 나레이션이 나올 때 원본 소리를 줄여서 섞는다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MIX_SAMPLE_RATE as SR
from .media import load_audio


def speech_bounds(wav: np.ndarray, pad_sec: float = 0.05) -> tuple[int, int]:
    """앞뒤 무음을 뺀 말소리 범위 (샘플). Chatterbox는 말이 끝난 뒤 1초 가까이 아주 작은 잡음을 남기므로,
    한 샘플이 아니라 20ms 단위 소리 크기가 가장 큰 곳의 5%를 넘는 구간만 말로 본다."""
    frame = int(0.02 * SR)
    n = len(wav) // frame
    if n == 0:
        return 0, 0
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > max(0.003, 0.05 * float(rms.max())))
    if loud.size == 0:
        return 0, 0
    pad = int(pad_sec * SR)
    return max(0, loud[0] * frame - pad), min(len(wav), (loud[-1] + 1) * frame + pad)


def load_speech(path: Path) -> tuple[np.ndarray, float]:
    """합성한 문장 음성을 앞뒤 무음 없이 읽는다. (음성, 앞에서 잘라 낸 초)를 돌려준다 (단어 시간 보정용)."""
    wav = load_audio(path, SR, 1)[:, 0]
    a, b = speech_bounds(wav)
    return wav[a:b], a / SR


def place(track: np.ndarray, wav: np.ndarray, start: float) -> None:
    s = int(start * SR)
    e = min(len(track), s + len(wav))
    track[s:e] += wav[: e - s]


def duck_envelope(narration: np.ndarray, duck_gain: float, ramp_sec: float = 0.15) -> np.ndarray:
    """나레이션이 나오는 동안은 duck_gain, 나머지는 1.0인 원본 소리 크기 곡선."""
    frame = int(0.02 * SR)
    n = len(narration) // frame + 1
    padded = np.pad(np.abs(narration), (0, n * frame - len(narration)))
    active = padded.reshape(n, frame).max(axis=1) > 0.01
    # 쉼표나 문장 사이 쉼에 원본 소리가 튀어나오지 않도록 0.6초 이하 틈은 메운다
    hold = int(0.6 / 0.02)
    idx = np.flatnonzero(active)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= hold:
            active[a:b] = True
    gain = np.where(active, duck_gain, 1.0).astype(np.float32)
    k = max(1, int(ramp_sec / 0.02))
    gain = np.convolve(np.pad(gain, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    return np.repeat(gain, frame)[: len(narration)].astype(np.float32)


def mix(original: np.ndarray, narration: np.ndarray, original_gain: float, duck_gain: float, out_path: Path) -> None:
    """original은 (샘플, 2) 스테레오, narration은 모노. 길이는 narration에 맞춘다."""
    n = len(narration)
    if len(original) < n:
        original = np.pad(original, ((0, n - len(original)), (0, 0)))
    env = duck_envelope(narration, duck_gain) * original_gain
    out = original[:n] * env[:, None] + narration[:, None]
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0.98:  # 클리핑 방지
        out *= 0.98 / peak
    sf.write(out_path, out, SR)
