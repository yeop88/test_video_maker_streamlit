"""피사체를 따라 자르기: 가로 영상을 세로로 자를 때 자르는 위치를 피사체 쪽으로 부드럽게 옮긴다.

1. 영상을 초당 ANALYSIS_FPS장, 폭 ANALYSIS_WIDTH로 줄여 읽는다.
2. 장면마다 피사체의 가로 위치(0~1)를 찾는다.
   - 얼굴이 있으면 얼굴 (OpenCV Haar, 큰 얼굴일수록 가중치)
   - 없으면 눈에 띄는 곳(spectral residual saliency) + 움직이는 곳(앞 장면과의 차이)
3. 흔들리지 않게 다듬는다: 중앙값 필터 → 작은 움직임 무시(데드존) → 이동 속도 제한.
4. ffmpeg crop 필터의 x를 시간 t에 대한 식(구간별 직선)으로 만든다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .media import ensure_ffmpeg, run_ffmpeg

ANALYSIS_FPS = 4
ANALYSIS_WIDTH = 256
KEYFRAME_SEC = 0.5  # crop 식의 꺾이는 점 간격
MAX_SPEED = 0.35  # 초당 최대 이동 (화면 폭 대비)
DEADZONE = 0.04  # 이보다 작은 움직임은 따라가지 않음 (화면 폭 대비)

_face_detector = None


def _frames(video: Path, width: int, height: int) -> np.ndarray:
    """(장면 수, 높이, 폭) 회색조 배열."""
    raw = run_ffmpeg([
        "-i", str(video), "-vf", f"fps={ANALYSIS_FPS},scale={width}:{height},format=gray",
        "-f", "rawvideo", "-",
    ])
    frame_size = width * height
    n = len(raw) // frame_size
    return np.frombuffer(raw[: n * frame_size], np.uint8).reshape(n, height, width)


def _faces(gray: np.ndarray) -> float | None:
    global _face_detector
    import cv2

    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = _face_detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(16, 16))
    if len(faces) == 0:
        return None
    weights = np.array([w * h for _, _, w, h in faces], float)
    centers = np.array([x + w / 2 for x, _, w, _ in faces], float)
    return float((centers * weights).sum() / weights.sum() / gray.shape[1])


def _saliency(gray: np.ndarray) -> np.ndarray:
    """spectral residual 방식의 눈에 띄는 정도 지도 (64×64로 계산 후 원래 크기)."""
    import cv2

    small = cv2.resize(gray.astype(np.float32), (64, 64))
    spectrum = np.fft.fft2(small)
    log_amp = np.log(np.abs(spectrum) + 1e-8)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))) ** 2
    sal = cv2.GaussianBlur(sal, (9, 9), 2.5)
    return cv2.resize(sal / (sal.max() + 1e-8), (gray.shape[1], gray.shape[0]))


def _centroid(weight_map: np.ndarray) -> float | None:
    columns = weight_map.sum(axis=0)
    columns = np.clip(columns - np.percentile(columns, 50), 0, None)  # 배경 수준은 빼고
    if columns.sum() <= 1e-6:
        return None
    return float((np.arange(len(columns)) * columns).sum() / columns.sum() / len(columns))


def subject_positions(video: Path, src_w: int, src_h: int) -> list[tuple[float, float]]:
    """[(시간 초, 피사체 가로 위치 0~1)]."""
    ensure_ffmpeg()
    width = ANALYSIS_WIDTH
    height = max(16, int(round(width * src_h / src_w / 2)) * 2)
    frames = _frames(video, width, height)
    positions, prev = [], None
    for i, gray in enumerate(frames):
        x = _faces(gray)
        if x is None:
            sal = _saliency(gray)
            if prev is not None:  # 움직이는 곳에 가중치
                motion = np.abs(gray.astype(np.float32) - prev.astype(np.float32))
                if motion.max() > 8:
                    sal = sal + 0.7 * motion / motion.max()
            x = _centroid(sal)
        positions.append((i / ANALYSIS_FPS, 0.5 if x is None else x))
        prev = gray
    return positions


def smooth(positions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not positions:
        return []
    xs = np.array([p for _, p in positions])
    k = 2  # 앞뒤 2장(1초) 중앙값
    med = np.array([np.median(xs[max(0, i - k): i + k + 1]) for i in range(len(xs))])
    out, cur = [], med[0]
    step = MAX_SPEED / ANALYSIS_FPS
    for (t, _), target in zip(positions, med):
        if abs(target - cur) > DEADZONE:
            cur += float(np.clip(target - cur, -step, step))
        out.append((t, cur))
    return out


def crop_x_expression(track: list[tuple[float, float]], scaled_w: int, out_w: int) -> str:
    """시간 t에 따른 crop x(픽셀) 식. 피사체가 가운데 오도록 하되 화면 밖으로 나가지 않게."""
    max_x = max(0, scaled_w - out_w)
    keys = []
    last_t = -1.0
    for t, c in track:
        if t - last_t >= KEYFRAME_SEC or not keys:
            keys.append((t, int(round(min(max_x, max(0.0, c * scaled_w - out_w / 2))))))
            last_t = t
    if len(keys) == 1:
        return str(keys[0][1])
    expr = str(keys[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(zip(keys, keys[1:]))):
        expr = f"if(lt(t,{t1:.2f}),{x0}+({x1 - x0})*(t-{t0:.2f})/{t1 - t0:.2f},{expr})"
    return f"if(lt(t,{keys[0][0]:.2f}),{keys[0][1]},{expr})"
