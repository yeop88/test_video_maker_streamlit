"""Whisper 음성 인식."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .media import load_audio

ASR_SAMPLE_RATE = 16000


@dataclass
class Segment:
    start: float
    end: float
    text: str


def load_whisper():
    import torch
    from transformers import pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    return pipeline("automatic-speech-recognition", model=config.WHISPER_MODEL, dtype=dtype, device=device)


def detect_language(asr_pipe, audio) -> tuple[str, str]:
    """앞 30초로 언어를 감지해 (코드, 이름)을 돌려준다. 예: ("en", "english")"""
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    features = asr_pipe.feature_extractor(
        audio[: 30 * ASR_SAMPLE_RATE], sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(asr_pipe.model.device, asr_pipe.model.dtype)
    token_id = asr_pipe.model.detect_language(features)[0]
    code = asr_pipe.tokenizer.decode(token_id).strip("<|>")
    return code, LANGUAGES.get(code, code)


def transcribe(asr_pipe, audio_path: Path) -> tuple[list[Segment], str]:
    """구간별 텍스트와 감지된 언어 이름(예: "english")을 돌려준다."""
    audio = load_audio(audio_path, ASR_SAMPLE_RATE, 1)[:, 0]
    # 음악·효과음만 있는 영상에서 Whisper는 "Thank you." 같은 말을 지어낸다.
    # 목소리 감지(Silero VAD)로 사람 목소리가 있는 구간을 먼저 찾고, 없으면 인식하지 않는다.
    voices = voice_regions(audio)
    if not voices:
        return [], ""
    voiced = np.concatenate([audio[int(a * ASR_SAMPLE_RATE) : int(b * ASR_SAMPLE_RATE)] for a, b in voices])
    code, language = detect_language(asr_pipe, voiced)  # 앞부분이 음악이어도 목소리로 언어를 판단

    # Whisper는 한 번에 30초까지만 듣는다. transformers의 chunk_length_s는 조각을 겹쳐 인식한 뒤
    # 이어 붙이는데, 경계의 대사가 두 번 들어가거나 통째로 빠지는 일이 있었다.
    # 그래서 조용한 지점에서 겹치지 않게 직접 자르고, 조각마다 따로 인식한다.
    words = _recognize_words(asr_pipe, audio, split_at_silence(audio), code)

    # Whisper는 가끔 짧은 문장을 통째로 건너뛴다. 말소리는 있는데 받아쓴 단어가 거의 없는
    # 구간은 그 부분만 잘라 다시 인식해서 채워 넣는다.
    pad = int(0.3 * ASR_SAMPLE_RATE)
    missing = [
        (max(0, int(a * ASR_SAMPLE_RATE) - pad), min(len(audio), int(b * ASR_SAMPLE_RATE) + pad))
        for a, b in voices
        if _coverage(a, b, words) < 0.3
    ]
    if missing:
        recovered = [
            w for w in _recognize_words(asr_pipe, audio, missing, code)
            # 이미 받아쓴 단어와 겹치는 것(앞뒤 여유 구간에서 다시 들린 단어)과 환각 문구는 버린다
            if _coverage(w[0], max(w[1], w[0] + 0.01), words) < 0.5 and not _is_hallucination(w[2])
        ]
        words = sorted(words + recovered)
    # 목소리 구간 밖에서 받아쓴 단어는 배경음을 말로 잘못 들은 것
    words = [w for w in words if _coverage(w[0], max(w[1], w[0] + 0.01), [(a, b, "") for a, b in voices]) >= 0.3]

    segments: list[Segment] = []
    for seg in group_sentences(words):
        if segments and segments[-1].text == seg.text:  # 무음 구간에서 같은 문장을 반복하는 환각 제거
            continue
        segments.append(seg)
    return snap_to_speech(merge_short(segments), audio), language


def _recognize_words(asr_pipe, audio: np.ndarray, pieces: list[tuple[int, int]], code: str):
    """샘플 범위 조각들을 인식해 (시작, 끝, 단어) 목록을 원본 시간 기준으로 돌려준다."""
    outputs = asr_pipe(
        [{"raw": audio[a:b], "sampling_rate": ASR_SAMPLE_RATE} for a, b in pieces],
        batch_size=8,
        # 문장 단위 타임스탬프는 부정확해서, 단어 단위로 받아 문장으로 직접 묶는다
        return_timestamps="word",
        # 언어를 고정해야 조각마다 언어를 잘못 바꿔 인식하지 않는다
        generate_kwargs={"task": "transcribe", "language": code},
    )
    words = []
    for (a, b), out in zip(pieces, outputs):
        offset, piece_end = a / ASR_SAMPLE_RATE, b / ASR_SAMPLE_RATE
        for w in out.get("chunks") or []:
            start, end = w["timestamp"]
            if start is None or not w["text"].strip():
                continue
            end = min(offset + end if end is not None else piece_end, piece_end)
            # 앞 문장을 건너뛰면 다음 첫 단어가 그 자리까지 늘어나 잡힌다 (예: "처음에는" 4글자가 4.4초).
            # 끝 시간은 정확하므로, 글자 수에 비해 지나치게 긴 단어는 끝 기준으로 줄인다.
            # (일본어·중국어는 띄어쓰기가 없어 문장 전체가 한 단어로 오므로 글자 수로 판단)
            chars = len(re.sub(r"[\s\W]", "", w["text"]))
            start = max(offset + start, end - max(MAX_WORD_SEC, SEC_PER_CHAR * chars))
            words.append((start, end, w["text"]))
    return words


MAX_WORD_SEC = 1.2
SEC_PER_CHAR = 0.35  # 아주 느린 말도 글자당 이 정도를 넘지 않는다


def _coverage(a: float, b: float, words) -> float:
    """말소리 구간 [a, b] 중 받아쓴 단어가 덮는 비율."""
    covered = sum(max(0.0, min(b, we) - max(a, ws)) for ws, we, _ in words)
    return covered / (b - a)


# 무음·잡음에서 Whisper가 지어내는 대표적인 문구 (다시 인식한 구간에서만 걸러낸다)
HALLUCINATIONS = {"감사합니다", "시청해주셔서감사합니다", "thankyou", "thanksforwatching", "ご視聴ありがとうございました"}


def _is_hallucination(text: str) -> bool:
    return re.sub(r"[\s\W]", "", text.lower()) in HALLUCINATIONS


FRAME_SEC = 0.02


def frame_rms(audio: np.ndarray) -> np.ndarray:
    """20ms 단위 소리 크기."""
    hop = int(FRAME_SEC * ASR_SAMPLE_RATE)
    frames = len(audio) // hop
    return np.sqrt((audio[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1))


def split_at_silence(audio: np.ndarray, max_sec: float = 28.0, min_sec: float = 15.0) -> list[tuple[int, int]]:
    """음성을 max_sec 이하 조각(샘플 범위)으로 나눈다. 각 조각은 min_sec~max_sec 사이의 가장 조용한 곳에서 끊는다."""
    rms = frame_rms(audio)
    # 0.3초 동안의 평균 크기가 가장 작은 지점 = 말과 말 사이
    smooth = np.convolve(rms, np.ones(15) / 15, mode="same")
    pieces, start = [], 0
    total = len(audio)
    while total - start > max_sec * ASR_SAMPLE_RATE:
        lo = int((start / ASR_SAMPLE_RATE + min_sec) / FRAME_SEC)
        hi = int((start / ASR_SAMPLE_RATE + max_sec) / FRAME_SEC)
        cut = (lo + int(np.argmin(smooth[lo:hi]))) * int(FRAME_SEC * ASR_SAMPLE_RATE)
        pieces.append((start, cut))
        start = cut
    pieces.append((start, total))
    return pieces


def speech_threshold(rms: np.ndarray) -> float:
    """배경음이 깔린 영상에서도 말소리만 넘도록, 잡음 바닥과 큰 소리 사이의 25% 지점."""
    noise, loud = np.percentile(rms, 10), np.percentile(rms, 95)
    return noise + 0.25 * (loud - noise)


_vad_model = None


def voice_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """사람 목소리가 있는 구간(초) 목록. 소리 크기가 아니라 Silero VAD로 판단해서 배경음악은 제외된다."""
    global _vad_model
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _vad_model is None:
        _vad_model = load_silero_vad()  # 패키지에 들어 있는 작은 모델 (다운로드 없음)
    stamps = get_speech_timestamps(
        torch.from_numpy(audio), _vad_model, sampling_rate=ASR_SAMPLE_RATE, return_seconds=True
    )
    return [(float(t["start"]), float(t["end"])) for t in stamps]


def snap_to_speech(segments: list[Segment], audio: np.ndarray, max_shift: float = 1.5) -> list[Segment]:
    """구간 시작을 실제로 말소리가 시작되는 지점으로 옮긴다.

    단어 타임스탬프는 첫 단어 시작을 음성 조각 맨 앞(0초 등)으로 잡는 경우가 있어서,
    그대로 쓰면 화자가 말하기 전에 더빙이 먼저 나온다.
    """
    rms = frame_rms(audio)
    frames = len(rms)
    if frames == 0:
        return segments
    threshold = speech_threshold(rms)

    snapped = []
    for seg in segments:
        a = int(seg.start / FRAME_SEC)
        b = min(int(min(seg.start + max_shift, seg.end) / FRAME_SEC), frames)
        above = np.flatnonzero(rms[a:b] >= threshold)
        start = (a + above[0]) * FRAME_SEC if above.size else seg.start
        snapped.append(Segment(max(seg.start, start - 0.05), seg.end, seg.text))
    return snapped


SENTENCE_END = (".", "?", "!", "。", "？", "！", "…")


def group_sentences(
    words: list[tuple[float, float, str]], pause: float = 0.8, soft_pause: float = 0.4, max_len: float = 12.0
) -> list[Segment]:
    """단어를 더빙 대사 단위로 묶는다.

    문장부호에서 끊고, 문장부호가 없는 언어(일본어 등)는 말 사이의 쉼으로 끊는다.
    """
    segments: list[Segment] = []
    current: list[tuple[float, float, str]] = []

    def flush():
        text = "".join(w[2] for w in current).strip()
        if text and current[-1][1] - current[0][0] >= 0.2:
            segments.append(Segment(current[0][0], current[-1][1], text))
        current.clear()

    for word in words:
        if current:
            gap = word[0] - current[-1][1]
            length = current[-1][1] - current[0][0]
            if gap >= pause or (gap >= soft_pause and length >= 3.0) or length >= max_len:
                flush()
        current.append(word)
        if word[2].strip().endswith(SENTENCE_END):
            flush()
    if current:
        flush()
    return segments


def merge_short(segments: list[Segment], min_dur: float = 1.0, max_gap: float = 0.3) -> list[Segment]:
    """너무 짧은 구간은 바로 뒤 구간과 합친다 (더빙이 뚝뚝 끊기지 않게)."""
    merged: list[Segment] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if prev and prev.end - prev.start < min_dur and seg.start - prev.end <= max_gap:
            merged[-1] = Segment(prev.start, seg.end, f"{prev.text} {seg.text}")
        else:
            merged.append(seg)
    return merged

