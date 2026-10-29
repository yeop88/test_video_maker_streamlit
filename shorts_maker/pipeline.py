"""영상 한 개를 처리하는 단계별 작업. app.py는 화면만 담당하고 실제 처리는 여기서 한다.

1단계 analyze: 음성 인식 + 장면 캡처·설명 → 타임라인
2단계 plan:    LLM이 쇼츠에 쓸 구간과 구간별 나레이션 대본을 정한다 (사용자가 표에서 수정)
3단계 make_short: 나레이션 음성 합성 → 구간 길이 맞추기 → 자르고 잇기 → 원본 소리 줄여 섞기 → 세로 변환 + 자막
"""
from __future__ import annotations

import hashlib
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from PIL import Image

from . import config, extras, llm, media, mixing, music, reframe, subtitles
from .config import MIX_SAMPLE_RATE as SR

LEAD_SEC = 0.3  # 구간이 시작되고 나레이션이 나오기까지 쉼
MAX_HOLD_SEC = 4.0  # 나레이션이 원본 끝을 넘을 때 마지막 화면을 멈춰 늘리는 최대 길이
TAIL_SEC = 0.4  # 나레이션이 끝나고 다음 구간으로 넘어가기까지 쉼
MIN_CLIP_SEC = 1.5
MAX_CLIP_SEC = 15.5  # 이보다 긴 구간은 나눈다 (긴 구간 하나에 항목이 여러 개면 나레이션이 화면과 어긋나기 쉬움)


@dataclass
class Job:
    dir: Path
    video: Path  # 분석할 구간만 자른 영상
    audio: Path  # 원본 오디오 (44.1kHz 스테레오)
    duration: float
    language: str = ""
    segments: list[dict] = field(default_factory=list)  # 원본 대사: start, end, text
    voices: list[tuple[float, float]] = field(default_factory=list)  # 목소리가 있는 구간 (VAD)
    frames: list[dict] = field(default_factory=list)  # 장면: t, path, caption


Progress = Callable[[int, int], None]


def _no_progress(done: int, total: int) -> None:
    pass


def save_upload(upload_id: str, filename: str, data: bytes) -> Path:
    folder = config.WORK_DIR / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"source{Path(filename).suffix.lower()}"
    if not path.exists():
        path.write_bytes(data)
    return path


def save_voice_sample(filename: str, data: bytes) -> Path:
    """Chatterbox용 목소리 샘플을 WAV로 바꿔 저장한다 (같은 파일이면 재사용)."""
    folder = config.WORK_DIR / "voices"
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(data).hexdigest()[:12]
    wav = folder / f"{key}.wav"
    if not wav.exists():
        raw = folder / f"{key}{Path(filename).suffix.lower()}"
        raw.write_bytes(data)
        media.extract_audio(raw, wav, sr=24000, channels=1)
    return wav


def prepare_job(source: Path, start: float, end: float) -> Job:
    folder = source.parent / "job"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    video = folder / "input.mp4"
    media.trim(source, video, start, end)
    info = media.probe(video)
    audio = folder / "audio.wav"
    if info.has_audio:
        media.extract_audio(video, audio)
    else:  # 소리 없는 영상도 나레이션은 입힐 수 있으므로 무음 트랙을 만든다
        sf.write(audio, np.zeros((int(info.duration * SR), 2), np.float32), SR)
    return Job(dir=folder, video=video, audio=audio, duration=info.duration)


# ───────────────────────── 1단계: 영상 이해 ─────────────────────────


def transcribe(job: Job, asr_pipe) -> None:
    from . import asr

    wav = media.load_audio(job.audio, asr.ASR_SAMPLE_RATE, 1)[:, 0]
    if wav.size == 0 or float(np.abs(wav).max()) < 0.01:  # 무음이면 인식하지 않음
        return
    job.voices = asr.voice_regions(wav)
    segments, language = asr.transcribe(asr_pipe, job.audio)
    job.language = language
    job.segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]


def capture_frames(job: Job) -> None:
    interval = max(config.MIN_FRAME_INTERVAL, job.duration / config.MAX_FRAMES)
    folder = job.dir / "frames"
    folder.mkdir(exist_ok=True)
    job.frames = []
    t = interval / 2
    while t < job.duration:
        path = folder / f"{len(job.frames):03d}.jpg"
        media.extract_frame(job.video, t, path, config.FRAME_WIDTH)
        if path.exists():
            job.frames.append({"t": round(t, 2), "path": path, "caption": ""})
        t += interval


def describe_frames(job: Job, model: str, on_progress: Progress = _no_progress) -> None:
    for i, frame in enumerate(job.frames):
        frame["caption"] = llm.describe_frame(model, frame["path"])
        on_progress(i + 1, len(job.frames))


def timeline(job: Job) -> str:
    return llm.timeline_text(job.frames, job.segments)


# ───────────────────────── 2단계: 쇼츠 구성 ─────────────────────────


def plan(job: Job, model: str, target_sec: float, style: str,
         span: tuple[float, float] | None = None, focus: str = "") -> tuple[str, list[dict]]:
    """(제목, 구간 목록)을 돌려준다. 구간은 start, end, narration.
    span을 주면 원본의 그 범위 안에서만 고른다 (한 영상을 여러 편으로 나눈 경우). focus는 그 편의 주제.

    구간 고르기와 나레이션 쓰기를 나눈다. 한 번에 시키면 LLM이 다른 구간의 내용을 엉뚱한 화면에
    붙이는 일이 있어서, 나레이션은 구간마다 그 시간의 장면·대사만 보여 주고 쓰게 한다.
    """
    lo, hi = span or (0.0, job.duration)
    text = llm.timeline_text(job.frames, job.segments, lo, hi)
    selection = llm.select_clips(model, text, job.duration, target_sec, span=span, focus=focus)
    chosen = sorted(selection.clips, key=lambda c: c.start)
    clips = [{"start": max(lo, c.start), "end": min(hi, c.end), "narration": ""} for c in chosen]
    clips = normalize_clips(job, clips)
    # 다시 요청해도 길이가 크게 틀리면 코드로 맞춘다
    total = sum(c["end"] - c["start"] for c in clips)
    if total > 1.3 * target_sec:  # 구간 끝을 같은 비율로 줄인다
        ratio = target_sec / total
        for c in clips:
            c["end"] = c["start"] + max(MIN_CLIP_SEC, (c["end"] - c["start"]) * ratio)
        clips = normalize_clips(job, clips)
    elif total < 0.75 * target_sec:  # 구간마다 앞뒤 빈 곳으로 똑같이 늘린다
        share = (target_sec - total) / len(clips)
        for i, c in enumerate(clips):
            after = (clips[i + 1]["start"] if i + 1 < len(clips) else hi) - c["end"]
            before = c["start"] - (clips[i - 1]["end"] if i > 0 else lo)
            grow_end = min(max(0.0, after), share)
            c["end"] += grow_end
            c["start"] -= min(max(0.0, before), share - grow_end)
        clips = normalize_clips(job, clips)
    clips = add_missing_items(job, clips, (lo, hi))
    return selection.title, narrate(job, split_long_clips(job, clips), model, style)


def plan_series(job: Job, model: str, count: int, target_sec: float, style: str,
                on_progress: Progress = _no_progress) -> list[dict]:
    """한 영상에서 서로 다른 주제의 쇼츠 count편을 구성한다. [{id, title, range, focus, clips}]"""
    import uuid

    if count <= 1:
        title, clips = plan(job, model, target_sec, style)
        return [{"id": uuid.uuid4().hex[:8], "title": title, "range": [0.0, job.duration], "focus": "", "clips": clips}]
    topics = llm.suggest_topics(model, timeline(job), job.duration, count, target_sec)
    shorts = []
    for i, topic in enumerate(topics):
        title, clips = plan(job, model, target_sec, style, span=(topic.start, topic.end), focus=topic.angle)
        if clips:
            shorts.append({"id": uuid.uuid4().hex[:8], "title": title or topic.title,
                           "range": [topic.start, topic.end], "focus": topic.angle, "clips": clips})
        on_progress(i + 1, len(topics))
    return shorts


def add_missing_items(job: Job, clips: list[dict], span: tuple[float, float] | None = None) -> list[dict]:
    """번호 붙은 목록 영상(화면 "TIP 2", 대사 "Second")에서 고른 구간에 빠진 번호가 있으면 그 부분을 구간으로 넣는다.
    LLM이 핵심 내용을 다 담았다고 답하고도 둘째 팁 구간을 빼먹은 일이 있었다."""
    lo, hi = span or (0.0, job.duration)
    where: dict[int, list[float]] = {}
    for f in job.frames:
        if not lo <= f["t"] <= hi:
            continue
        for n in llm.excerpt_ordinals(f["caption"]):
            where.setdefault(n, []).append(f["t"])
    for s in job.segments:
        if s["end"] < lo or s["start"] > hi:
            continue
        for n in llm.excerpt_ordinals(s["text"]):
            where.setdefault(n, []).extend([s["start"], s["end"]])
    items = sorted(where)
    if len(items) < 2 or items != list(range(items[0], items[0] + len(items))):  # 이어지는 번호가 아니면 목록이 아님
        return clips
    covered = set()
    for c in clips:
        covered |= llm.excerpt_ordinals(llm.timeline_text(job.frames, job.segments, c["start"], c["end"]))
    added = []
    for n in items:
        if n in covered:
            continue
        a, b = min(where[n]) - 0.5, max(where[n]) + 1.5
        b = min(b, a + 12.0)
        if any(c["start"] < b and a < c["end"] for c in clips + added):  # 겹치면 넣지 않음
            continue
        added.append({"start": max(lo, a), "end": min(hi, b), "narration": ""})
    if not added:
        return clips
    return normalize_clips(job, sorted(clips + added, key=lambda c: c["start"]))


def split_long_clips(job: Job, clips: list[dict]) -> list[dict]:
    """MAX_CLIP_SEC보다 긴 구간을 가운데에서 가장 가까운 대사 사이(말이 없는 곳)에서 나눈다.
    LLM에게 나누라고 다시 요청해도 긴 구간을 고르는 일이 있어서 코드로 마무리한다."""
    out = []
    queue = list(clips)
    while queue:
        clip = queue.pop(0)
        a, b = clip["start"], clip["end"]
        if b - a <= MAX_CLIP_SEC:
            out.append(clip)
            continue
        # 후보: 문장 사이 빈틈의 가운데. 없으면 그냥 가운데
        spans = sorted(_sentence_spans(job))
        gaps = [(x[1] + y[0]) / 2 for x, y in zip(spans, spans[1:]) if a + 3 < (x[1] + y[0]) / 2 < b - 3]
        cut = min(gaps, key=lambda g: abs(g - (a + b) / 2)) if gaps else (a + b) / 2
        second = {k: v for k, v in clip.items() if k != "uid"}  # 새 카드 ID는 화면에서 붙인다
        queue[:0] = [{**clip, "end": round(cut, 2)}, {**second, "start": round(cut, 2)}]
    return out


def narrate(job: Job, clips: list[dict], model: str, style: str) -> list[dict]:
    """구간마다 그 시간의 장면 설명과 대사를 보여 주고 나레이션을 쓴다."""
    items = [
        {"start": c["start"], "end": c["end"], "excerpt": llm.timeline_text(job.frames, job.segments, c["start"], c["end"])}
        for c in clips
    ]
    texts = llm.write_narrations(model, items, style)
    return [{**c, "narration": t} for c, t in zip(clips, texts)]


def _sentence_spans(job: Job) -> list[tuple[float, float]]:
    """Whisper 구간을 문장 단위로 다시 묶는다. Whisper는 실행마다 "공기가 정말 맑고,"처럼 쉼표에서
    구간을 나누기도 해서, 그 끝에 맞추면 문장 중간에서 끊겼다. 문장 끝 부호로 끝나지 않고 바로 다음 구간이
    이어지면(0.6초 이내) 한 문장으로 본다."""
    from .asr import SENTENCE_END

    spans: list[list[float]] = []
    open_sentence = False
    for seg in job.segments:
        if spans and open_sentence and seg["start"] - spans[-1][1] <= 0.6:
            spans[-1][1] = seg["end"]
        else:
            spans.append([seg["start"], seg["end"]])
        open_sentence = not seg["text"].strip().endswith(SENTENCE_END)
    return [tuple(s) for s in spans]


def _speech_spans(job: Job) -> list[tuple[float, float]]:
    return _sentence_spans(job) + list(job.voices)


def _safe_cut(job: Job, t: float, low: float, high: float, prefer_later: bool) -> float:
    """t가 대사 중간이면 low~high 안에서 가장 가까운 대사 경계로 옮긴다."""
    for _ in range(5):  # 옮긴 곳이 또 다른 대사 중간일 수 있음
        inside = next(((lo, hi) for lo, hi in _speech_spans(job) if lo < t < hi), None)
        if inside is None:
            return t
        options = [inside[1] + 0.15, inside[0] - 0.1]
        if not prefer_later:
            options.reverse()
        moved = next((x for x in options if low <= x <= high), None)
        if moved is None:
            return t
        t = moved
    return t


def normalize_clips(job: Job, clips: list[dict]) -> list[dict]:
    """영상 범위 안으로 맞추고, 원본 대사 중간에서 끊기지 않게 경계를 옮기고, 바로 앞 구간과 겹치면 합친다.

    순서는 그대로 둔다 (사용자가 카드 순서를 바꿀 수 있음). 시간 순 정렬은 AI 구성(plan)에서만 한다.
    """
    fixed = []
    for clip in clips:
        a = min(max(0.0, float(clip["start"])), job.duration)
        b = min(max(0.0, float(clip["end"])), job.duration)
        # 3초 안쪽이면 대사 경계까지 넓힌다. Whisper 문장 끝 시간은 실제보다 조금 이를 때가 있어서
        # 목소리 감지(VAD) 구간 경계에도 한 번 더 맞춘다.
        for lo, hi in _speech_spans(job):
            if lo < a < hi and a - lo <= 3.0:
                a = max(0.0, lo - 0.1)
            if lo < b < hi and hi - b <= 3.0:
                b = min(job.duration, hi + 0.15)
        if b - a >= MIN_CLIP_SEC:
            fixed.append({**clip, "start": round(a, 2), "end": round(b, 2)})

    merged: list[dict] = []
    for clip in fixed:
        if merged and merged[-1]["start"] <= clip["start"] < merged[-1]["end"]:
            prev = merged[-1]
            prev["end"] = max(prev["end"], clip["end"])
            prev["narration"] = f"{prev['narration']} {clip['narration']}".strip()
        else:
            merged.append(clip)
    return merged


# ───────────────────────── 3단계: 쇼츠 만들기 ─────────────────────────


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?…~])\s+", text.strip()) if s.strip()]


def fit_clips(job: Job, clips: list[dict], speech_sec: list[float]) -> list[tuple[float, float]]:
    """나레이션이 구간보다 길면 구간 끝을 늘리고 (다음 구간과 겹치지 않게), 모자라면 앞을 늘린다.

    그래도 넘친 나레이션은 다음 구간에서 이어서 나오므로, 넘친 만큼 다음 구간이 더 길어야 한다.
    """
    spans = []
    overflow = 0.0
    for i, (clip, speech) in enumerate(zip(clips, speech_sec)):
        a, b = clip["start"], clip["end"]
        start_at = max(LEAD_SEC, overflow - TAIL_SEC + config.NARRATION_GAP) if overflow > 0 else LEAD_SEC
        need = start_at + speech + TAIL_SEC if speech > 0 else overflow
        # 다음·앞 구간이 시간상으로도 뒤·앞에 있을 때만 그 경계를 넘지 않게 한다 (카드 순서를 바꾼 경우 대비)
        nxt = clips[i + 1]["start"] if i + 1 < len(clips) else job.duration
        limit = nxt if nxt > a else job.duration
        floor = spans[-1][1] if spans and spans[-1][1] <= a else 0.0
        if b - a < need:  # 늘린 끝이 원본 대사 중간이면 대사가 끝나는 곳까지 (안 되면 시작 전까지)
            b = _safe_cut(job, min(limit, a + need), a + MIN_CLIP_SEC, limit, prefer_later=True)
        if b - a < need:
            a = _safe_cut(job, max(floor, b - need), floor, b - MIN_CLIP_SEC, prefer_later=False)
        spans.append((a, b))
        overflow = max(0.0, need - (b - a))
    return spans


def make_short(
    job: Job,
    title: str,
    clips: list[dict],
    engine,
    original_gain: float,
    duck_gain: float,
    layout: str,
    font_size: int,
    margin_percent: int,
    color: str,
    on_progress: Progress = _no_progress,
    out_dir: Path | None = None,
    subtitle_style: str = "plain",
    highlight_color: str = "#FFE600",
    music_path: Path | None = None,
    music_gain: float = 0.25,
    music_duck: float = 0.4,
) -> tuple[Path, list[dict]]:
    """out_dir에 결과를 저장한다 (기본은 job.dir). 다시 분석해도 지워지지 않게 작업 폴더의 results/를 쓴다."""
    if not clips:
        raise RuntimeError("쇼츠에 넣을 구간이 없습니다.")

    # 1) 문장마다 나레이션 음성을 만든다 (문장 단위라 자막 시간을 정확히 맞출 수 있음)
    sentences = [split_sentences(c["narration"]) for c in clips]
    total_sentences = sum(len(s) for s in sentences)
    voices: list[list[np.ndarray]] = []
    timings: list[list[tuple[list | None, float]]] = []  # 문장마다 (엔진이 준 단어 시간, 앞에서 잘라 낸 초)
    done = 0
    for group in sentences:
        wavs, times = [], []
        for text in group:
            path = engine.synthesize_cached(text, job.dir / "tts")
            wav, lead = mixing.load_speech(path)
            wavs.append(wav)
            times.append((engine.word_timings(path), lead))
            done += 1
            on_progress(done, total_sentences)
        voices.append(wavs)
        timings.append(times)
    speech_sec = [
        sum(len(w) for w in wavs) / SR + config.NARRATION_GAP * max(0, len(wavs) - 1) for wavs in voices
    ]

    # 2) 나레이션 길이에 맞춰 구간을 늘린다
    spans = fit_clips(job, clips, speech_sec)
    total = sum(b - a for a, b in spans)

    # 3) 나레이션을 배치한다. 앞 나레이션이 길어 넘치면 다음 나레이션은 그 뒤에 이어서 나온다.
    # 마지막 나레이션이 원본 끝을 넘으면 마지막 화면을 멈춘 채로 최대 MAX_HOLD_SEC 늘린다 (나레이션을 자르지 않게)
    narration = np.zeros(int((total + MAX_HOLD_SEC) * SR) + 1, np.float32)
    rows, report = [], []
    offset, cursor = 0.0, 0.0
    for clip, (a, b), group, wavs, times, sec in zip(clips, spans, sentences, voices, timings, speech_sec):
        cursor = max(cursor, offset + LEAD_SEC)
        first = cursor
        for text, wav, (words, lead) in zip(group, wavs, times):
            mixing.place(narration, wav, cursor)
            dur = len(wav) / SR
            rows.append({"start": cursor, "end": cursor + dur, "text": text,
                         "words": subtitles.word_times(text, cursor, dur, words, lead)})
            cursor += dur + config.NARRATION_GAP
        end_sec = cursor - config.NARRATION_GAP if wavs else first
        if not wavs:
            status = "나레이션 없음"
        elif end_sec > total + MAX_HOLD_SEC:
            status = f"{end_sec - total - MAX_HOLD_SEC:.1f}초 잘림"
        elif end_sec > total:
            status = f"마지막 화면 {end_sec - total + TAIL_SEC:.1f}초 멈춤"
        elif end_sec > offset + (b - a):
            status = "다음 장면까지 이어짐"
        elif (b - a) - (clip["end"] - clip["start"]) > 0.05:
            status = "나레이션에 맞춰 늘림"
        else:
            status = "OK"
        report.append({
            "원본 구간": f"{a:.1f}~{b:.1f}초",
            "쇼츠 위치": f"{offset:.1f}~{offset + b - a:.1f}초",
            "나레이션(초)": round(sec, 1),
            "상태": status,
        })
        offset += b - a
    hold = min(MAX_HOLD_SEC, max(0.0, max((r["end"] for r in rows), default=0.0) + TAIL_SEC - total))
    hold = round(hold, 2) if hold > 0.05 else 0.0
    total += hold
    narration = narration[: int(total * SR) + 1]
    rows = [r for r in rows if r["start"] < total]
    for r in rows:
        r["end"] = min(r["end"], total)
        r["words"] = [(a, min(b, total)) for a, b in r.get("words", []) if a < total]

    # 4) 원본 소리를 같은 구간으로 잘라 이어 붙이고, 나레이션이 나올 때만 줄여서 섞는다
    original = media.load_audio(job.audio, SR, 2)
    original = np.concatenate([original[int(a * SR) : int(b * SR)] for a, b in spans])
    sf.write(job.dir / "narration.wav", narration, SR)  # 확인용
    mixed = job.dir / "mixed.wav"
    bed = music.bed(music_path, total) if music_path else None
    music_track = mixing.mix(original, narration, original_gain, duck_gain, mixed, bed, music_gain, music_duck)
    if music_track is not None:
        sf.write(job.dir / "music.wav", music_track, SR)  # 확인용 (섞인 배경음악만)

    # 5) 영상 자르고 잇기 → 세로 변환 + 자막
    cut = job.dir / "cut.mp4"
    media.cut_and_concat(job.video, spans, cut, hold_sec=hold)

    font_file, family = subtitles.find_korean_font()
    fonts = job.dir / "fonts"
    fonts.mkdir(exist_ok=True)
    if not (fonts / font_file.name).exists():
        shutil.copy(font_file, fonts / font_file.name)
    w, h = config.OUTPUT_SIZE
    cues = subtitles.build_cues(rows, subtitles.chars_per_line(font_size, w))
    subtitles.write_ass(
        cues, job.dir / "subs.ass", family, font_size, int(h * margin_percent / 100), color, (w, h),
        title=title, title_end=total,
        highlight=highlight_color if subtitle_style == "karaoke" else None,
    )
    out_dir = out_dir or job.dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shorts_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    crop_x = None
    if layout == "smart":  # 피사체 따라 자르기 (세로로 자를 여유가 있는 가로 영상일 때만)
        info = media.probe(cut)
        scaled_w = int(round(info.width * h / info.height / 2)) * 2
        if scaled_w > w + 8:
            track = reframe.smooth(reframe.subject_positions(cut, info.width, info.height))
            crop_x = reframe.crop_x_expression(track, scaled_w, w)
    media.render_short(cut, mixed, "subs.ass", "fonts", out, layout, (w, h), cwd=job.dir, crop_x=crop_x)
    # 함께 내보내는 파일: 같은 이름의 .srt(자막)와 .jpg(썸네일)
    extras.write_srt(rows, out.with_suffix(".srt"))
    frames = [
        Image.frombytes("RGB", (w, h), media.layout_frame(cut, total * (i + 0.5) / 6, layout, (w, h), crop_x)[: w * h * 3])
        for i in range(6)
    ]
    extras.make_thumbnail(frames, title or (clips[0]["narration"][:20] if clips else ""), font_file, out.with_suffix(".jpg"))
    return out, report
