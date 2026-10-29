"""배경음악 보관함: 올린 음악과 AI로 만든 음악을 workdir/music/에 모아 두고 여러 작업에서 다시 쓴다.

곡마다 <id>.wav(44.1kHz 스테레오)와 <id>.json(이름, 출처, 만든 때, AI 생성이면 프롬프트)을 저장한다.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import config, media
from .config import MIX_SAMPLE_RATE as SR

MUSIC_TYPES = ["mp3", "wav", "m4a", "ogg", "flac"]


def library_dir() -> Path:
    return config.WORK_DIR / "music"


def list_tracks() -> list[dict]:
    """[{id, name, source, created, seconds, prompt, path}] 최신 순."""
    folder = library_dir()
    tracks = []
    for meta in folder.glob("*.json") if folder.exists() else []:
        wav = meta.with_suffix(".wav")
        if not wav.exists():
            continue
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tracks.append({**info, "id": meta.stem, "path": wav})
    return sorted(tracks, key=lambda t: t.get("created", ""), reverse=True)


def track_path(track_id: str) -> Path | None:
    path = library_dir() / f"{track_id}.wav"
    return path if track_id and path.exists() else None


def _save(track_id: str, wav_src: Path, name: str, source: str, prompt: str = "") -> str:
    folder = library_dir()
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"{track_id}.wav"
    if wav_src != dst:
        media.extract_audio(wav_src, dst, sr=SR, channels=2)
    seconds = round(media.probe_audio_seconds(dst), 1)
    meta = {"name": name, "source": source, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": seconds, "prompt": prompt}
    (folder / f"{track_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return track_id


def add_upload(filename: str, data: bytes) -> str:
    """올린 음악 파일을 보관함에 넣고 곡 ID를 돌려준다 (같은 파일이면 재사용)."""
    track_id = "up_" + hashlib.sha1(data).hexdigest()[:10]
    if track_path(track_id):
        return track_id
    folder = library_dir()
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / f"{track_id}_src{Path(filename).suffix.lower()}"
    raw.write_bytes(data)
    try:
        return _save(track_id, raw, Path(filename).stem, "upload")
    finally:
        raw.unlink(missing_ok=True)


def add_generated(wav: Path, name: str, prompt: str) -> str:
    track_id = "ai_" + hashlib.sha1(f"{prompt}|{time.time()}".encode()).hexdigest()[:10]
    return _save(track_id, wav, name, "ai", prompt)


def delete(track_id: str) -> None:
    for suffix in (".wav", ".json"):
        (library_dir() / f"{track_id}{suffix}").unlink(missing_ok=True)


def bed(path: Path, seconds: float, fade_in: float = 0.5, fade_out: float = 1.5, crossfade: float = 1.0) -> np.ndarray:
    """배경음악을 쇼츠 길이로 맞춘 (샘플, 2) 배열. 짧으면 끝과 처음을 겹쳐(crossfade) 이어 반복한다."""
    music = media.load_audio(path, SR, 2)
    n = int(seconds * SR)
    if len(music) == 0:
        return np.zeros((n, 2), np.float32)
    xf = min(int(crossfade * SR), len(music) // 4)
    out = music.copy()
    while len(out) < n:
        if xf:
            ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
            tail = out[-xf:] * (1 - ramp) + music[:xf] * ramp
            out = np.concatenate([out[:-xf], tail, music[xf:]])
        else:
            out = np.concatenate([out, music])
    out = out[:n].copy()
    fi, fo = min(int(fade_in * SR), n // 2), min(int(fade_out * SR), n // 2)
    if fi:
        out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    if fo:
        out[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]
    return out.astype(np.float32)
