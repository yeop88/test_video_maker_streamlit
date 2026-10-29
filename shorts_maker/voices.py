"""목소리 보관함: 제로샷 음성 복제(Chatterbox)에 쓸 목소리 샘플을 이름 붙여 모아 둔다.

목소리마다 workdir/voices/library/<id>/에 sample.wav(24kHz 모노, 5~15초)와 meta.json을 둔다.
- upload / record: 본인 또는 동의받은 사람의 목소리 (동의 확인이 있어야 저장)
- virtual: Edge TTS 목소리의 음높이·음색(포먼트)을 바꿔 만든 가상 목소리 (실존 인물이 아님)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config, media

SR = 24000
MIN_SEC, MAX_SEC = 4.0, 15.0
VIRTUAL_SCRIPT = (
    "안녕하세요. 오늘도 함께해 주셔서 감사합니다. 지금부터 천천히, 그리고 또렷하게 이야기를 들려 드릴게요. "
    "어떤 장면이 기다리고 있을지 같이 살펴볼까요?"
)
SOURCES = {"upload": "올린 파일", "record": "녹음", "virtual": "가상 목소리"}


def library_dir() -> Path:
    return config.WORK_DIR / "voices" / "library"


def list_voices() -> list[dict]:
    """[{id, name, source, created, seconds, path, ...}] 최신 순."""
    voices = []
    for meta in library_dir().glob("*/meta.json") if library_dir().exists() else []:
        sample = meta.parent / "sample.wav"
        if not sample.exists():
            continue
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        voices.append({**info, "id": meta.parent.name, "path": sample})
    return sorted(voices, key=lambda v: v.get("created", ""), reverse=True)


def sample_path(voice_id: str) -> Path | None:
    path = library_dir() / voice_id / "sample.wav"
    return path if voice_id and path.exists() else None


def _clean_sample(src: Path) -> tuple[np.ndarray, float]:
    """24kHz 모노로 바꾸고 앞뒤 무음을 자르고 크기를 맞춘다. 15초가 넘으면 앞 15초만 쓴다."""
    wav = media.load_audio(src, SR, 1)[:, 0]
    frame = int(0.02 * SR)
    n = len(wav) // frame
    if n == 0:
        raise ValueError("소리가 없는 파일입니다.")
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > max(0.005, 0.05 * float(rms.max())))
    if loud.size == 0:
        raise ValueError("목소리가 들리지 않습니다. 더 크게 녹음해 주세요.")
    wav = wav[max(0, loud[0] * frame - frame * 5): (loud[-1] + 6) * frame][: int(MAX_SEC * SR)]
    seconds = len(wav) / SR
    if seconds < MIN_SEC:
        raise ValueError(f"목소리 샘플이 {seconds:.1f}초로 너무 짧습니다. {MIN_SEC:.0f}~{MAX_SEC:.0f}초로 말해 주세요.")
    peak = float(np.abs(wav).max())
    return (wav / peak * 0.9).astype(np.float32), round(seconds, 1)


def _save(name: str, wav: np.ndarray, seconds: float, source: str, extra: dict | None = None) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "", name).strip()[:30]
    if not name:
        raise ValueError("목소리 이름을 입력하세요.")
    voice_id = hashlib.sha1(wav.tobytes()).hexdigest()[:10]
    folder = library_dir() / voice_id
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / "sample.wav", wav, SR)
    meta = {"name": name, "source": source, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": seconds,
            **(extra or {})}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return voice_id


def add_sample(name: str, filename: str, data: bytes, source: str, consent: bool) -> str:
    """올린 파일이나 녹음을 보관함에 넣는다. 동의 확인이 없으면 저장하지 않는다."""
    if not consent:
        raise ValueError("본인 목소리이거나 목소리 주인의 동의를 받았는지 확인해 주세요.")
    folder = library_dir()
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / f"_upload{Path(filename).suffix.lower() or '.wav'}"
    raw.write_bytes(data)
    try:
        wav, seconds = _clean_sample(raw)
    finally:
        raw.unlink(missing_ok=True)
    return _save(name, wav, seconds, source, {"consent": True})


def create_virtual(name: str, base_voice: str, pitch: float, formant: float) -> str:
    """Edge TTS 목소리(base_voice)로 안내 문장을 읽힌 뒤 음높이(pitch 반음)·음색(formant 반음)을 바꿔
    실존 인물이 아닌 새 목소리 샘플을 만든다. Chatterbox가 이 샘플을 따라 해서 새 목소리로 말한다."""
    import edge_tts

    folder = library_dir()
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / "_virtual_base.mp3"
    asyncio.run(edge_tts.Communicate(VIRTUAL_SCRIPT, base_voice).save(str(base)))
    shifted = folder / "_virtual.wav"
    # 포먼트: 재생 속도를 바꿨다가(asetrate) 길이를 되돌리면(atempo) 음색과 음높이가 함께 바뀐다.
    # 음높이: 그다음 rubberband로 음색은 두고 음높이만 다시 맞춘다.
    f = 2 ** (formant / 12)
    p = 2 ** ((pitch - formant) / 12)
    chain = f"asetrate={SR * f:.0f},aresample={SR},atempo={1 / f:.5f},rubberband=pitch={p:.5f}:formant=preserved"
    try:
        media.run_ffmpeg(["-i", str(base), "-ac", "1", "-ar", str(SR), "-af", chain, str(shifted)])
        wav, seconds = _clean_sample(shifted)
    finally:
        base.unlink(missing_ok=True)
        shifted.unlink(missing_ok=True)
    return _save(name, wav, seconds, "virtual", {"base": base_voice, "pitch": pitch, "formant": formant})


def delete(voice_id: str) -> None:
    folder = library_dir() / voice_id
    if folder.parent == library_dir() and folder.exists():
        for f in folder.iterdir():
            f.unlink()
        folder.rmdir()
