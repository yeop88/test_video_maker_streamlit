"""한국어 나레이션 음성 합성 엔진."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .korean_numbers import read_numbers

EDGE_VOICES = {
    "선희 (여성)": "ko-KR-SunHiNeural",
    "인준 (남성)": "ko-KR-InJoonNeural",
    "현수 (남성, 다국어)": "ko-KR-HyunsuMultilingualNeural",
}


class _CachedEngine:
    """같은 설정과 문장이면 이미 만든 파일을 재사용한다 (대본 일부만 고쳐서 다시 만들 때 빠르게)."""

    ext = ".wav"

    def cache_key(self) -> str:
        raise NotImplementedError

    def synthesize(self, text: str, out_path: Path) -> None:
        raise NotImplementedError

    def synthesize_cached(self, text: str, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(f"{self.cache_key()}|{text}".encode()).hexdigest()[:16]
        path = cache_dir / f"{key}{self.ext}"
        if not path.exists():
            tmp = path.with_name(f"tmp_{path.name}")
            self.synthesize(text, tmp)
            words = tmp.with_suffix(".words.json")
            if words.exists():
                words.replace(path.with_suffix(".words.json"))
            tmp.replace(path)
        return path

    @staticmethod
    def word_timings(path: Path) -> list[dict] | None:
        """단어별 시간 [{start, end, text}] (음성 파일 기준 초). 엔진이 알려 주지 않으면 None."""
        side = path.with_suffix(".words.json")
        return json.loads(side.read_text(encoding="utf-8")) if side.exists() else None


class EdgeTTSEngine(_CachedEngine):
    """Microsoft Edge 온라인 TTS. GPU가 필요 없지만 인터넷 연결이 필요하다."""

    ext = ".mp3"

    def __init__(self, voice: str, rate_percent: int = 0):
        self.voice = voice
        self.rate = f"{rate_percent:+d}%"

    def cache_key(self) -> str:
        return f"edge2|{self.voice}|{self.rate}"  # edge2: 단어 시간(.words.json)을 함께 저장하는 형식

    def synthesize(self, text: str, out_path: Path) -> None:
        import edge_tts

        async def run() -> list[dict]:
            words = []
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, boundary="WordBoundary")
            with open(out_path, "wb") as f:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":  # 시간 단위는 100나노초
                        start = chunk["offset"] / 1e7
                        words.append({"start": start, "end": start + chunk["duration"] / 1e7, "text": chunk["text"]})
            return words

        words = asyncio.run(run())
        out_path.with_suffix(".words.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")


def load_chatterbox():
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    model.default_conds = model.conds  # 내장 기본 목소리 (목소리 샘플을 쓴 뒤 되돌릴 때 사용)
    return model


class ChatterboxEngine(_CachedEngine):
    """Chatterbox Multilingual. 목소리 샘플이 없으면 내장 기본 목소리를 쓴다."""

    def __init__(self, model, reference_wav: Path | None = None, exaggeration: float = 0.5, cfg_weight: float = 0.5):
        self.model = model
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.reference = reference_wav
        self._conditioned = False

    def cache_key(self) -> str:
        ref_hash = hashlib.sha1(self.reference.read_bytes()).hexdigest()[:12] if self.reference else "default"
        return f"chatterbox|{ref_hash}|{self.exaggeration}|{self.cfg_weight}"

    def synthesize(self, text: str, out_path: Path, max_attempts: int = 3) -> None:
        if not self._conditioned:  # 목소리 특징은 한 번만 추출 (모델은 앱 전체가 공유하므로 매번 설정)
            if self.reference:
                self.model.prepare_conditionals(str(self.reference), exaggeration=self.exaggeration)
            else:
                self.model.conds = self.model.default_conds
            self._conditioned = True

        # Chatterbox는 가끔 문장 끝에 없는 말을 덧붙인다 ("…해봐야겠어요. 네, 또.").
        # 문장 길이에 비해 음성이 지나치게 길면 다시 만들고, 가장 짧은 결과를 쓴다.
        chars = len(re.sub(r"[\s\W]", "", text))
        limit = 1.25 * (0.17 * chars + 0.4)  # 정상 음성은 글자당 0.14~0.20초
        best = None
        for _ in range(max_attempts):
            wav = self.model.generate(
                read_numbers(text),  # 숫자를 한글 읽기로 (자막은 그대로)
                language_id="ko",
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
            ).squeeze(0).numpy()
            loud = np.flatnonzero(np.abs(wav) > 0.01)
            seconds = (loud[-1] - loud[0]) / self.model.sr if loud.size else 0.0
            if best is None or seconds < best[0]:
                best = (seconds, wav)
            if seconds <= limit:
                break
        sf.write(out_path, best[1], self.model.sr, format="WAV")
