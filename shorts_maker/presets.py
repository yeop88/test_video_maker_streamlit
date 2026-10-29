"""사이드바 설정 프리셋: 저장, 불러오기, 삭제.

사용자 프리셋은 workdir/presets/<이름>.json에 저장한다.
화면 문구가 바뀌어도 깨지지 않도록 값은 코드값(엔진 "edge", 레이아웃 "blur", Edge 목소리 ID 등)으로 저장한다.
목소리 샘플 파일은 프리셋에 넣지 않는다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config
from .llm import STYLES
from .tts import EDGE_VOICES

# 숫자 설정의 (최솟값, 최댓값, 간격). app.py의 슬라이더도 이 값을 쓴다.
RANGES = {
    "target_sec": (15, 90, 5),
    "short_count": (1, 5, 1),
    "rate": (-20, 40, 5),
    "exaggeration": (0.25, 1.5, 0.05),
    "original_gain": (0.0, 1.0, 0.05),
    "duck_gain": (0.0, 1.0, 0.05),
    "bgm_gain": (0.0, 1.0, 0.05),
    "bgm_duck": (0.0, 1.0, 0.05),
    "font_size": (48, 110, 2),
    "margin_percent": (5, 50, 1),
}
CHOICES = {
    "style": list(STYLES),
    "engine": ["edge", "chatterbox"],
    "edge_voice": list(EDGE_VOICES.values()),
    "layout": ["blur", "crop", "smart"],
    "subtitle_style": ["plain", "karaoke"],
}

DEFAULTS = {
    "llm_model": config.DEFAULT_OLLAMA_MODEL,
    "style": "정보 전달 (차분한 설명)",
    "target_sec": 40,
    "short_count": 1,
    "engine": "edge",
    "edge_voice": "ko-KR-SunHiNeural",
    "rate": 10,
    "exaggeration": 0.5,
    "voice_id": "",  # 목소리 보관함의 목소리 ID (없음 = Chatterbox 기본 목소리)
    "original_gain": 0.8,
    "duck_gain": 0.2,
    "bgm_track": "",  # 배경음악 보관함의 곡 ID (없음 = "")
    "bgm_gain": 0.25,
    "bgm_duck": 0.4,
    "layout": "blur",
    "show_title": True,
    "font_size": 72,
    "margin_percent": 20,
    "color": "#FFFFFF",
    "subtitle_style": "plain",
    "highlight_color": "#FFE600",
}

# 기본 제공 프리셋 (덮어쓰거나 지울 수 없음). 빠진 값은 DEFAULTS를 따른다.
BUILTIN = {
    "기본": {},
    "감성 브이로그": {
        "style": "감성 스토리", "target_sec": 30, "rate": 0,
        "original_gain": 0.9, "duck_gain": 0.3, "font_size": 68, "margin_percent": 18,
    },
    "정보 요약 (빠르게)": {
        "style": "정보 전달 (차분한 설명)", "target_sec": 45, "edge_voice": "ko-KR-InJoonNeural", "rate": 20,
        "original_gain": 0.6, "duck_gain": 0.15, "font_size": 76, "color": "#FFE600",
    },
    "유쾌한 리뷰": {
        "style": "유쾌한 리뷰", "subtitle_style": "karaoke", "highlight_color": "#FFE600", "target_sec": 30, "edge_voice": "ko-KR-HyunsuMultilingualNeural", "rate": 15,
        "duck_gain": 0.25, "layout": "crop", "font_size": 80, "margin_percent": 25, "color": "#00E5FF",
    },
}

COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
BAD_NAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def preset_dir() -> Path:
    return config.WORK_DIR / "presets"


def _path(name: str) -> Path:
    return preset_dir() / f"{name}.json"


def is_builtin(name: str) -> bool:
    return name in BUILTIN


def list_presets() -> list[str]:
    """기본 제공 프리셋 다음에 사용자 프리셋(이름순)."""
    folder = preset_dir()
    user = sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []
    return list(BUILTIN) + [n for n in user if n not in BUILTIN]


def sanitize(values: dict) -> tuple[dict, list[str]]:
    """알려진 설정만 남기고 범위·선택지를 확인한다. (정리된 값, 버린 설정 이름)을 돌려준다.

    숫자는 범위를 벗어나면 범위 안으로 맞춘다. 모르는 설정이나 선택지에 없는 값은 버린다.
    """
    clean, dropped = {}, []
    for key, value in values.items():
        if key not in DEFAULTS:
            dropped.append(key)
            continue
        default = DEFAULTS[key]
        try:
            if key in RANGES:
                if isinstance(value, bool):
                    raise ValueError
                lo, hi, step = RANGES[key]
                number = min(hi, max(lo, float(value)))
                number = lo + round((number - lo) / step) * step  # 슬라이더 간격에 맞춤
                clean[key] = int(round(number)) if isinstance(default, int) else round(number, 2)
            elif key in CHOICES:
                if value not in CHOICES[key]:
                    raise ValueError
                clean[key] = value
            elif key in ("color", "highlight_color"):
                if not COLOR.match(str(value)):
                    raise ValueError
                clean[key] = str(value).upper()
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError
                clean[key] = value
            else:  # llm_model: 설치된 모델인지는 화면에서 확인
                clean[key] = str(value)
        except (TypeError, ValueError):
            dropped.append(key)
    return clean, dropped


def load(name: str) -> tuple[dict, list[str]]:
    """프리셋 값(DEFAULTS에 덮어씀)과 버린 설정 이름을 돌려준다."""
    if is_builtin(name):
        raw = BUILTIN[name]
    else:
        path = _path(name)
        if not path.exists():
            raise FileNotFoundError(f"프리셋 '{name}'을(를) 찾을 수 없습니다.")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"프리셋 '{name}' 파일이 손상되었습니다: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"프리셋 '{name}' 파일 형식이 올바르지 않습니다.")
    clean, dropped = sanitize(raw)
    return {**DEFAULTS, **clean}, dropped


def validate_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("프리셋 이름을 입력하세요.")
    if len(name) > 40:
        raise ValueError("프리셋 이름은 40자 이하로 해 주세요.")
    if BAD_NAME_CHARS.search(name) or name.startswith(".") or name.endswith("."):
        raise ValueError('프리셋 이름에 \\ / : * ? " < > | 문자는 쓸 수 없고, 마침표로 시작하거나 끝날 수 없습니다.')
    if name.upper() in WINDOWS_RESERVED:
        raise ValueError(f"'{name}'은(는) 파일 이름으로 쓸 수 없습니다.")
    if is_builtin(name):
        raise ValueError(f"'{name}'은(는) 기본 제공 프리셋이라 덮어쓸 수 없습니다. 다른 이름을 쓰세요.")
    return name


def save(name: str, values: dict) -> tuple[str, bool]:
    """(저장한 이름, 기존 프리셋을 덮어썼는지)를 돌려준다."""
    name = validate_name(name)
    clean, _ = sanitize(values)
    folder = preset_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = _path(name)
    existed = path.exists()
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return name, existed


def delete(name: str) -> None:
    if is_builtin(name):
        raise ValueError("기본 제공 프리셋은 지울 수 없습니다.")
    _path(name).unlink(missing_ok=True)
