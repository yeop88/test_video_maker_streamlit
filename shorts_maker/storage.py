"""저장 공간: workdir에서 무엇이 얼마나 차지하는지, 다시 만들 수 있는 캐시 지우기. (streamlit을 쓰지 않음)

- 작업(업로드·분석·결과)은 「최근 작업」에서 하나씩 지운다.
- AI 장면 캐시, 미리보기 캐시는 지워도 필요할 때 다시 만든다.
- 양자화 저장본(workdir/models/<모델>_nf4)은 지우면 다음에 쓸 때 원본에서 다시 양자화한다 (약 2분).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import config


def folder_bytes(path: Path) -> tuple[int, int]:
    """(바이트, 파일 수). 링크는 따라가지 않는다 (Hugging Face 캐시의 snapshots는 blobs를 가리키는 링크)."""
    total = count = 0
    for root, _, files in os.walk(path):
        for name in files:
            f = Path(root) / name
            try:
                if not f.is_symlink():
                    total += f.stat().st_size
                    count += 1
            except OSError:
                pass
    return total, count


def hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


CLEARABLE = {
    "gen_cache": ("🎨 AI 장면·음악 캐시", "지우면 2단계 AI 장면 카드의 미리보기가 사라지고, 쇼츠를 만들 때 다시 생성합니다."),
    "preview": ("🖼️ 미리보기 캐시", "화면·자막 미리보기와 카드 세로 미리보기 그림. 지워도 볼 때 다시 그립니다."),
    "exports": ("📦 내보낸 zip", "「📦 내보내기」로 만든 zip (작업마다 마지막 것만 남김). 다운로드한 뒤에는 지워도 됩니다."),
}


def usage() -> list[dict]:
    """[{key, label, path, bytes, files, note, clear}] 큰 것부터. key가 models/<이름>이면 양자화 저장본 하나."""
    work = config.WORK_DIR
    rows = []
    projects = [d for d in work.iterdir() if (d / "state.json").exists()] if work.exists() else []
    size = [folder_bytes(d) for d in projects]
    rows.append({"key": "projects", "label": f"📂 작업 {len(projects)}개 (올린 영상·분석·결과)", "path": work,
                 "bytes": sum(s[0] for s in size), "files": sum(s[1] for s in size),
                 "note": "「📂 최근 작업」의 🗑로 하나씩 지웁니다.", "clear": False})
    for key, (label, note) in CLEARABLE.items():
        b, n = folder_bytes(work / key)
        rows.append({"key": key, "label": label, "path": work / key, "bytes": b, "files": n, "note": note, "clear": True})
    models = work / "models"
    for d in sorted(models.iterdir()) if models.exists() else []:
        if d.is_dir():
            b, n = folder_bytes(d)
            rows.append({"key": f"models/{d.name}", "label": f"🧠 양자화 저장본 {d.name}", "path": d, "bytes": b, "files": n,
                         "note": "지우면 다음에 이 모델을 쓸 때 원본에서 다시 양자화합니다 (약 2분, 원본이 Hugging Face 캐시에 있어야 함).",
                         "clear": True})
    for key, label in (("music", "🎵 배경음악 보관함"), ("voices", "🎙️ 목소리 보관함"), ("fonts", "🔤 글꼴 보관함")):
        b, n = folder_bytes(work / key)
        rows.append({"key": key, "label": label, "path": work / key, "bytes": b, "files": n,
                     "note": "설정 팝업에서 하나씩 지웁니다.", "clear": False})
    hf = hf_cache_dir()
    b, n = folder_bytes(hf) if hf.exists() else (0, 0)
    rows.append({"key": "hf", "label": "🤗 Hugging Face 모델 캐시 (앱 밖)", "path": hf, "bytes": b, "files": n,
                 "note": "AI 생성 모델 원본. 지우려면 `hf cache delete` 명령을 쓰세요 (다시 받으려면 수십 GB).", "clear": False})
    return sorted(rows, key=lambda r: -r["bytes"])


def clear(key: str) -> int:
    """다시 만들 수 있는 캐시를 지우고 줄어든 바이트를 돌려준다. 쓰는 중인 파일은 건너뛴다."""
    work = config.WORK_DIR
    if key in CLEARABLE:
        target = work / key
    elif key.startswith("models/") and "/" not in key[7:] and key[7:] not in ("", ".", ".."):
        target = work / "models" / key[7:]
    else:
        raise ValueError(f"지울 수 없는 항목: {key}")
    if not target.exists():
        return 0
    before = folder_bytes(target)[0]
    # 먼저 이름을 바꾼다: 생성 프로세스가 파일을 쓰고 있으면 여기서 실패해서 반쯤 지워진 모델이 남지 않는다
    trash = target.with_name(target.name + ".deleting")
    shutil.rmtree(trash, ignore_errors=True)
    try:
        target.rename(trash)
    except OSError as e:
        raise RuntimeError("다른 작업이 쓰고 있어 지우지 못했습니다. 작업 큐가 쉴 때 다시 시도하세요.") from e
    shutil.rmtree(trash, ignore_errors=True)
    if key in CLEARABLE:
        target.mkdir(parents=True, exist_ok=True)
    return before


def disk_free() -> tuple[int, int]:
    """workdir가 있는 드라이브의 (남은 바이트, 전체 바이트)."""
    u = shutil.disk_usage(config.WORK_DIR if config.WORK_DIR.exists() else config.WORK_DIR.parent)
    return u.free, u.total


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f}{unit}" if unit in ("B", "KB") else f"{n:,.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"
