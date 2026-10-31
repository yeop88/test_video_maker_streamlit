"""작업(프로젝트) 저장과 불러오기.

프로젝트 하나 = workdir/<id>/ 폴더 하나. 상태는 그 폴더의 state.json에 저장해서,
브라우저를 새로고침하거나 앱을 다시 켜도 분석 결과·구성·나레이션·만든 쇼츠를 이어서 쓸 수 있다.

- kind "upload": 올린 영상으로 만드는 작업. 폴더 이름은 영상 내용 해시(같은 영상을 다시 올리면 같은 작업)
- kind "ai": 영상 없이 주제만으로 만드는 작업
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .pipeline import Job

STATE_FILE = "state.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Project:
    dir: Path
    kind: str  # "upload" | "ai"
    name: str  # 올린 파일 이름 또는 주제
    created: str = field(default_factory=now)
    updated: str = field(default_factory=now)
    source: str = ""  # 올린 영상 파일 이름 (dir 안)
    topic: str = ""  # AI 작업의 주제
    job: Job | None = None
    shorts: list[dict] = field(default_factory=list)  # [{id, title, clips: [...]}]
    results: list[dict] = field(default_factory=list)  # 만든 쇼츠 기록, 최신이 앞
    mtime: float = 0.0  # 마지막으로 읽거나 쓴 state.json의 수정 시각 (다른 곳에서 바꿨는지 알아보는 데 씀, 저장하지 않음)

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def source_path(self) -> Path | None:
        return self.dir / self.source if self.source else None


def _job_to_dict(job: Job) -> dict:
    return {
        "dir": str(job.dir),
        "video": str(job.video),
        "audio": str(job.audio),
        "duration": job.duration,
        "language": job.language,
        "segments": job.segments,
        "voices": [list(v) for v in job.voices],
        "frames": [{**f, "path": str(f["path"])} for f in job.frames],
    }


def _job_from_dict(d: dict) -> Job:
    return Job(
        dir=Path(d["dir"]),
        video=Path(d["video"]),
        audio=Path(d["audio"]),
        duration=float(d["duration"]),
        language=d.get("language", ""),
        segments=d.get("segments", []),
        voices=[tuple(v) for v in d.get("voices", [])],
        frames=[{**f, "path": Path(f["path"])} for f in d.get("frames", [])],
    )


def save(project: Project) -> None:
    project.updated = now()
    data = {
        "kind": project.kind,
        "name": project.name,
        "created": project.created,
        "updated": project.updated,
        "source": project.source,
        "topic": project.topic,
        "job": _job_to_dict(project.job) if project.job else None,
        "shorts": project.shorts,
        "results": project.results,
    }
    project.dir.mkdir(parents=True, exist_ok=True)
    path = project.dir / STATE_FILE
    tmp = path.with_name(f".{STATE_FILE}.tmp")
    with _lock_of(project.dir):
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        project.mtime = path.stat().st_mtime


_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock_of(folder: Path) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(str(Path(folder).resolve()), threading.RLock())


def update(folder: Path, change) -> Project:
    """저장된 최신 상태를 읽어 change(project)로 고치고 저장한다 (작업 큐가 쓰는 동안 화면에서 고친 내용을 덮어쓰지 않게)."""
    with _lock_of(folder):
        project = load(folder)
        change(project)
        save(project)
        return project


def changed(project: Project) -> bool:
    """화면에 불러온 뒤 다른 곳(작업 큐, 다른 브라우저)에서 저장했는지."""
    try:
        return (project.dir / STATE_FILE).stat().st_mtime != project.mtime
    except OSError:
        return True


def load(folder: Path) -> Project:
    path = folder / STATE_FILE
    mtime = path.stat().st_mtime
    data = json.loads(path.read_text(encoding="utf-8"))
    job = _job_from_dict(data["job"]) if data.get("job") else None
    if job and not job.video.exists():  # 중간 파일이 지워졌으면 분석부터 다시
        job = None
    return Project(
        dir=folder,
        kind=data.get("kind", "upload"),
        name=data.get("name", folder.name),
        created=data.get("created", now()),
        updated=data.get("updated", now()),
        source=data.get("source", ""),
        topic=data.get("topic", ""),
        job=job,
        shorts=data.get("shorts", []),
        results=[r for r in data.get("results", []) if Path(r.get("video", "")).exists()],
        mtime=mtime,
    )


def open_upload(upload_id: str, filename: str, data: bytes) -> tuple[Project, bool]:
    """올린 영상의 작업을 연다. (작업, 이전 상태를 불러왔는지)를 돌려준다."""
    folder = config.WORK_DIR / upload_id
    if (folder / STATE_FILE).exists():
        project = load(folder)
        if project.source_path and project.source_path.exists():
            return project, True
    folder.mkdir(parents=True, exist_ok=True)
    source = f"source{Path(filename).suffix.lower()}"
    if not (folder / source).exists():
        (folder / source).write_bytes(data)
    project = Project(dir=folder, kind="upload", name=filename, source=source)
    save(project)
    return project, False


def new_ai(topic: str) -> Project:
    folder = config.WORK_DIR / f"ai_{time.strftime('%Y%m%d_%H%M%S')}_{new_id()[:4]}"
    project = Project(dir=folder, kind="ai", name=topic.strip()[:40] or "AI 쇼츠", topic=topic)
    save(project)
    return project


def recent(limit: int = 20) -> list[dict]:
    """최근에 바뀐 작업 목록 (손상된 상태 파일은 건너뜀)."""
    items = []
    if not config.WORK_DIR.exists():
        return items
    for path in config.WORK_DIR.glob(f"*/{STATE_FILE}"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items.append({
            "dir": path.parent,
            "kind": data.get("kind", "upload"),
            "name": data.get("name", path.parent.name),
            "updated": data.get("updated", ""),
            "analyzed": bool(data.get("job")) or data.get("kind") == "ai",
            "shorts": len(data.get("shorts", [])),
            "results": len(data.get("results", [])),
        })
    return sorted(items, key=lambda x: x["updated"], reverse=True)[:limit]


# ───────────── 내보내기·가져오기 (zip) ─────────────
# state.json의 절대 경로를 "@project/<상대 경로>", "@gen_cache/<파일>"로 바꿔 담고, 가져올 때 새 위치로 되돌린다.
# AI 장면 카드의 미리보기(workdir/gen_cache)는 작업 폴더 밖에 있어서 zip의 gen_cache/에 함께 담는다.
EXPORT_VERSION = 1
STORED = {".mp4", ".mp3", ".m4a", ".png", ".jpg", ".jpeg", ".webp", ".zip", ".mov", ".mkv", ".webm"}  # 이미 압축된 파일
_GEN_IMAGE = re.compile(r"^(img_[0-9a-zA-Z]+)_[\d.]+s_m\d\.mp4$")


def _swap_paths(value, swap):
    """JSON 값 안의 문자열 경로를 swap(문자열) → 문자열로 바꾼다."""
    if isinstance(value, dict):
        return {k: _swap_paths(v, swap) for k, v in value.items()}
    if isinstance(value, list):
        return [_swap_paths(v, swap) for v in value]
    if isinstance(value, str):
        return swap(value)
    return value


def _inside(path: Path, folder: Path) -> Path | None:
    try:
        return path.resolve().relative_to(folder.resolve())
    except (ValueError, OSError):
        return None


def export_zip(project: Project, out: Path, include_results: bool = True, on_progress=None) -> Path:
    """작업 폴더(올린 영상, 분석, 구성, 결과)와 AI 장면을 zip 하나로. 결과 영상은 빼고 담을 수도 있다."""
    import zipfile

    folder = project.dir
    gen = config.WORK_DIR / "gen_cache"
    with _lock_of(folder):
        data = json.loads((folder / STATE_FILE).read_text(encoding="utf-8"))
    extra: set[Path] = set()

    def swap(text: str) -> str:
        if len(text) < 4 or not (":" in text[:3] or text.startswith("/")):
            return text
        path = Path(text)
        rel = _inside(path, folder)
        if rel is not None:
            return "@project/" + rel.as_posix()
        rel = _inside(path, gen)
        if rel is not None and path.exists():
            extra.add(path)
            base = _GEN_IMAGE.match(path.name)
            if base and (gen / f"{base.group(1)}.png").exists():  # 이미지 장면: 원본 그림도 (다른 길이로 다시 움직일 수 있게)
                extra.add(gen / f"{base.group(1)}.png")
            return "@gen_cache/" + rel.as_posix()
        return text

    data = _swap_paths(data, swap)
    if not include_results:
        data["results"] = []
    data["export"] = {"version": EXPORT_VERSION, "folder": folder.name, "exported": now()}
    skip = {(folder / STATE_FILE).resolve(), (folder / f".{STATE_FILE}.tmp").resolve()}
    if not include_results:
        skip |= {p.resolve() for p in (folder / "results").glob("*")} if (folder / "results").exists() else set()
    files = [p for p in folder.rglob("*") if p.is_file() and p.resolve() not in skip]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    total = len(files) + len(extra)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(STATE_FILE, json.dumps(data, ensure_ascii=False, indent=1))
        for i, path in enumerate(files + sorted(extra)):
            name = (f"files/{path.relative_to(folder).as_posix()}" if path in files
                    else f"gen_cache/{path.relative_to(gen).as_posix()}")
            z.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED)
            if on_progress:
                on_progress(i + 1, total)
    tmp.replace(out)
    return out


def import_zip(source: Path) -> Project:
    """내보낸 zip을 새 작업으로 가져온다. 같은 폴더 이름이 이미 있으면 뒤에 번호를 붙인다."""
    import zipfile

    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if STATE_FILE not in names:
            raise ValueError("이 앱에서 내보낸 작업 파일(zip)이 아닙니다 (state.json 없음).")
        data = json.loads(z.read(STATE_FILE).decode("utf-8"))
        info = data.pop("export", {})
        if int(info.get("version", 0)) > EXPORT_VERSION:
            raise ValueError("더 새로운 버전의 앱에서 내보낸 파일입니다. 앱을 업데이트하세요.")
        wanted = re.sub(r"[^0-9A-Za-z_\-]", "", str(info.get("folder", ""))) or f"import_{new_id()}"
        folder = config.WORK_DIR / wanted
        n = 2
        while folder.exists():
            folder = config.WORK_DIR / f"{wanted}_{n}"
            n += 1
        gen = config.WORK_DIR / "gen_cache"
        members = []
        for name in names:  # 폴더 밖으로 풀리는 이름(../, 절대 경로)은 거부
            if name == STATE_FILE or name.endswith("/"):
                continue
            prefix, _, rel = name.partition("/")
            root = {"files": folder, "gen_cache": gen}.get(prefix)
            target = (root / rel).resolve() if root and rel else None
            if target is None or _inside(target, root) is None:
                raise ValueError(f"작업 파일 안에 잘못된 경로가 있습니다: {name}")
            members.append((name, target))
        folder.mkdir(parents=True)
        try:
            for name, target in members:
                if target.exists() and name.startswith("gen_cache/"):
                    continue  # 같은 AI 장면이 이미 있음 (이름이 내용 해시)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def swap(text: str) -> str:
        if text.startswith("@project/"):
            return str(folder / text[len("@project/"):])
        if text.startswith("@gen_cache/"):
            return str(gen / text[len("@gen_cache/"):])
        return text

    data = _swap_paths(data, swap)
    data["name"] = data.get("name", folder.name)
    (folder / STATE_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return load(folder)


def delete(project_dir: Path) -> None:
    if project_dir.parent.resolve() == config.WORK_DIR.resolve() and (project_dir / STATE_FILE).exists():
        shutil.rmtree(project_dir)
