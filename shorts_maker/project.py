"""작업(프로젝트) 저장과 불러오기.

프로젝트 하나 = workdir/<id>/ 폴더 하나. 상태는 그 폴더의 state.json에 저장해서,
브라우저를 새로고침하거나 앱을 다시 켜도 분석 결과·구성·나레이션·만든 쇼츠를 이어서 쓸 수 있다.

- kind "upload": 올린 영상으로 만드는 작업. 폴더 이름은 영상 내용 해시(같은 영상을 다시 올리면 같은 작업)
- kind "ai": 영상 없이 주제만으로 만드는 작업
"""
from __future__ import annotations

import json
import shutil
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
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load(folder: Path) -> Project:
    data = json.loads((folder / STATE_FILE).read_text(encoding="utf-8"))
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


def delete(project_dir: Path) -> None:
    if project_dir.parent.resolve() == config.WORK_DIR.resolve() and (project_dir / STATE_FILE).exists():
        shutil.rmtree(project_dir)
