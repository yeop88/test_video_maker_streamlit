"""공용 작업 큐: 무거운 작업은 접속한 모든 사람이 같은 큐에 넣고, 앱 서버의 백그라운드 스레드 하나가 차례로 처리한다.

- GPU·CPU·메모리가 한정되어 있어 여럿이 동시에 눌러도 한 번에 한 작업만 돈다 (영상 분석, 구성, 나레이션,
  쇼츠 만들기, AI 장면·음악, 목소리 미리 듣기).
- 큐는 workdir/gen_queue.json에 저장한다 (새로고침·앱 재시작 뒤에도 남음, 하던 작업은 다시 대기로).
- 작업마다 넣은 사람(owner, 브라우저마다 다른 ID)을 적어 두고, 대기 중인 자기 작업만 빼거나 순서를 바꾼다.
- 실제 처리는 kind별 처리 함수(HANDLERS, tasks.py에서 등록)가 한다. 처리 함수는 report()로 진행 상황을 알리고,
  취소되면 report()가 Cancelled를 일으켜 멈춘다.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path

from . import config, generate

WAITING, RUNNING, DONE, FAILED, CANCELED = "waiting", "running", "done", "failed", "canceled"
STATUS_LABELS = {WAITING: "⏳ 대기", RUNNING: "🔄 진행 중", DONE: "✅ 완료", FAILED: "❌ 실패", CANCELED: "⏹ 취소"}
KIND_LABELS = {
    "analyze": "🔎 영상 분석 + 구성",
    "plan": "🧠 쇼츠 구성",
    "narrate": "✍️ 나레이션 다시 쓰기",
    "make": "🎬 쇼츠 만들기",
    "scene": "🎨 AI 장면",
    "music": "🎵 AI 배경음악",
    "voice": "🔈 목소리 미리 듣기",
}
DEFAULT_SEC = {"analyze": 240, "plan": 60, "narrate": 40, "make": 60, "scene": 30, "music": 60, "voice": 15}
KEEP_FINISHED = 60  # 끝난 항목은 최근 이만큼만 남긴다 (예상 시간 계산에도 씀)

Report = Callable[..., None]
HANDLERS: dict[str, Callable[[dict, Report], dict | None]] = {}

_lock = threading.RLock()
_runner: threading.Thread | None = None
_paused = False
_cancel = threading.Event()
_progress: dict[str, dict] = {}  # 항목 ID → {"note", "step", "total", "lines"} (진행 중인 항목, 저장하지 않음)


class Cancelled(Exception):
    pass


def queue_file() -> Path:
    return config.WORK_DIR / "gen_queue.json"


def _normalize(q: dict) -> dict:
    """이전 버전(AI 장면만 넣던 큐)의 항목을 지금 형식으로."""
    if "kind" not in q:
        q = {**q, "kind": "scene", "params": {"scene": {k: q.get(k) for k in ("prompt", "engine", "seed", "seconds")}},
             "owner": "", "project_dir": ""}
    q.setdefault("params", {})
    q.setdefault("result", None)
    return q


def _load() -> list[dict]:
    path = queue_file()
    if not path.exists():
        return []
    try:
        return [_normalize(q) for q in json.loads(path.read_text(encoding="utf-8"))]
    except (OSError, json.JSONDecodeError):
        return []


def _save(items: list[dict]) -> None:
    path = queue_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".gen_queue.json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def items() -> list[dict]:
    with _lock:
        return _load()


def get(item_id: str) -> dict | None:
    return next((q for q in items() if q["id"] == item_id), None)


def same_task(a: dict, kind: str, key: str) -> bool:
    return a["kind"] == kind and a.get("key") == key and a["status"] in (WAITING, RUNNING)


def add(kind: str, label: str, params: dict, owner: str = "", project: str = "", project_dir: str = "",
        key: str = "", card_uid: str = "") -> tuple[str, bool]:
    """작업을 큐 뒤에 넣는다. key가 같은 작업이 이미 대기·진행 중이면 넣지 않는다. (항목 ID, 새로 넣었는지)."""
    with _lock:
        queue = _load()
        if key:
            same = next((q for q in queue if same_task(q, kind, key)), None)
            if same:
                return same["id"], False
        item_id = uuid.uuid4().hex[:8]
        queue.append({
            "id": item_id, "kind": kind, "status": WAITING, "created": time.time(), "label": label,
            "params": params, "owner": owner, "project": project, "project_dir": project_dir, "key": key,
            "card_uid": card_uid, "error": "", "result": None,
        })
        _save(_trim(queue))
    ensure_runner()
    return item_id, True


def _trim(queue: list[dict]) -> list[dict]:
    finished = [q for q in queue if q["status"] in (DONE, FAILED, CANCELED)]
    drop = {q["id"] for q in finished[:-KEEP_FINISHED]} if len(finished) > KEEP_FINISHED else set()
    return [q for q in queue if q["id"] not in drop]


def remove(item_id: str) -> None:
    """대기·완료·실패 항목을 지운다. 진행 중이면 취소한다."""
    with _lock:
        queue = _load()
        target = next((q for q in queue if q["id"] == item_id), None)
        if target is None:
            return
        if target["status"] == RUNNING:
            target["status"] = CANCELED
            _save(queue)
            _cancel.set()
            if target["kind"] in ("scene", "music", "make"):
                generate.cancel()
            return
        _save([q for q in queue if q["id"] != item_id])


def move(item_id: str, step: int) -> None:
    """대기 중인 항목을 앞(-1)·뒤(+1)로 옮긴다 (대기 중인 항목끼리만)."""
    with _lock:
        queue = _load()
        waiting = [i for i, q in enumerate(queue) if q["status"] == WAITING]
        pos = next((k for k, i in enumerate(waiting) if queue[i]["id"] == item_id), None)
        if pos is None or not 0 <= pos + step < len(waiting):
            return
        a, b = waiting[pos], waiting[pos + step]
        queue[a], queue[b] = queue[b], queue[a]
        _save(queue)


def retry(item_id: str) -> None:
    with _lock:
        queue = _load()
        for q in queue:
            if q["id"] == item_id and q["status"] in (FAILED, CANCELED):
                q.update(status=WAITING, error="", created=time.time())
                queue.remove(q)
                queue.append(q)
                break
        _save(queue)
    ensure_runner()


def clear(statuses: tuple[str, ...], owner: str | None = None) -> None:
    """상태가 statuses인 항목을 지운다 (owner를 주면 그 사람 것만)."""
    with _lock:
        _save([q for q in _load() if not (q["status"] in statuses and (owner is None or q.get("owner") == owner))])


def set_paused(paused: bool) -> None:
    global _paused
    _paused = paused
    if not paused:
        ensure_runner()


def is_paused() -> bool:
    return _paused


def progress(item_id: str) -> dict:
    return _progress.get(item_id, {})


# ───────────── 예상 시간 ─────────────
def average_sec(queue: list[dict], kind: str) -> float:
    """최근 끝난 같은 종류 작업의 평균 시간 (없으면 기본값)."""
    spent = [q["finished"] - q["started"] for q in queue
             if q["kind"] == kind and q["status"] == DONE and q.get("started") and q.get("finished")][-5:]
    return sum(spent) / len(spent) if spent else DEFAULT_SEC.get(kind, 60)


def waits(queue: list[dict]) -> dict[str, float]:
    """대기·진행 중인 항목마다 시작(대기면)·끝(진행 중이면)까지 남은 예상 초."""
    out, ahead = {}, 0.0
    for q in queue:
        if q["status"] == RUNNING:
            left = max(5.0, average_sec(queue, q["kind"]) - (time.time() - q.get("started", time.time())))
            out[q["id"]] = left
            ahead += left
    for q in queue:
        if q["status"] == WAITING:
            out[q["id"]] = ahead
            ahead += average_sec(queue, q["kind"])
    return out


def position(queue: list[dict], item_id: str) -> int:
    """대기 순서 (1부터, 대기가 아니면 0)."""
    waiting = [q["id"] for q in queue if q["status"] == WAITING]
    return waiting.index(item_id) + 1 if item_id in waiting else 0


def for_project(project_dir: str) -> list[dict]:
    """이 작업(프로젝트)의 대기·진행 중인 큐 항목."""
    key = str(Path(project_dir).resolve())
    return [q for q in items() if q["status"] in (WAITING, RUNNING) and q.get("project_dir")
            and str(Path(q["project_dir"]).resolve()) == key]


def status_of_card(card_uid: str) -> dict | None:
    """카드의 가장 최근 AI 장면 큐 항목 (대기 순서 position 포함)."""
    queue = items()
    found = [q for q in queue if q.get("card_uid") == card_uid]
    if not found:
        return None
    return {**found[-1], "position": position(queue, found[-1]["id"])}


# ───────────── 처리 스레드 ─────────────
def _next() -> dict | None:
    with _lock:
        queue = _load()
        for q in queue:
            if q["status"] == WAITING:
                q["status"] = RUNNING
                q["started"] = time.time()
                _save(queue)
                return q
    return None


def _finish(item_id: str, status: str, error: str = "", result: dict | None = None) -> None:
    with _lock:
        queue = _load()
        for q in queue:
            if q["id"] == item_id and q["status"] == RUNNING:
                q.update(status=status, error=error[-1500:], finished=time.time(), result=result)
            elif q["id"] == item_id and q["status"] == CANCELED:
                q.update(finished=time.time())
        _save(queue)
    _progress.pop(item_id, None)


def _reporter(item_id: str) -> Report:
    def report(note: str | None = None, step: int | None = None, total: int | None = None, line: str | None = None):
        if _cancel.is_set():
            raise Cancelled()
        prog = _progress.setdefault(item_id, {"lines": []})
        if line:
            prog["lines"].append(line)
            prog.pop("step", None)
            prog.pop("total", None)
            prog["note"] = line
        if note is not None:
            prog["note"] = note
        if total:
            prog.update(step=step or 0, total=total)
    return report


def _loop() -> None:
    while True:
        item = None if _paused else _next()
        if item is None:
            generate.stop_idle()
            time.sleep(0.5)
            continue
        _cancel.clear()
        _progress[item["id"]] = {"lines": [], "note": "시작하는 중"}
        handler = HANDLERS.get(item["kind"])
        try:
            if handler is None:
                raise RuntimeError(f"처리할 수 없는 작업 종류입니다: {item['kind']}")
            result = handler(item, _reporter(item["id"]))
            _finish(item["id"], DONE, result=result)
        except (Cancelled, generate.Cancelled):
            _finish(item["id"], CANCELED)
        except Exception as e:  # 하나가 실패해도 큐는 계속
            detail = "".join(traceback.format_exception_only(type(e), e)).strip()
            _finish(item["id"], FAILED, f"{detail}\n\n{traceback.format_exc()[-1000:]}")


def ensure_runner() -> None:
    """큐를 처리하는 백그라운드 스레드를 (한 번만) 띄운다. 앱 재시작 전 하던 작업은 다시 대기로 돌린다."""
    global _runner
    with _lock:
        if _runner is not None and _runner.is_alive():
            return
        queue = _load()
        for q in queue:
            if q["status"] == RUNNING:
                q["status"] = WAITING
        _save(queue)
        _runner = threading.Thread(target=_loop, name="job-queue", daemon=True)
        _runner.start()
