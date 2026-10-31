"""공용 작업 큐: 무거운 작업은 접속한 모든 사람이 같은 큐에 넣고, 앱 서버의 백그라운드 스레드 하나가 차례로 처리한다.

- GPU·CPU·메모리가 한정되어 있어 여럿이 동시에 눌러도 한 번에 한 작업만 돈다 (영상 분석, 구성, 나레이션,
  쇼츠 만들기, AI 장면·음악, 목소리 미리 듣기).
- 큐는 workdir/gen_queue.json에 저장한다 (새로고침·앱 재시작 뒤에도 남음, 하던 작업은 다시 대기로).
- 작업마다 넣은 사람(owner, 브라우저마다 다른 ID)을 적어 두고, 대기 중인 자기 작업만 빼거나 순서를 바꾼다.
- 실제 처리는 kind별 처리 함수(HANDLERS, tasks.py에서 등록)가 한다. 처리 함수는 report()로 진행 상황을 알리고,
  취소되면 report()가 Cancelled를 일으켜 멈춘다.
- 처리 스레드는 프로세스에 하나, 큐를 처리하는 프로세스도 작업 폴더마다 하나다:
  - Streamlit이 앱 폴더의 파일이 바뀌면 모듈을 다시 불러오는데(새 모듈 객체), 그때 처리 스레드가 하나 더 생겨
    두 작업이 동시에 돈 적이 있다. 스레드·잠금·진행 상황·취소 신호·일시정지는 모듈이 아니라 _STATE(sys에 둠)에 둔다.
  - 같은 작업 폴더를 쓰는 앱 서버가 둘이면 잠금 파일(gen_queue.lock)을 잡은 프로세스만 큐를 처리한다.
"""
from __future__ import annotations

import json
import os
import sys
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
    "upload": "📣 업로드 문구",
}
DEFAULT_SEC = {"analyze": 240, "plan": 60, "narrate": 40, "make": 60, "scene": 30, "music": 60, "voice": 15,
               "upload": 20}
KEEP_FINISHED = 60  # 끝난 항목은 최근 이만큼만 남긴다 (예상 시간 계산에도 씀)
# AI 생성 프로세스를 쓰지 않는 작업 (시작 전에 쉬고 있는 생성 프로세스를 내려 GPU 메모리를 비운다).
# make는 아직 만들지 않은 AI 장면을 만들 수 있어서 빼고, scene·music은 생성 프로세스를 쓴다.
LLM_KINDS = {"analyze", "plan", "narrate", "upload", "voice"}

Report = Callable[..., None]
RUNNER_NAME = "job-queue"

# 모듈을 다시 불러와도 프로세스에 하나만 있어야 하는 것들 (sys는 다시 불러오지 않음)
_STATE: dict = sys.__dict__.setdefault("_shorts_job_queue", {})
_STATE.setdefault("lock", threading.RLock())
_STATE.setdefault("cancel", threading.Event())
_STATE.setdefault("progress", {})  # 항목 ID → {"note", "step", "total", "lines"} (진행 중인 항목, 저장하지 않음)
_STATE.setdefault("handlers", {})  # tasks.py가 등록 (다시 불러오면 새 함수로 덮어씀 → 처리 스레드도 새 코드를 씀)
_STATE.setdefault("paused", False)
_STATE.setdefault("runner", None)

HANDLERS: dict[str, Callable[[dict, Report], dict | None]] = _STATE["handlers"]
_lock: threading.RLock = _STATE["lock"]
_cancel: threading.Event = _STATE["cancel"]
_progress: dict[str, dict] = _STATE["progress"]


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
    # 임시 파일 이름은 프로세스·스레드마다 다르게 (같은 이름이면 앱 서버 둘이 동시에 쓸 때 서로의 파일을 바꿔치기함)
    tmp = path.with_name(f".gen_queue.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(20):  # Windows는 다른 프로세스가 읽는 중인 파일을 바꾸지 못함 → 잠깐 뒤 다시
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


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
    _STATE["paused"] = paused
    if not paused:
        ensure_runner()


def is_paused() -> bool:
    return _STATE["paused"]


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


def _canceled_in_file(item_id: str) -> bool:
    """다른 앱 서버 프로세스에서 취소했는지 (그 프로세스의 취소 신호는 여기까지 오지 않으므로 큐 파일로 확인)."""
    q = get(item_id)
    return q is None or q["status"] == CANCELED


def _reporter(item_id: str) -> Report:
    checked = [0.0]

    def report(note: str | None = None, step: int | None = None, total: int | None = None, line: str | None = None):
        if _cancel.is_set():
            raise Cancelled()
        if time.time() - checked[0] > 1.0:
            checked[0] = time.time()
            if _canceled_in_file(item_id):
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


def _process_lock():
    """이 작업 폴더의 큐를 처리할 권리 (다른 프로세스가 잡고 있으면 None). 프로세스가 끝나면 OS가 풀어 준다."""
    path = config.WORK_DIR / "gen_queue.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError:
        handle.close()
        return None


def _loop() -> None:
    held = None
    while held is None:  # 다른 앱 서버가 큐를 처리 중이면 그 서버가 끝날 때까지 기다린다
        held = _process_lock()
        if held is None:
            time.sleep(2)
    _STATE["lock_file"] = held
    with _lock:  # 큐를 처리하던 프로세스가 끝났다면 그때 하던 작업은 다시 대기로
        queue = _load()
        if any(q["status"] == RUNNING for q in queue):
            for q in queue:
                if q["status"] == RUNNING:
                    q["status"] = WAITING
            _save(queue)
    while True:
        item = None if _STATE["paused"] else _next()
        if item is None:
            generate.stop_idle()
            time.sleep(0.5)
            continue
        _cancel.clear()
        _progress[item["id"]] = {"lines": [], "note": "시작하는 중"}
        if item["kind"] in LLM_KINDS:  # LLM(Ollama)·음성 모델이 GPU를 쓰기 전에 쉬고 있는 생성 프로세스를 내린다
            generate.stop_idle(now=True)
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
    """큐를 처리하는 백그라운드 스레드를 프로세스에 하나만 띄운다 (모듈을 다시 불러와도 새로 띄우지 않음).
    앱 재시작 전 하던 작업은 스레드가 큐 처리 권리를 잡은 뒤 다시 대기로 돌린다."""
    with _lock:
        runner = _STATE.get("runner")
        if runner is not None and runner.is_alive():
            return
        if any(t.name == RUNNER_NAME and t.is_alive() for t in threading.enumerate()):
            return  # 이 보호가 생기기 전 코드가 띄운 스레드
        runner = threading.Thread(target=_loop, name=RUNNER_NAME, daemon=True)
        _STATE["runner"] = runner
        runner.start()
