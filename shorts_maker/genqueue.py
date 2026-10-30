"""AI 장면 생성 큐: 장면을 한꺼번에 넣어 두면 앱 서버의 백그라운드 스레드가 차례로 만든다.

- 큐는 workdir/gen_queue.json에 저장한다 (새로고침·앱 재시작 뒤에도 남음, 하던 작업은 다시 대기로).
- 대기 중인 항목은 언제든 지우거나 순서를 바꿀 수 있고, 만드는 중인 항목은 취소할 수 있다.
- 실제 생성은 generate.run_job(켜 둔 생성 프로세스)이 한다. 쇼츠 만들기도 같은 GPU_LOCK을 쓰므로 GPU를 둘이 동시에 쓰지 않는다.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from . import config, generate

WAITING, RUNNING, DONE, FAILED, CANCELED = "waiting", "running", "done", "failed", "canceled"
STATUS_LABELS = {WAITING: "⏳ 대기", RUNNING: "🔄 만드는 중", DONE: "✅ 완료", FAILED: "❌ 실패", CANCELED: "⏹ 취소"}

_lock = threading.RLock()
_runner: threading.Thread | None = None
_paused = False
_progress: dict[str, dict] = {}  # 항목 ID → {"step", "total", "note"} (만드는 중인 항목의 진행 상황, 저장하지 않음)


def queue_file() -> Path:
    return config.WORK_DIR / "gen_queue.json"


def _load() -> list[dict]:
    path = queue_file()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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


def scene_of(item: dict) -> dict:
    return {"prompt": item["prompt"], "engine": item["engine"], "seed": item["seed"], "seconds": item["seconds"]}


def add(scenes: list[dict]) -> int:
    """scenes: [{prompt, engine, seed, seconds, label, card_uid, project}] 를 큐 뒤에 넣는다.
    이미 만든 장면, 같은 장면이 이미 대기·진행 중이면 넣지 않는다. 넣은 개수를 돌려준다."""
    added = 0
    with _lock:
        queue = _load()
        busy = {generate.scene_clip_path(scene_of(q)) for q in queue if q["status"] in (WAITING, RUNNING)}
        for s in scenes:
            if not s.get("prompt", "").strip():
                continue
            path = generate.scene_clip_path(s)
            if path in busy or generate.cached_clip(s):
                continue
            queue.append({
                "id": uuid.uuid4().hex[:8], "status": WAITING, "created": time.strftime("%H:%M:%S"),
                "prompt": s["prompt"], "engine": s.get("engine", "image"), "seed": int(s.get("seed", 0)),
                "seconds": float(s.get("seconds", 5.0)), "label": s.get("label", ""),
                "card_uid": s.get("card_uid", ""), "project": s.get("project", ""), "error": "",
            })
            busy.add(path)
            added += 1
        _save(queue)
    ensure_runner()
    return added


def remove(item_id: str) -> None:
    """대기·완료·실패 항목을 지운다. 만드는 중이면 취소한다."""
    with _lock:
        queue = _load()
        target = next((q for q in queue if q["id"] == item_id), None)
        if target is None:
            return
        if target["status"] == RUNNING:
            target["status"] = CANCELED
            _save(queue)
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


def move_to_front(item_id: str) -> None:
    with _lock:
        queue = _load()
        target = next((q for q in queue if q["id"] == item_id and q["status"] == WAITING), None)
        if target is None:
            return
        queue.remove(target)
        first = next((i for i, q in enumerate(queue) if q["status"] == WAITING), len(queue))
        queue.insert(first, target)
        _save(queue)


def retry(item_id: str) -> None:
    with _lock:
        queue = _load()
        for q in queue:
            if q["id"] == item_id and q["status"] in (FAILED, CANCELED):
                q.update(status=WAITING, error="")
        _save(queue)
    ensure_runner()


def clear(statuses: tuple[str, ...]) -> None:
    with _lock:
        _save([q for q in _load() if q["status"] not in statuses])


def set_paused(paused: bool) -> None:
    global _paused
    _paused = paused
    if not paused:
        ensure_runner()


def is_paused() -> bool:
    return _paused


def progress(item_id: str) -> dict:
    return _progress.get(item_id, {})


def status_of_card(card_uid: str) -> dict | None:
    """카드의 가장 최근 큐 항목 (대기 순서 position 포함)."""
    queue = items()
    waiting = [q["id"] for q in queue if q["status"] == WAITING]
    found = [q for q in queue if q.get("card_uid") == card_uid]
    if not found:
        return None
    item = found[-1]
    return {**item, "position": waiting.index(item["id"]) + 1 if item["id"] in waiting else 0}


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


def _finish(item_id: str, status: str, error: str = "") -> None:
    with _lock:
        queue = _load()
        for q in queue:
            if q["id"] == item_id and q["status"] == RUNNING:
                q.update(status=status, error=error[-800:], finished=time.time())
        _save(queue)
    _progress.pop(item_id, None)


def _loop() -> None:
    while True:
        item = None if _paused else _next()
        if item is None:
            generate.stop_idle()
            time.sleep(1.0)
            continue
        scene = scene_of(item)
        try:
            kind, task, out = generate.scene_job(scene)
            if not out.exists():
                _progress[item["id"]] = {"note": "생성 프로세스 준비 중"}
                generate.run_job(
                    kind, task, request_id=item["id"],
                    on_step=lambda s, n, i=item["id"]: _progress.__setitem__(i, {"step": s, "total": n}),
                    on_loading=lambda m, i=item["id"]: _progress.__setitem__(i, {"note": f"{m} 불러오는 중"}),
                )
            generate.finish_scene(scene)
            _finish(item["id"], DONE)
        except generate.Cancelled:
            _finish(item["id"], CANCELED)
        except Exception as e:  # 하나가 실패해도 큐는 계속
            _finish(item["id"], FAILED, str(e))


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
        _runner = threading.Thread(target=_loop, name="gen-queue", daemon=True)
        _runner.start()
