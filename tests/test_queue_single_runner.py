"""공용 작업 큐가 한 번에 한 작업만 처리하는지 (GPU·Ollama 없이 몇 초 작업으로 확인).

1. 같은 프로세스에서 모듈을 다시 불러온 뒤(Streamlit이 파일 변경 때 하는 것) 작업을 더 넣어도 처리 스레드는 하나이고
   작업 시간이 겹치지 않는다. (예전에는 처리 스레드가 하나 더 생겨 두 작업이 동시에 돌았음)
2. 같은 작업 폴더를 쓰는 앱 서버 프로세스가 둘이어도 한쪽만 처리해서 작업 시간이 겹치지 않는다.

실행: .venv\\Scripts\\python tests\\test_queue_single_runner.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 자식 프로세스(두 번째 검사)도 이 파일을 불러오므로 같은 임시 폴더·기록 파일을 쓰게 환경 변수로 넘긴다
WORK = Path(os.environ.get("QUEUE_TEST_DIR") or tempfile.mkdtemp(prefix="queue_test_"))
os.environ["QUEUE_TEST_DIR"] = str(WORK)
os.environ.setdefault("SHORTS_WORK_DIR", str(WORK / "workdir"))
sys.path.insert(0, str(ROOT))
LOG = WORK / "runs.jsonl"
JOB_SEC = 1.0

results: list[bool] = []


def check(name: str, ok: bool, detail="") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def sleep_job(item: dict, report) -> dict:
    """가짜 작업: 시작·끝 시각과 처리한 프로세스·스레드를 기록한다."""
    start = time.time()
    time.sleep(JOB_SEC)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": item["id"], "pid": os.getpid(), "thread": threading.get_ident(),
                            "start": start, "end": time.time()}) + "\n")
    return {}


def runs() -> list[dict]:
    return [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines()] if LOG.exists() else []


def overlaps(items: list[dict]) -> list[tuple[str, str]]:
    items = sorted(items, key=lambda r: r["start"])
    return [(a["id"], b["id"]) for a, b in zip(items, items[1:]) if b["start"] < a["end"] - 0.01]


def wait_done(jq, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not any(q["status"] in (jq.WAITING, jq.RUNNING) for q in jq.items()):
            return True
        time.sleep(0.2)
    return False


def reload_package():
    """Streamlit의 파일 변경 처리처럼 앱 패키지 모듈을 sys.modules에서 지우고 다시 불러온다."""
    for name in [n for n in sys.modules if n == "shorts_maker" or n.startswith("shorts_maker.")]:
        del sys.modules[name]
    from shorts_maker import jobqueue

    return jobqueue


def test_reload() -> None:
    from shorts_maker import jobqueue as jq1

    jq1.HANDLERS["sleep"] = sleep_job
    for i in range(3):
        jq1.add("sleep", f"첫 모듈 {i}", {})
    time.sleep(0.3)  # 첫 작업이 도는 중에
    jq2 = reload_package()
    check("다시 불러온 모듈은 새 모듈 객체", jq2 is not jq1)
    jq2.HANDLERS["sleep"] = sleep_job
    jq2.ensure_runner()
    for i in range(2):
        jq2.add("sleep", f"다시 불러온 모듈 {i}", {})
    done = wait_done(jq2, 30)
    threads = [t for t in threading.enumerate() if t.name == "job-queue" and t.is_alive()]
    got = runs()
    check("모든 작업이 끝남", done and len(got) == 5, f"{len(got)}개")
    check("처리 스레드는 하나", len(threads) == 1 and len({r["thread"] for r in got}) == 1,
          f"살아 있는 스레드 {len(threads)}, 처리한 스레드 {len({r['thread'] for r in got})}")
    check("작업 시간이 겹치지 않음 (모듈 다시 불러온 뒤에도)", not overlaps(got), overlaps(got))
    check("같은 작업을 두 번 처리하지 않음", len({r["id"] for r in got}) == len(got))


CHILD = """
import os, sys, time
from pathlib import Path
sys.path.insert(0, {root!r})
sys.path.insert(0, {tests!r})
import test_queue_single_runner as t
from shorts_maker import jobqueue as jq
jq.HANDLERS["sleep"] = t.sleep_job
jq.ensure_runner()
(t.WORK / f"ready_{{os.getpid()}}").touch()
while not (t.WORK / "added").exists():
    time.sleep(0.1)
end = time.time() + 40
while time.time() < end and any(q["status"] in (jq.WAITING, jq.RUNNING) for q in jq.items()):
    time.sleep(0.2)
"""


def test_two_processes() -> None:
    """두 프로세스가 모두 처리 스레드를 띄운 뒤에 작업을 넣는다 (먼저 뜬 쪽이 혼자 다 처리해 버리지 않게)."""
    LOG.unlink(missing_ok=True)
    other = WORK / "workdir2"
    env = {**os.environ, "SHORTS_WORK_DIR": str(other), "PYTHONIOENCODING": "utf-8"}
    code = CHILD.format(root=str(ROOT), tests=str(ROOT / "tests"))
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env) for _ in range(2)]
    end = time.time() + 60
    while len(list(WORK.glob("ready_*"))) < 2 and time.time() < end:
        time.sleep(0.2)
    add = ("import sys; sys.path.insert(0, {root!r}); from shorts_maker import jobqueue as jq\n"
           "jq.ensure_runner = lambda: None  # 작업만 넣는 프로세스는 처리하지 않음\n"
           "[jq.add('sleep', f'p{{i}}', {{}}) for i in range(4)]").format(root=str(ROOT))
    subprocess.run([sys.executable, "-c", add], env=env, check=True)
    (WORK / "added").touch()
    for p in procs:
        p.wait(timeout=90)
    got = runs()
    check("두 프로세스: 작업 4개가 모두 한 번씩 끝남", len(got) == 4 and len({r["id"] for r in got}) == 4,
          [r["id"] for r in got])
    check("두 프로세스: 작업 시간이 겹치지 않음", not overlaps(got), overlaps(got))
    check("두 프로세스: 한 프로세스만 큐를 처리", len({r["pid"] for r in got}) == 1, {r["pid"] for r in got})


if __name__ == "__main__":
    test_reload()
    test_two_processes()
    print(f"\n{sum(results)}/{len(results)} 통과 (임시 폴더: {WORK})")
    sys.exit(0 if all(results) else 1)
