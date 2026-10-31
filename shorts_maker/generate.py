"""AI 생성: 장면 이미지·영상 클립, 배경음악. 실제 생성은 gen/worker.py를 생성 전용 가상환경에서 돌린다.

- .venv-gen: SDXL 계열 이미지(SDXL-Turbo, DreamShaper XL), LTX-Video 2B(동영상), MusicGen(음악)
- .venv-ltx: LTX-2.5 22B(동영상, 4bit/8bit 양자화), FLUX.2 [klein](이미지). diffusers 개발판과 최신 transformers가
  필요해서 따로 둔다.

생성 프로세스는 한 번 띄우면 모델을 올려 둔 채 요청을 하나씩 받는다 (GenService). 장면마다 프로세스를 새로
띄우면 LTX-Video를 불러오는 데만 장면마다 20초 넘게 걸렸다. 한동안 쓰지 않으면 끄고 GPU 메모리를 돌려준다.
두 환경의 프로세스는 GPU를 함께 쓰지 않게 하나가 일할 때 다른 하나를 끈다 (GPU_LOCK).
결과는 workdir/gen_cache/에 (종류, 프롬프트, 시드, 설정) 해시 이름으로 저장해서 같은 요청은 다시 만들지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from . import config, media

WORKER = config.ROOT / "gen" / "worker.py"


def _python(venv: str) -> Path:
    return config.ROOT / venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


GEN_PYTHON = _python(".venv-gen")
LTX_PYTHON = _python(".venv-ltx")
LTX_REPO = "Lightricks/LTX-2.5-Diffusers"

# AI 이미지 모델. env: 어느 생성 환경에서 돌리는지, size: 세로 이미지 크기 (모델이 잘 그리는 해상도),
# steps·guidance·scheduler: 모델 카드 권장값, quant: 16GB GPU에 맞추려고 양자화하는지, gated: 라이선스 동의 필요.
IMAGE_MODELS = {
    "sdxl_turbo": {"name": "SDXL-Turbo", "repo": "stabilityai/sdxl-turbo", "by": "Stability AI", "env": "gen",
                   "size": (576, 1024), "steps": 4, "guidance": 0.0, "license": "비상업 연구용",
                   "label": "SDXL-Turbo (가장 빠름, 화질 보통)", "speed": "장면당 몇 초", "download": "약 7GB"},
    "dreamshaper_lightning": {
        "name": "DreamShaper XL Lightning", "repo": "Lykon/dreamshaper-xl-lightning", "by": "Lykon", "env": "gen",
        "size": (832, 1216), "steps": 4, "guidance": 2.0, "scheduler": "dpm", "license": "OpenRAIL++",
        "label": "DreamShaper XL Lightning (빠름, 사진·일러스트 모두 좋음)", "speed": "장면당 몇 초", "download": "약 7GB"},
    "dreamshaper_turbo": {
        "name": "DreamShaper XL v2 Turbo", "repo": "Lykon/dreamshaper-xl-v2-turbo", "by": "Lykon", "env": "gen",
        "size": (832, 1216), "steps": 6, "guidance": 2.0, "scheduler": "dpm", "license": "OpenRAIL++",
        "label": "DreamShaper XL v2 Turbo (빠름, 디테일 조금 더)", "speed": "장면당 몇 초", "download": "약 7GB"},
    "flux2_klein_4b": {
        "name": "FLUX.2 [klein] 4B", "repo": "black-forest-labs/FLUX.2-klein-4B", "by": "Black Forest Labs", "env": "ltx",
        "size": (768, 1344), "steps": 4, "guidance": 1.0, "license": "Apache 2.0",
        "label": "FLUX.2 [klein] 4B (최신, 프롬프트를 잘 따름)", "speed": "장면당 10초 안팎", "download": "약 16GB"},
    "flux2_klein_9b": {
        "name": "FLUX.2 [klein] 9B (4bit)", "repo": "black-forest-labs/FLUX.2-klein-9B", "by": "Black Forest Labs",
        "env": "ltx", "size": (768, 1344), "steps": 4, "guidance": 1.0, "quant": "nf4", "gated": True,
        "license": "FLUX 비상업 라이선스",
        "label": "FLUX.2 [klein] 9B 4bit (최고 화질, 비상업용)", "speed": "장면당 20초 안팎", "download": "약 35GB"},
}

# 화면에 보여 줄 모델 정보 (어떤 AI가 무엇을 만드는지). "video"는 동영상 모델 설정에 따라 바뀐다 (set_video_model).
VIDEO_MODEL_INFO = {
    "ltx2b": {"name": "LTX-Video (2B)", "repo": "Lightricks/LTX-Video", "by": "Lightricks",
              "what": "약 4초 세로 동영상 (512×768, 24fps)", "speed": "장면당 1분 안팎"},
    "ltx25": {"name": "LTX-2.5 (22B)", "repo": LTX_REPO, "by": "Lightricks",
              "what": "약 5초 세로 동영상 (704×1280, 24fps, 2단계 생성)", "speed": "장면당 약 1분 15초 (4bit)"},
}
VIDEO_MODELS = {"ltx2b": "LTX-Video 2B (가벼움, 화질 낮음)", "ltx25": "LTX-2.5 22B (양자화, 화질 좋음, 느림)"}
LTX_QUANTS = {
    "nf4": "4bit (NF4, 추천) — GPU 약 12GB, 장면당 약 1분 15초",
    "fp8": "8bit (fp8) — RAM에 두고 나눠 올림, 장면당 약 10분 (RAM 64GB 이상)",
}
MODEL_INFO = {
    "image": {},
    "video": dict(VIDEO_MODEL_INFO["ltx2b"]),
    "music": {"name": "MusicGen small", "repo": "facebook/musicgen-small", "by": "Meta",
              "what": "최대 30초 배경음악 (가사 없음)", "speed": "20초 곡에 약 40초"},
}
ENGINES: dict[str, str] = {}
IMAGE_SIZE = (576, 1024)  # SDXL-Turbo 세로 이미지
VIDEO_SIZE = (512, 768)  # LTX-Video 2B 세로 영상 (32의 배수)
VIDEO_FRAMES, VIDEO_FPS, VIDEO_STEPS = 97, 24, 30  # 약 4초
LTX25_SIZE = (704, 1280)  # LTX-2.5 최종 해상도 (1단계는 절반 352×640). ComfyUI 워크플로의 1280×720을 세로로
LTX25_FRAMES = 121  # 24fps 5초 (8의 배수 + 1)
LTX25_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"  # ComfyUI 워크플로 값
STYLE_SUFFIX = "vertical composition, cinematic, high detail, natural light"
NEGATIVE = "text, watermark, logo, letters, low quality, blurry, deformed"
IDLE_STOP_SEC = 90  # 이만큼 쓰지 않으면 생성 프로세스를 끄고 GPU 메모리를 돌려준다
LTX_IDLE_STOP_SEC = 600  # LTX-2.5는 불러오는 데 오래 걸려서 더 오래 켜 둔다

settings = {"video_model": "ltx2b", "ltx_quant": "nf4", "image_model": "sdxl_turbo"}
# 생성 작업마다 쓸 장치 ("gpu" | "hybrid" | "cpu"). 요청에 실어 보내므로 한 프로세스에서 이미지는 CPU, 동영상은 GPU도 된다.
task_devices = {"image": "gpu", "video": "gpu", "music": "gpu"}
Progress = Callable[[int, int], None]


def _refresh_labels() -> None:
    image = IMAGE_MODELS[settings["image_model"]]
    MODEL_INFO["image"] = {"name": image["name"], "repo": image["repo"], "by": image["by"], "speed": image["speed"],
                           "what": f"세로 이미지 1장({image['size'][0]}×{image['size'][1]}) → 천천히 확대·이동하는 클립"}
    info = dict(VIDEO_MODEL_INFO[settings["video_model"]])
    if settings["video_model"] == "ltx25":
        info["name"] = f"LTX-2.5 (22B, {'4bit' if settings['ltx_quant'] == 'nf4' else '8bit'})"
    MODEL_INFO["video"] = info
    ENGINES.clear()
    ENGINES["image"] = f"AI 이미지 + 움직임 — {MODEL_INFO['image']['name']} (빠름)"
    ENGINES["video"] = f"AI 동영상 — {MODEL_INFO['video']['name']} (느림)"


_refresh_labels()


def set_video_model(model: str, quant: str) -> None:
    """AI 동영상에 쓸 모델("ltx2b" | "ltx25")과 LTX-2.5 양자화("nf4" | "fp8")."""
    settings.update(video_model=model if model in VIDEO_MODEL_INFO else "ltx2b",
                    ltx_quant=quant if quant in LTX_QUANTS else "nf4")
    _refresh_labels()


def set_image_model(model: str) -> None:
    """AI 이미지에 쓸 모델 (IMAGE_MODELS의 키)."""
    settings["image_model"] = model if model in IMAGE_MODELS else "sdxl_turbo"
    _refresh_labels()


def model_label(kind: str) -> str:
    info = MODEL_INFO[kind]
    return f"{info['name']} ({info['by']}, `{info['repo']}`)"


_probe: tuple[bool, bool] | None = None  # (생성 환경을 불러올 수 있음, GPU 있음)


def available() -> tuple[bool, str]:
    """(쓸 수 있는지, 안내 문구). 생성 환경이 없으면 설치 방법을 알려 준다."""
    global _probe
    if not GEN_PYTHON.exists():
        return False, "AI 생성 환경이 없습니다. README의 「AI 생성 기능 설치」대로 .venv-gen을 만드세요."
    if _probe is None:
        proc = subprocess.run([str(GEN_PYTHON), "-c", "import torch, diffusers, transformers; print(torch.cuda.is_available())"],
                              capture_output=True, text=True)
        _probe = (proc.returncode == 0, proc.stdout.strip() == "True")
    loads, has_gpu = _probe
    if not loads:
        return False, "AI 생성 환경(.venv-gen)을 불러오지 못했습니다."
    if any(d != "cpu" for d in task_devices.values()) and not has_gpu:
        return False, "생성 환경에서 GPU를 찾지 못했습니다. 설정 「💻 컴퓨터·AI 모델」에서 계산 장치를 「CPU만」으로 바꾸면 (느리지만) 쓸 수 있습니다."
    return True, ""


_gated_checked: dict[str, tuple[float, bool, str]] = {}


def gated_status(repo: str, refresh: bool = False) -> tuple[bool, str]:
    """.venv-ltx 환경과, 게이트 저장소(라이선스 동의가 필요한 모델)에 접근할 수 있는지.
    성공은 계속 기억하고, 실패는 30초 뒤에 다시 확인한다 (로그인·동의를 마치면 곧 반영되게)."""
    if not LTX_PYTHON.exists():
        return False, "최신 생성 환경이 없습니다. README의 「LTX-2.5·FLUX.2 설치」대로 .venv-ltx를 만드세요."
    checked = _gated_checked.get(repo)
    if checked and not refresh and (checked[1] or time.time() - checked[0] < 30):
        return checked[1], checked[2]
    page = f"https://huggingface.co/{repo}"
    try:
        from huggingface_hub import auth_check, get_token

        if not get_token():
            ok, msg = False, ("Hugging Face에 로그인되어 있지 않습니다. 터미널에서 `.venv-ltx\\Scripts\\hf auth login`으로 "
                              f"로그인하고, {page} 에서 라이선스에 동의하세요.")
        else:
            auth_check(repo)
            ok, msg = True, ""
    except Exception as e:  # 동의하지 않았거나(GatedRepoError) 인터넷이 안 됨
        name = type(e).__name__
        ok, msg = False, (f"{page} 에서 라이선스에 동의해야 합니다 (게이트 모델)." if "Gated" in name
                          else f"{repo} 저장소를 확인하지 못했습니다: {name}")
    _gated_checked[repo] = (time.time(), ok, msg)
    return ok, msg


def ltx_status(refresh: bool = False) -> tuple[bool, str]:
    return gated_status(LTX_REPO, refresh)


def image_available() -> tuple[bool, str]:
    """지금 고른 이미지 모델을 쓸 수 있는지."""
    model = IMAGE_MODELS[settings["image_model"]]
    if model["env"] == "gen":
        return available()
    if model.get("gated"):
        return gated_status(model["repo"])
    if not LTX_PYTHON.exists():
        return False, "최신 생성 환경이 없습니다. README의 「LTX-2.5·FLUX.2 설치」대로 .venv-ltx를 만드세요."
    return True, ""


def video_available() -> tuple[bool, str]:
    """지금 고른 동영상 모델을 쓸 수 있는지."""
    if settings["video_model"] == "ltx25":
        if task_devices["video"] == "cpu":
            return False, "LTX-2.5는 GPU에서만 돌립니다. AI 동영상 장치를 GPU로 바꾸거나 LTX-Video 2B를 고르세요."
        return ltx_status()
    return available()


def scene_available(engine: str) -> tuple[bool, str]:
    return video_available() if engine == "video" else image_available()


def cache_dir() -> Path:
    folder = config.WORK_DIR / "gen_cache"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def models_dir() -> Path:
    """양자화해서 저장한 모델 (LTX-2.5 4bit 등)."""
    folder = config.WORK_DIR / "models"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def _free_gpu() -> None:
    """생성 모델이 GPU를 쓰도록 Ollama에 올라간 모델을 내린다 (다음 LLM 요청 때 다시 올라옴)."""
    try:
        import ollama

        for m in ollama.ps().models:
            ollama.generate(model=m.model, prompt="", keep_alive=0)
    except Exception:
        pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


class Cancelled(RuntimeError):
    pass


# 모듈을 다시 불러와도(Streamlit이 파일 변경 시) 생성 프로세스·GPU 잠금은 프로세스에 하나만 (jobqueue._STATE와 같은 이유)
_SHARED: dict = sys.__dict__.setdefault("_shorts_generate", {})
GPU_LOCK = _SHARED.setdefault("gpu_lock", threading.Lock())  # 생성 프로세스들이 한 번에 하나만 GPU를 쓰게


class GenService:
    """켜 둔 생성 프로세스 하나. 요청은 한 번에 하나씩 (GPU_LOCK), 다른 스레드에서 cancel()로 지금 작업을 멈출 수 있다."""

    def __init__(self, name: str, python: Path, kinds: set[str]):
        self.name, self.python, self.kinds = name, python, kinds
        self.lock = GPU_LOCK
        self.proc: subprocess.Popen | None = None
        self.events: queue.Queue = queue.Queue()
        self.last_used = 0.0
        self.loaded_kind: str | None = None
        self.current_id: str | None = None
        self._cancelled = False
        self.proc_args: list[str] = []

    def args(self) -> list[str]:
        if self.name == "ltx":
            return ["--quant", settings["ltx_quant"], "--models-dir", str(models_dir())]
        return []  # 장치는 요청마다 실어 보낸다 (task["device"])

    def _start(self) -> None:
        ok, message = ((True, "") if LTX_PYTHON.exists() else (False, "최신 생성 환경(.venv-ltx)이 없습니다.")) \
            if self.name == "ltx" else available()
        if not ok:
            raise RuntimeError(message)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        self.proc_args = self.args()
        self.proc = subprocess.Popen(
            [str(self.python), str(WORKER), "--serve", *self.proc_args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", env=env, cwd=config.ROOT,
        )
        self.events = queue.Queue()
        self.loaded_kind = None
        threading.Thread(target=self._read, args=(self.proc, self.events), daemon=True).start()
        event = self._wait(timeout=120)
        if event.get("event") != "ready":
            raise RuntimeError("AI 생성 프로세스를 시작하지 못했습니다.")

    @staticmethod
    def _read(proc: subprocess.Popen, events: queue.Queue) -> None:
        for line in proc.stdout:
            if line.startswith("@@"):
                try:
                    events.put(json.loads(line[2:]))
                except json.JSONDecodeError:
                    pass
        events.put({"event": "exit"})

    def _wait(self, timeout: float | None = None) -> dict:
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return {"event": "timeout"}

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _shutdown(self) -> None:
        """(GPU_LOCK을 잡은 상태에서) 프로세스를 끄고 GPU 메모리를 돌려준다."""
        if self.alive():
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.proc = None

    def run(self, kind: str, task: dict, request_id: str = "", on_step: Progress | None = None,
            on_loading: Callable[[str], None] | None = None) -> None:
        """작업 하나를 끝날 때까지 실행한다. 실패하면 RuntimeError, cancel()로 멈추면 Cancelled."""
        if task.get("device", "gpu") != "cpu":
            _free_gpu()  # GPU를 쓰는 생성 전에 Ollama 모델을 GPU에서 내린다
        with self.lock:
            for other in SERVICES:  # 다른 생성 환경이 GPU에 모델을 올려 두고 있으면 끈다
                if other is not self:
                    other._shutdown()
            if self.alive() and self.proc_args != self.args():  # 양자화 설정을 바꿨으면 다시 켬
                self.proc.kill()
                self.proc.wait()
            if not self.alive():
                self._start()
            request_id = request_id or _key(kind, json.dumps(task, ensure_ascii=False), time.time())
            self.current_id, self._cancelled = request_id, False
            try:
                self.proc.stdin.write(json.dumps({"id": request_id, "kind": kind, "task": task}, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
                while True:
                    event = self._wait()
                    if event.get("event") == "exit":
                        self.proc = None
                        if self._cancelled:
                            raise Cancelled("취소했습니다.")
                        raise RuntimeError("AI 생성 프로세스가 중간에 종료되었습니다 (GPU·RAM 메모리 부족일 수 있음).")
                    if event.get("id") not in (request_id, None):
                        continue
                    if event["event"] == "loading" and on_loading:
                        on_loading(event.get("model", ""))
                    elif event["event"] == "step" and on_step:
                        on_step(event["step"], event["total"])
                    elif event["event"] == "done":
                        self.loaded_kind = kind
                        return
                    elif event["event"] == "error":
                        raise RuntimeError(f"AI 생성({kind}) 실패:\n{event.get('error', '')}")
            finally:
                self.current_id = None
                self.last_used = time.time()

    def cancel(self) -> None:
        """지금 만드는 작업을 멈춘다 (프로세스를 끄므로 다음 작업 때 모델을 다시 불러옴)."""
        if self.alive():
            self._cancelled = True
            self.proc.kill()

    def stop_if_idle(self, idle_sec: float = IDLE_STOP_SEC) -> bool:
        """한동안 쓰지 않았으면 끄고 GPU 메모리를 돌려준다."""
        if self.alive() and self.current_id is None and time.time() - self.last_used > idle_sec:
            if self.lock.acquire(blocking=False):
                try:
                    self._shutdown()
                finally:
                    self.lock.release()
                return True
        return False


service = _SHARED.setdefault("service", GenService("gen", GEN_PYTHON, {"image", "video", "music"}))
ltx_service = _SHARED.setdefault("ltx_service", GenService("ltx", LTX_PYTHON, {"ltx25", "flux2"}))
SERVICES = [service, ltx_service]


def service_for(kind: str) -> GenService:
    return ltx_service if kind in ltx_service.kinds else service


def run_job(kind: str, task: dict, **kwargs) -> None:
    service_for(kind).run(kind, task, **kwargs)


def cancel() -> None:
    """지금 만드는 작업을 멈춘다."""
    for s in SERVICES:
        if s.current_id is not None:
            s.cancel()


def stop_idle(now: bool = False) -> None:
    """한동안 쓰지 않은 생성 프로세스를 끈다. now=True면 기다리지 않고 지금 쉬고 있는 것을 모두 끈다
    (LLM·음성 작업 전에: 쉬는 LTX-2.5 프로세스도 GPU 메모리 몇 GB를 잡고 있어 Ollama와 합치면 16GB를 넘었음)."""
    service.stop_if_idle(0 if now else IDLE_STOP_SEC)
    ltx_service.stop_if_idle(0 if now else LTX_IDLE_STOP_SEC)


def set_task_devices(devices: dict) -> None:
    """생성 작업(image·video·music)마다 쓸 장치."""
    task_devices.update({k: v for k, v in devices.items() if k in task_devices})


# ───────────────────────── 장면 ─────────────────────────


def scene_job(scene: dict) -> tuple[str, dict, Path]:
    """장면 {prompt, engine, seed, seconds} → (종류, 생성 작업, 생성 결과 파일). 결과 파일 이름은 요청 내용의 해시.
    AI 동영상은 그때의 동영상 모델 설정(LTX-Video 2B / LTX-2.5)을 따른다."""
    prompt = f"{scene['prompt'].strip()}, {STYLE_SUFFIX}"
    seed = int(scene.get("seed", 0))
    if scene.get("engine", "image") == "video" and settings["video_model"] == "ltx25":
        quant = settings["ltx_quant"]
        out = cache_dir() / f"ltx25_{_key('ltx25', quant, prompt, seed, LTX25_SIZE, LTX25_FRAMES)}.mp4"
        return "ltx25", {"prompt": prompt, "negative": LTX25_NEGATIVE, "seed": seed, "width": LTX25_SIZE[0],
                         "height": LTX25_SIZE[1], "frames": LTX25_FRAMES, "fps": VIDEO_FPS, "out": str(out),
                         "device": "gpu"}, out
    if scene.get("engine", "image") == "video":
        out = cache_dir() / f"vid_{_key('video', prompt, seed, VIDEO_SIZE, VIDEO_FRAMES)}.mp4"
        return "video", {"prompt": prompt, "negative": NEGATIVE, "seed": seed, "width": VIDEO_SIZE[0],
                         "height": VIDEO_SIZE[1], "frames": VIDEO_FRAMES, "fps": VIDEO_FPS,
                         "steps": VIDEO_STEPS, "out": str(out), "device": task_devices["video"]}, out
    model_id = settings["image_model"]
    model = IMAGE_MODELS[model_id]
    w, h = model["size"]
    # SDXL-Turbo는 예전 캐시 이름을 그대로 쓴다 (이미 만든 장면을 다시 만들지 않게)
    key = _key("image", prompt, seed, IMAGE_SIZE) if model_id == "sdxl_turbo" else _key("image", model_id, prompt, seed)
    out = cache_dir() / f"img_{key}.png"
    task = {"model": model["repo"], "prompt": prompt, "seed": seed, "width": w, "height": h,
            "steps": model["steps"], "guidance": model["guidance"], "out": str(out), "device": task_devices["image"]}
    if model["env"] == "ltx":
        return "flux2", {**task, "quant": model.get("quant", "")}, out
    return "image", {**task, "negative": NEGATIVE, "scheduler": model.get("scheduler", "")}, out


def scene_size(scene: dict) -> tuple[int, int]:
    """장면 클립 크기 (쇼츠 출력 크기, 기본 1080x1920)."""
    size = scene.get("size")
    return (int(size[0]), int(size[1])) if size else config.OUTPUT_SIZE


def scene_clip_path(scene: dict) -> Path:
    """장면의 최종 클립 경로 (이미지는 길이·크기별 움직임 클립, 영상은 생성 결과 그대로).
    "_fit": 그림 비율을 지켜 자르는 지금 방식 (예전 클립은 그림을 화면 비율로 늘려 좌우로 눌려 보였음)."""
    kind, _, out = scene_job(scene)
    if kind not in ("image", "flux2"):
        return out
    seconds = round(float(scene.get("seconds", 5.0)), 1)
    w, h = scene_size(scene)
    tag = "" if (w, h) == tuple(config.OUTPUT_SIZE) else f"_{w}x{h}"
    return out.with_name(f"{out.stem}_{seconds}s_m{int(scene.get('seed', 0)) % 6}_fit{tag}.mp4")


def cached_clip(scene: dict) -> Path | None:
    """이미 만들어 둔 장면 클립 (없으면 None)."""
    clip = scene_clip_path(scene)
    if clip.exists():
        return clip
    kind, _, out = scene_job(scene)
    if kind in ("image", "flux2") and out.exists():  # 이미지는 있는데 이 길이의 움직임 클립이 없으면 바로 만든다
        return finish_scene(scene)
    return None


def finish_scene(scene: dict) -> Path:
    """생성 결과를 장면 클립으로 (이미지는 seconds 길이 움직임 클립)."""
    kind, _, out = scene_job(scene)
    clip = scene_clip_path(scene)
    if kind in ("image", "flux2") and not clip.exists():
        image_to_clip(out, round(float(scene.get("seconds", 5.0)), 1), clip, motion=int(scene.get("seed", 0)),
                      size=scene_size(scene))
    return clip


def image_to_clip(image: Path, seconds: float, out: Path, motion: int = 0, size: tuple[int, int] | None = None) -> None:
    """정지 이미지를 천천히 확대·이동하는 클립으로 (켄 번스 효과, 기본 1080x1920). motion으로 방향을 바꾼다.
    그림은 비율을 지켜 화면을 채우고 넘치는 쪽을 가운데 기준으로 자른다 (늘려서 맞추면 피사체가 눌려 보임)."""
    w, h = size or config.OUTPUT_SIZE
    frames = max(1, int(round(seconds * 30)))
    zoom = "min(zoom+0.0009,1.18)" if motion % 2 == 0 else "if(eq(on,0),1.18,max(zoom-0.0009,1.0))"
    x = ["iw/2-(iw/zoom/2)", "(iw-iw/zoom)*on/{f}", "(iw-iw/zoom)*(1-on/{f})"][motion % 3].format(f=frames)
    media.run_ffmpeg([
        "-loop", "1", "-i", str(image),
        "-vf", f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},zoompan=z='{zoom}':x='{x}':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps=30,"
               "format=yuv420p",
        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out),
    ])


def scene_clips(scenes: list[dict], on_progress: Progress | None = None) -> list[Path]:
    """scenes: [{prompt, engine("image"|"video"), seed, seconds}] → 장면마다 영상 클립 경로.
    이미 만든 장면(큐에서 먼저 만든 것 포함)은 다시 만들지 않는다."""
    todo = [s for s in scenes if cached_clip(s) is None]
    for i, scene in enumerate(todo):
        kind, task, out = scene_job(scene)
        if not out.exists():
            run_job(kind, task)
        if on_progress:
            on_progress(i + 1, len(todo))
    return [finish_scene(s) for s in scenes]


def music(prompt: str, seconds: float, on_progress: Progress | None = None) -> Path:
    """MusicGen 배경음악 (최대 30초, 더 긴 쇼츠는 배경음악 반복으로 채움)."""
    seconds = min(30.0, max(5.0, seconds))
    out = cache_dir() / f"music_{_key('music', prompt, seconds)}.wav"
    if not out.exists():
        run_job("music", {"prompt": prompt, "seconds": seconds, "out": str(out), "device": task_devices["music"]})
        if on_progress:
            on_progress(1, 1)
    return out
