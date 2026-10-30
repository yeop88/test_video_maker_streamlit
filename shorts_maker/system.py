"""컴퓨터 사양 확인과 계산 장치(CPU / GPU / 둘 다)에 맞는 추천 모델.

사양은 표준 라이브러리와 nvidia-smi로만 읽는다 (torch를 불러오지 않아 첫 화면이 느려지지 않음).
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

from . import config

DEVICES = {
    "gpu": "GPU만 (가장 빠름)",
    "hybrid": "GPU + CPU 함께 (GPU 메모리 아끼기)",
    "split": "작업별로 나누기 (예: 영상은 GPU, 나머지는 CPU)",
    "cpu": "CPU만 (GPU 없이, 느림)",
}
# 「작업별로 나누기」에서 장치를 따로 고르는 작업
TASKS = {
    "llm": "장면 이해·대본 (Ollama LLM)",
    "asr": "대사 인식 (Whisper)",
    "tts": "나레이션 음성 (Chatterbox)",
    "image": "AI 이미지",
    "video": "AI 동영상",
    "music": "AI 배경음악",
}
TASK_DEVICES = {"gpu": "GPU", "cpu": "CPU"}
SPLIT_DEFAULT = {"llm": "cpu", "asr": "cpu", "tts": "cpu", "image": "cpu", "video": "gpu", "music": "cpu"}
DEVICE_HELP = (
    "GPU만: 모델을 GPU에 올려 가장 빠릅니다.\n\n"
    "GPU + CPU 함께: 큰 모델은 RAM에 두고 쓸 부분만 GPU로 옮깁니다 (GPU 메모리가 작을 때, 조금 느림). "
    "Ollama는 GPU에 다 안 들어가는 층을 CPU에서 돌립니다.\n\n"
    "작업별로 나누기: 작업마다 GPU·CPU를 고릅니다. 예를 들어 AI 동영상만 GPU에 두고 나머지를 CPU로 돌리면, "
    "동영상을 만드는 동안 다른 작업이 GPU 메모리를 차지하지 않습니다.\n\n"
    "CPU만: GPU를 쓰지 않습니다. 대사 인식·장면 이해·AI 이미지가 몇 배~수십 배 느리고, AI 동영상은 사실상 쓸 수 없습니다."
)


@dataclass
class Gpu:
    name: str
    vram_gb: float
    driver: str = ""


@dataclass
class Specs:
    os: str
    cpu: str
    cores: int
    ram_gb: float
    ram_free_gb: float
    disk_free_gb: float
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def vram_gb(self) -> float:
        return max((g.vram_gb for g in self.gpus), default=0.0)


def _os_name() -> str:
    if platform.system() == "Windows":  # Windows 11도 release가 "10"으로 나와서 빌드 번호로 구분
        build = int(platform.version().split(".")[-1] or 0)
        return f"Windows {'11' if build >= 22000 else platform.release()}"
    return f"{platform.system()} {platform.release()}"


def _cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "알 수 없음"


def _ram_gb() -> tuple[float, float]:
    """(전체, 남은) RAM GB."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / 1024 ** 3, status.ullAvailPhys / 1024 ** 3
    try:
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                name, value = line.split(":", 1)
                info[name] = int(value.split()[0]) / 1024 ** 2
        return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)
    except OSError:
        return 0.0, 0.0


def _gpus() -> list[Gpu]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append(Gpu(parts[0], round(float(parts[1]) / 1024, 1), parts[2] if len(parts) > 2 else ""))
            except ValueError:
                pass
    return gpus


def specs() -> Specs:
    total, free = _ram_gb()
    disk = shutil.disk_usage(config.WORK_DIR if config.WORK_DIR.exists() else config.ROOT)
    return Specs(
        os=_os_name(), cpu=_cpu_name(), cores=os.cpu_count() or 1,
        ram_gb=round(total, 1), ram_free_gb=round(free, 1), disk_free_gb=round(disk.free / 1024 ** 3),
        gpus=_gpus(),
    )


def suggested_device(s: Specs) -> str:
    """이 컴퓨터에 맞는 계산 장치."""
    if s.vram_gb >= 12:
        return "gpu"
    if s.vram_gb >= 6:
        return "hybrid"
    return "cpu"


# 계산 장치·동영상 모델·양자화는 컴퓨터마다 다르므로 프리셋(다른 PC와 나눌 수 있는 스타일 설정)에 넣지 않고
# 이 컴퓨터에 따로 저장한다 (workdir/device.json).
MACHINE_CHOICES = {
    "video_model": ("ltx2b", "ltx25"), "ltx_quant": ("nf4", "fp8"),
    "image_model": ("sdxl_turbo", "dreamshaper_lightning", "dreamshaper_turbo", "flux2_klein_4b", "flux2_klein_9b"),
}


def _device_file():
    return config.WORK_DIR / "device.json"


def load_machine(s: Specs) -> dict:
    """{device, split, video_model, ltx_quant, image_model}. 저장된 값이 없거나 잘못되면 이 컴퓨터에 맞는 기본값."""
    try:
        saved = json.loads(_device_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    values = {"device": suggested_device(s), "video_model": "ltx2b", "ltx_quant": "nf4", "image_model": "sdxl_turbo"}
    if saved.get("device") in DEVICES:
        values["device"] = saved["device"]
    for key, choices in MACHINE_CHOICES.items():
        if saved.get(key) in choices:
            values[key] = saved[key]
    split = saved.get("split") if isinstance(saved.get("split"), dict) else {}
    values["split"] = {t: split.get(t) if split.get(t) in TASK_DEVICES else SPLIT_DEFAULT[t] for t in TASKS}
    return values


def task_device(device: str, split: dict, task: str) -> str:
    """작업 하나가 쓸 장치 ("gpu" | "hybrid" | "cpu"). 「작업별로 나누기」면 작업마다 고른 값."""
    if device == "split":
        return split.get(task, SPLIT_DEFAULT[task])
    return device


def save_machine(**values) -> None:
    try:
        saved = json.loads(_device_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    saved.update(values)
    _device_file().parent.mkdir(parents=True, exist_ok=True)
    _device_file().write_text(json.dumps(saved), encoding="utf-8")


def recommend(task: str, device: str, s: Specs) -> dict:
    """작업과 그 작업의 장치에 맞는 추천.
    {"text": 화면 문구, "choice": 설정에 넣을 값(없으면 None), "key": 바꿀 설정 이름, "need": 필요한 GPU 메모리 GB}.
    계산 장치를 바꾸면 추천도 바뀐다 (GPU면 GPU 메모리에 맞는 큰 모델, CPU면 가볍고 빠른 모델)."""
    gpu = device != "cpu" and bool(s.gpus)
    vram, ram = s.vram_gb, s.ram_gb
    if task == "llm":
        if gpu and vram >= 8:
            return {"text": "qwen3.5:9b — 화면을 보는 비전 모델 (약 6.6GB)", "choice": "qwen3.5:9b", "key": "llm_model", "need": 7}
        return {"text": "qwen3.5:4b — 비전 모델 (약 3.4GB). " + ("CPU에서 9b보다 빠름 (9b는 짧은 답에 약 12초, 장면 설명이 많으면 수 분)" if not gpu else "GPU 메모리가 작아서"),
                "choice": "qwen3.5:4b", "key": "llm_model", "need": 4}
    if task == "asr":
        return {"text": "whisper-large-v3-turbo (fp16, 약 1.6GB)" if gpu else
                "whisper-large-v3-turbo (fp32) — 대사 6초에 약 4초 (GPU보다 몇 배 느림, 긴 영상은 오래 걸림)",
                "choice": None, "key": None, "need": 2}
    if task == "tts":
        if gpu and vram >= 6:
            return {"text": "Chatterbox — 목소리 복제·감정 조절 (약 3GB)", "choice": "chatterbox", "key": "engine", "need": 4}
        return {"text": "Edge TTS — 인터넷 음성 합성이라 PC 성능과 상관없이 빠름 (목소리 복제가 필요하면 Chatterbox도 CPU에서 문장당 약 8초)",
                "choice": "edge", "key": "engine", "need": 0}
    if task == "image":
        if gpu and vram >= 14:
            return {"text": "FLUX.2 [klein] 9B 4bit — 최고 화질, 장당 약 7초", "choice": "flux2_klein_9b", "key": "image_model", "need": 12}
        if gpu and vram >= 12:
            return {"text": "FLUX.2 [klein] 4B — 프롬프트를 잘 따름, 장당 약 7초", "choice": "flux2_klein_4b", "key": "image_model", "need": 12}
        if gpu and vram >= 8:
            return {"text": "DreamShaper XL Lightning — 장당 1~2초", "choice": "dreamshaper_lightning", "key": "image_model", "need": 8}
        if gpu:
            return {"text": "SDXL-Turbo (GPU + CPU 함께) — 가장 가벼움", "choice": "sdxl_turbo", "key": "image_model", "need": 6}
        return {"text": "SDXL-Turbo (fp32) — CPU에서 가장 빠름, 장당 20~30초 (DreamShaper XL 약 1분, FLUX.2 4B 약 2분 30초)",
                "choice": "sdxl_turbo", "key": "image_model", "need": 0}
    if task == "video":
        if gpu and vram >= 14 and ram >= 32:
            return {"text": "LTX-2.5 22B 4bit — 화질이 훨씬 좋음, 장면당 약 1분 15초 (처음 한 번 약 71GB 내려받기)",
                    "choice": "ltx25", "key": "video_model", "need": 13}
        if gpu and vram >= 10:
            return {"text": "LTX-Video 2B — 장면당 1분 안팎, 화질 낮음", "choice": "ltx2b", "key": "video_model", "need": 10}
        return {"text": "추천하지 않음 — CPU나 작은 GPU에서는 장면 하나에 수십 분~몇 시간. 「AI 이미지 + 움직임」을 쓰세요",
                "choice": "ltx2b", "key": "video_model", "need": 0}
    return {"text": "MusicGen small — 20초 곡에 약 40초" if gpu else "MusicGen small (fp32) — 10초 곡에 약 50초",
            "choice": None, "key": None, "need": 3}


def fits(need_gb: float, s: Specs, device: str) -> str:
    """이 PC·장치 설정에서 추천 모델을 쓸 수 있는지 한 줄 판정."""
    if device == "cpu" or not s.gpus:
        return "CPU로 실행"
    if s.vram_gb >= need_gb:
        return "✅ GPU 메모리 충분"
    if device == "hybrid" and s.ram_gb >= 32:
        return "⚠️ GPU+CPU로 가능 (느림)"
    return "❌ GPU 메모리 부족"
