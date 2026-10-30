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
    "cpu": "CPU만 (GPU 없이, 느림)",
}
DEVICE_HELP = (
    "GPU만: 모델을 GPU에 올려 가장 빠릅니다.\n\n"
    "GPU + CPU 함께: 큰 모델은 RAM에 두고 쓸 부분만 GPU로 옮깁니다 (GPU 메모리가 작을 때, 조금 느림). "
    "Ollama는 GPU에 다 안 들어가는 층을 CPU에서 돌립니다.\n\n"
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
    """{device, video_model, ltx_quant}. 저장된 값이 없거나 잘못되면 이 컴퓨터에 맞는 기본값."""
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
    return values


def save_machine(**values) -> None:
    try:
        saved = json.loads(_device_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    saved.update(values)
    _device_file().parent.mkdir(parents=True, exist_ok=True)
    _device_file().write_text(json.dumps(saved), encoding="utf-8")


# 작업별 추천 모델: (작업, GPU 추천, CPU 추천, 필요한 GPU 메모리 GB(GPU 추천 기준))
RECOMMENDED = [
    ("장면 이해·구간 고르기·나레이션·대본 (Ollama LLM)",
     "qwen3.5:9b — 화면을 보는 비전 모델, 약 6.6GB. GPU 메모리 6GB 이하면 qwen3.5:4b",
     "qwen3.5:4b (약 3.4GB, RAM 8GB 이상) — 9b는 장면 하나 설명에 수십 초. 더 느린 PC는 qwen3.5:2b", 7),
    ("대사 인식 (Whisper)",
     "whisper-large-v3-turbo (fp16, 약 1.6GB) — 지금 쓰는 모델",
     "whisper-large-v3-turbo도 되지만 영상 길이의 1~3배 시간. 1분 이하 영상을 권장", 2),
    ("나레이션 음성",
     "Chatterbox Multilingual (약 3GB) — 목소리 샘플로 복제, 감정 조절",
     "Edge TTS — 인터넷 음성 합성이라 PC 성능과 상관없이 빠름 (Chatterbox는 CPU에서 문장당 수십 초)", 4),
    ("AI 장면 — 이미지 + 움직임",
     "FLUX.2 [klein] 4B (최신, 프롬프트를 잘 따름, 약 13GB를 번갈아) 또는 DreamShaper XL Lightning (약 7GB, 몇 초). "
     "최고 화질은 FLUX.2 [klein] 9B 4bit (비상업). 가장 빠른 것은 SDXL-Turbo",
     "SDXL-Turbo (fp32) — 장면당 30초~1분 정도. CPU에서 쓸 수 있는 유일한 AI 장면 방식", 8),
    ("AI 장면 — 동영상",
     "LTX-2.5 22B 4bit (GPU 16GB 이상 + RAM 32GB 이상, 처음 한 번 약 71GB 내려받기) — 화질이 훨씬 좋음, 장면당 몇 분. "
     "8bit는 RAM 64GB 이상. GPU 12GB 이하면 LTX-Video 2B (장면당 1분 안팎, 화질 낮음)",
     "추천하지 않음 — 장면 하나에 수십 분~몇 시간. AI 이미지 + 움직임을 쓰세요", 16),
    ("배경음악",
     "MusicGen small (약 2GB) — 20초 곡에 약 40초",
     "MusicGen small — 20초 곡에 몇 분", 3),
]


def fits(need_gb: float, s: Specs, device: str) -> str:
    """이 PC·장치 설정에서 GPU 추천 모델을 쓸 수 있는지 한 줄 판정."""
    if device == "cpu" or not s.gpus:
        return "CPU 추천을 씀"
    if s.vram_gb >= need_gb:
        return "✅ GPU 추천 가능"
    if device == "hybrid" and s.ram_gb >= 32:
        return "⚠️ GPU+CPU로 가능 (느림)"
    return "❌ GPU 메모리 부족"
