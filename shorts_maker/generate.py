"""AI 생성: 장면 이미지·영상 클립, 배경음악. 실제 생성은 gen/worker.py를 생성 전용 가상환경(.venv-gen)에서 돌린다.

결과는 workdir/gen_cache/에 (종류, 프롬프트, 시드, 설정) 해시 이름으로 저장해서 같은 요청은 다시 만들지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from . import config, media

GEN_DIR = config.ROOT / ".venv-gen"
GEN_PYTHON = GEN_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
WORKER = config.ROOT / "gen" / "worker.py"

# 장면 생성 방식
ENGINES = {
    "image": "AI 이미지 + 움직임 (빠름, 장면당 몇 초)",
    "video": "AI 동영상 (LTX-Video, 장면당 1분 안팎)",
}
IMAGE_SIZE = (576, 1024)  # SDXL-Turbo 세로 이미지
VIDEO_SIZE = (512, 768)  # LTX-Video 세로 영상 (32의 배수)
VIDEO_FRAMES, VIDEO_FPS = 97, 24  # 약 4초
STYLE_SUFFIX = "vertical composition, cinematic, high detail, natural light"
NEGATIVE = "text, watermark, logo, letters, low quality, blurry, deformed"

_available: tuple[bool, str] | None = None
Progress = Callable[[int, int], None]


def available() -> tuple[bool, str]:
    """(쓸 수 있는지, 안내 문구). 생성 환경이 없으면 설치 방법을 알려 준다."""
    global _available
    if _available is None:
        if not GEN_PYTHON.exists():
            _available = (False, "AI 생성 환경이 없습니다. README의 「AI 생성 기능 설치」대로 .venv-gen을 만드세요.")
        else:
            proc = subprocess.run([str(GEN_PYTHON), "-c", "import torch, diffusers, transformers; print(torch.cuda.is_available())"],
                                  capture_output=True, text=True)
            ok = proc.returncode == 0 and proc.stdout.strip() == "True"
            _available = (ok, "" if ok else "AI 생성 환경(.venv-gen)을 불러오지 못했거나 GPU를 찾지 못했습니다.")
    return _available


def cache_dir() -> Path:
    folder = config.WORK_DIR / "gen_cache"
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


def run_worker(kind: str, tasks: list[dict], on_progress: Progress | None = None) -> None:
    ok, message = available()
    if not ok:
        raise RuntimeError(message)
    _free_gpu()
    spec = cache_dir() / f"job_{kind}_{_key(json.dumps(tasks, ensure_ascii=False))}.json"
    spec.write_text(json.dumps({"kind": kind, "tasks": tasks}, ensure_ascii=False), encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([str(GEN_PYTHON), str(WORKER), str(spec)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", env=env, cwd=config.ROOT)
    for line in proc.stdout:  # "PROGRESS 2 5"
        parts = line.split()
        if len(parts) == 3 and parts[0] == "PROGRESS" and on_progress:
            on_progress(int(parts[1]), int(parts[2]))
    err = proc.stderr.read()
    proc.wait()
    spec.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"AI 생성({kind}) 실패:\n{err[-2500:]}")


def image_to_clip(image: Path, seconds: float, out: Path, motion: int = 0) -> None:
    """정지 이미지를 천천히 확대·이동하는 1080x1920 클립으로 (켄 번스 효과). motion으로 방향을 바꾼다."""
    w, h = config.OUTPUT_SIZE
    frames = max(1, int(round(seconds * 30)))
    zoom = "min(zoom+0.0009,1.18)" if motion % 2 == 0 else "if(eq(on,0),1.18,max(zoom-0.0009,1.0))"
    x = ["iw/2-(iw/zoom/2)", "(iw-iw/zoom)*on/{f}", "(iw-iw/zoom)*(1-on/{f})"][motion % 3].format(f=frames)
    media.run_ffmpeg([
        "-loop", "1", "-i", str(image),
        "-vf", f"scale={w * 2}:{h * 2},zoompan=z='{zoom}':x='{x}':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps=30,"
               "format=yuv420p",
        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out),
    ])


def scene_clips(scenes: list[dict], on_progress: Progress | None = None) -> list[Path]:
    """scenes: [{prompt, engine("image"|"video"), seed, seconds}] → 장면마다 영상 클립 경로 (캐시 사용).
    image는 이미지 한 장을 seconds 길이 움직임 클립으로, video는 약 4초 클립 그대로 (길이 맞춤은 렌더링 때)."""
    folder = cache_dir()
    outs, todo = [], {"image": [], "video": []}
    for s in scenes:
        prompt = f"{s['prompt'].strip()}, {STYLE_SUFFIX}"
        engine = s.get("engine", "image")
        if engine == "video":
            out = folder / f"vid_{_key('video', prompt, s.get('seed', 0), VIDEO_SIZE, VIDEO_FRAMES)}.mp4"
            if not out.exists():
                todo["video"].append({"prompt": prompt, "negative": NEGATIVE, "seed": s.get("seed", 0),
                                      "width": VIDEO_SIZE[0], "height": VIDEO_SIZE[1], "frames": VIDEO_FRAMES,
                                      "fps": VIDEO_FPS, "out": str(out)})
        else:
            out = folder / f"img_{_key('image', prompt, s.get('seed', 0), IMAGE_SIZE)}.png"
            if not out.exists():
                todo["image"].append({"prompt": prompt, "negative": NEGATIVE, "seed": s.get("seed", 0),
                                      "width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1], "out": str(out)})
        outs.append(out)
    total = sum(len(v) for v in todo.values())
    done = 0

    def step(d: int, n: int) -> None:
        if on_progress:
            on_progress(done + d, total)

    for kind in ("image", "video"):
        if todo[kind]:
            run_worker(kind, todo[kind], step)
            done += len(todo[kind])
    clips = []
    for s, out in zip(scenes, outs):  # 이미지는 움직임 클립으로 바꾼다 (길이별로 캐시)
        if out.suffix == ".png":
            seconds = round(float(s.get("seconds", 5.0)), 1)
            clip = out.with_name(f"{out.stem}_{seconds}s_m{s.get('seed', 0) % 6}.mp4")
            if not clip.exists():
                image_to_clip(out, seconds, clip, motion=s.get("seed", 0))
            out = clip
        clips.append(out)
    return clips


def music(prompt: str, seconds: float, on_progress: Progress | None = None) -> Path:
    """MusicGen 배경음악 (최대 30초, 더 긴 쇼츠는 배경음악 반복으로 채움)."""
    seconds = min(30.0, max(5.0, seconds))
    out = cache_dir() / f"music_{_key('music', prompt, seconds)}.wav"
    if not out.exists():
        run_worker("music", [{"prompt": prompt, "seconds": seconds, "out": str(out)}], on_progress)
    return out


def scene_thumbnail(clip: Path) -> Path:
    """카드에 보여 줄 장면 첫 화면 JPG."""
    thumb = clip.with_suffix(".jpg")
    if not thumb.exists():
        media.run_ffmpeg(["-ss", "0.5", "-i", str(clip), "-frames:v", "1", "-vf", "scale=270:-2", str(thumb)])
    return thumb


if __name__ == "__main__":  # 빠른 확인: python -m shorts_maker.generate "프롬프트"
    print(available())
    print(scene_clips([{"prompt": sys.argv[1] if len(sys.argv) > 1 else "a calm lake at sunrise", "seconds": 4}]))
