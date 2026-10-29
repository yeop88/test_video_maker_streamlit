"""ffmpeg / ffprobe 래퍼."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def ensure_ffmpeg() -> None:
    """시스템에 ffmpeg가 없으면 static-ffmpeg 바이너리를 PATH에 추가한다."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    import static_ffmpeg

    static_ffmpeg.add_paths()  # 첫 실행 때 바이너리를 내려받음


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> bytes:
    ensure_ffmpeg()
    cmd =["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")[-2000:]
        raise RuntimeError(f"ffmpeg 실행 실패:\n{err}")
    return proc.stdout


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    ensure_ffmpeg()
    proc =subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("영상 파일을 읽을 수 없습니다. 손상되었거나 지원하지 않는 형식입니다.")
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if video is None:
        raise RuntimeError("영상 스트림이 없는 파일입니다.")
    return VideoInfo(
        duration=float(data["format"]["duration"]),
        width=int(video["width"]),
        height=int(video["height"]),
        has_audio=any(s["codec_type"] == "audio" for s in streams),
    )


def probe_audio_seconds(path: Path) -> float:
    """오디오 파일 길이(초). 영상 스트림이 없어도 된다."""
    ensure_ffmpeg()
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                          capture_output=True)
    try:
        return float(proc.stdout.decode().strip())
    except ValueError:
        raise RuntimeError("음악 파일을 읽을 수 없습니다. 손상되었거나 지원하지 않는 형식입니다.") from None


def trim(src: Path, dst: Path, start: float, end: float) -> None:
    """정확한 구간으로 자르기 위해 다시 인코딩한다."""
    run_ffmpeg([
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", str(dst),
    ])


def extract_audio(src: Path, dst: Path, sr: int = 44100, channels: int = 2) -> None:
    run_ffmpeg(["-i", str(src), "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])


def extract_frame(src: Path, t: float, dst: Path, width: int) -> None:
    """t초 화면을 width 폭의 JPG로 저장한다 (-ss를 입력 앞에 둬서 빠르게 찾음)."""
    run_ffmpeg(["-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(dst)])


def cut_and_concat(src: Path, clips: list[tuple[float, float]], dst: Path, hold_sec: float = 0.0) -> None:
    """원본에서 여러 구간을 잘라 순서대로 이어 붙인 영상(소리 없음)을 만든다. hold_sec만큼 마지막 화면을 멈춰 늘린다."""
    parts = [
        f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,fps=30,format=yuv420p[v{i}]"
        for i, (a, b) in enumerate(clips)
    ]
    inputs = "".join(f"[v{i}]" for i in range(len(clips)))
    hold = f",tpad=stop_mode=clone:stop_duration={hold_sec:.3f}" if hold_sec > 0 else ""
    graph = ";".join(parts) + f";{inputs}concat=n={len(clips)}:v=1:a=0{hold}[out]"
    run_ffmpeg([
        "-i", str(src), "-filter_complex", graph, "-map", "[out]", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst),
    ])


def load_audio(path: Path, sr: int, channels: int = 1, tempo: float = 1.0) -> np.ndarray:
    """오디오를 (샘플 수, 채널) float32 배열로 읽는다. tempo > 1이면 음높이 유지한 채 빠르게."""
    filters = ["-af", f"atempo={tempo:.4f}"] if abs(tempo - 1.0) > 1e-3 else []
    raw = run_ffmpeg(["-i", str(path), *filters, "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"])
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).copy()


def render_short(
    video: Path,
    audio: Path,
    ass_name: str,
    fonts_dirname: str,
    out: Path,
    layout: str,
    size: tuple[int, int],
    cwd: Path,
    crop_x: str | None = None,
) -> None:
    """세로 영상으로 변환하고 자막을 입힌다. layout "smart"는 crop_x(시간 t에 대한 식) 위치를 잘라 낸다.

    ass 필터는 Windows 경로(C:\\...)의 콜론을 파싱하지 못하므로,
    자막과 폰트는 cwd 기준 상대 경로로 넘긴다.
    """
    w, h = size
    if layout == "smart" and crop_x is not None:  # 높이를 맞추고, 피사체를 따라 가로 위치를 옮기며 자름
        base = f"[0:v]scale=-2:{h},crop={w}:{h}:x='{crop_x}':y=0,setsar=1[base]"
    elif layout in ("crop", "smart"):
        base = f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1[base]"
    else:  # blur: 원본은 가운데, 위아래는 흐린 배경
        base = (
            "[0:v]split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
            "boxblur=25:2,eq=brightness=-0.08[bg];"
            f"[fgsrc]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
        )
    graph = f"{base};[base]ass={ass_name}:fontsdir={fonts_dirname}[out]"
    run_ffmpeg(
        [
            "-i", str(video), "-i", str(audio),
            "-filter_complex", graph,
            "-map", "[out]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(out),
        ],
        cwd=cwd,
    )
