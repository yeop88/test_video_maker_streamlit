"""피사체 따라 자르기(layout "smart") 확인. GPU 불필요, 30초 정도.

사용법 (프로젝트 루트에서):
    .venv\\Scripts\\python tests\\test_smart_crop.py

1. 흰 원이 왼쪽 끝 → 오른쪽 끝으로 움직이는 1280×720 영상: 세로로 자른 결과에서 원이 늘 화면 가운데 쪽에 있어야 한다.
   (가운데 크롭이면 원이 화면 밖으로 나가는 순간이 있음)
2. 가만히 있는 원: 자르는 위치가 흔들리지 않아야 한다.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from shorts_maker import media, reframe

W, H, FPS, SECONDS = 1280, 720, 30, 8
OUT_W, OUT_H = 1080, 1920
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def disc_video(path: Path, x_of_t) -> None:
    """어두운 잡음 배경 위에 흰 원(반지름 70)이 x_of_t(t) 위치에 있는 영상."""
    rng = np.random.default_rng(0)
    background = (rng.integers(20, 50, (H, W), dtype=np.uint8))
    yy, xx = np.mgrid[0:H, 0:W]
    frames = []
    for i in range(SECONDS * FPS):
        frame = background.copy()
        cx = x_of_t(i / FPS)
        frame[(xx - cx) ** 2 + (yy - H / 2) ** 2 < 70 ** 2] = 240
        frames.append(np.repeat(frame[:, :, None], 3, axis=2))
    raw = path.with_suffix(".rgb")
    raw.write_bytes(np.stack(frames).tobytes())
    media.run_ffmpeg(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", str(raw),
                      "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)])
    raw.unlink()


def render(video: Path, folder: Path, layout: str) -> Path:
    """앱과 같은 render_short로 세로 영상을 만든다 (소리·자막은 비움)."""
    (folder / "fonts").mkdir(exist_ok=True)
    (folder / "empty.ass").write_text("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
                                      "[V4+ Styles]\nFormat: Name, Fontname, Fontsize\nStyle: Default,Arial,40\n\n"
                                      "[Events]\nFormat: Layer, Start, End, Style, Text\n", encoding="utf-8")
    audio = folder / "silence.wav"
    media.run_ffmpeg(["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo", "-t", str(SECONDS), str(audio)])
    crop_x = None
    if layout == "smart":
        info = media.probe(video)
        track = reframe.smooth(reframe.subject_positions(video, info.width, info.height))
        crop_x = reframe.crop_x_expression(track, int(round(info.width * OUT_H / info.height / 2)) * 2, OUT_W)
    out = folder / f"{video.stem}_{layout}.mp4"
    media.render_short(video, audio, "empty.ass", "fonts", out, layout, (OUT_W, OUT_H), cwd=folder, crop_x=crop_x)
    return out


def disc_x_in_output(video: Path, t: float) -> float | None:
    """결과 영상 t초 장면에서 흰 원의 가로 위치 (0~1, 화면에 없으면 None)."""
    raw = media.run_ffmpeg(["-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=270:480,format=gray",
                            "-f", "rawvideo", "-"])
    frame = np.frombuffer(raw[: 270 * 480], np.uint8).reshape(480, 270)
    ys, xs = np.nonzero(frame > 180)
    return float(xs.mean() / 270) if xs.size > 30 else None


folder = Path(tempfile.mkdtemp(prefix="smart_crop_"))
moving = folder / "moving.mp4"
disc_video(moving, lambda t: 120 + (W - 240) * t / SECONDS)  # 8초 동안 왼쪽 끝 → 오른쪽 끝
smart = render(moving, folder, "smart")
center = render(moving, folder, "crop")
times = [0.5, 2.0, 4.0, 6.0, 7.5]
smart_x = [disc_x_in_output(smart, t) for t in times]
center_x = [disc_x_in_output(center, t) for t in times]
print("피사체 따라:", [None if x is None else round(x, 2) for x in smart_x])
print("가운데 크롭:", [None if x is None else round(x, 2) for x in center_x])
check("결과가 1080x1920", (media.probe(smart).width, media.probe(smart).height) == (OUT_W, OUT_H))
check("피사체가 늘 화면 안", all(x is not None for x in smart_x))
check("피사체가 가운데 쪽(0.2~0.8)", all(x is not None and 0.2 <= x <= 0.8 for x in smart_x))
check("(비교) 가운데 크롭은 피사체를 놓침", any(x is None for x in center_x))

still = folder / "still.mp4"
disc_video(still, lambda t: 900)
track = reframe.smooth(reframe.subject_positions(still, W, H))
xs = [c for _, c in track]
check("가만히 있는 피사체에서 흔들리지 않음", max(xs) - min(xs) < 0.02, f"{min(xs):.3f}~{max(xs):.3f}")
check("가만히 있는 피사체 쪽으로 자름", abs(np.median(xs) - 900 / W) < 0.08, f"{np.median(xs):.3f} (정답 {900 / W:.3f})")

print(f"\n{sum(results)}/{len(results)} 통과 (임시 폴더: {folder})")
sys.exit(0 if all(results) else 1)
