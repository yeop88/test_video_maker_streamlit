"""LTX-2.5 동영상 생성 확인 (.venv-ltx에서 실행, Hugging Face 로그인·라이선스 동의 필요).

    .venv-ltx\\Scripts\\python tests\\test_ltx25.py [nf4|fp8] [결과 폴더]

처음에는 약 75GB를 내려받고 양자화해서 workdir/models에 저장하느라 오래 걸린다.
확인하는 것: 704x1280 · 121프레임 · 24fps 세로 영상, 화면이 실제로 움직임, GPU 메모리 최대치, 걸린 시간.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen"))

import truststore  # noqa: E402

truststore.inject_into_ssl()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ltx25 import LTX25  # noqa: E402

QUANT = sys.argv[1] if len(sys.argv) > 1 else "nf4"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "workdir" / "test_ltx25"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "workdir" / "models"
results = []


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def check(name: str, ok: bool, detail="") -> None:
    results.append(bool(ok))
    log(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def probe(path: Path) -> dict:
    import imageio_ffmpeg

    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True).stderr
    import re

    size = re.search(r"Video:.*?(\d{3,4})x(\d{3,4})", out)
    fps = re.search(r"([\d.]+) fps", out)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    return {"size": (int(size[1]), int(size[2])) if size else None, "fps": float(fps[1]) if fps else 0,
            "seconds": int(dur[2]) * 60 + float(dur[3]) if dur else 0}


t = time.time()
model = LTX25(QUANT, str(MODELS), log)
load_sec = time.time() - t
check(f"LTX-2.5 불러오기 ({QUANT})", True, f"{load_sec:.0f}초")

prompts = [
    ("ocean", "ocean waves rolling onto a rocky shore at sunset, golden light, spray in the air, "
              "slow camera push in, vertical composition, cinematic, high detail"),
    ("cafe", "a cup of hot tea on a wooden table by a rainy window, steam rising, raindrops sliding down the glass, "
             "warm indoor light, vertical composition, cinematic"),
]
report = {"quant": QUANT, "load_sec": round(load_sec), "scenes": []}
for name, prompt in prompts:
    out = OUT / f"ltx25_{QUANT}_{name}.mp4"
    torch.cuda.reset_peak_memory_stats()
    steps = []
    t = time.time()
    model.run({"prompt": prompt, "negative": "pc game, console game, video game, cartoon, childish, ugly", "seed": 42,
               "width": 704, "height": 1280, "frames": 121, "fps": 24, "out": str(out)},
              lambda s, n: steps.append(s))
    sec = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    info = probe(out)
    import imageio.v3 as iio

    frames = iio.imread(out, plugin="pyav")
    motion = float(np.abs(frames[5].astype(int) - frames[-5].astype(int)).mean())
    iio.imwrite(OUT / f"ltx25_{QUANT}_{name}_mid.png", frames[len(frames) // 2])
    check(f"{name}: 704x1280 세로 · 약 5초", info["size"] == (704, 1280) and 4.8 <= info["seconds"] <= 5.2,
          f"{info}, 프레임 {len(frames)}")
    check(f"{name}: 화면이 움직임", motion > 3, f"처음·끝 프레임 차이 {motion:.1f}")
    check(f"{name}: 단계 11개 진행 알림", steps and steps[-1] == 11, f"{steps}")
    log(f"{name}: {sec:.0f}초, GPU 최대 {peak:.1f}GB")
    report["scenes"].append({"name": name, "seconds": round(sec), "peak_gpu_gb": round(peak, 1), "motion": round(motion, 1),
                             "video": info})

(OUT / f"report_{QUANT}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{sum(results)}/{len(results)} 통과")
sys.exit(0 if all(results) else 1)
