"""AI 생성 확인: 장면 이미지 클립, 장면 영상 클립, 배경음악. 생성 환경(.venv-gen)과 GPU 필요, 처음에는 2~3분.

사용법 (프로젝트 루트에서):
    .venv\\Scripts\\python tests\\test_generate.py

캐시(workdir/gen_cache)를 피하려고 실행할 때마다 다른 시드를 쓴다.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from shorts_maker import generate, media

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


ok, message = generate.available()
check("생성 환경 사용 가능", ok, message)
if not ok:
    sys.exit(1)
seed = int(time.time()) % 100000

t = time.time()
steps = []
clips = generate.scene_clips(
    [{"prompt": "a calm mountain lake at sunrise", "engine": "image", "seed": seed, "seconds": 4.5},
     {"prompt": "a cup of coffee steaming on a wooden table", "engine": "image", "seed": seed + 1, "seconds": 3.0}],
    on_progress=lambda d, n: steps.append((d, n)))
infos = [media.probe(c) for c in clips]
check("이미지 장면 → 1080x1920 클립", all((i.width, i.height) == (1080, 1920) for i in infos))
check("이미지 장면 길이 = 요청 길이", abs(infos[0].duration - 4.5) < 0.1 and abs(infos[1].duration - 3.0) < 0.1,
      [round(i.duration, 2) for i in infos])
check("진행 상황 알림", steps and steps[-1] == (2, 2), steps)
again = generate.scene_clips([{"prompt": "a calm mountain lake at sunrise", "engine": "image", "seed": seed, "seconds": 4.5}])
check("같은 요청은 캐시 사용", again[0] == clips[0])
print(f"  이미지 2장: {time.time() - t:.0f}초")

t = time.time()
video = generate.scene_clips([{"prompt": "ocean waves on a rocky shore at sunset", "engine": "video", "seed": seed}])[0]
info = media.probe(video)
check("AI 동영상 장면 (512x768, 약 4초)", (info.width, info.height) == generate.VIDEO_SIZE and 3.5 < info.duration < 4.5,
      f"{info.width}x{info.height} {info.duration:.2f}초, {time.time() - t:.0f}초 걸림")
frames = [media.run_ffmpeg(["-ss", f"{x}", "-i", str(video), "-frames:v", "1", "-vf", "scale=64:96,format=gray",
                            "-f", "rawvideo", "-"]) for x in (0.3, 3.5)]
diff = float(np.abs(np.frombuffer(frames[0], np.uint8).astype(int) - np.frombuffer(frames[1], np.uint8).astype(int)).mean())
check("AI 동영상이 실제로 움직임", diff > 3, f"처음·끝 장면 차이 {diff:.1f}")

t = time.time()
song = generate.music(f"calm ambient piano, soft, relaxing, take {seed}", 8)
wav = media.load_audio(song, 32000, 1)[:, 0]
check("AI 음악 길이", abs(len(wav) / 32000 - 8) < 0.5, f"{len(wav) / 32000:.1f}초, {time.time() - t:.0f}초 걸림")
check("AI 음악 소리 크기", 0.03 < float(np.sqrt((wav ** 2).mean())) < 0.5)

print(f"\n{sum(results)}/{len(results)} 통과")
sys.exit(0 if all(results) else 1)
