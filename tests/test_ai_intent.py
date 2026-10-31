"""영상 없이 AI로 만든 쇼츠가 주제(사용자가 쓴 프롬프트)의 의도와 맞는지 검증한다.

주제마다: AI 대본 → 장면 생성 → (선택) 쇼츠 만들기
- 대본: 제목·나레이션에 주제의 핵심 낱말이 있는가, 장면 영어 프롬프트마다 주인공 피사체가 들어 있는가
- 장면: 만든 장면의 가운데 화면을 비전 LLM에게 보여 주고 「주인공 피사체가 보이는가」「장면 설명과 맞는가」를 묻는다
- 쇼츠: 고른 출력 크기(--size, 예: 480x640)로 만들어지는가, 소리·자막·길이

GPU, Ollama(qwen3.5:9b), AI 생성 환경(.venv-gen / .venv-ltx)이 필요하다. 생성 모델은 이 컴퓨터 설정(workdir/device.json)을 따른다.
장면 화면 모음(<주제>_scenes.png)은 사람이 직접 보고 확인한다.

실행: .venv\\Scripts\\python tests\\test_ai_intent.py <결과 폴더> [--size 480x640] [--engine image|video|topic] [--no-render]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ollama  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from shorts_maker import config, generate, llm, media, pipeline, system, tts  # noqa: E402

MODEL = "qwen3.5:9b"
TOPICS = [
    {"name": "airplane", "topic": "비행기가 만들어지는 과정의 핵심을 빠르게 설명하는 쇼츠", "engine": "video",
     "ko": ["비행기", "항공기"], "en": ["airplane", "aircraft", "airliner", "plane", "jet"],
     "subject": "an airplane, or clearly recognizable airplane parts (fuselage, wing, jet engine, cockpit, airplane cabin)"},
    {"name": "kimchi_rice", "topic": "초보자를 위한 김치볶음밥 만드는 법", "engine": "image",
     "ko": ["김치볶음밥", "볶음밥"], "en": ["kimchi", "fried rice"],
     "subject": "kimchi fried rice, or its ingredients/cooking (kimchi, rice, frying pan)"},
    {"name": "autumn_lake", "topic": "가을 단풍이 물든 산속 호수의 고요한 아침", "engine": "image",
     "ko": ["단풍", "호수"], "en": ["autumn", "maple", "lake", "fall foliage"],
     "subject": "an autumn mountain lake landscape with colorful fall foliage"},
    {"name": "cat_day", "topic": "고양이가 하루 동안 하는 일", "engine": "image",
     "ko": ["고양이"], "en": ["cat", "kitten"],
     "subject": "a cat"},
]

JUDGE = """You check whether one frame of an AI-generated video scene matches what the creator intended.
Topic of the whole short video: {topic}
Main subject that should be visible in every scene: {subject}
Intended content of this scene (Korean): {scene}

Look only at the image. Answer in JSON:
- what_is_shown: what the image actually shows, one short English sentence
- subject_visible: true only if the main subject is clearly visible in the image
- matches_scene: true only if the image shows the main content of the intended scene (ignore minor style, color or camera differences)"""


class Verdict(BaseModel):
    what_is_shown: str
    subject_visible: bool
    matches_scene: bool


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def judge(image_png: bytes, t: dict, scene_ko: str) -> Verdict | None:
    path = OUT / "_judge.png"
    path.write_bytes(image_png)
    for _ in range(3):
        resp = ollama.chat(model=MODEL, format=Verdict.model_json_schema(), think=False, keep_alive="2m",
                           options={"temperature": 0, "num_predict": 400},
                           messages=[{"role": "user", "images": [str(path)],
                                      "content": JUDGE.format(topic=t["topic"], subject=t["subject"], scene=scene_ko)}])
        try:
            return Verdict.model_validate_json(resp.message.content)
        except Exception:
            continue
    return None


def frame_png(video: Path, t: float, width: int = 360) -> bytes:
    return media.run_ffmpeg(["-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2",
                             "-f", "image2pipe", "-vcodec", "png", "-"])


def contains_any(text: str, words: list[str]) -> bool:
    low = text.lower()
    return any(w.lower() in low for w in words)


def run_topic(t: dict, engine: str, size: tuple[int, int] | None, render: bool) -> dict:
    log(f"===== {t['name']} ({engine}) : {t['topic']} =====")
    r = {"name": t["name"], "topic": t["topic"], "engine": engine}
    t0 = time.time()
    title, cards = pipeline.plan_ai(MODEL, t["topic"], 30, llm.STYLES["정보 전달 (차분한 설명)"], engine)
    r["script_sec"] = round(time.time() - t0, 1)
    r["title"] = title
    r["scenes"] = [{k: c.get(k) for k in ("prompt_ko", "prompt", "narration", "seconds", "engine")} for c in cards]
    narration = " ".join(c["narration"] for c in cards)
    with_subject = [contains_any(c["prompt"], t["en"]) for c in cards]
    r["prompt_has_subject"] = with_subject
    log(f"대본 {r['script_sec']}초: 「{title}」 장면 {len(cards)}개, 프롬프트에 주인공 {sum(with_subject)}/{len(cards)}")

    t1 = time.time()
    scenes = [{"prompt": c["prompt"], "engine": c["engine"], "seed": int(c["seed"]), "seconds": float(c["seconds"]),
               **({"size": list(size)} if size else {})} for c in cards]
    clips = generate.scene_clips(scenes)
    r["generate_sec"] = round(time.time() - t1, 1)
    log(f"장면 생성 {r['generate_sec']}초")

    verdicts, tiles = [], []
    for i, (c, clip) in enumerate(zip(cards, clips)):
        dur = media.probe(clip).duration
        png = frame_png(clip, dur / 2)
        v = judge(png, t, c["prompt_ko"])
        verdicts.append(v.model_dump() if v else None)
        tiles.append((Image.open(io.BytesIO(png)).convert("RGB"), i, v))
        log(f"  장면 {i}: 주인공 {'O' if v and v.subject_visible else 'X'} / 설명 {'O' if v and v.matches_scene else 'X'}"
            f"  | 의도: {c['prompt_ko'][:40]} | 보임: {v.what_is_shown[:70] if v else '?'}")
    r["verdicts"] = verdicts
    ok = [v for v in verdicts if v]
    r["subject_rate"] = round(sum(v["subject_visible"] for v in ok) / max(1, len(cards)), 2)
    r["match_rate"] = round(sum(v["matches_scene"] for v in ok) / max(1, len(cards)), 2)

    # 사람이 직접 볼 장면 모음 (O/X 표시)
    w, h = tiles[0][0].size
    cols = 4
    sheet = Image.new("RGB", (w * cols, (h + 34) * ((len(tiles) + cols - 1) // cols)), "black")
    for n, (im, i, v) in enumerate(tiles):
        x, y = (n % cols) * w, (n // cols) * (h + 34)
        sheet.paste(im.resize((w, h)), (x, y + 34))
        mark = f"{i}: subject {'O' if v and v.subject_visible else 'X'} scene {'O' if v and v.matches_scene else 'X'}"
        ImageDraw.Draw(sheet).text((x + 4, y + 4), mark, fill="yellow")
        ImageDraw.Draw(sheet).text((x + 4, y + 18), cards[i]["prompt"][:58], fill="white")
    sheet.save(OUT / f"{t['name']}_{TAG}_scenes.png")

    checks = {
        "title_or_narration_on_topic": contains_any(title + " " + narration, t["ko"]),
        "subject_visible>=0.8": r["subject_rate"] >= 0.8,
        "scene_matches>=0.75": r["match_rate"] >= 0.75,
    }
    if render:
        t2 = time.time()
        out, report = pipeline.make_short(None, title, cards, tts.EdgeTTSEngine("ko-KR-SunHiNeural", 0), 0.8, 0.2,
                                          "crop", 72, 24, "#FFFFFF", work_dir=OUT / f"{t['name']}_{TAG}_work",
                                          out_dir=OUT, subtitle_style="karaoke", title_style="bold",
                                          **({"size": size} if size else {}))
        r["render_sec"] = round(time.time() - t2, 1)
        info = media.probe(out)
        r["output"] = f"{info.width}x{info.height} {info.duration:.1f}s audio={info.has_audio}"
        r["output_path"] = str(out)
        want = size or config.OUTPUT_SIZE
        checks["output_size"] = (info.width, info.height) == tuple(want) and info.has_audio
        checks["length_20_45s"] = 20 <= info.duration <= 45
        checks["no_narration_cut"] = not any("잘림" in x["상태"] for x in report)
        for k, ts in enumerate((1.5, info.duration / 2)):
            (OUT / f"{t['name']}_{TAG}_frame{k}.png").write_bytes(frame_png(out, ts, width=info.width))
        log(f"쇼츠 {r['render_sec']}초: {r['output']}")
    r["checks"] = checks
    r["passed"] = all(checks.values())
    log(f"{t['name']} passed={r['passed']} 주인공 {r['subject_rate']:.0%} 설명 {r['match_rate']:.0%} {checks}")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--size", default="")
    ap.add_argument("--engine", default="topic", help="image | video | topic(주제마다 정한 방식)")
    ap.add_argument("--topics", default="", help="쉼표로 고른 주제 이름 (비우면 전부)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    TAG = args.tag
    size = tuple(int(x) for x in args.size.lower().split("x")) if args.size else None

    machine = system.load_machine(system.specs())  # 이 컴퓨터의 생성 모델 설정 (LLM은 시간 때문에 GPU)
    generate.set_image_model(machine.get("image_model", "sdxl_turbo"))
    generate.set_video_model(machine.get("video_model", "ltx2b"), machine.get("ltx_quant", "nf4"))
    generate.set_task_devices({"image": "gpu", "video": "gpu", "music": "gpu"})
    llm.DEVICE = "gpu"
    ok, message = generate.available()
    if not ok:
        sys.exit(f"AI 생성 환경이 없습니다: {message}")
    log(f"이미지 {generate.settings['image_model']}, 동영상 {generate.settings['video_model']}, 크기 {size or config.OUTPUT_SIZE}")

    chosen = [t for t in TOPICS if not args.topics or t["name"] in args.topics.split(",")]
    results = []
    for t in chosen:
        engine = t["engine"] if args.engine == "topic" else args.engine
        results.append(run_topic(t, engine, size, not args.no_render))
        (OUT / f"results_{TAG}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    generate.stop_idle()
    total = sum(len(r["scenes"]) for r in results)
    subj = sum(r["subject_rate"] * len(r["scenes"]) for r in results) / max(1, total)
    match = sum(r["match_rate"] * len(r["scenes"]) for r in results) / max(1, total)
    log("ALL DONE " + " ".join(f"{r['name']}={'PASS' if r['passed'] else 'FAIL'}" for r in results)
        + f" | 장면 {total}개: 주인공 보임 {subj:.0%}, 설명과 맞음 {match:.0%}")
