"""작업 큐(jobqueue)가 차례로 처리하는 작업들. 화면(streamlit)을 쓰지 않고 백그라운드 스레드에서 돈다.

작업마다 넣을 때의 설정(settings)을 함께 받는다: 프리셋 설정 이름(presets.DEFAULTS) + llm_model + devices(작업별 장치).
작업 결과는 project.update로 저장된 최신 상태에 반영해서, 그동안 화면에서 고친 내용을 덮어쓰지 않는다.
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

from . import config, generate, jobqueue, llm, media, music, pipeline, project, tts, voices

PREVIEW_TEXT = "안녕하세요. 이 목소리로 쇼츠 나레이션을 읽어 드릴게요."
VOICE_LABELS = {voice_id: label for label, voice_id in tts.EDGE_VOICES.items()}
LAYOUT_SHORT = {"blur": "블러 배경", "crop": "가운데 크롭", "smart": "피사체 따라 크롭"}

# ───────────── 모델 (큐 스레드에서만 쓰므로 한 번에 하나씩) ─────────────
_models: dict[tuple, object] = {}
_models_lock = threading.Lock()


def _model(name: str, device: str, loader):
    with _models_lock:
        key = (name, device)
        if key not in _models:
            for old in [k for k in _models if k[0] == name]:  # 다른 장치에 올린 같은 모델은 내린다
                _models.pop(old)
            _models[key] = loader(device)
        return _models[key]


def whisper(device: str):
    from . import asr

    return _model("whisper", device, asr.load_whisper)


def chatterbox(device: str):
    return _model("chatterbox", device, tts.load_chatterbox)


def unload_models() -> None:
    """계산 장치를 바꾸면 불러 둔 Whisper·Chatterbox를 내린다 (다음에 쓸 때 새 장치로 불러옴)."""
    with _models_lock:
        _models.clear()


# ───────────── 공통 ─────────────
def _settings(item: dict) -> dict:
    s = item["params"]["settings"]
    llm.DEVICE = s["devices"]["llm"]
    return s


def _style(s: dict) -> str:
    return llm.STYLES[s["style"]]


def _quiet(report):
    """생성 프로세스 콜백용: 취소는 generate.cancel()로 따로 하므로 여기서는 예외를 내지 않는다."""
    def call(**kwargs):
        try:
            report(**kwargs)
        except jobqueue.Cancelled:
            pass
    return call


def _short(p: project.Project, short_id: str) -> dict:
    short = next((s for s in p.shorts if s["id"] == short_id), None)
    if short is None:
        raise RuntimeError("이 쇼츠가 지워졌습니다.")
    return short


def _with_uids(clips: list[dict]) -> list[dict]:
    for c in clips:
        c.setdefault("uid", uuid.uuid4().hex[:8])
    return clips


def _folder(item: dict) -> Path:
    return Path(item["project_dir"])


# ───────────── 분석·구성·나레이션 ─────────────
def analyze(item: dict, report) -> dict:
    s, p = _settings(item), item["params"]
    folder = _folder(item)
    proj = project.load(folder)
    model = s["llm_model"]
    report(line="✂️ 구간 자르기, 오디오 추출")
    job = pipeline.prepare_job(proj.source_path, p["start"], p["end"])
    report(line="📝 대사 인식 (Whisper)")
    pipeline.transcribe(job, whisper(s["devices"]["asr"]))
    report(line="🖼️ 장면 캡처")
    pipeline.capture_frames(job)
    report(line=f"👀 장면 이해 ({model})")
    pipeline.describe_frames(job, model, on_progress=lambda d, n: report(step=d, total=n, note=f"👀 장면 이해 {d} / {n}"))
    project.update(folder, lambda pr: setattr(pr, "job", job))  # 구성에서 실패해도 분석 결과는 남긴다
    _plan_all(folder, s, report)
    return {}


def _plan_all(folder: Path, s: dict, report) -> None:
    proj = project.load(folder)
    count = int(s["short_count"])
    report(line=f"🧠 서로 다른 주제로 쇼츠 {count}편 나누기 + 편마다 구간·나레이션" if count > 1
           else "🧠 하이라이트 구간 고르기 + 나레이션 대본 쓰기")
    shorts = pipeline.plan_series(proj.job, s["llm_model"], count, s["target_sec"], _style(s),
                                  on_progress=lambda d, n: report(step=d, total=n, note=f"🧠 {d} / {n}편 구성"))
    if not shorts or not any(x["clips"] for x in shorts):
        raise RuntimeError("쓸 만한 구간을 고르지 못했습니다. 다시 구성하거나 쇼츠 길이를 바꿔 보세요.")
    for x in shorts:
        _with_uids(x["clips"])
    project.update(folder, lambda pr: setattr(pr, "shorts", shorts))


def plan(item: dict, report) -> dict:
    """구성 다시 만들기. short_id가 없으면 전체(여러 편이면 주제 나누기부터), 있으면 그 편만. AI 작업은 주제로 대본부터."""
    s, p = _settings(item), item["params"]
    folder = _folder(item)
    proj = project.load(folder)
    short_id = p.get("short_id")
    model = s["llm_model"]
    if proj.kind == "ai":
        engine = p.get("engine")
        if short_id and not engine:
            engines = {c.get("engine", "image") for c in _short(proj, short_id)["clips"] if c.get("kind") == "ai"}
            engine = engines.pop() if len(engines) == 1 else "mix"
        report(line=f"✨ {model}(으)로 대본과 장면 구성 쓰기")
        title, cards = pipeline.plan_ai(model, proj.topic, s["target_sec"], _style(s), engine or "image")

        def apply(pr: project.Project) -> None:
            if short_id and any(x["id"] == short_id for x in pr.shorts):
                _short(pr, short_id).update(title=title, clips=cards)
            else:
                pr.shorts = [{"id": uuid.uuid4().hex[:8], "title": title, "range": None, "focus": "", "clips": cards}]
        project.update(folder, apply)
        return {}
    if proj.job is None:
        raise RuntimeError("영상 분석 결과가 없습니다. 1단계부터 다시 하세요.")
    if not short_id:
        _plan_all(folder, s, report)
        return {}
    short = _short(proj, short_id)
    report(line="🧠 이 쇼츠의 구간과 나레이션 다시 구성")
    span = tuple(short.get("range") or (0.0, proj.job.duration))
    title, clips = pipeline.plan(proj.job, model, s["target_sec"], _style(s),
                                 span=span if len(proj.shorts) > 1 else None, focus=short.get("focus", ""))
    project.update(folder, lambda pr: _short(pr, short_id).update(title=title, clips=_with_uids(clips)))
    return {}


def narrate(item: dict, report) -> dict:
    """구간 시간은 그대로 두고 나레이션만 다시 쓴다."""
    s, p = _settings(item), item["params"]
    folder = _folder(item)
    proj = project.load(folder)
    short = _short(proj, p["short_id"])
    clips = [c for c in short["clips"] if c.get("kind") == "ai" or c["end"] > c["start"]]
    report(line=f"✍️ 구간 {len(clips)}개 나레이션 쓰기")
    texts = {c["uid"]: c["narration"] for c in pipeline.narrate(proj.job, clips, s["llm_model"], _style(s))}

    def apply(pr: project.Project) -> None:
        for c in _short(pr, p["short_id"])["clips"]:
            if c.get("uid") in texts:
                c["narration"] = texts[c["uid"]]
    project.update(folder, apply)
    return {}


# ───────────── 쇼츠 만들기 ─────────────
def _ensure_prompt(clip: dict, model: str | None) -> None:
    """장면 설명(한글 가능)을 생성 모델용 영어 프롬프트로 (설명이 바뀌었을 때만 다시 번역)."""
    text = clip.get("prompt_ko", "").strip()
    if text and clip.get("prompt_src") != text:
        clip["prompt"] = llm.english_prompt(model, text) if model else text
        clip["prompt_src"] = text


def settings_summary(s: dict, cards: list[dict]) -> str:
    """결과 기록에 남길 설정 한 줄 요약 (나란히 비교할 때 무엇이 달랐는지 보이게)."""
    if s["engine"] == "edge":
        voice = f"{VOICE_LABELS.get(s['edge_voice'], s['edge_voice'])} {s['rate']:+d}%"
    else:
        chosen = next((v for v in voices.list_voices() if v["id"] == s["voice_id"]), None)
        voice = f"Chatterbox {chosen['name'] if chosen else '기본 목소리'}"
    subs = "단어 강조 자막" if s["subtitle_style"] == "karaoke" else "기본 자막"
    track = next((t for t in music.list_tracks() if t["id"] == s["bgm_track"]), None)
    bgm = f"배경음악 {track['name']} {s['bgm_gain']:.2f}" if track else "배경음악 없음"
    text = (f"{s['style']} · {voice} · {LAYOUT_SHORT.get(s['layout'], s['layout'])} · {subs} · "
            f"원본 소리 {s['original_gain']:.2f}/{s['duck_gain']:.2f} · {bgm}")
    kinds = sorted({c.get("engine", "image") for c in cards if c.get("kind") == "ai"})
    if kinds:
        text += f" · LLM {s['llm_model']} · AI 장면 {', '.join(generate.MODEL_INFO[k]['name'] for k in kinds)}"
    return text


def tts_engine(s: dict):
    if s["engine"] == "edge":
        return tts.EdgeTTSEngine(s["edge_voice"], s["rate"])
    return tts.ChatterboxEngine(chatterbox(s["devices"]["tts"]), voices.sample_path(s["voice_id"]), s["exaggeration"])


def make(item: dict, report) -> dict:
    """쇼츠를 만들어 결과 기록에 넣는다 (short_ids 순서대로). 카드는 작업이 시작될 때 저장된 최신 값을 쓴다."""
    s, p = _settings(item), item["params"]
    folder = _folder(item)
    engine = tts_engine(s)
    made = []
    for n, short_id in enumerate(p["short_ids"]):
        proj = project.load(folder)
        short = _short(proj, short_id)
        title = short["title"]
        prefix = f"[{n + 1}/{len(p['short_ids'])}] " if len(p["short_ids"]) > 1 else ""
        cards = [c for c in short["clips"] if c.get("kind") == "ai" or c["end"] > c["start"]]
        for c in cards:
            if c.get("kind") == "ai":
                if not c.get("prompt_ko", "").strip():
                    raise RuntimeError(f"'{title}'에 장면 설명이 비어 있는 AI 장면이 있습니다.")
                _ensure_prompt(c, s["llm_model"])
        clips = pipeline.normalize_clips(proj.job, cards) if proj.job else cards
        if not clips:
            raise RuntimeError(f"'{title}'에 쓸 구간이 없습니다.")
        video, rep = pipeline.make_short(
            proj.job, title if s["show_title"] else "", clips, engine, s["original_gain"], s["duck_gain"],
            s["layout"], s["font_size"], s["margin_percent"], s["color"],
            on_progress=lambda d, t: report(step=d, total=t, note=f"{prefix}🎙️ {title} · {d} / {t} 문장"),
            out_dir=folder / "results",
            subtitle_style=s["subtitle_style"], highlight_color=s["highlight_color"],
            title_style=s.get("title_style", "box"), font=s.get("subtitle_font", ""),
            music_path=music.track_path(s["bgm_track"]), music_gain=s["bgm_gain"], music_duck=s["bgm_duck"],
            work_dir=proj.job.dir if proj.job else folder / "render", on_status=lambda m: report(line=prefix + m),
        )
        result = {
            "id": project.new_id(), "short_id": short_id, "title": title, "video": str(video), "report": rep,
            "created": project.now(), "duration": round(media.probe(video).duration, 1),
            "summary": settings_summary(s, cards), "owner": item.get("owner", ""),
            "script": " ".join(c["narration"].strip() for c in clips if c.get("narration", "").strip()),
        }
        prompts = {c["uid"]: (c.get("prompt"), c.get("prompt_src")) for c in cards if c.get("kind") == "ai"}

        def apply(pr: project.Project, result=result, prompts=prompts) -> None:
            pr.results.insert(0, result)
            for x in pr.shorts:
                for c in x["clips"]:
                    if c.get("uid") in prompts and prompts[c["uid"]][0]:
                        c["prompt"], c["prompt_src"] = prompts[c["uid"]]
        project.update(folder, apply)
        made.append(result["id"])
    return {"results": made}


def upload_text(item: dict, report) -> dict:
    """만든 쇼츠의 업로드 제목 후보·설명·해시태그를 써서 결과 기록에 붙인다."""
    s, p = _settings(item), item["params"]
    folder = _folder(item)
    proj = project.load(folder)
    result = next((r for r in proj.results if r["id"] == p["result_id"]), None)
    if result is None:
        raise RuntimeError("이 결과가 지워졌습니다.")
    script = result.get("script", "")
    if not script:  # 이전 버전 결과: 지금 카드의 나레이션
        short = next((x for x in proj.shorts if x["id"] == result.get("short_id")), None)
        script = " ".join(c["narration"] for c in (short or {}).get("clips", []) if c.get("narration"))
    if not script.strip() and not result["title"].strip():
        raise RuntimeError("나레이션과 제목이 없어 업로드 문구를 쓸 수 없습니다.")
    report(line=f"📣 {s['llm_model']}(으)로 제목 후보·설명·해시태그 쓰기")
    text = llm.write_upload_text(s["llm_model"], result["title"], script).model_dump()

    def apply(pr: project.Project) -> None:
        for r in pr.results:
            if r["id"] == p["result_id"]:
                r["upload"] = text
    project.update(folder, apply)
    return {}


# ───────────── AI 장면·음악·목소리 ─────────────
def scene_of(params: dict) -> dict:
    return {k: params[k] for k in ("prompt", "engine", "seed", "seconds")}


def scene(item: dict, report) -> dict:
    """AI 장면 하나. 카드에서 넣은 장면이면 설명을 영어로 바꾸고, 다 만들면 카드에 미리보기를 붙인다."""
    p = item["params"]
    if "scene" in p:  # 이전 버전 큐 항목
        sc = p["scene"]
    else:
        s = _settings(item)
        card = {"prompt_ko": p["prompt_ko"], "prompt": p.get("prompt", ""), "prompt_src": p.get("prompt_src", "")}
        if card["prompt_src"] != card["prompt_ko"].strip():
            report(line="🔤 장면 설명을 영어 프롬프트로")
        _ensure_prompt(card, s["llm_model"])
        sc = {"prompt": card["prompt"], "engine": p["engine"], "seed": int(p["seed"]), "seconds": float(p["seconds"])}
    kind, task, out = generate.scene_job(sc)
    if not out.exists():
        quiet = _quiet(report)
        report(line=f"🎨 {generate.MODEL_INFO['video' if sc['engine'] == 'video' else 'image']['name']}(으)로 만들기")
        generate.run_job(kind, task, request_id=item["id"],
                         on_step=lambda st, n: quiet(step=st, total=n, note=f"{st} / {n} 단계"),
                         on_loading=lambda m: quiet(note=f"{m} 불러오는 중"))
    path = generate.finish_scene(sc)
    if item.get("project_dir") and item.get("card_uid"):
        def apply(pr: project.Project) -> None:
            for x in pr.shorts:
                for c in x["clips"]:
                    same = (c.get("uid") == item["card_uid"] and c.get("prompt_ko", "").strip() == p["prompt_ko"].strip()
                            and c.get("engine") == p["engine"] and int(c.get("seed", 0)) == int(p["seed"]))
                    if same:
                        c.update(prompt=sc["prompt"], prompt_src=p["prompt_ko"].strip(), preview=str(path))
        if (_folder(item) / project.STATE_FILE).exists():
            project.update(_folder(item), apply)
    return {"path": str(path)}


def ai_music(item: dict, report) -> dict:
    s, p = _settings(item), item["params"]
    desc = p["desc"]
    report(line="🔤 음악 설명을 영어 프롬프트로")
    prompt = llm.english_prompt(s["llm_model"], desc, "music") if s["llm_model"] else desc
    report(line=f"🎵 MusicGen으로 {p['seconds']:.0f}초 만들기")
    wav = generate.music(prompt, float(p["seconds"]))
    return {"track": music.add_generated(wav, desc[:20], prompt), "prompt": prompt}


def voice_preview(item: dict, report) -> dict:
    s, p = _settings(item), item["params"]
    report(line="🔈 Chatterbox로 예시 문장 읽기")
    engine = tts.ChatterboxEngine(chatterbox(s["devices"]["tts"]), voices.sample_path(p["voice_id"]), p["exaggeration"])
    return {"path": str(engine.synthesize_cached(PREVIEW_TEXT, config.WORK_DIR / "voices" / "preview"))}


jobqueue.HANDLERS.update(analyze=analyze, plan=plan, narrate=narrate, make=make, scene=scene, music=ai_music,
                         voice=voice_preview, upload=upload_text)
