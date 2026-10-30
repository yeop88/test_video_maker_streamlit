"""AI 쇼츠 나레이션 스튜디오 — 영상을 이해해 하이라이트를 고르고, AI 나레이션과 한글 자막을 입혀 쇼츠로 만든다."""
import hashlib
from pathlib import Path

# pyarrow는 맨 먼저 불러온다. Whisper·Chatterbox(torch/CUDA) 등을 올린 뒤 처음으로 pyarrow DLL을 불러오면
# Windows에서 서버 프로세스가 access violation으로 죽는 일이 있었다 (결과 표 st.dataframe에서 처음 불러올 때).
import pyarrow  # noqa: F401
import streamlit as st
import streamlit.components.v1 as components

import shorts_maker
from shorts_maker import config, generate, genqueue, llm, media, music, pipeline, presets, project, system, tts, voices

st.set_page_config(page_title="AI 쇼츠 나레이션 스튜디오", page_icon="🎬", layout="wide")

ENGINE_LABELS = {"edge": "Edge TTS (기본 목소리, 인터넷 필요)", "chatterbox": "Chatterbox (GPU, 목소리 샘플 사용 가능)"}
LAYOUT_LABELS = {
    "blur": "블러 배경 (가로 영상·글자 많은 화면 추천)",
    "crop": "가운데 크롭 (화면 꽉 차게, 양옆 잘림)",
    "smart": "피사체 따라 크롭 (사람·움직임을 따라감)",
}
VOICE_LABELS = {voice_id: label for label, voice_id in tts.EDGE_VOICES.items()}
SUBTITLE_LABELS = {"plain": "기본", "karaoke": "단어 강조 (읽는 낱말 색칠)"}
VIDEO_TYPES = ["mp4", "mov", "webm", "mkv", "m4v", "avi"]
UI_DIR = Path(__file__).parent / "ui"


@st.cache_resource(show_spinner="Whisper 모델을 불러오는 중... (처음 한 번은 다운로드로 몇 분 걸립니다)")
def get_whisper(device: str):
    from shorts_maker import asr

    return asr.load_whisper(device)


@st.cache_resource(show_spinner="Chatterbox 모델을 불러오는 중... (처음 한 번은 다운로드로 몇 분 걸립니다)")
def get_chatterbox(device: str):
    return tts.load_chatterbox(device)


@st.cache_data(ttl=60, show_spinner=False)
def get_specs() -> system.Specs:
    return system.specs()


@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models() -> list[str]:
    return llm.list_models()


media.ensure_ffmpeg()

try:
    models = get_ollama_models()
except Exception:
    models = []

# ───────────────────────── 설정 상태 / 프리셋 ─────────────────────────
# 설정 위젯은 모두 session_state의 같은 이름 키에 값을 둔다 (값 = 프리셋에 저장하는 코드값).
ss = st.session_state
for key, value in presets.DEFAULTS.items():
    # 지금 화면에 없는 위젯(예: Edge 목소리를 Chatterbox 선택 중)의 값은 Streamlit이 지우므로 매번 다시 넣어 둔다
    ss[key] = ss[key] if key in ss else value
if models and ss.llm_model not in models:
    ss.llm_model = config.DEFAULT_OLLAMA_MODEL if config.DEFAULT_OLLAMA_MODEL in models else models[0]
specs = get_specs()
machine = system.load_machine(specs)  # 계산 장치·동영상 모델·양자화 (프리셋에는 넣지 않고 이 컴퓨터에 저장)
for key in ("device", "video_model", "ltx_quant", "image_model"):
    ss[key] = ss[key] if key in ss else machine[key]
for task in system.TASKS:  # 「작업별로 나누기」의 작업별 장치 (숨겨진 위젯 값은 지워지므로 매번 다시 넣음)
    ss[f"dev_{task}"] = ss[f"dev_{task}"] if f"dev_{task}" in ss else machine["split"][task]


def dev(task: str) -> str:
    """작업 하나가 쓸 장치 ("gpu" | "hybrid" | "cpu")."""
    return system.task_device(ss.device, {t: ss[f"dev_{t}"] for t in system.TASKS}, task)


def apply_devices() -> None:
    llm.DEVICE = dev("llm")
    generate.set_task_devices({t: dev(t) for t in ("image", "video", "music")})


apply_devices()
generate.set_video_model(ss.video_model, ss.ltx_quant)
generate.set_image_model(ss.image_model)


def change_device() -> None:
    """계산 장치를 바꾸면 저장하고, 다른 장치에 올려 둔 Whisper·Chatterbox를 내린다."""
    system.save_machine(device=ss.device, split={t: ss[f"dev_{t}"] for t in system.TASKS})
    get_whisper.clear()
    get_chatterbox.clear()
    apply_devices()
    ss.preset_msg = ("info", f"계산 장치: {system.DEVICES[ss.device]}. 모델은 다음에 쓸 때 이 장치로 다시 불러옵니다. "
                             "추천 모델도 장치에 맞게 바뀌었습니다 (「✨ 추천 모델로 맞추기」로 한 번에 바꿀 수 있음).")


def recommendations() -> dict:
    """작업마다 지금 장치에 맞는 추천."""
    return {t: system.recommend(t, dev(t), specs) for t in system.TASKS}


def current_model(task: str) -> str:
    if task == "llm":
        return ss.llm_model or "(없음)"
    if task == "asr":
        return "whisper-large-v3-turbo"
    if task == "tts":
        return "Chatterbox" if ss.engine == "chatterbox" else "Edge TTS"
    if task == "image":
        return generate.IMAGE_MODELS[ss.image_model]["name"]
    if task == "video":
        return generate.MODEL_INFO["video"]["name"]
    return "MusicGen small"


def is_recommended(task: str, rec: dict) -> bool:
    return rec["key"] is None or ss.get(rec["key"]) == rec["choice"]


def apply_recommended() -> None:
    """지금 장치에 맞는 추천 모델로 한 번에 바꾼다 (설치되지 않은 LLM은 건너뜀)."""
    changed, skipped = [], []
    for task, rec in recommendations().items():
        if rec["key"] is None or ss.get(rec["key"]) == rec["choice"]:
            continue
        if rec["key"] == "llm_model" and rec["choice"] not in models:
            skipped.append(f"{rec['choice']} (설치 안 됨: `ollama pull {rec['choice']}`)")
            continue
        ss[rec["key"]] = rec["choice"]
        changed.append(f"{system.TASKS[task]} → {current_model(task)}")
    if ss.video_model == "ltx25":
        ss.ltx_quant = "nf4"
    system.save_machine(video_model=ss.video_model, ltx_quant=ss.ltx_quant, image_model=ss.image_model)
    generate.set_video_model(ss.video_model, ss.ltx_quant)
    generate.set_image_model(ss.image_model)
    text = "추천 모델로 맞췄습니다: " + ", ".join(changed) if changed else "이미 모두 추천 모델입니다."
    if skipped:
        text += " · 건너뜀: " + ", ".join(skipped)
    ss.preset_msg = ("success" if changed else "info", text)


def change_video_model() -> None:
    """AI 동영상 모델·양자화를 바꾸면 저장한다. 이미 만든 장면은 그대로 두고 새로 만드는 장면부터 적용된다."""
    system.save_machine(video_model=ss.video_model, ltx_quant=ss.ltx_quant, image_model=ss.image_model)
    generate.set_video_model(ss.video_model, ss.ltx_quant)
    generate.set_image_model(ss.image_model)


def current_settings() -> dict:
    return {key: ss[key] for key in presets.DEFAULTS}


def apply_preset(name: str) -> None:
    try:
        values, dropped = presets.load(name)
    except (FileNotFoundError, ValueError) as e:
        ss.preset_msg = ("error", str(e))
        return
    notes = [f"설정 {', '.join(dropped)}은(는) 올바르지 않아 기본값을 썼습니다"] if dropped else []
    if values["voice_id"] and not voices.sample_path(values["voice_id"]):
        notes.append("프리셋의 목소리가 보관함에 없어 기본 목소리를 씁니다")
        values["voice_id"] = ""
    if values["bgm_track"] and not music.track_path(values["bgm_track"]):
        notes.append("프리셋의 배경음악이 보관함에 없어 배경음악 없이 둡니다")
        values["bgm_track"] = ""
    if models and values["llm_model"] not in models:
        notes.append(f"LLM '{values['llm_model']}'이(가) 설치되어 있지 않아 지금 모델을 그대로 둡니다")
        values["llm_model"] = ss.llm_model
    ss.update(values)
    ss.preset_loaded = (name, values)
    ss.preset_msg = ("warning", f"'{name}' 불러옴. " + ". ".join(notes)) if notes else ("success", f"'{name}' 프리셋을 불러왔습니다.")


def save_preset() -> None:
    try:
        name, existed = presets.save(ss.preset_new_name, current_settings())
    except ValueError as e:
        ss.preset_msg = ("error", str(e))
        return
    ss.preset_loaded = (name, current_settings())
    ss.preset_select = name
    ss.preset_msg = ("success", f"'{name}' 프리셋을 {'덮어썼습니다' if existed else '저장했습니다'}.")


def add_music_upload() -> None:
    file = ss.get("bgm_upload")
    if file is None:
        return
    try:
        ss.bgm_track = music.add_upload(file.name, file.getvalue())
        ss.preset_msg = ("success", f"'{file.name}'을(를) 배경음악 보관함에 넣었습니다.")
    except RuntimeError as e:
        ss.preset_msg = ("error", str(e))


PREVIEW_TEXT = "안녕하세요. 이 목소리로 쇼츠 나레이션을 읽어 드릴게요."


def add_voice(kind: str) -> None:
    """목소리 보관함에 새 목소리를 넣고 바로 고른다 (파일·녹음은 동의 확인 필요)."""
    try:
        if kind == "virtual":
            voice_id = voices.create_virtual(ss.get("new_voice_name_virtual", ""), ss.virtual_base,
                                             ss.virtual_pitch, ss.virtual_formant)
        else:
            file = ss.get("new_voice_file" if kind == "upload" else "new_voice_rec")
            if file is None:
                raise ValueError("목소리 파일을 올리거나 녹음해 주세요.")
            name = ss.get("new_voice_name_file" if kind == "upload" else "new_voice_name_rec", "")
            consent = ss.get("consent_file" if kind == "upload" else "consent_rec", False)
            voice_id = voices.add_sample(name, getattr(file, "name", "recording.wav"), file.getvalue(), kind, consent)
    except (ValueError, RuntimeError) as e:
        ss.preset_msg = ("error", str(e))
        return
    ss.voice_id = voice_id
    ss.pop("voice_preview", None)
    ss.preset_msg = ("success", "목소리를 보관함에 넣고 골랐습니다. 「▶ 미리 듣기」로 들어 보세요.")


def delete_voice() -> None:
    voices.delete(ss.voice_id)
    ss.voice_id = ""
    ss.pop("voice_preview", None)
    ss.preset_msg = ("success", "목소리를 지웠습니다.")


def delete_preset(name: str) -> None:
    try:
        presets.delete(name)
    except ValueError as e:
        ss.preset_msg = ("error", str(e))
        return
    if ss.get("preset_loaded", ("",))[0] == name:
        ss.pop("preset_loaded")
    ss.preset_select = "기본"
    ss.preset_msg = ("success", f"'{name}' 프리셋을 지웠습니다.")


def setting_slider(label: str, key: str, **kwargs):
    """프리셋 범위(최솟값, 최댓값, 간격)를 쓰는 슬라이더. 값은 session_state[key]에 있다."""
    lo, hi, step = presets.RANGES[key]
    return st.slider(label, min_value=lo, max_value=hi, step=step, key=key, **kwargs)


# ───────────────────────── 설정 팝업 (분류별) ─────────────────────────
# 사이드바에는 프리셋과 분류별 요약만 두고, 자세한 설정은 버튼을 누르면 뜨는 팝업(st.dialog)에서 바꾼다.
# 팝업이 닫혀 있으면 위젯이 그려지지 않지만, 설정 값은 위쪽에서 매번 session_state에 다시 넣어 두므로 유지된다.
DEVICE_SHORT = {"gpu": "GPU", "hybrid": "GPU+CPU", "cpu": "CPU"}


def show_message() -> None:
    """콜백이 남긴 알림 (팝업 안이든 사이드바든 먼저 그리는 쪽에서 한 번 보여 줌)."""
    msg = ss.pop("preset_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])


def close_button() -> None:
    st.divider()
    if st.button("완료", type="primary", width="stretch", key="dialog_done"):
        st.rerun()  # 팝업을 닫고 바뀐 설정을 화면 전체에 반영


def device_summary() -> str:
    if ss.device == "split":
        gpu_tasks = [system.TASKS[t].split(" (")[0] for t in system.TASKS if dev(t) != "cpu"]
        return f"작업별: {', '.join(gpu_tasks)}만 GPU" if gpu_tasks else "작업별: 모두 CPU"
    return system.DEVICES[ss.device].split(" (")[0]


def computer_settings() -> None:
    """💻 컴퓨터·AI 모델: 사양, 계산 장치(작업별), 추천 모델, AI 이미지·동영상 모델."""
    show_message()
    gpu_text = ", ".join(f"{g.name} {g.vram_gb:.0f}GB" for g in specs.gpus) or "NVIDIA GPU 없음"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GPU (VRAM)", f"{specs.vram_gb:.0f} GB" if specs.gpus else "없음", help=gpu_text)
    c2.metric("RAM", f"{specs.ram_gb:.0f} GB", f"남은 {specs.ram_free_gb:.0f} GB", delta_color="off")
    c3.metric("CPU 코어", f"{specs.cores}개", help=specs.cpu)
    c4.metric("작업 폴더 디스크 여유", f"{specs.disk_free_gb:,} GB")
    st.caption(f"{specs.os} · CPU {specs.cpu} · GPU {gpu_text}" + (f" (드라이버 {specs.gpus[0].driver})" if specs.gpus else ""))

    tab_device, tab_models = st.tabs(["계산 장치와 추천 모델", "AI 이미지·동영상 모델"])
    with tab_device:
        st.radio("계산 장치", list(system.DEVICES), format_func=system.DEVICES.get, key="device", on_change=change_device,
                 help=system.DEVICE_HELP)
        suggested = system.suggested_device(specs)
        if ss.device != suggested:
            st.caption(f"이 컴퓨터 추천: {system.DEVICES[suggested]}")
        if ss.device == "split":
            with st.container(border=True):
                cols = st.columns(3)
                for i, (task, label) in enumerate(system.TASKS.items()):
                    cols[i % 3].radio(label, list(system.TASK_DEVICES), format_func=system.TASK_DEVICES.get,
                                      key=f"dev_{task}", horizontal=True, on_change=change_device)
        if any(dev(t) != "cpu" for t in system.TASKS) and not specs.gpus:
            st.warning("NVIDIA GPU를 찾지 못했습니다. GPU를 고르면 CPU로 돌아가거나 AI 생성이 막힙니다.")
        # 작업마다 지금 장치에 맞는 추천 (장치를 바꾸면 추천도 바뀜). 문구가 길어서 줄바꿈되는 표(st.table)로.
        recs = recommendations()
        st.markdown("**추천 모델** (장치를 바꾸면 추천도 바뀝니다)")
        st.table([{"작업": label, "장치": DEVICE_SHORT[dev(t)], "추천 모델 (지금 장치 기준)": recs[t]["text"],
                   "지금 고른 모델": current_model(t),
                   "상태": ("✅ 추천대로" if is_recommended(t, recs[t]) else "↔ 추천과 다름") + " · "
                           + system.fits(recs[t]["need"], specs, dev(t))}
                  for t, label in system.TASKS.items()], hide_index=True)
        st.button("✨ 추천 모델로 맞추기", on_click=apply_recommended, width="stretch",
                  disabled=all(is_recommended(t, r) for t, r in recs.items()))
        st.caption("이 설정은 프리셋에 넣지 않고 이 컴퓨터에 저장합니다. GPU 메모리는 모델을 하나씩 쓸 때 기준입니다.")
    with tab_models:
        recs = recommendations()
        st.radio("AI 이미지 모델", list(generate.IMAGE_MODELS), format_func=lambda k: generate.IMAGE_MODELS[k]["label"],
                 key="image_model", on_change=change_video_model,
                 help="AI 장면을 「AI 이미지 + 움직임」으로 만들 때 쓰는 모델입니다. 바꾸면 새로 만드는 장면부터 적용됩니다. "
                      "처음 쓰는 모델은 내려받느라 오래 걸립니다.")
        st.caption(f"추천 ({DEVICE_SHORT[dev('image')]}): {recs['image']['text']}")
        image_ok, image_msg = generate.image_available()
        if not image_ok:
            st.warning(image_msg)
        st.radio("AI 동영상 모델", list(generate.VIDEO_MODELS), format_func=generate.VIDEO_MODELS.get, key="video_model",
                 on_change=change_video_model,
                 help="AI 장면을 「AI 동영상」으로 만들 때 쓰는 모델입니다. 바꾸면 새로 만드는 장면부터 적용됩니다.")
        st.caption(f"추천 ({DEVICE_SHORT[dev('video')]}): {recs['video']['text']}")
        if dev("video") == "cpu" and ss.video_model == "ltx25":
            st.warning("LTX-2.5는 GPU에서만 돌립니다. AI 동영상 장치를 GPU로 바꾸거나 LTX-Video 2B를 고르세요.")
        if ss.video_model == "ltx25":
            st.radio("LTX-2.5 양자화", list(generate.LTX_QUANTS), format_func=generate.LTX_QUANTS.get, key="ltx_quant",
                     on_change=change_video_model,
                     help="22B 모델은 원본(bf16 42GB)이 16GB GPU에 들어가지 않아 줄여서 올립니다. "
                          "처음 쓸 때 약 71GB를 내려받고 4bit로 양자화해 workdir/models에 저장합니다 (다음부터는 바로 불러옴).")
            ltx_ok, ltx_msg = generate.ltx_status()
            if not ltx_ok:
                st.warning(ltx_msg)
    close_button()


def script_settings() -> None:
    """🧠 대본·구성: LLM, 말투, 길이, 편 수."""
    show_message()
    if not models:
        st.error("Ollama에 연결할 수 없습니다. Ollama를 실행한 뒤 새로고침하세요.")
    else:
        st.selectbox("LLM (Ollama)", models, key="llm_model",
                     help="장면 이미지를 이해해야 하므로 vision을 지원하는 모델(qwen3.5, gemma4 등)을 고르세요.")
        st.caption(f"추천 ({DEVICE_SHORT[dev('llm')]}): {recommendations()['llm']['text']}")
    st.selectbox("나레이션 말투", presets.CHOICES["style"], key="style")
    setting_slider("쇼츠 길이 목표 (초)", "target_sec")
    setting_slider("쇼츠 개수 (한 영상에서)", "short_count",
                   help="2편 이상이면 긴 영상을 서로 다른 주제로 나눠 편마다 쇼츠를 만듭니다.")
    close_button()


def voice_settings() -> None:
    """🎙️ 나레이션 목소리: 엔진, 목소리, 속도·감정, 목소리 보관함."""
    show_message()
    st.radio("음성 엔진", presets.CHOICES["engine"], format_func=ENGINE_LABELS.get, key="engine")
    st.caption(f"추천 ({DEVICE_SHORT[dev('tts')]}): {recommendations()['tts']['text']}")
    if ss.engine == "edge":
        st.selectbox("목소리", presets.CHOICES["edge_voice"], format_func=VOICE_LABELS.get, key="edge_voice")
        setting_slider("말하기 속도 (%)", "rate")
    else:
        library = {v["id"]: v for v in voices.list_voices()}
        if ss.voice_id not in library:
            ss.voice_id = ""
        col_voice, col_del_voice = st.columns([5, 1], vertical_alignment="bottom")
        col_voice.selectbox(
            "복제할 목소리", [""] + list(library), key="voice_id",
            format_func=lambda i: "기본 목소리" if not i else
            f"{'🤖' if library[i]['source'] == 'virtual' else '🎙️'} {library[i]['name']}",
            help="보관함의 목소리 샘플을 Chatterbox가 제로샷으로 따라 해서 나레이션을 읽습니다.",
        )
        col_del_voice.button("🗑", key="del_voice", disabled=not ss.voice_id, help="이 목소리를 보관함에서 지우기",
                             on_click=delete_voice)
        setting_slider("감정 표현 강도", "exaggeration")
        if st.button("▶ 미리 듣기", width="stretch", help="고른 목소리로 예시 문장을 읽어 봅니다 (몇 초)."):
            with st.spinner("목소리를 만드는 중..."):
                engine = tts.ChatterboxEngine(get_chatterbox(dev("tts")), voices.sample_path(ss.voice_id), ss.exaggeration)
                ss.voice_preview = str(engine.synthesize_cached(PREVIEW_TEXT, config.WORK_DIR / "voices" / "preview"))
        if ss.get("voice_preview") and Path(ss.voice_preview).exists():
            st.audio(ss.voice_preview)
        with st.expander("➕ 새 목소리 만들기"):
            tab_file, tab_rec, tab_virtual = st.tabs(["파일", "녹음", "가상 목소리"])
            with tab_file:
                st.text_input("이름", key="new_voice_name_file", placeholder="예: 내 목소리")
                st.file_uploader("목소리 파일 (5~15초)", type=["wav", "mp3", "m4a"], key="new_voice_file")
                st.checkbox("본인 목소리이거나 동의를 받았습니다", key="consent_file")
                st.button("보관함에 넣기", key="add_voice_file", on_click=add_voice, args=("upload",), width="stretch")
            with tab_rec:
                st.text_input("이름", key="new_voice_name_rec", placeholder="예: 내 목소리 (녹음)")
                st.caption(f"이 문장을 또박또박 읽어 주세요 (5~15초): {voices.VIRTUAL_SCRIPT}")
                st.audio_input("녹음", key="new_voice_rec")
                st.checkbox("본인 목소리이거나 동의를 받았습니다", key="consent_rec")
                st.button("보관함에 넣기", key="add_voice_rec", on_click=add_voice, args=("record",), width="stretch")
            with tab_virtual:
                st.caption("Edge TTS 목소리의 음높이와 음색을 바꿔 실존 인물이 아닌 새 목소리를 만듭니다.")
                st.text_input("이름", key="new_voice_name_virtual", placeholder="예: 차분한 해설자")
                st.selectbox("바탕 목소리", list(tts.EDGE_VOICES.values()), format_func=VOICE_LABELS.get, key="virtual_base")
                st.slider("음높이 (반음)", -6.0, 6.0, 0.0, 0.5, key="virtual_pitch", help="낮추면 굵고 낮은 목소리, 높이면 밝은 목소리")
                st.slider("음색 (반음)", -4.0, 4.0, 0.0, 0.5, key="virtual_formant",
                          help="목소리 울림(포먼트)을 바꿉니다. 낮추면 더 어른스럽고, 높이면 더 가볍게 들립니다.")
                st.button("만들기", key="add_voice_virtual", on_click=add_voice, args=("virtual",), width="stretch")
        st.caption("목소리 샘플은 본인 또는 동의받은 사람의 목소리만 사용하세요.")
    close_button()


def sound_settings() -> None:
    """🔊 소리·배경음악: 원본 소리, 배경음악 보관함, AI 음악, 음악 올리기."""
    show_message()
    setting_slider("원본 소리 크기", "original_gain")
    setting_slider("나레이션 중 원본 소리 크기", "duck_gain",
                   help="나레이션이 나오는 동안 원본 소리를 이 비율로 줄입니다 (원본 소리 크기에 곱해짐).")
    st.markdown("**🎵 배경음악**")
    tracks = {t["id"]: t for t in music.list_tracks()}
    if "pending_bgm" in ss:  # 방금 만든 AI 음악을 고른다 (위젯을 그리기 전에만 값을 바꿀 수 있음)
        ss.bgm_track = ss.pop("pending_bgm")
    if ss.bgm_track not in tracks:
        ss.bgm_track = ""
    st.selectbox(
        "배경음악", [""] + list(tracks), key="bgm_track", label_visibility="collapsed",
        format_func=lambda i: "없음" if not i else
        f"{'✨' if tracks[i]['source'] == 'ai' else '📁'} {tracks[i]['name']} ({tracks[i]['seconds']}초)",
    )
    if ss.bgm_track:
        st.audio(str(tracks[ss.bgm_track]["path"]))
        setting_slider("배경음악 크기", "bgm_gain")
        setting_slider("나레이션 중 배경음악 크기", "bgm_duck", help="나레이션이 나오는 동안 배경음악을 이 비율로 줄입니다.")
    with st.expander("✨ AI로 음악 만들기"):
        gen_ok, gen_msg = generate.available()
        st.text_input("어떤 음악?", key="ai_music_desc", placeholder="예: 잔잔한 피아노, 평화로운 아침")
        st.slider("길이 (초)", 10, 30, 20, 5, key="ai_music_sec", help="쇼츠가 더 길면 겹쳐 이어 반복합니다.")
        if st.button("음악 만들기", disabled=not gen_ok, help=gen_msg or "MusicGen으로 배경음악을 만듭니다.",
                     width="stretch"):
            desc = ss.get("ai_music_desc", "").strip() or "calm ambient background music"
            with st.spinner("음악을 만드는 중..."):
                try:
                    prompt = llm.english_prompt(ss.llm_model, desc, "music") if models else desc
                    wav = generate.music(prompt, float(ss.get("ai_music_sec", 20)))
                    ss.pending_bgm = music.add_generated(wav, desc[:20], prompt)
                    ss.preset_msg = ("success", f"AI 음악을 만들어 보관함에 넣었습니다: {prompt}")
                except RuntimeError as e:
                    ss.preset_msg = ("error", str(e)[:300])
            st.rerun(scope="fragment")  # 팝업은 그대로 두고 새 곡을 골라 다시 그림
    with st.expander("음악 올리기"):
        st.file_uploader("음악 파일", type=music.MUSIC_TYPES, key="bgm_upload", label_visibility="collapsed")
        st.button("보관함에 넣기", on_click=add_music_upload, disabled=ss.get("bgm_upload") is None, width="stretch",
                  help="올린 음악은 workdir/music/에 저장되어 다른 작업에서도 고를 수 있습니다. 권리가 있는 음악만 쓰세요.")
    close_button()


def screen_settings() -> None:
    """🎬 화면·자막: 세로 변환, 제목, 자막 스타일·크기·위치·색."""
    show_message()
    st.radio("세로 변환 방식", presets.CHOICES["layout"], format_func=LAYOUT_LABELS.get, key="layout")
    st.checkbox("위쪽에 제목 표시", key="show_title")
    st.radio("자막 스타일", presets.CHOICES["subtitle_style"], format_func=SUBTITLE_LABELS.get, key="subtitle_style",
             help="단어 강조: 나레이션이 읽는 낱말을 다른 색으로 조금 크게 보여 줍니다.")
    if ss.subtitle_style == "karaoke":
        st.color_picker("강조 색", key="highlight_color")
    setting_slider("자막 크기", "font_size")
    setting_slider("자막 위치 (아래에서 %)", "margin_percent")
    st.color_picker("자막 색", key="color")
    close_button()


def voice_summary() -> str:
    if ss.engine == "edge":
        return f"Edge · {VOICE_LABELS.get(ss.edge_voice, ss.edge_voice)} · 속도 {ss.rate:+d}%"
    chosen = next((v for v in voices.list_voices() if v["id"] == ss.voice_id), None)
    return f"Chatterbox · {chosen['name'] if chosen else '기본 목소리'}"


def sound_summary() -> str:
    track = next((t for t in music.list_tracks() if t["id"] == ss.bgm_track), None)
    return f"원본 {ss.original_gain:.2f} · " + (f"배경음악 {track['name']}" if track else "배경음악 없음")


# 분류: (버튼 이름, 팝업 제목, 팝업 내용, 요약)
SETTING_GROUPS = [
    ("🧠 대본·구성", "대본·구성 설정", script_settings,
     lambda: f"{ss.llm_model or 'LLM 없음'} · {ss.style.split(' (')[0]} · {ss.target_sec}초 · {ss.short_count}편"),
    ("🎙️ 나레이션 목소리", "나레이션 목소리 설정", voice_settings, voice_summary),
    ("🔊 소리·배경음악", "소리·배경음악 설정", sound_settings, sound_summary),
    ("🎬 화면·자막", "화면·자막 설정", screen_settings,
     lambda: f"{LAYOUT_LABELS[ss.layout].split(' (')[0]} · {SUBTITLE_LABELS[ss.subtitle_style].split(' (')[0]} 자막 · {ss.font_size}"),
    ("💻 컴퓨터·AI 모델", "컴퓨터와 AI 모델 설정", computer_settings,
     lambda: f"{device_summary()} · {generate.IMAGE_MODELS[ss.image_model]['name']} · {generate.MODEL_INFO['video']['name']}"),
]
DIALOGS = {label: st.dialog(title, width="large")(body) for label, title, body, _ in SETTING_GROUPS}

# ───────────────────────── 사이드바: 프리셋 + 설정 분류 ─────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    st.subheader("💾 프리셋")
    names = presets.list_presets()
    if ss.get("preset_select") not in names:
        ss.preset_select = names[0]
    selected = st.selectbox("프리셋", names, key="preset_select", format_func=lambda n: f"⭐ {n}" if presets.is_builtin(n) else n)
    col_load, col_delete = st.columns(2)
    col_load.button("불러오기", on_click=apply_preset, args=(selected,), width="stretch")
    col_delete.button(
        "삭제", on_click=delete_preset, args=(selected,), width="stretch",
        disabled=presets.is_builtin(selected), help="⭐ 기본 제공 프리셋은 지울 수 없습니다.",
    )
    with st.expander("현재 설정을 프리셋으로 저장"):
        st.text_input("프리셋 이름", key="preset_new_name", placeholder="예: 여행 브이로그용", max_chars=40)
        st.button("저장", on_click=save_preset, type="primary", width="stretch",
                  help="같은 이름이 있으면 덮어씁니다. 목소리 샘플 파일은 저장되지 않습니다.")
    loaded = ss.get("preset_loaded")
    if loaded:
        changed = loaded[1] != current_settings()
        st.caption(f"현재: {loaded[0]}" + (" (수정됨)" if changed else ""))
    show_message()

    st.subheader("🛠️ 세부 설정")
    st.caption("누르면 팝업 창에서 바꿀 수 있습니다.")
    open_dialog = None
    for label, _, _, summary in SETTING_GROUPS:
        if st.button(label, key=f"open_{label}", width="stretch"):
            open_dialog = label
        st.caption(summary())
    if open_dialog:
        DIALOGS[open_dialog]()

    st.divider()
    st.caption("⚠️ 본인이 권리를 가진 영상이나 허락받은 영상만 사용하세요. 목소리 샘플은 본인 또는 동의받은 사람의 목소리만 사용하세요.")
    st.caption(f"v{shorts_maker.__version__}")

llm_model = ss.llm_model if models else None

style_label, target_sec = ss.style, ss.target_sec
show_title, font_size, margin_percent, color = ss.show_title, ss.font_size, ss.margin_percent, ss.color
original_gain, duck_gain = ss.original_gain, ss.duck_gain


# ───────────────────────── 작업 열기 (새로고침해도 이어서) ─────────────────────────
st.title("🎬 AI 쇼츠 나레이션 스튜디오")
st.caption(
    "영상을 올리면 AI가 **장면과 대사를 이해**해서 하이라이트 구간을 고르고, "
    "**한국어 나레이션**과 **한글 자막**을 입힌 세로 쇼츠/릴스(1080×1920)를 만들어 줍니다. "
    "영상이 없으면 **주제만으로 AI가 대본과 장면을 만들어** 쇼츠를 만듭니다."
)
MODE_LABELS = {"upload": "📼 영상으로 만들기", "ai": "✨ 영상 없이 AI로 만들기"}
ss.setdefault("mode", "upload")


def refresh_cards() -> None:
    """구간 카드 입력칸을 저장된 값으로 다시 그리게 한다 (카드 위젯 키에 붙는 번호를 바꿈)."""
    ss.card_rev = ss.get("card_rev", 0) + 1


def set_project(p: project.Project, message: str = "") -> None:
    """작업을 화면 상태로 불러온다."""
    ss.project = p
    if ss.get("mode") != p.kind:  # 최근 작업에서 연 경우 (콜백 안이라 위젯 값을 바꿀 수 있음)
        ss.mode = p.kind
    for short in p.shorts:
        for clip in short["clips"]:
            clip.setdefault("uid", project.new_id())  # 이전 버전 작업에는 카드 ID가 없음
    refresh_cards()
    if message:
        ss.flash = message


def open_recent(folder: Path) -> None:
    try:
        set_project(project.load(folder), f"'{folder.name}' 작업을 불러왔습니다.")
    except (OSError, ValueError, KeyError) as e:
        ss.flash = f"작업을 불러오지 못했습니다: {e}"
    ss.upload_id = None


def close_project() -> None:
    ss.pop("project", None)


def folder_mb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1024 ** 2


def delete_recent(folder: Path, name: str) -> None:
    """작업 폴더를 통째로 지운다. 지금 열려 있는 작업이면 닫는다.
    (upload_id는 그대로 두어, 업로드 칸에 남은 파일 때문에 방금 지운 작업이 다시 만들어지지 않게 한다)"""
    current = ss.get("project")
    if current is not None and Path(current.dir).resolve() == folder.resolve():
        ss.pop("project", None)
    try:
        project.delete(folder)
        ss.flash = f"'{name}' 작업을 지웠습니다."
    except OSError as e:
        ss.flash = f"작업을 지우지 못했습니다 (다른 프로그램이 파일을 쓰는 중일 수 있음): {e}"


@st.dialog("작업 지우기")
def confirm_delete_recent(folder: Path, name: str, results: int) -> None:
    st.markdown(f"**{name}** 작업을 지울까요?")
    st.caption(f"폴더 `{folder.name}` · {folder_mb(folder):,.0f}MB")
    st.warning(f"올린 영상, 분석 결과, 쇼츠 구성, 만든 쇼츠 {results}개(영상·자막·썸네일)가 모두 지워지며 되돌릴 수 없습니다. "
               "필요한 결과는 먼저 다운로드하세요.")
    col_yes, col_no = st.columns(2)
    if col_yes.button("🗑 지우기", type="primary", width="stretch"):
        delete_recent(folder, name)
        st.rerun()
    if col_no.button("취소", width="stretch"):
        st.rerun()


recent_items = project.recent(limit=50)
with st.expander(f"📂 최근 작업 ({len(recent_items)})", expanded=False):
    if not recent_items:
        st.caption("아직 작업이 없습니다.")
    with st.container(height=min(420, 60 + 56 * len(recent_items)) if recent_items else "content", border=False):
        for item in recent_items:
            col_name, col_info, col_open, col_del = st.columns([4, 3, 1, 0.6], vertical_alignment="center")
            icon = "✨" if item["kind"] == "ai" else "📼"
            col_name.markdown(f"{icon} **{item['name']}**")
            status = "분석 완료" if item["analyzed"] else "분석 전"
            col_info.caption(f"{item['updated']} · {status} · 만든 쇼츠 {item['results']}개")
            col_open.button("열기", key=f"open_{item['dir'].name}", on_click=open_recent, args=(item["dir"],))
            if col_del.button("🗑", key=f"delrecent_{item['dir'].name}", help="이 작업 지우기 (확인 후)"):
                confirm_delete_recent(item["dir"], item["name"], item["results"])

st.radio("만드는 방식", list(MODE_LABELS), format_func=MODE_LABELS.get, horizontal=True, key="mode",
         on_change=close_project, label_visibility="collapsed")

uploaded = None
if ss.mode == "upload":
    # 드래그 앤 드롭: 업로드 칸을 크게 보여 주고, 화면 어디에 놓아도 업로드되게 한다 (ui/dropzone.*)
    st.html(f"<style>{(UI_DIR / 'dropzone.css').read_text(encoding='utf-8')}</style>")
    components.html(f"<script>{(UI_DIR / 'dropzone.js').read_text(encoding='utf-8')}</script>", height=0)
    with st.container(key="video_drop"):
        uploaded = st.file_uploader("영상 업로드", type=VIDEO_TYPES)

if uploaded is not None:
    data = uploaded.getvalue()
    upload_id = hashlib.sha1(data).hexdigest()[:12]
    if ss.get("upload_id") != upload_id:  # 새로 올린 영상 (같은 영상이면 이전 작업을 그대로 불러옴)
        ss.upload_id = upload_id
        opened, restored = project.open_upload(upload_id, uploaded.name, data)
        set_project(opened, "이전에 작업한 영상이라 분석 결과와 구성을 불러왔습니다." if restored and opened.job else "")

flash = ss.pop("flash", None)
if flash:
    st.toast(flash)

proj: project.Project | None = ss.get("project")
if proj is not None and proj.kind != ss.mode:
    proj = None
if proj is None and ss.mode == "upload":
    st.info(f"영상을 올리면 시작합니다. 긴 영상은 분석할 구간을 {config.MAX_SOURCE_SECONDS // 60}분 이하로 골라 주세요.")
    st.stop()
AI_ENGINES = {**generate.ENGINES, "mix": "섞기 (동영상·이미지 번갈아)"}


def model_table() -> None:
    """어떤 단계를 어떤 AI 모델이 하는지 보여 준다."""
    rows = [
        {"단계": "대본·장면 구성, 장면 설명 → 영어 프롬프트", "모델": f"{llm_model or '(Ollama 모델 없음)'}",
         "만든 곳": "Ollama (설정 「🧠 대본·구성」에서 바꿈)", "하는 일": "주제로 제목·장면별 나레이션·장면 설명을 씀"},
        *[{"단계": label, "모델": generate.MODEL_INFO[k]["name"], "만든 곳": f"{generate.MODEL_INFO[k]['by']} · {generate.MODEL_INFO[k]['repo']}",
           "하는 일": f"{generate.MODEL_INFO[k]['what']} ({generate.MODEL_INFO[k]['speed']})"}
          for k, label in (("image", "장면: AI 이미지 + 움직임"), ("video", "장면: AI 동영상"), ("music", "배경음악 (설정 「🔊 소리·배경음악」)"))],
        {"단계": "나레이션 음성", "모델": "Edge TTS" if ss.engine == "edge" else "Chatterbox Multilingual",
         "만든 곳": "설정 「🎙️ 나레이션 목소리」", "하는 일": "나레이션을 읽고 단어 시간으로 자막을 맞춤"},
    ]
    with st.expander("🤖 사용하는 AI 모델", expanded=True):
        st.table(rows, hide_index=True)


if proj is None:  # 영상 없이 AI로: 주제 → 대본·장면 구성
    gen_ok, gen_msg = generate.available()
    if not gen_ok:
        st.warning(gen_msg)
    st.text_area("어떤 쇼츠를 만들까요?", key="ai_topic", height=90,
                 placeholder="예: 초보자를 위한 집에서 하는 아침 스트레칭 / 비 오는 날 혼자 즐기는 카페의 분위기")
    st.radio("장면 만드는 방식", list(AI_ENGINES), format_func=AI_ENGINES.get, key="ai_engine", horizontal=True,
             help="AI 이미지 + 움직임은 장면당 몇 초, AI 동영상은 장면당 1분 안팎 걸립니다.")
    model_table()
    st.caption("쇼츠 길이·말투·목소리·배경음악·자막은 사이드바 「🛠️ 세부 설정」을 씁니다. "
               "AI가 쓴 대본은 사실과 다를 수 있으니 2단계에서 꼭 확인하세요.")
    if st.button("1단계 · AI 대본 + 장면 구성", type="primary",
                 disabled=llm_model is None or not ss.get("ai_topic", "").strip()):
        with st.status(f"{llm_model}(으)로 대본과 장면 구성을 쓰는 중...", expanded=True) as status:
            try:
                new_project = project.new_ai(ss.ai_topic)
                title, cards = pipeline.plan_ai(llm_model, ss.ai_topic, target_sec, llm.STYLES[style_label], ss.ai_engine)
                new_project.shorts = [{"id": project.new_id(), "title": title, "range": None, "focus": "", "clips": cards}]
                project.save(new_project)
                set_project(new_project)
                status.update(label="1단계 완료", state="complete", expanded=False)
            except Exception as e:
                status.update(label="1단계 실패", state="error")
                st.exception(e)
                st.stop()
        st.rerun()
    st.stop()


@st.cache_data(show_spinner=False)
def probe(path: str) -> media.VideoInfo:
    return media.probe(Path(path))


run_analyze = False
if proj.kind == "ai":
    col_topic, col_close = st.columns([6, 1])
    col_topic.markdown(f"**✨ {proj.topic}**")
    col_close.button("닫기", on_click=close_project, help="작업은 저장되어 있어서 「최근 작업」에서 다시 열 수 있습니다.")
else:
    source = proj.source_path
    info = probe(str(source))
    col_video, col_options = st.columns([1, 2])
    with col_video:
        st.video(str(source))
    with col_options:
        head, close = st.columns([5, 1])
        head.markdown(f"**{proj.name}** · {info.duration:.1f}초 · {info.width}×{info.height}" + ("" if info.has_audio else " · 소리 없음"))
        close.button("닫기", on_click=close_project, help="작업은 저장되어 있어서 「최근 작업」에서 다시 열 수 있습니다.")
        total = round(info.duration, 1)
        start, end = st.slider("분석할 구간 (초)", 0.0, total, (0.0, min(total, float(config.MAX_SOURCE_SECONDS))), 0.5)
        if end - start > config.MAX_SOURCE_SECONDS:
            st.warning(f"{config.MAX_SOURCE_SECONDS // 60}분이 넘으면 분석이 오래 걸리고 장면 설명이 듬성듬성해집니다.")
        label = "1단계 · 다시 분석 + 쇼츠 구성" if proj.job else "1단계 · 영상 분석 + 쇼츠 구성"
        run_analyze = st.button(label, type="primary", disabled=llm_model is None)


def card_key(uid: str, field: str) -> str:
    return f"card{ss.card_rev}_{uid}_{field}"


def sync_cards(short: dict) -> None:
    """카드 입력칸에 쓴 값을 저장된 구간에 반영한다 (버튼 콜백에서 순서를 바꾸기 전에 호출)."""
    for clip in short["clips"]:
        for field in ("start", "end", "narration", "prompt_ko", "seconds", "engine"):
            key = card_key(clip["uid"], field)
            if key in ss:
                clip[field] = ss[key]


def card_action(p: project.Project, short: dict, uid: str, action: str) -> None:
    sync_cards(short)
    items = short["clips"]
    i = next(k for k, c in enumerate(items) if c["uid"] == uid)
    if action == "up" and i > 0:
        items[i - 1], items[i] = items[i], items[i - 1]
    elif action == "down" and i + 1 < len(items):
        items[i + 1], items[i] = items[i], items[i + 1]
    elif action == "delete":
        items.pop(i)
    project.save(p)
    refresh_cards()


def add_card(p: project.Project, short: dict, duration: float) -> None:
    sync_cards(short)
    source = [c for c in short["clips"] if c.get("kind") != "ai"]
    start = source[-1]["end"] if source else 0.0
    start = min(start, max(0.0, duration - 5.0))
    short["clips"].append({"uid": project.new_id(), "start": round(start, 1),
                           "end": round(min(duration, start + 5.0), 1), "narration": ""})
    project.save(p)
    refresh_cards()


def add_ai_card(p: project.Project, short: dict) -> None:
    sync_cards(short)
    short["clips"].append({"uid": project.new_id(), "kind": "ai", "prompt_ko": "", "prompt": "", "engine": "image",
                           "seed": int(project.new_id()[:4], 16), "seconds": 5.0, "narration": ""})
    project.save(p)
    refresh_cards()


def ensure_prompt(clip: dict) -> None:
    """장면 설명(한글 가능)을 생성 모델용 영어 프롬프트로 (설명이 바뀌었을 때만 다시 번역)."""
    text = clip.get("prompt_ko", "").strip()
    if text and clip.get("prompt_src") != text:
        clip["prompt"] = llm.english_prompt(llm_model, text) if llm_model else text
        clip["prompt_src"] = text


def card_scene(clip: dict) -> dict:
    """AI 장면 카드 → 생성 요청 {prompt(영어), engine, seed, seconds}."""
    return {"prompt": clip.get("prompt", ""), "engine": clip.get("engine", "image"),
            "seed": int(clip.get("seed", 0)), "seconds": float(clip.get("seconds", 5.0))}


def enqueue_cards(p: project.Project, short: dict, uids: set[str], new_seed: bool = False) -> int:
    """AI 장면 카드들을 생성 큐에 넣는다 (설명은 영어 프롬프트로 번역). 넣은 개수를 돌려준다."""
    sync_cards(short)
    scenes = []
    for clip in short["clips"]:
        if clip.get("kind") != "ai" or clip["uid"] not in uids or not clip.get("prompt_ko", "").strip():
            continue
        if new_seed:
            clip["seed"] = int(clip.get("seed", 0)) + 1
            clip.pop("preview", None)
        ensure_prompt(clip)
        scenes.append({**card_scene(clip), "label": clip["prompt_ko"].replace("*", "").strip()[:40], "card_uid": clip["uid"],
                       "project": f"{p.name} · {short['title'][:20]}"})
    project.save(p)
    refresh_cards()
    return genqueue.add(scenes)


def card_queue_item(clip: dict) -> dict | None:
    """이 카드의 지금 설정(설명·방식·시드)으로 넣은 큐 항목 (다른 설정으로 넣었던 것은 무시)."""
    item = genqueue.status_of_card(clip["uid"])
    if item and clip.get("prompt") and generate.scene_clip_path(genqueue.scene_of(item)) == generate.scene_clip_path(card_scene(clip)):
        return item
    return None


def ai_card(p: project.Project, short: dict, clip: dict, i: int) -> None:
    """AI 장면 카드: 미리보기·큐에 넣기, 방식·길이, 장면 설명, 나레이션."""
    uid = clip["uid"]
    if clip.get("prompt") and not (clip.get("preview") and Path(clip["preview"]).exists()):
        made = generate.cached_clip(card_scene(clip))  # 큐에서 다 만들었으면 미리보기로
        if made:
            clip["preview"] = str(made)
            project.save(p)
    with st.container(border=True, key=f"cardbox_{uid}"):
        col_media, col_opts, col_text = st.columns([1.3, 1, 3])
        with col_media:
            preview = Path(clip.get("preview", ""))
            item = card_queue_item(clip)
            if clip.get("preview") and preview.exists():
                st.video(str(preview))
            elif item and item["status"] in (genqueue.WAITING, genqueue.RUNNING):
                where = f" ({item['position']}번째)" if item["position"] else ""
                st.info(f"{genqueue.STATUS_LABELS[item['status']]}{where} · 아래 「AI 장면 생성 큐」")
            elif item and item["status"] == genqueue.FAILED:
                st.error("❌ 만들기 실패 · 아래 큐에서 이유를 볼 수 있습니다")
            else:
                st.info("🎨 아직 만들지 않은 AI 장면")
            gen_ok, gen_msg = generate.scene_available(clip.get("engine", "image"))
            col_make, col_again = st.columns(2)
            make = col_make.button("➕ 큐에 넣기", key=f"gen_{uid}", disabled=not gen_ok,
                                   help=gen_msg or "이 장면을 생성 큐에 넣습니다. 큐가 차례로 만들어 이 카드에 보여 줍니다.",
                                   width="stretch")
            again = col_again.button("🎲 다른 그림", key=f"regen_{uid}", disabled=not gen_ok,
                                     help="같은 설명, 다른 시드로 큐에 넣습니다.", width="stretch")
            if make or again:
                sync_cards(short)
                if not clip.get("prompt_ko", "").strip():
                    ss.flash = "장면 설명을 먼저 써 주세요."
                else:
                    with st.spinner("장면 설명을 영어 프롬프트로 바꾸는 중..."):
                        added = enqueue_cards(p, short, {uid}, new_seed=again)
                    ss.flash = "큐에 넣었습니다." if added else "이미 만들었거나 큐에 있는 장면입니다."
                st.rerun()
        with col_opts:
            st.markdown(f"**장면 {i + 1}** · ✨ AI")
            engine = st.selectbox("만드는 방식", list(generate.ENGINES), format_func=lambda e: generate.ENGINES[e].rsplit(" (", 1)[0],
                                  index=list(generate.ENGINES).index(clip.get("engine", "image")), key=card_key(uid, "engine"))
            seconds = st.number_input("길이(초)", 2.0, 15.0, float(clip.get("seconds", 5.0)), 0.5, format="%.1f",
                                      key=card_key(uid, "seconds"), help="나레이션이 더 길면 나레이션 길이만큼 늘어납니다.")
            card_buttons(p, short, uid, i)
        with col_text:
            prompt_ko = st.text_area("장면 설명 (한글로 써도 됩니다)", clip.get("prompt_ko", ""), height=68,
                                     key=card_key(uid, "prompt_ko"), placeholder="예: 안개 낀 호숫가의 작은 나무 오두막, 새벽")
            narration = st.text_area("나레이션 (자막)", clip["narration"], height=90, key=card_key(uid, "narration"))
            st.caption(f"{len(narration)}자 / 권장 {int(seconds * config.NARRATION_CHARS_PER_SECOND)}자")
    if (engine, seconds, prompt_ko, narration) != (clip.get("engine"), clip.get("seconds"), clip.get("prompt_ko"), clip["narration"]):
        if engine != clip.get("engine") or prompt_ko != clip.get("prompt_ko"):
            clip.pop("preview", None)  # 설명이나 방식을 바꾸면 미리보기는 다시 만들어야 함
        clip.update(engine=engine, seconds=seconds, prompt_ko=prompt_ko, narration=narration)
        project.save(p)


def card_buttons(p: project.Project, short: dict, uid: str, i: int) -> None:
    b_up, b_down, b_del = st.columns(3)
    b_up.button("⬆", key=f"up_{uid}", help="앞으로", disabled=i == 0, on_click=card_action, args=(p, short, uid, "up"))
    b_down.button("⬇", key=f"down_{uid}", help="뒤로", disabled=i + 1 == len(short["clips"]),
                  on_click=card_action, args=(p, short, uid, "down"))
    b_del.button("🗑", key=f"del_{uid}", help="이 카드 빼기", on_click=card_action, args=(p, short, uid, "delete"))


def clip_cards(p: project.Project, short: dict, job: pipeline.Job | None) -> list[dict]:
    """카드마다: 원본 구간(구간 재생, 시작·끝) 또는 AI 장면, 나레이션, 순서 바꾸기·삭제. 고친 값을 저장하고 카드 목록을 돌려준다."""
    cps = config.NARRATION_CHARS_PER_SECOND
    for i, clip in enumerate(short["clips"]):
        uid = clip["uid"]
        if clip.get("kind") == "ai":
            ai_card(p, short, clip, i)
            continue
        with st.container(border=True, key=f"cardbox_{uid}"):
            col_media, col_time, col_text = st.columns([1.3, 1, 3])
            with col_media:
                st.video(str(job.video), start_time=float(clip["start"]), end_time=float(clip["end"]))
            with col_time:
                st.markdown(f"**구간 {i + 1}**")
                start = st.number_input("시작(초)", 0.0, job.duration, float(clip["start"]), 0.1, format="%.1f",
                                        key=card_key(uid, "start"))
                end = st.number_input("끝(초)", 0.0, job.duration, float(clip["end"]), 0.1, format="%.1f",
                                      key=card_key(uid, "end"))
                card_buttons(p, short, uid, i)
            with col_text:
                narration = st.text_area("나레이션 (자막)", clip["narration"], height=130, key=card_key(uid, "narration"))
                limit = int(max(0.0, end - start) * cps)
                note = f"{len(narration)}자 / 권장 {limit}자 ({end - start:.1f}초)"
                if end <= start:
                    st.caption(f"⚠️ 끝이 시작보다 앞입니다. 이 구간은 빠집니다. · {note}")
                elif len(narration) > limit * 1.2:
                    st.caption(f"⚠️ {note} — 길면 구간이 자동으로 늘어납니다")
                else:
                    st.caption(note)
        if (start, end, narration) != (clip["start"], clip["end"], clip["narration"]):
            clip.update(start=start, end=end, narration=narration)  # 고친 값을 바로 저장 (새로고침해도 남음)
            project.save(p)
    col_add, col_add_ai = st.columns(2)
    if job is not None:
        col_add.button("➕ 구간 추가", key=f"add_{short['id']}", on_click=add_card, args=(p, short, job.duration),
                       help="마지막 구간 뒤 5초를 새 카드로 추가합니다. 시간을 고쳐 원하는 장면을 고르세요.", width="stretch")
    gen_ok, gen_msg = generate.available()
    col_add_ai.button("✨ AI 장면 추가", key=f"addai_{short['id']}", on_click=add_ai_card, args=(p, short),
                      disabled=not gen_ok, width="stretch",
                      help=gen_msg or "원본에 없는 장면(인트로·마무리·설명 그림 등)을 AI로 만들어 넣습니다.")
    return [c for c in short["clips"] if c.get("kind") == "ai" or c["end"] > c["start"]]


def make_plan(p: project.Project, on_progress=None) -> None:
    """사이드바 「쇼츠 개수」만큼 주제를 나눠 편마다 구간·나레이션을 구성한다 (1편이면 영상 전체에서)."""
    shorts = pipeline.plan_series(p.job, llm_model, ss.short_count, target_sec, llm.STYLES[style_label],
                                  on_progress=on_progress or (lambda done, n: None))
    if not shorts or not any(s["clips"] for s in shorts):
        raise RuntimeError("쓸 만한 구간을 고르지 못했습니다. 다시 구성하거나 쇼츠 길이를 바꿔 보세요.")
    p.shorts = shorts
    project.save(p)
    set_project(p)


if run_analyze:
    with st.status("영상을 이해하는 중...", expanded=True) as status:
        try:
            st.write("✂️ 구간 자르기, 오디오 추출")
            job = pipeline.prepare_job(source, start, end)
            st.write("📝 대사 인식 (Whisper)")
            pipeline.transcribe(job, get_whisper(dev("asr")))
            st.write("🖼️ 장면 캡처")
            pipeline.capture_frames(job)
            st.write(f"👀 장면 이해 ({llm_model})")
            bar = st.progress(0.0)
            pipeline.describe_frames(
                job, llm_model, on_progress=lambda done, n: bar.progress(done / n, text=f"{done} / {n} 장면")
            )
            proj.job = job
            project.save(proj)  # 구성에서 실패해도 분석 결과는 남긴다
            if ss.short_count > 1:
                st.write(f"🧠 서로 다른 주제로 쇼츠 {ss.short_count}편 나누기 + 편마다 구간·나레이션")
                plan_bar = st.progress(0.0)
                make_plan(proj, on_progress=lambda done, n: plan_bar.progress(done / n, text=f"{done} / {n}편 구성"))
            else:
                st.write("🧠 하이라이트 구간 고르기 + 나레이션 대본 쓰기")
                make_plan(proj)
            status.update(label="1단계 완료", state="complete", expanded=False)
        except Exception as e:
            status.update(label="1단계 실패", state="error")
            st.exception(e)
            st.stop()

job = proj.job
if proj.kind == "upload":
    if job is None:
        st.stop()
    with st.expander(f"🔎 AI가 이해한 내용 (장면 {len(job.frames)}개, 대사 {len(job.segments)}개)"):
        st.caption(f"대사 언어: {job.language or '대사 없음'}")
        st.text(pipeline.timeline(job) or "(내용 없음)")

# ───────────────────────── 2단계: 구성 수정 / 3단계: 쇼츠 만들기 ─────────────────────────
def settings_summary() -> str:
    """결과 기록에 남길 설정 한 줄 요약 (나란히 비교할 때 무엇이 달랐는지 보이게)."""
    if ss.engine == "edge":
        voice = f"{VOICE_LABELS.get(ss.edge_voice, ss.edge_voice)} {ss.rate:+d}%"
    else:
        chosen = next((v for v in voices.list_voices() if v["id"] == ss.voice_id), None)
        voice = f"Chatterbox {chosen['name'] if chosen else '기본 목소리'}"
    layout = {"blur": "블러 배경", "crop": "가운데 크롭", "smart": "피사체 따라 크롭"}.get(ss.layout, ss.layout)
    subs = "단어 강조 자막" if ss.subtitle_style == "karaoke" else "기본 자막"
    track = next((t for t in music.list_tracks() if t["id"] == ss.bgm_track), None)
    bgm = f"배경음악 {track['name']} {ss.bgm_gain:.2f}" if track else "배경음악 없음"
    return f"{ss.style} · {voice} · {layout} · {subs} · 원본 소리 {ss.original_gain:.2f}/{ss.duck_gain:.2f} · {bgm}"


def ai_models_summary(cards: list[dict]) -> str:
    kinds = sorted({c.get("engine", "image") for c in cards if c.get("kind") == "ai"})
    if not kinds:
        return ""
    return f" · LLM {llm_model} · AI 장면 {', '.join(generate.MODEL_INFO[k]['name'] for k in kinds)}"


def make_engine():
    if ss.engine == "edge":
        return tts.EdgeTTSEngine(ss.edge_voice, ss.rate)
    return tts.ChatterboxEngine(get_chatterbox(dev("tts")), voices.sample_path(ss.voice_id), ss.exaggeration)


def build_short(p: project.Project, short: dict, engine) -> None:
    """쇼츠 한 편을 만들어 결과 기록에 넣는다."""
    cards = [c for c in short["clips"] if c.get("kind") == "ai" or c["end"] > c["start"]]
    for c in cards:
        if c.get("kind") == "ai":
            if not c.get("prompt_ko", "").strip():
                raise RuntimeError(f"'{short['title']}'에 장면 설명이 비어 있는 AI 장면이 있습니다.")
            ensure_prompt(c)
    clips = pipeline.normalize_clips(p.job, cards) if p.job else cards
    if not clips:
        raise RuntimeError(f"'{short['title']}'에 쓸 구간이 없습니다.")
    bar = st.progress(0.0, text=f"🎙️ {short['title']} 나레이션 음성")
    video, report = pipeline.make_short(
        p.job, short["title"] if show_title else "", clips, engine, original_gain, duck_gain,
        ss.layout, font_size, margin_percent, color,
        on_progress=lambda done, n: bar.progress(done / n, text=f"🎙️ {short['title']} · {done} / {n} 문장"),
        out_dir=p.dir / "results",
        subtitle_style=ss.subtitle_style, highlight_color=ss.highlight_color,
        music_path=music.track_path(ss.bgm_track), music_gain=ss.bgm_gain, music_duck=ss.bgm_duck,
        work_dir=p.job.dir if p.job else p.dir / "render", on_status=st.write,
    )
    p.results.insert(0, {
        "id": project.new_id(), "short_id": short["id"], "title": short["title"],
        "video": str(video), "report": report, "created": project.now(),
        "duration": round(media.probe(video).duration, 1), "summary": settings_summary() + ai_models_summary(cards),
    })
    project.save(p)


def replan_short(p: project.Project, short: dict) -> None:
    """이 쇼츠 한 편만 다시 구성 (여러 편이면 그 편의 원본 범위·주제 안에서, AI 작업이면 주제로 대본부터)."""
    if p.kind == "ai":
        engines = {c.get("engine", "image") for c in short["clips"] if c.get("kind") == "ai"}
        engine = engines.pop() if len(engines) == 1 else "mix"
        title, clips = pipeline.plan_ai(llm_model, p.topic, target_sec, llm.STYLES[style_label], engine)
        short.update(title=title, clips=clips)
        project.save(p)
        set_project(p)
        return
    span = tuple(short.get("range") or (0.0, p.job.duration))
    title, clips = pipeline.plan(p.job, llm_model, target_sec, llm.STYLES[style_label],
                                 span=span if len(p.shorts) > 1 else None, focus=short.get("focus", ""))
    short.update(title=title, clips=clips)
    project.save(p)
    set_project(p)


def short_editor(p: project.Project, short: dict, job: pipeline.Job) -> None:
    sid = short["id"]
    col_title, col_replan = st.columns([3, 1])
    with col_title:
        title = st.text_input("제목 (화면 위쪽)", value=short["title"], disabled=not show_title, key=f"title_{sid}")
        if title != short["title"]:
            short["title"] = title
            project.save(p)
        if len(p.shorts) > 1 and short.get("range"):
            a, b = short["range"]
            st.caption(f"원본 {a:.0f}~{b:.0f}초 부분 · {short.get('focus', '')}")
    with col_replan:
        st.write("")
        replan_label = ("🔄 대본 다시 만들기" if p.kind == "ai" else
                        "🔄 이 쇼츠 다시 구성" if len(p.shorts) > 1 else "🔄 구성 다시 만들기")
        if st.button(replan_label, key=f"replan_{sid}", disabled=llm_model is None,
                     help="같은 분석 결과로 이 쇼츠의 구간과 나레이션을 다시 만듭니다. 「🧠 대본·구성」의 말투·길이가 반영됩니다."):
            with st.spinner("다시 구성하는 중..."):
                replan_short(p, short)
            st.rerun()

    raw_clips = clip_cards(p, short, job)
    # 쇼츠를 만들 때 쓰는 카드 (원본 구간은 범위 맞춤, 대사 경계, 겹침 합치기)
    clips = pipeline.normalize_clips(job, raw_clips) if job else raw_clips

    col_info, col_renarrate = st.columns([3, 1])
    with col_info:
        source = [c for c in clips if c.get("kind") != "ai"]
        ai = [c for c in clips if c.get("kind") == "ai"]
        parts = []
        if source:
            parts.append(f"원본 구간 {len(source)}개 {sum(c['end'] - c['start'] for c in source):.1f}초")
        if ai:
            parts.append(f"AI 장면 {len(ai)}개 {sum(float(c.get('seconds', 5)) for c in ai):.1f}초")
        st.caption(" · ".join(parts) or "카드가 없습니다.")
        if len(clips) < len(raw_clips):
            st.caption("⚠️ 겹치는 구간은 쇼츠를 만들 때 하나로 합쳐집니다.")
    with col_renarrate:
        if st.button(
            "✍️ 나레이션만 다시 쓰기", key=f"renarr_{sid}", disabled=not clips or llm_model is None,
            help="카드의 구간 시간은 그대로 두고, 구간마다 그 시간의 장면·대사를 보고 나레이션을 새로 씁니다.",
        ):
            with st.spinner("나레이션을 쓰는 중..."):
                short["clips"] = pipeline.narrate(job, raw_clips, llm_model, llm.STYLES[style_label])
                project.save(p)
                set_project(p)
            st.rerun()

    ai_cards = [c for c in short["clips"] if c.get("kind") == "ai"]
    if ai_cards:
        not_made = [c for c in ai_cards if not (c.get("preview") and Path(c["preview"]).exists())]
        gen_ok, gen_msg = generate.available()
        if st.button(f"🎬 AI 장면 {len(not_made)}개 모두 큐에 넣기", key=f"enqueue_all_{sid}",
                     disabled=not gen_ok or not not_made,
                     help=gen_msg or "아직 만들지 않은 AI 장면을 전부 생성 큐에 넣습니다. 큐가 차례로 만들고, "
                                     "그동안 다른 카드를 고치거나 큐 순서를 바꿀 수 있습니다."):
            with st.spinner("장면 설명을 영어 프롬프트로 바꾸는 중..."):
                added = enqueue_cards(p, short, {c["uid"] for c in not_made})
            ss.flash = f"AI 장면 {added}개를 큐에 넣었습니다." if added else "모두 이미 만들었거나 큐에 있습니다."
            st.rerun()

    if st.button("3단계 · 나레이션 + 쇼츠 만들기", type="primary", disabled=not clips, key=f"make_{sid}"):
        with st.status("쇼츠를 만드는 중...", expanded=True) as status:
            try:
                build_short(p, short, make_engine())
                status.update(label="완성!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="만들기 실패", state="error")
                st.exception(e)


st.divider()
st.subheader("2단계 · 쇼츠 구성과 나레이션 확인")
st.caption(
    "AI가 고른 원본 구간과 구간별 나레이션입니다. 카드에서 구간을 재생해 보고 시간과 나레이션을 고치거나, "
    "순서를 바꾸고 구간을 추가·삭제할 수 있습니다. 나레이션이 구간보다 길면 쇼츠를 만들 때 구간을 자동으로 늘립니다. "
    "고친 내용은 바로 저장됩니다."
)
if not proj.shorts and proj.kind == "upload":
    if st.button("🧠 쇼츠 구성 만들기", type="primary", disabled=llm_model is None):
        with st.spinner("구성하는 중..."):
            make_plan(proj)
        st.rerun()
    st.stop()

if proj.kind == "upload" and (len(proj.shorts) > 1 or ss.short_count > 1):
    col_count, col_all_plan, col_all_make = st.columns([2, 1, 1])
    col_count.markdown(f"**쇼츠 {len(proj.shorts)}편** · 한 영상에서 서로 다른 주제로 나눴습니다.")
    if col_all_plan.button(f"🔄 {ss.short_count}편으로 다시 나누기", disabled=llm_model is None,
                           help="「🧠 대본·구성」의 「쇼츠 개수」만큼 주제를 다시 나누고 편마다 구성합니다."):
        with st.spinner(f"쇼츠 {ss.short_count}편으로 나누는 중..."):
            make_plan(proj)
        st.rerun()
    if col_all_make.button("🎬 모든 쇼츠 만들기", type="primary", disabled=len(proj.shorts) < 2):
        with st.status(f"쇼츠 {len(proj.shorts)}편을 만드는 중...", expanded=True) as status:
            try:
                engine = make_engine()
                for item in proj.shorts:
                    build_short(proj, item, engine)
                status.update(label=f"{len(proj.shorts)}편 완성!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="만들기 실패", state="error")
                st.exception(e)

if len(proj.shorts) == 1:
    short_editor(proj, proj.shorts[0], job)
else:
    tabs = st.tabs([f"{i + 1}. {s['title'][:14] or '(제목 없음)'}" for i, s in enumerate(proj.shorts)])
    for tab, item in zip(tabs, proj.shorts):
        with tab:
            short_editor(proj, item, job)


# ───────────────────────── AI 장면 생성 큐 ─────────────────────────
def queue_rows() -> None:
    """큐 목록: 순서 바꾸기·지우기·취소·다시 시도. 끝난 항목이 생기면 카드를 새로 그린다."""
    items = genqueue.items()
    finished = {q["id"] for q in items if q["status"] in (genqueue.DONE, genqueue.FAILED)}
    if ss.get("queue_finished") is not None and finished - ss.queue_finished:
        ss.queue_finished = finished
        st.rerun(scope="app")  # 카드 미리보기·상태를 새로 그림
    ss.queue_finished = finished
    waiting = [q for q in items if q["status"] == genqueue.WAITING]
    running = [q for q in items if q["status"] == genqueue.RUNNING]
    done = [q for q in items if q["status"] == genqueue.DONE]
    failed = [q for q in items if q["status"] in (genqueue.FAILED, genqueue.CANCELED)]

    col_info, col_pause, col_clear_done, col_clear_wait = st.columns([3, 1, 1, 1])
    paused = genqueue.is_paused()
    col_info.markdown(f"만드는 중 **{len(running)}** · 대기 **{len(waiting)}** · 완료 {len(done)} · 실패·취소 {len(failed)}"
                      + (" · ⏸ **일시정지됨**" if paused else ""))
    col_pause.button("▶ 이어서 만들기" if paused else "⏸ 일시정지", key="q_pause", width="stretch",
                     on_click=genqueue.set_paused, args=(not paused,),
                     help="일시정지하면 지금 만드는 장면까지만 만들고 멈춥니다.")
    col_clear_done.button("완료 비우기", key="q_clear_done", width="stretch", disabled=not (done or failed),
                          on_click=genqueue.clear, args=((genqueue.DONE, genqueue.FAILED, genqueue.CANCELED),))
    col_clear_wait.button("대기 전부 빼기", key="q_clear_wait", width="stretch", disabled=not waiting,
                          on_click=genqueue.clear, args=((genqueue.WAITING,),))
    if not items:
        st.caption("큐가 비어 있습니다. AI 장면 카드의 「➕ 큐에 넣기」나 「🎬 모두 큐에 넣기」로 넣으세요.")
        return

    order = running + waiting + failed + done
    for q in order:
        with st.container(border=True, key=f"q_{q['id']}"):
            col_state, col_text, col_btns = st.columns([1.2, 4, 2])
            kind = "video" if q["engine"] == "video" else "image"
            if q["status"] == genqueue.WAITING:
                col_state.markdown(f"⏳ **{waiting.index(q) + 1}번째**")
            else:
                col_state.markdown(genqueue.STATUS_LABELS[q["status"]])
            label = (q["label"] or q["prompt"][:40]).replace("*", "").strip()  # 설명에 ** 가 있으면 굵게 표시가 깨짐
            col_text.markdown(f"**{label}** · {q['seconds']:.1f}초 · 시드 {q['seed']}")
            col_text.caption(f"{generate.MODEL_INFO[kind]['name']} · {q['project']} · 넣은 시각 {q['created']}")
            if q["status"] == genqueue.RUNNING:
                prog = genqueue.progress(q["id"])
                if prog.get("total"):
                    col_text.progress(prog["step"] / prog["total"], text=f"{prog['step']} / {prog['total']} 단계")
                else:
                    col_text.caption(prog.get("note", "준비 중"))
            if q.get("error"):
                with col_text.expander("실패 이유"):
                    st.code(q["error"][-600:])
            b1, b2, b3 = col_btns.columns(3)
            if q["status"] == genqueue.WAITING:
                pos = waiting.index(q)
                b1.button("⬆", key=f"qup_{q['id']}", help="앞으로", disabled=pos == 0,
                          on_click=genqueue.move, args=(q["id"], -1))
                b2.button("⬇", key=f"qdown_{q['id']}", help="뒤로", disabled=pos + 1 == len(waiting),
                          on_click=genqueue.move, args=(q["id"], 1))
                b3.button("🗑", key=f"qdel_{q['id']}", help="큐에서 빼기", on_click=genqueue.remove, args=(q["id"],))
            elif q["status"] == genqueue.RUNNING:
                b3.button("⏹", key=f"qstop_{q['id']}", help="지금 만드는 장면 취소", on_click=genqueue.remove, args=(q["id"],))
            else:
                if q["status"] in (genqueue.FAILED, genqueue.CANCELED):
                    b1.button("🔁", key=f"qretry_{q['id']}", help="다시 시도", on_click=genqueue.retry, args=(q["id"],))
                b3.button("🗑", key=f"qdel_{q['id']}", help="목록에서 지우기", on_click=genqueue.remove, args=(q["id"],))


genqueue.ensure_runner()
queue_items = genqueue.items()
if any(c.get("kind") == "ai" for s in proj.shorts for c in s["clips"]) or queue_items:
    st.divider()
    active = any(q["status"] in (genqueue.WAITING, genqueue.RUNNING) for q in queue_items)
    st.subheader("AI 장면 생성 큐")
    st.caption("넣은 순서대로 한 장면씩 만듭니다. 대기 중인 장면은 순서를 바꾸거나 뺄 수 있고, 다른 카드를 고치는 동안에도 "
               "계속 만듭니다. 큐는 저장되어 새로고침해도 남습니다. 쇼츠를 만들 때 이미 만든 장면은 다시 만들지 않습니다.")
    # 만드는 중일 때만 2초마다 새로 고침
    st.fragment(queue_rows, run_every=2 if active else None)()

# ───────────────────────── 결과 기록 ─────────────────────────
def delete_result(p: project.Project, result_id: str) -> None:
    for r in [r for r in p.results if r["id"] == result_id]:
        video = Path(r["video"])
        for path in (video, video.with_suffix(".srt"), video.with_suffix(".jpg")):
            path.unlink(missing_ok=True)
        p.results.remove(r)
    project.save(p)
    ss.flash = "결과를 지웠습니다."


RESULTS_PER_ROW = 3
if proj.results:
    st.divider()
    st.subheader(f"결과 기록 ({len(proj.results)}개)")
    st.caption("만든 쇼츠가 최신 순으로 쌓입니다. 말투·목소리·화면 설정을 바꿔 가며 만들어 나란히 비교해 보세요.")
    latest = proj.results[0]["report"]
    if any("잘림" in r["상태"] or "이어짐" in r["상태"] for r in latest):
        st.warning("가장 최근 결과에서 나레이션이 구간보다 길어 다음 장면으로 넘어가거나 잘린 곳이 있습니다. 2단계에서 나레이션을 줄이거나 구간을 늘려 보세요.")
    for row_start in range(0, len(proj.results), RESULTS_PER_ROW):
        cols = st.columns(RESULTS_PER_ROW)
        for col, result in zip(cols, proj.results[row_start:row_start + RESULTS_PER_ROW]):
            with col, st.container(border=True, key=f"result_{result['id']}"):
                badge = "🆕 " if result is proj.results[0] else ""
                st.markdown(f"**{badge}{result['title'] or '(제목 없음)'}**")
                video_path = Path(result["video"])
                tab_video, tab_thumb = st.tabs(["영상", "썸네일"])
                with tab_video:
                    st.video(result["video"])
                with tab_thumb:
                    if video_path.with_suffix(".jpg").exists():
                        st.image(str(video_path.with_suffix(".jpg")), width="stretch")
                    else:
                        st.caption("썸네일이 없습니다 (이전 버전에서 만든 결과).")
                st.caption(f"{result['created']} · {result.get('duration', '?')}초")
                if result.get("summary"):
                    st.caption(result["summary"])
                stamp = result["created"].replace("-", "").replace(":", "").replace(" ", "_")
                col_dl, col_del = st.columns([3, 1])
                col_dl.download_button(
                    "⬇️ 다운로드", data=Path(result["video"]).read_bytes(), key=f"dl_{result['id']}",
                    file_name=f"{Path(proj.name).stem}_{stamp}.mp4", mime="video/mp4",
                    type="primary" if result is proj.results[0] else "secondary", width="stretch",
                )
                col_del.button("🗑", key=f"delres_{result['id']}", help="이 결과 지우기",
                               on_click=delete_result, args=(proj, result["id"]))
                col_srt, col_jpg = st.columns(2)
                for col_x, suffix, label, mime in ((col_srt, ".srt", "자막 SRT", "application/x-subrip"),
                                                   (col_jpg, ".jpg", "썸네일 JPG", "image/jpeg")):
                    extra = video_path.with_suffix(suffix)
                    if extra.exists():
                        col_x.download_button(f"⬇️ {label}", data=extra.read_bytes(), key=f"dl{suffix}_{result['id']}",
                                              file_name=f"{Path(proj.name).stem}_{stamp}{suffix}", mime=mime, width="stretch")
                with st.expander("구간별 나레이션 길이"):
                    st.dataframe(result["report"], hide_index=True, width="stretch")
