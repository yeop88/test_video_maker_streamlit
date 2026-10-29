"""AI 쇼츠 나레이션 스튜디오 — 영상을 이해해 하이라이트를 고르고, AI 나레이션과 한글 자막을 입혀 쇼츠로 만든다."""
import hashlib
from pathlib import Path

import streamlit as st

import shorts_maker
from shorts_maker import config, llm, media, pipeline, presets, tts

st.set_page_config(page_title="AI 쇼츠 나레이션 스튜디오", page_icon="🎬", layout="wide")

ENGINE_LABELS = {"edge": "Edge TTS (기본 목소리, 인터넷 필요)", "chatterbox": "Chatterbox (GPU, 목소리 샘플 사용 가능)"}
LAYOUT_LABELS = {"blur": "블러 배경 (가로 영상·글자 많은 화면 추천)", "crop": "가운데 크롭 (화면 꽉 차게, 양옆 잘림)"}
VOICE_LABELS = {voice_id: label for label, voice_id in tts.EDGE_VOICES.items()}
STATE_KEYS = ("job", "title", "plan", "result")


@st.cache_resource(show_spinner="Whisper 모델을 불러오는 중... (처음 한 번은 다운로드로 몇 분 걸립니다)")
def get_whisper():
    from shorts_maker import asr

    return asr.load_whisper()


@st.cache_resource(show_spinner="Chatterbox 모델을 불러오는 중... (처음 한 번은 다운로드로 몇 분 걸립니다)")
def get_chatterbox():
    return tts.load_chatterbox()


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


def current_settings() -> dict:
    return {key: ss[key] for key in presets.DEFAULTS}


def apply_preset(name: str) -> None:
    try:
        values, dropped = presets.load(name)
    except (FileNotFoundError, ValueError) as e:
        ss.preset_msg = ("error", str(e))
        return
    notes = [f"설정 {', '.join(dropped)}은(는) 올바르지 않아 기본값을 썼습니다"] if dropped else []
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


# ───────────────────────── 사이드바: 설정 ─────────────────────────
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
    msg = ss.pop("preset_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])

    st.subheader("영상 이해 / 대본")
    if not models:
        st.error("Ollama에 연결할 수 없습니다. Ollama를 실행한 뒤 새로고침하세요.")
    llm_model = st.selectbox(
        "LLM (Ollama)", models, key="llm_model",
        help="장면 이미지를 이해해야 하므로 vision을 지원하는 모델(qwen3.5, gemma4 등)을 고르세요.",
    ) if models else None
    st.selectbox("나레이션 말투", presets.CHOICES["style"], key="style")
    setting_slider("쇼츠 길이 목표 (초)", "target_sec")

    st.subheader("나레이션 목소리")
    st.radio("음성 엔진", presets.CHOICES["engine"], format_func=ENGINE_LABELS.get, key="engine")
    voice_sample = None
    if ss.engine == "edge":
        st.selectbox("목소리", presets.CHOICES["edge_voice"], format_func=VOICE_LABELS.get, key="edge_voice")
        setting_slider("말하기 속도 (%)", "rate")
    else:
        setting_slider("감정 표현 강도", "exaggeration")
        voice_sample = st.file_uploader(
            "목소리 샘플 (선택, 5~15초)", type=["wav", "mp3", "m4a"],
            help="비워 두면 Chatterbox 기본 목소리를 씁니다. 본인 목소리나 동의받은 목소리만 올리세요. 프리셋에는 저장되지 않습니다.",
        )

    st.subheader("소리")
    setting_slider("원본 소리 크기", "original_gain")
    setting_slider(
        "나레이션 중 원본 소리 크기", "duck_gain",
        help="나레이션이 나오는 동안 원본 소리를 이 비율로 줄입니다 (원본 소리 크기에 곱해짐).",
    )

    st.subheader("화면 / 자막")
    st.radio("세로 변환 방식", presets.CHOICES["layout"], format_func=LAYOUT_LABELS.get, key="layout")
    st.checkbox("위쪽에 제목 표시", key="show_title")
    setting_slider("자막 크기", "font_size")
    setting_slider("자막 위치 (아래에서 %)", "margin_percent")
    st.color_picker("자막 색", key="color")

    st.divider()
    st.caption("⚠️ 본인이 권리를 가진 영상이나 허락받은 영상만 사용하세요. 목소리 샘플은 본인 또는 동의받은 사람의 목소리만 사용하세요.")
    st.caption(f"v{shorts_maker.__version__}")

style_label, target_sec = ss.style, ss.target_sec
show_title, font_size, margin_percent, color = ss.show_title, ss.font_size, ss.margin_percent, ss.color
original_gain, duck_gain = ss.original_gain, ss.duck_gain

# ───────────────────────── 업로드 ─────────────────────────
st.title("🎬 AI 쇼츠 나레이션 스튜디오")
st.caption(
    "영상을 올리면 AI가 **장면과 대사를 이해**해서 하이라이트 구간을 고르고, "
    "**한국어 나레이션**과 **한글 자막**을 입힌 세로 쇼츠/릴스(1080×1920)를 만들어 줍니다."
)

uploaded = st.file_uploader("영상 업로드", type=["mp4", "mov", "webm", "mkv", "m4v"])
if uploaded is None:
    st.info(f"영상을 올리면 시작합니다. 긴 영상은 분석할 구간을 {config.MAX_SOURCE_SECONDS // 60}분 이하로 골라 주세요.")
    st.stop()

data = uploaded.getvalue()
upload_id = hashlib.sha1(data).hexdigest()[:12]
if st.session_state.get("upload_id") != upload_id:  # 새 영상이면 이전 결과 초기화
    for key in STATE_KEYS:
        st.session_state.pop(key, None)
    source = pipeline.save_upload(upload_id, uploaded.name, data)
    st.session_state.update(upload_id=upload_id, source=source, info=media.probe(source))

source: Path = st.session_state.source
info: media.VideoInfo = st.session_state.info

col_video, col_options = st.columns([1, 2])
with col_video:
    st.video(data)
with col_options:
    st.markdown(f"**길이** {info.duration:.1f}초 · **해상도** {info.width}×{info.height}" + ("" if info.has_audio else " · 소리 없음"))
    total = round(info.duration, 1)
    start, end = st.slider("분석할 구간 (초)", 0.0, total, (0.0, min(total, float(config.MAX_SOURCE_SECONDS))), 0.5)
    if end - start > config.MAX_SOURCE_SECONDS:
        st.warning(f"{config.MAX_SOURCE_SECONDS // 60}분이 넘으면 분석이 오래 걸리고 장면 설명이 듬성듬성해집니다.")
    run_analyze = st.button("1단계 · 영상 분석 + 쇼츠 구성", type="primary", disabled=llm_model is None)


def make_plan(job: pipeline.Job) -> None:
    title, clips = pipeline.plan(job, llm_model, target_sec, llm.STYLES[style_label])
    if not clips:
        raise RuntimeError("쓸 만한 구간을 고르지 못했습니다. 다시 구성하거나 쇼츠 길이를 바꿔 보세요.")
    st.session_state.title = title
    st.session_state.plan = pipeline.plan_dataframe(clips)
    st.session_state.pop("result", None)
    st.session_state.pop("editor", None)


if run_analyze:
    for key in STATE_KEYS:
        st.session_state.pop(key, None)
    with st.status("영상을 이해하는 중...", expanded=True) as status:
        try:
            st.write("✂️ 구간 자르기, 오디오 추출")
            job = pipeline.prepare_job(source, start, end)
            st.write("📝 대사 인식 (Whisper)")
            pipeline.transcribe(job, get_whisper())
            st.write("🖼️ 장면 캡처")
            pipeline.capture_frames(job)
            st.write(f"👀 장면 이해 ({llm_model})")
            bar = st.progress(0.0)
            pipeline.describe_frames(
                job, llm_model, on_progress=lambda done, n: bar.progress(done / n, text=f"{done} / {n} 장면")
            )
            st.write("🧠 하이라이트 구간 고르기 + 나레이션 대본 쓰기")
            st.session_state.job = job
            make_plan(job)
            status.update(label="1단계 완료", state="complete", expanded=False)
        except Exception as e:
            status.update(label="1단계 실패", state="error")
            st.exception(e)
            st.stop()

job: pipeline.Job | None = st.session_state.get("job")
if job is None:
    st.stop()

with st.expander(f"🔎 AI가 이해한 내용 (장면 {len(job.frames)}개, 대사 {len(job.segments)}개)"):
    st.caption(f"대사 언어: {job.language or '대사 없음'}")
    st.text(pipeline.timeline(job) or "(내용 없음)")

# ───────────────────────── 2단계: 구성 수정 ─────────────────────────
st.divider()
st.subheader("2단계 · 쇼츠 구성과 나레이션 확인")
st.caption(
    "AI가 고른 원본 구간과 구간별 나레이션입니다. 시간과 대본을 고치거나, 행을 추가·삭제할 수 있습니다. "
    "나레이션이 구간보다 길면 구간을 자동으로 늘립니다. **권장 글자 수**는 구간 길이 기준입니다."
)
col_title, col_replan = st.columns([3, 1])
with col_title:
    title = st.text_input("제목 (화면 위쪽)", value=st.session_state.get("title", ""), disabled=not show_title)
with col_replan:
    st.write("")
    if st.button("🔄 구성 다시 만들기", help="같은 분석 결과로 구간과 나레이션을 다시 만듭니다. 사이드바의 말투·길이가 반영됩니다."):
        with st.spinner("다시 구성하는 중..."):
            make_plan(job)
        st.rerun()

edited = st.data_editor(
    st.session_state.plan,
    column_config={
        "start": st.column_config.NumberColumn("시작(초)", step=0.1, format="%.1f", width="small", min_value=0.0),
        "end": st.column_config.NumberColumn("끝(초)", step=0.1, format="%.1f", width="small", min_value=0.0),
        "narration": st.column_config.TextColumn("나레이션 (자막)", width="large"),
        "max_chars": st.column_config.NumberColumn("권장 글자 수", disabled=True, width="small"),
    },
    hide_index=True,
    num_rows="dynamic",
    width="stretch",
    key="editor",
)
clips = pipeline.plan_from_dataframe(job, edited)
col_info, col_renarrate = st.columns([3, 1])
with col_info:
    st.caption(f"구간 {len(clips)}개 · 원본 구간 합계 {sum(c['end'] - c['start'] for c in clips):.1f}초")
with col_renarrate:
    if st.button(
        "✍️ 나레이션만 다시 쓰기", disabled=not clips or llm_model is None,
        help="표의 구간 시간은 그대로 두고, 구간마다 그 시간의 장면·대사를 보고 나레이션을 새로 씁니다.",
    ):
        with st.spinner("나레이션을 쓰는 중..."):
            st.session_state.plan = pipeline.plan_dataframe(
                pipeline.narrate(job, clips, llm_model, llm.STYLES[style_label])
            )
            st.session_state.pop("editor", None)
        st.rerun()

# ───────────────────────── 3단계: 쇼츠 만들기 ─────────────────────────
if st.button("3단계 · 나레이션 + 쇼츠 만들기", type="primary", disabled=not clips):
    with st.status("쇼츠를 만드는 중...", expanded=True) as status:
        try:
            if ss.engine == "edge":
                engine = tts.EdgeTTSEngine(ss.edge_voice, ss.rate)
            else:
                reference = pipeline.save_voice_sample(voice_sample.name, voice_sample.getvalue()) if voice_sample else None
                engine = tts.ChatterboxEngine(get_chatterbox(), reference, ss.exaggeration)
            st.write("🎙️ 나레이션 음성 만들기")
            bar = st.progress(0.0)
            video, report = pipeline.make_short(
                job, title if show_title else "", clips, engine, original_gain, duck_gain,
                ss.layout, font_size, margin_percent, color,
                on_progress=lambda done, n: bar.progress(done / n, text=f"{done} / {n} 문장"),
            )
            st.session_state.result = {"video": video, "report": report}
            status.update(label="완성!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="만들기 실패", state="error")
            st.exception(e)

# ───────────────────────── 결과 ─────────────────────────
result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("결과")
    col_out, col_report = st.columns([1, 2])
    with col_out:
        st.video(str(result["video"]))
        st.download_button(
            "⬇️ 쇼츠 다운로드", data=Path(result["video"]).read_bytes(),
            file_name=f"{Path(uploaded.name).stem}_shorts.mp4", mime="video/mp4", type="primary",
        )
    with col_report:
        report = result["report"]
        st.dataframe(report, hide_index=True, width="stretch")
        if any("잘림" in r["상태"] or "이어짐" in r["상태"] for r in report):
            st.warning("나레이션이 구간보다 길어서 다음 장면으로 넘어가거나 잘린 곳이 있습니다. 2단계에서 대본을 줄이거나 구간을 늘려 보세요.")
        else:
            st.success("모든 나레이션이 자기 장면 안에 들어갔습니다.")
