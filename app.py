"""AI 쇼츠 나레이션 스튜디오 — 영상을 이해해 하이라이트를 고르고, AI 나레이션과 한글 자막을 입혀 쇼츠로 만든다."""
import hashlib
from pathlib import Path

# pyarrow는 맨 먼저 불러온다. Whisper·Chatterbox(torch/CUDA) 등을 올린 뒤 처음으로 pyarrow DLL을 불러오면
# Windows에서 서버 프로세스가 access violation으로 죽는 일이 있었다 (결과 표 st.dataframe에서 처음 불러올 때).
import pyarrow  # noqa: F401
import streamlit as st
import streamlit.components.v1 as components

import shorts_maker
from shorts_maker import config, llm, media, pipeline, presets, project, tts

st.set_page_config(page_title="AI 쇼츠 나레이션 스튜디오", page_icon="🎬", layout="wide")

ENGINE_LABELS = {"edge": "Edge TTS (기본 목소리, 인터넷 필요)", "chatterbox": "Chatterbox (GPU, 목소리 샘플 사용 가능)"}
LAYOUT_LABELS = {"blur": "블러 배경 (가로 영상·글자 많은 화면 추천)", "crop": "가운데 크롭 (화면 꽉 차게, 양옆 잘림)"}
VOICE_LABELS = {voice_id: label for label, voice_id in tts.EDGE_VOICES.items()}
SUBTITLE_LABELS = {"plain": "기본", "karaoke": "단어 강조 (읽는 낱말 색칠)"}
VIDEO_TYPES = ["mp4", "mov", "webm", "mkv", "m4v", "avi"]
UI_DIR = Path(__file__).parent / "ui"


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
    st.radio("자막 스타일", presets.CHOICES["subtitle_style"], format_func=SUBTITLE_LABELS.get, key="subtitle_style",
             help="단어 강조: 나레이션이 읽는 낱말을 다른 색으로 조금 크게 보여 줍니다.")
    if ss.subtitle_style == "karaoke":
        st.color_picker("강조 색", key="highlight_color")
    setting_slider("자막 크기", "font_size")
    setting_slider("자막 위치 (아래에서 %)", "margin_percent")
    st.color_picker("자막 색", key="color")

    st.divider()
    st.caption("⚠️ 본인이 권리를 가진 영상이나 허락받은 영상만 사용하세요. 목소리 샘플은 본인 또는 동의받은 사람의 목소리만 사용하세요.")
    st.caption(f"v{shorts_maker.__version__}")

style_label, target_sec = ss.style, ss.target_sec
show_title, font_size, margin_percent, color = ss.show_title, ss.font_size, ss.margin_percent, ss.color
original_gain, duck_gain = ss.original_gain, ss.duck_gain


# ───────────────────────── 작업 열기 (새로고침해도 이어서) ─────────────────────────
st.title("🎬 AI 쇼츠 나레이션 스튜디오")
st.caption(
    "영상을 올리면 AI가 **장면과 대사를 이해**해서 하이라이트 구간을 고르고, "
    "**한국어 나레이션**과 **한글 자막**을 입힌 세로 쇼츠/릴스(1080×1920)를 만들어 줍니다."
)


def refresh_cards() -> None:
    """구간 카드 입력칸을 저장된 값으로 다시 그리게 한다 (카드 위젯 키에 붙는 번호를 바꿈)."""
    ss.card_rev = ss.get("card_rev", 0) + 1


def set_project(p: project.Project, message: str = "") -> None:
    """작업을 화면 상태로 불러온다."""
    ss.project = p
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


recent_items = project.recent()
with st.expander(f"📂 최근 작업 ({len(recent_items)})", expanded=False):
    if not recent_items:
        st.caption("아직 작업이 없습니다.")
    for item in recent_items[:10]:
        col_name, col_info, col_open = st.columns([4, 3, 1])
        icon = "✨" if item["kind"] == "ai" else "📼"
        col_name.markdown(f"{icon} **{item['name']}**")
        status = "분석 완료" if item["analyzed"] else "분석 전"
        col_info.caption(f"{item['updated']} · {status} · 만든 쇼츠 {item['results']}개")
        col_open.button("열기", key=f"open_{item['dir'].name}", on_click=open_recent, args=(item["dir"],))

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
if proj is None:
    st.info(f"영상을 올리면 시작합니다. 긴 영상은 분석할 구간을 {config.MAX_SOURCE_SECONDS // 60}분 이하로 골라 주세요.")
    st.stop()


@st.cache_data(show_spinner=False)
def probe(path: str) -> media.VideoInfo:
    return media.probe(Path(path))


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
        for field in ("start", "end", "narration"):
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
    start = short["clips"][-1]["end"] if short["clips"] else 0.0
    start = min(start, max(0.0, duration - 5.0))
    short["clips"].append({"uid": project.new_id(), "start": round(start, 1),
                           "end": round(min(duration, start + 5.0), 1), "narration": ""})
    project.save(p)
    refresh_cards()


def clip_cards(p: project.Project, short: dict, job: pipeline.Job) -> list[dict]:
    """구간마다 카드 하나: 구간 재생, 시작·끝, 나레이션, 순서 바꾸기·삭제. 고친 값을 저장하고 구간 목록을 돌려준다."""
    cps = config.NARRATION_CHARS_PER_SECOND
    for i, clip in enumerate(short["clips"]):
        uid = clip["uid"]
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
                b_up, b_down, b_del = st.columns(3)
                b_up.button("⬆", key=f"up_{uid}", help="앞으로", disabled=i == 0,
                            on_click=card_action, args=(p, short, uid, "up"))
                b_down.button("⬇", key=f"down_{uid}", help="뒤로", disabled=i + 1 == len(short["clips"]),
                              on_click=card_action, args=(p, short, uid, "down"))
                b_del.button("🗑", key=f"del_{uid}", help="이 구간 빼기", on_click=card_action, args=(p, short, uid, "delete"))
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
    st.button("➕ 구간 추가", on_click=add_card, args=(p, short, job.duration),
              help="마지막 구간 뒤 5초를 새 카드로 추가합니다. 시간을 고쳐 원하는 장면을 고르세요.")
    return [c for c in short["clips"] if c["end"] > c["start"]]


def make_plan(p: project.Project) -> None:
    title, clips = pipeline.plan(p.job, llm_model, target_sec, llm.STYLES[style_label])
    if not clips:
        raise RuntimeError("쓸 만한 구간을 고르지 못했습니다. 다시 구성하거나 쇼츠 길이를 바꿔 보세요.")
    p.shorts = [{"id": project.new_id(), "title": title, "clips": clips}]
    project.save(p)
    set_project(p)


if run_analyze:
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
            proj.job = job
            project.save(proj)  # 구성에서 실패해도 분석 결과는 남긴다
            st.write("🧠 하이라이트 구간 고르기 + 나레이션 대본 쓰기")
            make_plan(proj)
            status.update(label="1단계 완료", state="complete", expanded=False)
        except Exception as e:
            status.update(label="1단계 실패", state="error")
            st.exception(e)
            st.stop()

job = proj.job
if job is None:
    st.stop()

with st.expander(f"🔎 AI가 이해한 내용 (장면 {len(job.frames)}개, 대사 {len(job.segments)}개)"):
    st.caption(f"대사 언어: {job.language or '대사 없음'}")
    st.text(pipeline.timeline(job) or "(내용 없음)")

# ───────────────────────── 2단계: 구성 수정 ─────────────────────────
st.divider()
st.subheader("2단계 · 쇼츠 구성과 나레이션 확인")
st.caption(
    "AI가 고른 원본 구간과 구간별 나레이션입니다. 카드에서 구간을 재생해 보고 시간과 나레이션을 고치거나, "
    "순서를 바꾸고 구간을 추가·삭제할 수 있습니다. 나레이션이 구간보다 길면 쇼츠를 만들 때 구간을 자동으로 늘립니다. "
    "고친 내용은 바로 저장됩니다."
)
if not proj.shorts:
    if st.button("🧠 쇼츠 구성 만들기", type="primary", disabled=llm_model is None):
        with st.spinner("구성하는 중..."):
            make_plan(proj)
        st.rerun()
    st.stop()
short = proj.shorts[0]

col_title, col_replan = st.columns([3, 1])
with col_title:
    title = st.text_input("제목 (화면 위쪽)", value=short["title"], disabled=not show_title, key=f"title_{short['id']}")
with col_replan:
    st.write("")
    if st.button("🔄 구성 다시 만들기", help="같은 분석 결과로 구간과 나레이션을 다시 만듭니다. 사이드바의 말투·길이가 반영됩니다."):
        with st.spinner("다시 구성하는 중..."):
            make_plan(proj)
        st.rerun()

raw_clips = clip_cards(proj, short, job)
clips = pipeline.normalize_clips(job, raw_clips)  # 쇼츠를 만들 때 쓰는 구간 (범위 맞춤, 대사 경계, 겹침 합치기)
if title != short["title"]:
    short["title"] = title
    project.save(proj)

col_info, col_renarrate = st.columns([3, 1])
with col_info:
    st.caption(f"구간 {len(clips)}개 · 원본 구간 합계 {sum(c['end'] - c['start'] for c in clips):.1f}초")
    if len(clips) < len([c for c in raw_clips if c["end"] > c["start"]]):
        st.caption("⚠️ 겹치는 구간은 쇼츠를 만들 때 하나로 합쳐집니다.")
with col_renarrate:
    if st.button(
        "✍️ 나레이션만 다시 쓰기", disabled=not clips or llm_model is None,
        help="카드의 구간 시간은 그대로 두고, 구간마다 그 시간의 장면·대사를 보고 나레이션을 새로 씁니다.",
    ):
        with st.spinner("나레이션을 쓰는 중..."):
            short["clips"] = pipeline.narrate(job, raw_clips, llm_model, llm.STYLES[style_label])
            project.save(proj)
            set_project(proj)
        st.rerun()

def settings_summary() -> str:
    """결과 기록에 남길 설정 한 줄 요약 (나란히 비교할 때 무엇이 달랐는지 보이게)."""
    if ss.engine == "edge":
        voice = f"{VOICE_LABELS.get(ss.edge_voice, ss.edge_voice)} {ss.rate:+d}%"
    else:
        voice = f"Chatterbox {'목소리 샘플' if voice_sample else '기본 목소리'}"
    layout = {"blur": "블러 배경", "crop": "가운데 크롭"}.get(ss.layout, ss.layout)
    subs = "단어 강조 자막" if ss.subtitle_style == "karaoke" else "기본 자막"
    return f"{ss.style} · {voice} · {layout} · {subs} · 원본 소리 {ss.original_gain:.2f}/{ss.duck_gain:.2f}"


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
                out_dir=proj.dir / "results",
                subtitle_style=ss.subtitle_style, highlight_color=ss.highlight_color,
            )
            proj.results.insert(0, {
                "id": project.new_id(), "short_id": short["id"], "title": title,
                "video": str(video), "report": report, "created": project.now(),
                "duration": round(media.probe(video).duration, 1), "summary": settings_summary(),
            })
            project.save(proj)
            status.update(label="완성!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="만들기 실패", state="error")
            st.exception(e)

# ───────────────────────── 결과 기록 ─────────────────────────
def delete_result(p: project.Project, result_id: str) -> None:
    for r in [r for r in p.results if r["id"] == result_id]:
        Path(r["video"]).unlink(missing_ok=True)
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
                st.video(result["video"])
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
                with st.expander("구간별 나레이션 길이"):
                    st.dataframe(result["report"], hide_index=True, width="stretch")
