"""AI 쇼츠 나레이션 스튜디오 — 영상을 이해해 하이라이트를 고르고, AI 나레이션과 한글 자막을 입혀 쇼츠로 만든다."""
import copy
import hashlib
import time
import uuid
from pathlib import Path

# pyarrow는 맨 먼저 불러온다. Whisper·Chatterbox(torch/CUDA) 등을 올린 뒤 처음으로 pyarrow DLL을 불러오면
# Windows에서 서버 프로세스가 access violation으로 죽는 일이 있었다 (결과 표 st.dataframe에서 처음 불러올 때).
import pyarrow  # noqa: F401
# OpenCV(cv2)도 같은 이유로 먼저 불러온다. 큐 스레드가 쇼츠 썸네일 점수를 매기며 처음 불러오다
# access violation으로 서버가 죽은 적이 있다 (2026-10-31, torch를 올린 뒤).
import cv2  # noqa: F401
import streamlit as st

import shorts_maker
from shorts_maker import (config, generate, jobqueue, llm, media, music, pipeline, presets, preview, project, subtitles,
                          fonts, storage, system, tasks, tts, voices)

st.set_page_config(page_title="AI 쇼츠 나레이션 스튜디오", page_icon="🎬", layout="wide")

ENGINE_LABELS = {"edge": "Edge TTS (기본 목소리, 인터넷 필요)", "chatterbox": "Chatterbox (GPU, 목소리 샘플 사용 가능)"}
LAYOUT_LABELS = {
    "blur": "블러 배경 (가로 영상·글자 많은 화면 추천)",
    "crop": "가운데 크롭 (화면 꽉 차게, 양옆 잘림)",
    "smart": "피사체 따라 크롭 (사람·움직임을 따라감)",
}
VOICE_LABELS = {voice_id: label for label, voice_id in tts.EDGE_VOICES.items()}
SUBTITLE_LABELS = {"plain": "기본", "karaoke": "단어 강조 (읽는 낱말 색칠)"}
TITLE_LABELS = {"bold": "크고 굵게 (핵심 낱말 강조)", "box": "반투명 상자", "band": "위쪽 띠"}
OUTPUT_LABELS = {"1080x1920": "1080×1920 (9:16, 쇼츠·릴스 표준)", "480x640": "480×640 (3:4, 작게)"}
VIDEO_TYPES = ["mp4", "mov", "webm", "mkv", "m4v", "avi"]
UI_DIR = Path(__file__).parent / "ui"


@st.cache_data(ttl=60, show_spinner=False)
def get_specs() -> system.Specs:
    return system.specs()


@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models() -> list[str]:
    return llm.list_models()


media.ensure_ffmpeg()
jobqueue.ensure_runner()  # 모든 접속자가 같은 작업 큐를 쓴다 (한 번에 한 작업)
system.ensure_sampler()  # CPU·RAM·GPU 사용량을 1초마다 잰다

# 브라우저(탭)마다 다른 ID: 큐에 넣은 작업이 누구 것인지 구분한다. 주소에 넣어 두어 새로고침해도 유지된다.
if not st.query_params.get("u"):
    st.query_params["u"] = uuid.uuid4().hex[:8]
ME = st.query_params["u"]

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
if ss.get("pending_llm") in models:  # 팝업에서 방금 설치한 모델 (위젯을 그리기 전에 옮김)
    ss.llm_model = ss.pop("pending_llm")
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
    tasks.unload_models()
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
            skipped.append(f"{rec['choice']} (설치 안 됨: 「⬇️ {rec['choice']} 설치」 버튼으로 받을 수 있음)")
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


def add_font_upload(key: str) -> None:
    """올린 글꼴을 보관함에 넣고 고른다. 업로드 칸은 키를 바꿔 비운다 (남은 파일 때문에 다음 업로드가 꼬이지 않게)."""
    file = ss.get(key)
    ss.font_rev = ss.get("font_rev", 0) + 1
    if file is None:
        return
    try:
        ss.subtitle_font = fonts.add_upload(file.name, file.getvalue())  # 콜백 안이라 위젯 값을 바꿀 수 있음
        ss.preset_msg = ("success", f"'{file.name}' 글꼴을 넣고 골랐습니다.")
    except ValueError as e:
        ss.preset_msg = ("error", str(e))


def delete_font() -> None:
    fonts.delete(ss.subtitle_font)
    ss.subtitle_font = ""
    ss.preset_msg = ("success", "글꼴을 보관함에서 지웠습니다. 자동 글꼴로 돌아갑니다.")


# ───────────────────────── 공용 작업 큐에 넣기 ─────────────────────────
def task_settings() -> dict:
    """작업을 넣을 때의 설정 (작업은 나중에 돌아도 넣은 때의 설정으로 만든다)."""
    return {**current_settings(), "llm_model": ss.llm_model if models else None,
            "devices": {t: dev(t) for t in system.TASKS}}


def clock(t) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) if isinstance(t, (int, float)) else str(t)


def minutes(sec: float) -> str:
    return f"약 {sec / 60:.0f}분" if sec >= 90 else f"약 {max(5, round(sec / 5) * 5):.0f}초"


def enqueue(kind: str, label: str, params: dict, p: project.Project | None = None, key: str = "",
            card_uid: str = "", notify: bool = True) -> str:
    """작업을 공용 큐에 넣고 항목 ID를 돌려준다. 같은 작업이 이미 대기·진행 중이면 그것을 돌려준다."""
    item_id, added = jobqueue.add(kind, label, {**params, "settings": task_settings()}, owner=ME,
                                  project=p.name if p else "", project_dir=str(p.dir) if p else "", key=key,
                                  card_uid=card_uid)
    if notify:
        queue = jobqueue.items()
        pos = jobqueue.position(queue, item_id)
        wait = jobqueue.waits(queue).get(item_id, 0)
        where = f"대기 {pos}번째, {minutes(wait)} 뒤 시작 예상" if pos > 1 or wait > 5 else "바로 시작합니다"
        ss.flash = f"{'큐에 넣었습니다' if added else '이미 큐에 있습니다'}: {jobqueue.KIND_LABELS[kind]} ({where})"
    return item_id


def task_line(q: dict, queue: list[dict]) -> str:
    """큐 항목 한 줄 상태: 진행 중이면 진행 내용, 대기면 순서와 예상 시작."""
    if q["status"] == jobqueue.RUNNING:
        prog = jobqueue.progress(q["id"])
        spent = time.time() - q.get("started", time.time())
        return f"🔄 {prog.get('note', '진행 중')} · {spent:.0f}초 지남"
    if q["status"] == jobqueue.WAITING:
        return f"⏳ 대기 {jobqueue.position(queue, q['id'])}번째 · {minutes(jobqueue.waits(queue).get(q['id'], 0))} 뒤 시작 예상"
    return jobqueue.STATUS_LABELS[q["status"]]


def task_progress(q: dict) -> None:
    """진행 중인 항목의 단계 목록과 진행 막대."""
    prog = jobqueue.progress(q["id"])
    for line in prog.get("lines", [])[-6:]:
        st.caption(line)
    if prog.get("total"):
        st.progress(min(prog["step"] / prog["total"], 1.0), text=prog.get("note", ""))


def watch_in_dialog(state_key: str, dialog: str, on_done) -> None:
    """팝업 안에서 큐에 넣은 작업(목소리 미리 듣기, AI 음악)을 지켜본다. 끝나면 on_done(결과)을 부르고 같은 팝업을 다시 연다."""
    item_id = ss.get(state_key)
    if not item_id:
        return

    @st.fragment(run_every=1)
    def watch() -> None:
        queue = jobqueue.items()
        q = next((x for x in queue if x["id"] == item_id), None)
        if q is None or q["status"] in (jobqueue.CANCELED, jobqueue.FAILED):
            ss.pop(state_key, None)
            if q is not None:
                st.error(f"{jobqueue.KIND_LABELS[q['kind']]} 실패: {q['error'].splitlines()[0] if q['error'] else '취소됨'}")
            return
        if q["status"] == jobqueue.DONE:
            ss.pop(state_key, None)
            on_done(q["result"] or {})
            ss.reopen_dialog = dialog
            st.rerun()
        st.info(f"{jobqueue.KIND_LABELS[q['kind']]} · {task_line(q, queue)}")
    watch()




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


def llm_missing() -> str | None:
    """지금 장치의 추천 LLM이 Ollama에 없으면 그 이름 (Ollama에 연결되지 않으면 None)."""
    choice = recommendations()["llm"]["choice"]
    return choice if models and choice and choice not in models else None


def install_llm_button(name: str, dialog: str) -> None:
    """설치되지 않은 Ollama 모델을 내려받는 버튼. 다 받으면 그 모델을 고르고 같은 팝업을 다시 연다."""
    if not st.button(f"⬇️ {name} 설치", key=f"install_llm_{dialog}", width="stretch",
                     help="Ollama에서 모델을 내려받습니다 (`ollama pull`과 같음). 몇 GB라 몇 분 걸릴 수 있습니다."):
        return
    bar = st.progress(0.0, text=f"{name} 내려받는 중...")
    try:
        for status, done, total in llm.pull_model(name):
            size = f" {done / 1024 ** 3:.1f} / {total / 1024 ** 3:.1f} GB" if total else ""
            bar.progress(min(done / total, 1.0) if total else 0.0, text=f"{name}: {status}{size}")
    except Exception as e:  # 이름 오류, 인터넷 끊김 등
        st.error(f"{name}을(를) 내려받지 못했습니다: {e}")
        return
    get_ollama_models.clear()
    ss.pending_llm = name
    ss.reopen_dialog = dialog
    ss.preset_msg = ("success", f"{name}을(를) 설치하고 LLM으로 골랐습니다.")
    st.rerun()


def device_summary() -> str:
    if ss.device == "split":
        gpu_tasks = [system.TASKS[t].split(" (")[0] for t in system.TASKS if dev(t) != "cpu"]
        return f"작업별: {', '.join(gpu_tasks)}만 GPU" if gpu_tasks else "작업별: 모두 CPU"
    return system.DEVICES[ss.device].split(" (")[0]


def usage_now() -> dict | None:
    history = system.usage_history()
    return history[-1] if history else None


def usage_summary() -> str:
    u = usage_now()
    if u is None:
        return "사용량을 재는 중..."
    text = f"CPU {u['cpu']:.0f}% · RAM {u['ram_used']:.0f}/{u['ram_total']:.0f}GB"
    if u["gpu"] is not None:
        text += f" · GPU {u['gpu']:.0f}% · VRAM {u['vram_used']:.1f}/{u['vram_total']:.0f}GB"
    return text


def resource_monitor() -> None:
    """실시간 자원 사용량: 지금 값, 최근 2분 그래프, 프로그램별 CPU·RAM (1초마다 새로 고침)."""
    u = usage_now()
    if u is None:
        st.caption("사용량을 재는 중... (psutil이 없으면 표시되지 않습니다)")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPU 사용", f"{u['cpu']:.0f}%", help=f"{specs.cpu} · 코어 {specs.cores}개")
    c2.metric("RAM 사용", f"{u['ram_used']:.1f} GB", f"전체 {u['ram_total']:.0f} GB 중 {u['ram_used'] / u['ram_total']:.0%}",
              delta_color="off")
    if u["gpu"] is not None:
        c3.metric("GPU 사용", f"{u['gpu']:.0f}%", f"{u['temp']:.0f}°C · {u['power']:.0f}W", delta_color="off")
        c4.metric("GPU 메모리 (VRAM)", f"{u['vram_used']:.1f} GB",
                  f"전체 {u['vram_total']:.0f} GB 중 {u['vram_used'] / u['vram_total']:.0%}", delta_color="off")
    else:
        c3.metric("GPU", "없음")
    history = system.usage_history()
    now = history[-1]["t"]
    chart = {"초 전": [round(h["t"] - now) for h in history], "CPU %": [h["cpu"] for h in history],
             "RAM %": [100 * h["ram_used"] / h["ram_total"] for h in history]}
    if u["gpu"] is not None:
        chart["GPU %"] = [h["gpu"] for h in history]
        chart["VRAM %"] = [100 * h["vram_used"] / h["vram_total"] for h in history]
    st.line_chart(chart, x="초 전", y=[k for k in chart if k != "초 전"], height=180, y_label="%", x_label="초 전 (최근 2분)")
    rows = system.usage_processes()
    if rows:
        st.table(rows, hide_index=True)
    running = next((q for q in jobqueue.items() if q["status"] == jobqueue.RUNNING), None)
    st.caption(f"측정 {clock(u['t'])} · "
               + ("지금 큐에서 도는 작업: " + f"{jobqueue.KIND_LABELS[running['kind']]} · {running['label']}" if running
                  else "지금 큐에서 도는 작업 없음")
               + " · 프로그램별 GPU 메모리는 Windows에서 알 수 없어 GPU는 전체 사용량만 보여 줍니다.")


def computer_settings() -> None:
    """💻 컴퓨터·AI 모델: 사양, 실시간 사용량, 계산 장치(작업별), 추천 모델, AI 이미지·동영상 모델."""
    show_message()
    gpu_text = ", ".join(f"{g.name} {g.vram_gb:.0f}GB" for g in specs.gpus) or "NVIDIA GPU 없음"
    st.caption(f"{specs.os} · CPU {specs.cpu} ({specs.cores}코어) · RAM {specs.ram_gb:.0f}GB · GPU {gpu_text}"
               + (f" (드라이버 {specs.gpus[0].driver})" if specs.gpus else "") + f" · 작업 폴더 디스크 여유 {specs.disk_free_gb:,}GB")
    st.markdown("**📈 실시간 사용량** (1초마다)")
    st.fragment(resource_monitor, run_every=1)()

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
                   "상태": ("✅ 추천대로" if is_recommended(t, recs[t]) else
                           "⬇️ 추천 모델 설치 안 됨" if t == "llm" and llm_missing() else "↔ 추천과 다름") + " · "
                           + system.fits(recs[t]["need"], specs, dev(t))}
                  for t, label in system.TASKS.items()], hide_index=True)
        if missing := llm_missing():
            st.info(f"추천 LLM **{missing}**이(가) 아직 설치되어 있지 않습니다.")
            install_llm_button(missing, "💻 컴퓨터·AI 모델")
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
        if missing := llm_missing():
            install_llm_button(missing, "🧠 대본·구성")
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
        if st.button("▶ 미리 듣기", width="stretch", disabled=bool(ss.get("voice_task")),
                     help="고른 목소리로 예시 문장을 읽어 봅니다. 공용 작업 큐에 들어가 차례가 되면 만듭니다 (몇 초)."):
            ss.voice_task = enqueue("voice", f"목소리 미리 듣기 · {ss.voice_id or '기본 목소리'}",
                                    {"voice_id": ss.voice_id, "exaggeration": ss.exaggeration}, notify=False)
        watch_in_dialog("voice_task", "🎙️ 나레이션 목소리", lambda r: ss.__setitem__("voice_preview", r.get("path", "")))
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
        if st.button("음악 만들기", disabled=not gen_ok or bool(ss.get("music_task")), width="stretch",
                     help=gen_msg or "MusicGen으로 배경음악을 만듭니다. 공용 작업 큐에 들어가 차례가 되면 만듭니다."):
            desc = ss.get("ai_music_desc", "").strip() or "calm ambient background music"
            ss.music_task = enqueue("music", f"AI 배경음악 · {desc[:30]}",
                                    {"desc": desc, "seconds": float(ss.get("ai_music_sec", 20))}, notify=False)

        def music_done(result: dict) -> None:
            ss.pending_bgm = result["track"]  # 다시 연 팝업이 위젯을 그리기 전에 고른다
            ss.preset_msg = ("success", f"AI 음악을 만들어 보관함에 넣었습니다: {result.get('prompt', '')}")
        watch_in_dialog("music_task", "🔊 소리·배경음악", music_done)
    with st.expander("음악 올리기"):
        st.file_uploader("음악 파일", type=music.MUSIC_TYPES, key="bgm_upload", label_visibility="collapsed")
        st.button("보관함에 넣기", on_click=add_music_upload, disabled=ss.get("bgm_upload") is None, width="stretch",
                  help="올린 음악은 workdir/music/에 저장되어 다른 작업에서도 고를 수 있습니다. 권리가 있는 음악만 쓰세요.")
    close_button()


def screen_settings() -> None:
    """🎬 화면·자막: 세로 변환, 제목, 자막 스타일·크기·위치·색."""
    show_message()
    col_set, col_view = st.columns([3, 2])
    with col_set:
        st.radio("출력 크기", presets.CHOICES["output_size"], format_func=OUTPUT_LABELS.get, key="output_size",
                 horizontal=True, help="둘 다 세로 화면입니다. 자막·제목은 같은 비율로 줄어듭니다. "
                                       "480×640은 파일이 작고 만들기가 빠르지만 쇼츠·릴스 화면(9:16)보다 위아래가 짧습니다.")
        st.radio("세로 변환 방식", presets.CHOICES["layout"], format_func=LAYOUT_LABELS.get, key="layout")
        st.checkbox("위쪽에 제목 표시", key="show_title")
        st.radio("제목 모양", presets.CHOICES["title_style"], format_func=TITLE_LABELS.get, key="title_style",
                 horizontal=True, disabled=not ss.show_title)
        st.radio("자막 스타일", presets.CHOICES["subtitle_style"], format_func=SUBTITLE_LABELS.get, key="subtitle_style",
                 help="단어 강조: 나레이션이 읽는 낱말을 다른 색으로 조금 크게 보여 줍니다.", horizontal=True)
        font_list = {f["id"]: f for f in fonts.list_fonts()}
        if ss.subtitle_font not in font_list:  # 다른 컴퓨터의 프리셋 등으로 이 컴퓨터에 없는 글꼴
            ss.subtitle_font = ""
        col_font, col_font_del = st.columns([4, 1], vertical_alignment="bottom")
        col_font.selectbox("글꼴 (자막·제목·썸네일)", [""] + list(font_list), key="subtitle_font",
                           format_func=lambda i: "자동 (맑은 고딕 등)" if not i else
                           f"{'📁 ' if font_list[i]['source'] == 'upload' else ''}{font_list[i]['name']}")
        col_font_del.button("🗑", key="font_delete", on_click=delete_font, width="stretch",
                            disabled=not ss.subtitle_font.startswith("up_"), help="올린 글꼴을 보관함에서 지우기")
        upload_key = f"font_upload_{ss.get('font_rev', 0)}"
        st.file_uploader("글꼴 올리기 (TTF·OTF, 올리면 바로 보관함에 넣고 고름)", type=fonts.FONT_TYPES, key=upload_key,
                         on_change=add_font_upload, args=(upload_key,),
                         help="한글이 들어 있는 글꼴만 넣을 수 있습니다. 영상에 써도 되는 라이선스인지 확인하세요 "
                              "(눈누·네이버 나눔글꼴 등 무료 글꼴도 사용 범위를 확인).")
        setting_slider("자막 크기", "font_size")
        setting_slider("자막 위치 (아래에서 %)", "margin_percent",
                       help=f"쇼츠 앱은 아래쪽 약 {subtitles.SAFE_BOTTOM:.0%}를 제목·채널·설명으로 가립니다. "
                            "미리보기의 빨간 부분을 피하세요.")
        col_c1, col_c2 = st.columns(2)
        col_c1.color_picker("자막 색", key="color")
        col_c2.color_picker("강조 색", key="highlight_color", help="제목의 핵심 낱말(굵은 제목)과 단어 강조 자막에 씁니다.")
    with col_view:
        st.checkbox("앱 UI가 가리는 곳 보기", key="show_safe_guide", value=True,
                    help="쇼츠 앱의 상태 표시줄·제목·버튼이 덮는 대략적인 곳 (공식 수치가 없어 참고용)")
        with st.spinner("미리보기를 그리는 중… (가변 글꼴은 처음 한 번 굵게 바꾸느라 10초쯤 걸림)"):
            image = style_preview()
        st.image(image, caption="미리보기 (실제 쇼츠와 같은 방법으로 그림)", width="stretch")
        if ss.margin_percent < subtitles.SAFE_BOTTOM * 100:
            st.caption(f"⚠️ 자막이 아래쪽 앱 UI({subtitles.SAFE_BOTTOM:.0%})에 가려질 수 있습니다.")
    close_button()


@st.cache_data(show_spinner=False, max_entries=8)
def preview_background(video: str, t: float, layout: str, size: tuple[int, int]) -> bytes:
    return media.layout_frame(Path(video), t, "crop" if layout == "smart" else layout, size)


def style_preview() -> bytes:
    """화면·자막 설정 미리보기. 열어 둔 작업이 있으면 첫 원본 구간의 장면과 그 나레이션을 쓴다."""
    background, title, text = None, preview.SAMPLE_TITLE, preview.SAMPLE_TEXT
    p = ss.get("project")
    if p is not None and p.shorts:
        short = p.shorts[0]
        title = short["title"] or title
        clip = next((c for c in short["clips"] if c.get("kind") != "ai" and c["end"] > c["start"]), None)
        if clip and p.job is not None:
            background = preview_background(str(p.job.video), (clip["start"] + clip["end"]) / 2, ss.layout,
                                            config.output_size(ss.output_size))
        text = next((c["narration"] for c in short["clips"] if c.get("narration")), text)
    return preview.render(background, title, text, ss.font_size, ss.margin_percent, ss.color, ss.subtitle_style,
                          ss.highlight_color, ss.title_style, ss.show_title, ss.get("show_safe_guide", True),
                          font=ss.subtitle_font, size=config.output_size(ss.output_size))


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
     lambda: ("" if ss.output_size == presets.DEFAULTS["output_size"] else f"{ss.output_size.replace('x', '×')} · ")
             + f"{LAYOUT_LABELS[ss.layout].split(' (')[0]} · "
             + (f"제목 {TITLE_LABELS[ss.title_style].split(' (')[0]} · " if ss.show_title else "제목 없음 · ")
             + f"{SUBTITLE_LABELS[ss.subtitle_style].split(' (')[0]} 자막 · {ss.font_size}"
             + (f" · {next((f['name'] for f in fonts.list_fonts() if f['id'] == ss.subtitle_font), '')}" if ss.subtitle_font else "")),
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
    open_dialog = ss.pop("reopen_dialog", None)  # 팝업 안 작업(모델 설치 등) 뒤 같은 팝업을 다시 엶
    if not open_dialog:
        show_message()  # 다시 여는 팝업이 있으면 알림은 팝업 안에서 보여 줌

    st.subheader("🛠️ 세부 설정")
    st.caption("누르면 팝업 창에서 바꿀 수 있습니다.")
    for label, _, _, summary in SETTING_GROUPS:
        if st.button(label, key=f"open_{label}", width="stretch"):
            open_dialog = label
        st.caption(summary())
    if open_dialog:
        DIALOGS[open_dialog]()

    st.subheader("📈 이 컴퓨터 지금")

    @st.fragment(run_every=2)
    def sidebar_usage() -> None:
        st.caption(usage_summary())
        queue = jobqueue.items()
        running = sum(q["status"] == jobqueue.RUNNING for q in queue)
        waiting = sum(q["status"] == jobqueue.WAITING for q in queue)
        st.caption(f"작업 큐: 진행 {running} · 대기 {waiting}" + (" · ⏸ 일시정지" if jobqueue.is_paused() else ""))
    sidebar_usage()

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
    "**한국어 나레이션**과 **한글 자막**을 입힌 세로 쇼츠/릴스(기본 1080×1920)를 만들어 줍니다. "
    "영상이 없으면 **주제만으로 AI가 대본과 장면을 만들어** 쇼츠를 만듭니다."
)

# ───────────────────────── 공용 작업 큐 (모든 접속자가 같은 큐) ─────────────────────────
def queue_rows(queue: list[dict]) -> None:
    """큐 목록. 내 작업만 빼기·순서 바꾸기·취소·다시 시도를 할 수 있다 (다른 사람 작업은 보기만)."""
    waiting = [q for q in queue if q["status"] == jobqueue.WAITING]
    running = [q for q in queue if q["status"] == jobqueue.RUNNING]
    finished = [q for q in queue if q["status"] in (jobqueue.DONE, jobqueue.FAILED, jobqueue.CANCELED)
                and q.get("owner") == ME][::-1][:8]
    col_info, col_pause, col_clear_done, col_clear_wait = st.columns([3, 1, 1, 1])
    paused = jobqueue.is_paused()
    mine = sum(q.get("owner") == ME for q in waiting + running)
    col_info.caption(f"진행 {len(running)} · 대기 {len(waiting)} (내 작업 {mine}) · 넣은 순서대로 한 번에 하나씩 처리합니다."
                     + (" · ⏸ **일시정지됨**" if paused else ""))
    col_pause.button("▶ 이어서" if paused else "⏸ 일시정지", key="q_pause", width="stretch",
                     on_click=jobqueue.set_paused, args=(not paused,),
                     help="모든 사람의 큐를 멈춥니다. 지금 하는 작업까지만 끝내고 멈춥니다.")
    col_clear_done.button("내 완료 비우기", key="q_clear_done", width="stretch", disabled=not finished,
                          on_click=jobqueue.clear, args=((jobqueue.DONE, jobqueue.FAILED, jobqueue.CANCELED), ME))
    col_clear_wait.button("내 대기 전부 빼기", key="q_clear_wait", width="stretch",
                          disabled=not any(q.get("owner") == ME for q in waiting),
                          on_click=jobqueue.clear, args=((jobqueue.WAITING,), ME))
    if not (waiting or running or finished):
        st.caption("큐가 비어 있습니다.")
        return
    for q in running + waiting + finished:
        own = q.get("owner") == ME
        with st.container(border=True, key=f"q_{q['id']}"):
            col_state, col_text, col_btns = st.columns([1.3, 4, 1.6])
            if q["status"] == jobqueue.WAITING:
                col_state.markdown(f"⏳ **{waiting.index(q) + 1}번째**")
            else:
                col_state.markdown(jobqueue.STATUS_LABELS[q["status"]])
            col_state.caption("👤 내 작업" if own else "다른 사용자")
            label = q["label"].replace("*", "").strip()  # 설명에 ** 가 있으면 굵게 표시가 깨짐
            col_text.markdown(f"**{jobqueue.KIND_LABELS.get(q['kind'], q['kind'])}** · {label}")
            spent = (f" · {q['finished'] - q['started']:.0f}초 걸림" if q.get("finished") and q.get("started") else "")
            col_text.caption(f"{q['project'] or '-'} · 넣은 시각 {clock(q['created'])}{spent}")
            if q["status"] in (jobqueue.RUNNING, jobqueue.WAITING):
                col_text.caption(task_line(q, queue))
            if q["status"] == jobqueue.RUNNING:
                with col_text:
                    task_progress(q)
            if q.get("error") and own:
                with col_text.expander("실패 이유"):
                    st.code(q["error"][-1200:])
            if not own:
                continue
            b1, b2, b3 = col_btns.columns(3)
            if q["status"] == jobqueue.WAITING:
                pos = waiting.index(q)
                prev_mine = pos > 0 and waiting[pos - 1].get("owner") == ME
                next_mine = pos + 1 < len(waiting) and waiting[pos + 1].get("owner") == ME
                b1.button("⬆", key=f"qup_{q['id']}", help="앞으로 (내 작업끼리만)", disabled=not prev_mine,
                          on_click=jobqueue.move, args=(q["id"], -1))
                b2.button("⬇", key=f"qdown_{q['id']}", help="뒤로 (내 작업끼리만)", disabled=not next_mine,
                          on_click=jobqueue.move, args=(q["id"], 1))
                b3.button("🗑", key=f"qdel_{q['id']}", help="큐에서 빼기", on_click=jobqueue.remove, args=(q["id"],))
            elif q["status"] == jobqueue.RUNNING:
                b3.button("⏹", key=f"qstop_{q['id']}", help="지금 하는 작업 취소", on_click=jobqueue.remove, args=(q["id"],))
            else:
                if q["status"] in (jobqueue.FAILED, jobqueue.CANCELED):
                    b1.button("🔁", key=f"qretry_{q['id']}", help="다시 시도 (큐 맨 뒤로)", on_click=jobqueue.retry,
                              args=(q["id"],))
                b3.button("🗑", key=f"qdel_{q['id']}", help="목록에서 지우기", on_click=jobqueue.remove, args=(q["id"],))


def queue_panel() -> None:
    """화면 위쪽의 공용 큐. 내 작업이나 지금 연 작업의 큐 항목이 끝나면 화면 전체를 새로 그린다."""
    queue = jobqueue.items()
    open_dir = str(Path(ss.project.dir).resolve()) if ss.get("project") is not None else ""
    # 목소리 미리 듣기·AI 음악은 팝업이 직접 지켜보므로 빼다 (화면 전체를 새로 그리면 열린 팝업이 닫힘)
    watched = {q["id"] for q in queue if q["status"] in (jobqueue.DONE, jobqueue.FAILED, jobqueue.CANCELED)
               and q["kind"] not in ("voice", "music")
               and (q.get("owner") == ME or (q.get("project_dir") and str(Path(q["project_dir"]).resolve()) == open_dir))}
    if ss.get("queue_finished") is not None and watched - ss.queue_finished:
        ss.queue_finished = watched
        st.rerun(scope="app")  # 결과·카드·미리보기를 새로 그림
    ss.queue_finished = watched
    running = next((q for q in queue if q["status"] == jobqueue.RUNNING), None)
    waiting = [q for q in queue if q["status"] == jobqueue.WAITING]
    mine_active = [q for q in queue if q["status"] in (jobqueue.WAITING, jobqueue.RUNNING) and q.get("owner") == ME]
    head = (f"🚦 작업 큐 · 🔄 {jobqueue.KIND_LABELS[running['kind']]} ({running['project'] or running['label']})"
            if running else "🚦 작업 큐 · 쉬는 중") + f" · 대기 {len(waiting)}" + (f" · 내 작업 {len(mine_active)}" if mine_active else "")
    with st.expander(head, expanded=bool(mine_active)):
        queue_rows(queue)


MODE_LABELS = {"upload": "📼 영상으로 만들기", "ai": "✨ 영상 없이 AI로 만들기"}
ss.setdefault("mode", "upload")

# 큐가 돌 때는 2초마다, 쉴 때도 5초마다 확인한다 (다른 사람이 넣은 작업도 보이게)
queue_active = any(q["status"] in (jobqueue.WAITING, jobqueue.RUNNING) for q in jobqueue.items())
st.fragment(queue_panel, run_every=2 if queue_active else 5)()


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
        opened = project.load(folder)
        set_project(opened, f"'{opened.name}' 작업을 불러왔습니다.")
    except (OSError, ValueError, KeyError) as e:
        ss.flash = f"작업을 불러오지 못했습니다: {e}"
    # 업로드 칸에 남아 있는 파일이 방금 연 작업을 다시 덮어쓰지 않게, 그 파일은 이미 처리한 것으로 둔다
    file = ss.get("video_file")
    ss.upload_id = hashlib.sha1(file.getvalue()).hexdigest()[:12] if file is not None else None


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


@st.cache_data(show_spinner="디스크 사용량을 재는 중…", ttl=30)
def storage_usage(rev: int) -> list[dict]:
    return storage.usage()


@st.dialog("🧹 저장 공간", width="large")
def storage_dialog() -> None:
    """workdir에서 무엇이 얼마나 차지하는지, 다시 만들 수 있는 캐시 지우기."""
    show_message()
    free, total = storage.disk_free()
    st.progress(1 - free / total, text=f"드라이브 남은 공간 {storage.human(free)} / {storage.human(total)}")
    busy = any(q["status"] in (jobqueue.WAITING, jobqueue.RUNNING) for q in jobqueue.items())
    rows = storage_usage(ss.get("storage_rev", 0))
    for row in rows:
        col_name, col_size, col_btn = st.columns([5, 1.4, 1.2], vertical_alignment="center")
        col_name.markdown(f"**{row['label']}**")
        col_name.caption(row["note"])
        col_size.markdown(f"{storage.human(row['bytes'])}  \n<small>파일 {row['files']:,}개</small>", unsafe_allow_html=True)
        if row["clear"] and col_btn.button("지우기", key=f"clear_{row['key']}", disabled=busy or not row["bytes"],
                                           width="stretch", help="작업 큐가 쉴 때만 지울 수 있습니다." if busy else None):
            try:
                freed = storage.clear(row["key"])
                ss.preset_msg = ("success", f"{row['label']} {storage.human(freed)}를 지웠습니다.")
            except (RuntimeError, ValueError) as e:
                ss.preset_msg = ("error", str(e))
            ss.storage_rev = ss.get("storage_rev", 0) + 1
            st.rerun(scope="fragment")
    if busy:
        st.caption("작업 큐가 도는 동안에는 캐시를 지울 수 없습니다 (만드는 중인 장면이 지워지지 않게).")
    close_button()


def import_project() -> None:
    """「📦 작업 가져오기」에 올린 zip을 새 작업으로 풀고 연다 (같은 파일은 한 번만)."""
    file = ss.get("import_file")
    if file is None or ss.get("imported_file") == file.file_id:
        return
    ss.imported_file = file.file_id
    tmp = config.WORK_DIR / "exports" / f"import_{uuid.uuid4().hex[:8]}.zip"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_bytes(file.getvalue())
        opened = project.import_zip(tmp)
        set_project(opened, f"'{opened.name}' 작업을 가져왔습니다.")
    except (ValueError, OSError, KeyError) as e:
        ss.flash = f"가져오지 못했습니다: {e}"
    except Exception as e:  # zip이 깨졌을 때 등
        ss.flash = f"가져오지 못했습니다 (zip 파일을 확인하세요): {e}"
    finally:
        tmp.unlink(missing_ok=True)


recent_items = project.recent(limit=50)
with st.expander(f"📂 최근 작업 ({len(recent_items)})", expanded=False):
    if not recent_items:
        st.caption("아직 작업이 없습니다.")
    # 많을 때만 높이를 정해 스크롤 (적을 때 높이를 정하면 아래가 비어 보임)
    with st.container(height=420 if len(recent_items) > 6 else "content", border=False):
        for item in recent_items:
            col_name, col_info, col_open, col_del = st.columns([4, 3, 1, 0.6], vertical_alignment="center")
            icon = "✨" if item["kind"] == "ai" else "📼"
            col_name.markdown(f"{icon} **{item['name']}**")
            status = "분석 완료" if item["analyzed"] else "분석 전"
            col_info.caption(f"{item['updated']} · {status} · 만든 쇼츠 {item['results']}개")
            col_open.button("열기", key=f"open_{item['dir'].name}", on_click=open_recent, args=(item["dir"],))
            if col_del.button("🗑", key=f"delrecent_{item['dir'].name}", help="이 작업 지우기 (확인 후)"):
                confirm_delete_recent(item["dir"], item["name"], item["results"])
    st.file_uploader("📦 작업 가져오기 (내보낸 zip)", type=["zip"], key="import_file", on_change=import_project,
                     help="다른 PC나 이전에 「📦」로 내보낸 작업을 새 작업으로 가져옵니다.")
    if st.button("🧹 저장 공간 보기", help="작업·캐시·모델이 디스크를 얼마나 쓰는지 보고, 다시 만들 수 있는 캐시를 지웁니다."):
        ss.storage_rev = ss.get("storage_rev", 0) + 1  # 열 때마다 새로 잼
        storage_dialog()

st.radio("만드는 방식", list(MODE_LABELS), format_func=MODE_LABELS.get, horizontal=True, key="mode",
         on_change=close_project, label_visibility="collapsed")

uploaded = None
if ss.mode == "upload":
    # 드래그 앤 드롭: 업로드 칸을 크게 보여 주고, 화면 어디에 놓아도 업로드되게 한다 (ui/dropzone.*).
    # 스크립트는 iframe(지원 중단된 components.html) 없이 본 화면에서 바로 실행한다.
    st.html(f"<style>{(UI_DIR / 'dropzone.css').read_text(encoding='utf-8')}</style>"
            f"<script>{(UI_DIR / 'dropzone.js').read_text(encoding='utf-8')}</script>", unsafe_allow_javascript=True)
    # 최근 작업에서 연 작업이 있으면 업로드 칸은 접어 둔다 (방금 올린 파일이 있으면 그대로 보여 줌)
    opened = ss.get("project")
    folded = opened is not None and opened.kind == "upload" and ss.get("video_file") is None
    with st.expander("📼 다른 영상 올리기", expanded=False) if folded else st.container():
        with st.container(key="video_drop"):
            uploaded = st.file_uploader("영상 업로드", type=VIDEO_TYPES, key="video_file")

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
if proj is not None and project.changed(proj):  # 작업 큐나 다른 브라우저가 저장했으면 최신 상태로
    try:
        proj = ss.project = project.load(proj.dir)
        refresh_cards()
    except (OSError, ValueError, KeyError):  # 다른 사람이 지운 작업
        proj = None
        ss.pop("project", None)
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
        new_project = project.new_ai(ss.ai_topic)
        enqueue("plan", f"AI 대본 · {new_project.name}", {"engine": ss.ai_engine}, new_project, key=f"plan:{new_project.dir.name}")
        set_project(new_project, ss.pop("flash", ""))
        st.rerun()
    st.stop()


@st.cache_data(show_spinner=False)
def probe(path: str) -> media.VideoInfo:
    return media.probe(Path(path))


@st.dialog("📦 작업 내보내기")
def export_dialog(p: project.Project) -> None:
    """작업 하나(올린 영상, 분석, 구성, AI 장면, 결과)를 zip으로 내려받는다. 다른 PC에서 「작업 가져오기」로 이어서 작업."""
    st.markdown(f"**{p.name}**")
    with_results = st.checkbox(f"만든 쇼츠 영상도 담기 ({len(p.results)}개)", value=True, key="export_results",
                               disabled=not p.results)
    st.caption("올린 영상, 분석 결과(대사·장면 설명), 카드 구성, AI 장면 그림이 들어갑니다. "
               "설정(목소리·자막 등)은 프리셋으로 따로 내보내세요.")
    made = ss.get("export_file")
    if st.button("zip 만들기", type="primary", width="stretch"):
        folder = config.WORK_DIR / "exports"
        for old in folder.glob(f"{p.id}_*.zip"):  # 이 작업의 예전 내보내기는 지운다
            old.unlink(missing_ok=True)
        out = folder / f"{p.id}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        bar = st.progress(0.0, text="담는 중…")
        project.export_zip(p, out, with_results and bool(p.results),
                           on_progress=lambda d, n: bar.progress(d / n, text=f"담는 중… {d} / {n}"))
        made = ss.export_file = {"project": p.id, "path": str(out)}
        bar.empty()
    if made and made["project"] == p.id and Path(made["path"]).exists():
        path = Path(made["path"])
        st.download_button(f"⬇️ 다운로드 ({path.stat().st_size / 1024 ** 2:,.1f}MB)", data=path.read_bytes(),
                           file_name=f"{Path(p.name).stem[:30]}_{path.stem.split('_', 1)[1]}.zip", mime="application/zip",
                           on_click="ignore", width="stretch", key="export_download")
    close_button()


run_analyze = False
if proj.kind == "ai":
    col_topic, col_export, col_close = st.columns([6, 0.6, 1])
    col_topic.markdown(f"**✨ {proj.topic}**")
    if col_export.button("📦", key="export_open", help="이 작업을 zip으로 내보내기 (다른 PC로 옮기거나 보관)"):
        export_dialog(proj)
    col_close.button("닫기", on_click=close_project, help="작업은 저장되어 있어서 「최근 작업」에서 다시 열 수 있습니다.")
else:
    source = proj.source_path
    info = probe(str(source))
    col_video, col_options = st.columns([1, 2])
    with col_video:
        st.video(str(source))
    with col_options:
        head, col_export, close = st.columns([5, 0.6, 1])
        head.markdown(f"**{proj.name}** · {info.duration:.1f}초 · {info.width}×{info.height}" + ("" if info.has_audio else " · 소리 없음"))
        if col_export.button("📦", key="export_open", help="이 작업을 zip으로 내보내기 (다른 PC로 옮기거나 보관)"):
            export_dialog(proj)
        close.button("닫기", on_click=close_project, help="작업은 저장되어 있어서 「최근 작업」에서 다시 열 수 있습니다.")
        total = round(info.duration, 1)
        start, end = st.slider("분석할 구간 (초)", 0.0, total, (0.0, min(total, float(config.MAX_SOURCE_SECONDS))), 0.5)
        if end - start > config.MAX_SOURCE_SECONDS:
            st.warning(f"{config.MAX_SOURCE_SECONDS // 60}분이 넘으면 분석이 오래 걸리고 장면 설명이 듬성듬성해집니다.")
        label = "1단계 · 다시 분석 + 쇼츠 구성" if proj.job else "1단계 · 영상 분석 + 쇼츠 구성"
        busy = bool(jobqueue.for_project(str(proj.dir)))
        run_analyze = st.button(label, type="primary", disabled=llm_model is None or busy,
                                help="공용 작업 큐에 넣습니다. 앞에 작업이 있으면 차례를 기다립니다." if not busy else
                                "이 작업의 큐 항목이 끝난 뒤에 누를 수 있습니다.")
        if dev("llm") == "cpu" and specs.gpus:
            st.caption("⚠️ 장면 이해(LLM)가 CPU로 설정되어 있어 분석이 몇 배 느립니다 (33초 영상에 약 8분). "
                       "「💻 컴퓨터·AI 모델」에서 LLM을 GPU로 두면 빨라집니다.")


def project_tasks_box(p: project.Project) -> None:
    """이 작업의 큐 항목(대기 순서·진행 상황). 끝나면 큐 패널이 화면 전체를 새로 그린다."""
    queue = jobqueue.items()
    mine = [q for q in jobqueue.for_project(str(p.dir))]
    for q in mine:
        with st.container(border=True):
            st.markdown(f"**{jobqueue.KIND_LABELS[q['kind']]}** · {task_line(q, queue)}")
            if q["status"] == jobqueue.RUNNING:
                task_progress(q)
            elif q.get("owner") == ME:
                st.button("큐에서 빼기", key=f"ptask_del_{q['id']}", on_click=jobqueue.remove, args=(q["id"],))


def project_failures(p: project.Project) -> None:
    """이 작업에서 최근 10분 안에 실패한 내 큐 항목."""
    here = Path(p.dir).resolve()
    failed = [q for q in jobqueue.items() if q.get("owner") == ME and q["status"] == jobqueue.FAILED
              and q.get("project_dir") and Path(q["project_dir"]).resolve() == here
              and time.time() - q.get("finished", 0) < 600]
    if failed:
        q = failed[-1]
        st.error(f"{jobqueue.KIND_LABELS[q['kind']]} 실패: {q['error'].splitlines()[0] if q['error'] else ''} "
                 "(위 「🚦 작업 큐」에서 이유를 보고 🔁 다시 시도할 수 있습니다)")


def project_busy(p: project.Project, kinds: tuple[str, ...]) -> bool:
    """이 작업에 구성을 바꾸는 큐 항목이 대기·진행 중인지 (그동안 카드를 고치면 덮어써짐)."""
    return any(q["kind"] in kinds for q in jobqueue.for_project(str(p.dir)))


STEPPER_CSS = """<style>
.sm-steps{display:flex;align-items:center;gap:6px;margin:4px 0 2px;flex-wrap:wrap}
.sm-step{display:flex;align-items:center;gap:6px;padding:4px 12px 4px 6px;border-radius:999px;font-size:14px;
  border:1px solid rgba(250,250,250,.15);color:rgba(250,250,250,.55)}
.sm-step b{display:inline-flex;width:22px;height:22px;border-radius:50%;align-items:center;justify-content:center;
  font-size:12px;background:rgba(250,250,250,.12)}
.sm-step.done{color:rgba(250,250,250,.85)} .sm-step.done b{background:#2e7d4f;color:#fff}
.sm-step.now{color:#fff;border-color:#FF4B6E;background:rgba(255,75,110,.12)} .sm-step.now b{background:#FF4B6E;color:#fff}
.sm-step.running{color:#fff;border-color:#e0a030;background:rgba(224,160,48,.12)} .sm-step.running b{background:#e0a030;color:#111}
.sm-line{flex:0 0 18px;height:2px;background:rgba(250,250,250,.2)}
</style>"""


def stepper(p: project.Project) -> None:
    """지금 어느 단계인지: ① 분석·구성(AI 대본) → ② 편집 → ③ 만들기 → ④ 결과. 큐에서 도는 단계는 ⏳."""
    queued = {q["kind"] for q in jobqueue.for_project(str(p.dir))}
    steps = [
        ["AI 대본·장면 구성" if p.kind == "ai" else "분석·구성",
         "running" if queued & {"analyze", "plan"} else "done" if p.shorts else "todo"],
        ["카드 편집", "done" if p.results else "todo"],
        ["쇼츠 만들기", "running" if "make" in queued else "done" if p.results else "todo"],
        ["결과 받기", "done" if p.results else "todo"],
    ]
    if p.shorts and steps[0][1] == "done" and steps[2][1] == "todo":
        steps[1][1] = "now"  # 구성이 끝났고 만들기 전이면 편집 중
    elif not any(s == "running" for _, s in steps):
        for step in steps:
            if step[1] == "todo":
                step[1] = "now"
                break
    marks = {"done": "✓", "running": "⏳"}
    parts = [f'<div class="sm-step {state}"><b>{marks.get(state, i + 1)}</b>{name}</div>'
             for i, (name, state) in enumerate(steps)]
    st.html(STEPPER_CSS + '<div class="sm-steps">' + '<div class="sm-line"></div>'.join(parts) + "</div>")


stepper(proj)
if jobqueue.for_project(str(proj.dir)):
    st.fragment(project_tasks_box, run_every=1)(proj)
project_failures(proj)


def card_key(uid: str, field: str) -> str:
    return f"card{ss.card_rev}_{uid}_{field}"


def sync_cards(short: dict) -> None:
    """카드 입력칸에 쓴 값을 저장된 구간에 반영한다 (버튼 콜백에서 순서를 바꾸기 전에 호출)."""
    for clip in short["clips"]:
        for field in ("start", "end", "narration", "prompt_ko", "seconds", "engine"):
            key = card_key(clip["uid"], field)
            if key in ss:
                clip[field] = ss[key]


UNDO_LIMIT = 20


def remember(short: dict) -> None:
    """되돌리기용으로 이 쇼츠의 지금 제목·카드를 기억한다 (브라우저 세션마다, 쇼츠마다 최근 UNDO_LIMIT개)."""
    stack = ss.setdefault("undo", {}).setdefault(short["id"], [])
    snap = {"title": short["title"], "clips": copy.deepcopy(short["clips"])}
    if not stack or stack[-1] != snap:
        stack.append(snap)
        del stack[:-UNDO_LIMIT]


def undo(p: project.Project, short: dict) -> None:
    stack = ss.get("undo", {}).get(short["id"])
    if not stack:
        return
    snap = stack.pop()
    short["title"], short["clips"] = snap["title"], snap["clips"]
    project.save(p)
    refresh_cards()
    ss.flash = "↩️ 되돌렸습니다."


def card_action(p: project.Project, short: dict, uid: str, action: str) -> None:
    sync_cards(short)
    remember(short)
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
    remember(short)
    source = [c for c in short["clips"] if c.get("kind") != "ai"]
    start = source[-1]["end"] if source else 0.0
    start = min(start, max(0.0, duration - 5.0))
    short["clips"].append({"uid": project.new_id(), "start": round(start, 1),
                           "end": round(min(duration, start + 5.0), 1), "narration": ""})
    project.save(p)
    refresh_cards()


def add_ai_card(p: project.Project, short: dict) -> None:
    sync_cards(short)
    remember(short)
    short["clips"].append({"uid": project.new_id(), "kind": "ai", "prompt_ko": "", "prompt": "", "engine": "image",
                           "seed": int(project.new_id()[:4], 16), "seconds": 5.0, "narration": ""})
    project.save(p)
    refresh_cards()


def card_scene(clip: dict) -> dict:
    """AI 장면 카드 → 생성 요청 {prompt(영어), engine, seed, seconds}."""
    return {"prompt": clip.get("prompt", ""), "engine": clip.get("engine", "image"),
            "seed": int(clip.get("seed", 0)), "seconds": float(clip.get("seconds", 5.0))}


def scene_key(clip: dict) -> str:
    text = clip.get("prompt_ko", "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"scene:{clip['uid']}:{clip.get('engine', 'image')}:{int(clip.get('seed', 0))}:{float(clip.get('seconds', 5.0))}:{digest}"


def scene_made(clip: dict) -> bool:
    """지금 설명·방식·시드로 이미 만든 장면인지 (영어 프롬프트가 지금 설명을 번역한 것일 때만 확인)."""
    return (clip.get("prompt_src") == clip.get("prompt_ko", "").strip() and bool(clip.get("prompt"))
            and generate.cached_clip(card_scene(clip)) is not None)


def enqueue_cards(p: project.Project, short: dict, uids: set[str], new_seed: bool = False) -> int:
    """AI 장면 카드들을 공용 작업 큐에 넣는다 (영어 프롬프트 번역은 큐에서). 넣은 개수를 돌려준다."""
    sync_cards(short)
    added = 0
    for clip in short["clips"]:
        if clip.get("kind") != "ai" or clip["uid"] not in uids or not clip.get("prompt_ko", "").strip():
            continue
        if new_seed:
            clip["seed"] = int(clip.get("seed", 0)) + 1
            clip.pop("preview", None)
        if scene_made(clip):
            continue
        params = {k: clip.get(k, "") for k in ("prompt_ko", "prompt", "prompt_src", "subject")}
        params.update(engine=clip.get("engine", "image"), seed=int(clip.get("seed", 0)),
                      seconds=float(clip.get("seconds", 5.0)), short_id=short["id"])
        if any(jobqueue.same_task(q, "scene", scene_key(clip)) for q in jobqueue.items()):
            continue
        enqueue("scene", clip["prompt_ko"].replace("*", "").strip()[:40], params, p, key=scene_key(clip),
                card_uid=clip["uid"], notify=False)
        added += 1
    project.save(p)
    refresh_cards()
    return added


def card_queue_item(clip: dict) -> dict | None:
    """이 카드의 지금 설정(설명·방식·시드·길이)으로 넣은 큐 항목 (다른 설정으로 넣었던 것은 무시)."""
    item = jobqueue.status_of_card(clip["uid"])
    return item if item and item.get("key") == scene_key(clip) else None


def ai_card(p: project.Project, short: dict, clip: dict, i: int) -> None:
    """AI 장면 카드: 미리보기·큐에 넣기, 방식·길이, 장면 설명, 나레이션."""
    uid = clip["uid"]
    if scene_made(clip) and not (clip.get("preview") and Path(clip["preview"]).exists()):
        clip["preview"] = str(generate.cached_clip(card_scene(clip)))  # 큐에서 다 만들었으면 미리보기로
        project.save(p)
    with st.container(border=True, key=f"cardbox_{uid}"):
        col_media, col_opts, col_text = st.columns([1.3, 1, 3])
        with col_media:
            preview = Path(clip.get("preview", ""))
            item = card_queue_item(clip)
            if clip.get("preview") and preview.exists():
                st.video(str(preview))
            elif item and item["status"] in (jobqueue.WAITING, jobqueue.RUNNING):
                where = f" ({item['position']}번째)" if item["position"] else ""
                st.info(f"{jobqueue.STATUS_LABELS[item['status']]}{where} · 위 「🚦 작업 큐」")
            elif item and item["status"] == jobqueue.FAILED:
                st.error("❌ 만들기 실패 · 위 「🚦 작업 큐」에서 이유를 볼 수 있습니다")
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
        remember(short)
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


def card_preview(job: pipeline.Job, short: dict, clip: dict) -> bytes:
    """원본 구간 카드가 쇼츠에서 보일 모습 (시작 장면 + 제목 + 나레이션 첫 두 줄, 지금 화면·자막 설정)."""
    uid = clip["uid"]
    start = float(ss.get(card_key(uid, "start"), clip["start"]))
    narration = ss.get(card_key(uid, "narration"), clip["narration"]) or " "
    return scene_preview(str(job.video), round(min(start + 0.3, job.duration), 1), ss.layout, short["title"], narration,
                         ss.font_size, ss.margin_percent, ss.color, ss.subtitle_style, ss.highlight_color,
                         ss.title_style, ss.show_title, font=ss.subtitle_font, size=config.output_size(ss.output_size))


@st.cache_data(show_spinner=False, max_entries=200)
def scene_preview(video: str, t: float, layout: str, title: str, text: str, *style, font: str,
                  size: tuple[int, int]) -> bytes:
    """세로 장면 미리보기 PNG (작은 그림이라 많이 캐시해도 됨. 큰 배경 프레임은 캐시하지 않음).
    style = (font_size, margin_percent, color, subtitle_style, highlight_color, title_style, show_title)."""
    background = media.layout_frame(Path(video), t, "crop" if layout == "smart" else layout, size)
    return preview.render(background, title, text, *style, False, width=240, font=font, size=size)


def clip_cards(p: project.Project, short: dict, job: pipeline.Job | None, vertical: bool = False) -> list[dict]:
    """카드마다: 원본 구간(구간 재생 또는 세로 미리보기, 시작·끝) 또는 AI 장면, 나레이션, 순서 바꾸기·삭제.
    고친 값을 저장하고 카드 목록을 돌려준다."""
    cps = config.NARRATION_CHARS_PER_SECOND
    for i, clip in enumerate(short["clips"]):
        uid = clip["uid"]
        if clip.get("kind") == "ai":
            ai_card(p, short, clip, i)
            continue
        with st.container(border=True, key=f"cardbox_{uid}"):
            col_media, col_time, col_text = st.columns([1.3, 1, 3])
            with col_media:
                if vertical:
                    st.image(card_preview(job, short, clip), width="stretch")
                else:
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
            remember(short)
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


def enqueue_plan(p: project.Project, short: dict | None = None) -> None:
    """구성 다시 만들기를 큐에 넣는다 (short가 없으면 전체: 여러 편이면 주제 나누기부터)."""
    if short is None:
        label = f"{p.name} · 쇼츠 {ss.short_count}편 구성" if p.kind == "upload" else f"AI 대본 · {p.name}"
        params = {"engine": ss.get("ai_engine", "image")} if p.kind == "ai" else {}
        enqueue("plan", label, params, p, key=f"plan:{p.dir.name}")
    else:
        enqueue("plan", f"{p.name} · '{short['title'][:20]}' 다시 구성", {"short_id": short["id"]}, p,
                key=f"plan:{p.dir.name}:{short['id']}")


def enqueue_make(p: project.Project, shorts: list[dict]) -> None:
    label = f"'{shorts[0]['title'][:24]}'" if len(shorts) == 1 else f"쇼츠 {len(shorts)}편"
    enqueue("make", f"{p.name} · {label}", {"short_ids": [x["id"] for x in shorts]}, p,
            key=f"make:{ME}:{p.dir.name}:{','.join(x['id'] for x in shorts)}")  # 사람마다 설정이 달라 따로 만든다


if run_analyze:
    enqueue("analyze", f"{proj.name} · {start:.0f}~{end:.0f}초", {"start": start, "end": end}, proj,
            key=f"analyze:{proj.dir.name}")
    st.rerun()

if project_busy(proj, ("analyze", "plan")):
    st.info("이 작업의 분석·구성이 큐에서 끝나면 2단계 카드가 나옵니다. 그동안 다른 작업을 열거나 창을 닫아도 큐는 계속 돕니다.")
    st.stop()

job = proj.job
if proj.kind == "upload":
    if job is None:
        st.stop()
    with st.expander(f"🔎 AI가 이해한 내용 (장면 {len(job.frames)}개, 대사 {len(job.segments)}개)"):
        st.caption(f"대사 언어: {job.language or '대사 없음'}")
        st.text(pipeline.timeline(job) or "(내용 없음)")

# ───────────────────────── 2단계: 구성 수정 / 3단계: 쇼츠 만들기 ─────────────────────────
@st.cache_data(show_spinner=False, max_entries=16)
def cut_times(job_dir: str, frames: int, _job: pipeline.Job) -> list[float]:
    """원본의 컷 전환 시각 (타임라인 표시용, 작업 폴더·캡처 수로 캐시)."""
    try:
        return pipeline.scene_cuts(_job)
    except Exception:  # 캡처가 지워진 옛 작업
        return []


TIMELINE_CSS = """<style>
.sm-tl{margin:2px 0 10px;font-size:12px;color:rgba(250,250,250,.6)}
.sm-tl .row{display:flex;align-items:center;gap:8px;margin:5px 0}
.sm-tl .lab{flex:0 0 74px;text-align:right}
.sm-tl .track{position:relative;flex:1;height:24px;border-radius:5px;background:rgba(250,250,250,.07);overflow:hidden}
.sm-tl .seq{display:flex;gap:2px;background:none}
.sm-tl .blk{display:flex;align-items:center;justify-content:center;color:#fff;font-weight:600;min-width:14px;
  border-radius:4px;overflow:hidden;white-space:nowrap}
.sm-tl .src{background:#FF4B6E} .sm-tl .ai{background:#8a63d2}
.sm-tl .pos{position:absolute;top:2px;bottom:7px;border-radius:3px;box-shadow:inset -2px 0 0 #0e1117;
  display:flex;align-items:center;justify-content:center;color:#fff;font-weight:600;overflow:hidden}
.sm-tl .voice{position:absolute;bottom:0;height:4px;background:#3fa7d6;opacity:.7}
.sm-tl .range{position:absolute;top:0;bottom:0;background:rgba(250,250,250,.08);border-left:1px dashed rgba(250,250,250,.4);
  border-right:1px dashed rgba(250,250,250,.4)}
.sm-tl .cut{position:absolute;top:0;bottom:0;width:2px;background:#f2c94c}
.sm-tl .legend{margin-left:82px} .sm-tl .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px 0 10px;vertical-align:-1px}
</style>"""


def timeline(short: dict, job: pipeline.Job | None) -> None:
    """카드 순서(쇼츠에 나오는 차례, 길이 비율)와 원본에서 고른 곳(대사·컷 전환과 함께)을 막대로 보여 준다."""
    cards = [c for c in short["clips"] if c.get("kind") == "ai" or c["end"] > c["start"]]
    if not cards:
        return
    length = lambda c: float(c.get("seconds", 5.0)) if c.get("kind") == "ai" else c["end"] - c["start"]
    total = sum(length(c) for c in cards)
    def block(i: int, c: dict) -> str:
        ai = c.get("kind") == "ai"
        what = "AI 장면" if ai else f"원본 {c['start']:.1f}~{c['end']:.1f}초"
        return (f'<div class="blk {"ai" if ai else "src"}" style="flex:{length(c):.2f}" '
                f'title="{i + 1}. {what} · {length(c):.1f}초">{i + 1}</div>')

    seq = "".join(block(i, c) for i, c in enumerate(short["clips"]) if c in cards)
    rows = [f'<div class="row"><span class="lab">쇼츠 순서<br>{total:.0f}초</span><div class="track seq">{seq}</div></div>']
    legend = '<i style="background:#FF4B6E"></i>원본 구간'
    if any(c.get("kind") == "ai" for c in cards):
        legend += '<i style="background:#8a63d2"></i>AI 장면'
    if job is not None and any(c.get("kind") != "ai" for c in cards):
        d = max(job.duration, 0.1)
        pct = lambda t: f"{max(0.0, min(100.0, t / d * 100)):.2f}%"
        marks = []
        if short.get("range"):
            a, b = short["range"]
            marks.append(f'<div class="range" style="left:{pct(a)};width:{pct(b - a)}" title="이 쇼츠의 주제 부분 {a:.0f}~{b:.0f}초"></div>')
        marks += [f'<div class="voice" style="left:{pct(a)};width:{pct(b - a)}"></div>' for a, b in job.voices]
        cuts = cut_times(str(job.dir), len(job.frames), job)
        marks += [f'<div class="cut" style="left:{pct(t)}" title="컷 전환 {t:.1f}초"></div>' for t in cuts]
        marks += [f'<div class="pos src" style="left:{pct(c["start"])};width:{pct(c["end"] - c["start"])}" '
                  f'title="{i + 1}. {c["start"]:.1f}~{c["end"]:.1f}초">{i + 1}</div>'
                  for i, c in enumerate(short["clips"]) if c in cards and c.get("kind") != "ai"]
        rows.append(f'<div class="row"><span class="lab">원본 위치<br>0~{d:.0f}초</span><div class="track">{"".join(marks)}</div></div>')
        if job.voices:
            legend += '<i style="background:#3fa7d6"></i>목소리'
        if cuts:
            legend += '<i style="background:#f2c94c"></i>컷 전환 (구간을 늘려도 넘지 않음)'
    st.html(TIMELINE_CSS + f'<div class="sm-tl">{"".join(rows)}<div class="legend">{legend}</div></div>')


def short_editor(p: project.Project, short: dict, job: pipeline.Job) -> None:
    sid = short["id"]
    col_title, col_replan = st.columns([3, 1])
    with col_title:
        title = st.text_input("제목 (화면 위쪽)", value=short["title"], disabled=not show_title,
                              key=f"title{ss.card_rev}_{sid}")
        if title != short["title"]:
            remember(short)
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
                     help="같은 분석 결과로 이 쇼츠의 구간과 나레이션을 다시 만듭니다. 「🧠 대본·구성」의 말투·길이가 반영됩니다. "
                          "공용 작업 큐에 들어갑니다. 마음에 들지 않으면 「↩️ 되돌리기」로 돌아올 수 있습니다."):
            remember(short)
            enqueue_plan(p, short)
            st.rerun()

    timeline(short, job)
    col_vertical, col_undo = st.columns([3, 1], vertical_alignment="center")
    vertical = job is not None and col_vertical.toggle(
        "📱 카드를 세로 화면 미리보기로 보기", key=f"vertical_{sid}",
        help="원본 구간 카드에 영상 대신, 지금 화면·자막 설정으로 그린 쇼츠 모습(구간 시작 장면 + 제목 + 나레이션)을 보여 줍니다.")
    undo_slot = col_undo.empty()  # 카드에서 고친 것까지 세려고 카드를 그린 뒤에 채운다
    raw_clips = clip_cards(p, short, job, vertical)
    undo_count = len(ss.get("undo", {}).get(sid, []))
    undo_slot.button(f"↩️ 되돌리기 ({undo_count})", key=f"undo_{sid}", disabled=not undo_count, width="stretch",
                     on_click=undo, args=(p, short),
                     help="제목·카드를 고치기 전으로 한 단계씩 돌아갑니다 (순서 바꾸기·추가·삭제, 시간·나레이션 수정, "
                          "구성·나레이션 다시 만들기). 이 브라우저에서 고친 것만, 최근 20번까지.")
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
            help="카드의 구간 시간은 그대로 두고, 구간마다 그 시간의 장면·대사를 보고 나레이션을 새로 씁니다. "
                 "공용 작업 큐에 들어갑니다.",
        ):
            remember(short)
            enqueue("narrate", f"{p.name} · '{short['title'][:20]}' 나레이션", {"short_id": sid}, p,
                    key=f"narrate:{p.dir.name}:{sid}")
            st.rerun()

    ai_cards = [c for c in short["clips"] if c.get("kind") == "ai"]
    if ai_cards:
        not_made = [c for c in ai_cards if not (c.get("preview") and Path(c["preview"]).exists())]
        gen_ok, gen_msg = generate.available()
        if st.button(f"🎬 AI 장면 {len(not_made)}개 모두 큐에 넣기", key=f"enqueue_all_{sid}",
                     disabled=not gen_ok or not not_made,
                     help=gen_msg or "아직 만들지 않은 AI 장면을 전부 생성 큐에 넣습니다. 큐가 차례로 만들고, "
                                     "그동안 다른 카드를 고치거나 큐 순서를 바꿀 수 있습니다."):
            added = enqueue_cards(p, short, {c["uid"] for c in not_made})
            ss.flash = f"AI 장면 {added}개를 큐에 넣었습니다." if added else "모두 이미 만들었거나 큐에 있습니다."
            st.rerun()

    if st.button("3단계 · 나레이션 + 쇼츠 만들기", type="primary", disabled=not clips, key=f"make_{sid}",
                 help="공용 작업 큐에 넣습니다. 지금 설정(목소리·자막·배경음악 등)으로 만들고, 카드는 차례가 됐을 때의 값을 씁니다."):
        enqueue_make(p, [short])
        st.rerun()


st.divider()
st.subheader("2단계 · 쇼츠 구성과 나레이션 확인")
st.caption(
    "AI가 고른 원본 구간과 구간별 나레이션입니다. 카드에서 구간을 재생해 보고 시간과 나레이션을 고치거나, "
    "순서를 바꾸고 구간을 추가·삭제할 수 있습니다. 나레이션이 구간보다 길면 쇼츠를 만들 때 구간을 자동으로 늘립니다. "
    "고친 내용은 바로 저장됩니다."
)
if not proj.shorts:  # 구성이 아직 없거나 실패함
    if st.button("🧠 쇼츠 구성 만들기" if proj.kind == "upload" else "✨ AI 대본 + 장면 구성 만들기", type="primary",
                 disabled=llm_model is None):
        enqueue_plan(proj)
        st.rerun()
    st.stop()

if proj.kind == "upload" and (len(proj.shorts) > 1 or ss.short_count > 1):
    col_count, col_all_plan, col_all_make = st.columns([2, 1, 1])
    col_count.markdown(f"**쇼츠 {len(proj.shorts)}편** · 한 영상에서 서로 다른 주제로 나눴습니다.")
    if col_all_plan.button(f"🔄 {ss.short_count}편으로 다시 나누기", disabled=llm_model is None,
                           help="「🧠 대본·구성」의 「쇼츠 개수」만큼 주제를 다시 나누고 편마다 구성합니다."):
        enqueue_plan(proj)
        st.rerun()
    if col_all_make.button("🎬 모든 쇼츠 만들기", type="primary", disabled=len(proj.shorts) < 2):
        enqueue_make(proj, proj.shorts)
        st.rerun()

if len(proj.shorts) == 1:
    short_editor(proj, proj.shorts[0], job)
else:
    tabs = st.tabs([f"{i + 1}. {s['title'][:14] or '(제목 없음)'}" for i, s in enumerate(proj.shorts)])
    for tab, item in zip(tabs, proj.shorts):
        with tab:
            short_editor(proj, item, job)


# ───────────────────────── 결과 기록 ─────────────────────────
def delete_result(p: project.Project, result_id: str) -> None:
    for r in [r for r in p.results if r["id"] == result_id]:
        video = Path(r["video"])
        for path in (video, video.with_suffix(".srt"), video.with_suffix(".jpg")):
            path.unlink(missing_ok=True)
        p.results.remove(r)
    project.save(p)
    ss.flash = "결과를 지웠습니다."


def upload_box(p: project.Project, result: dict) -> None:
    """업로드할 때 쓸 제목 후보·설명·해시태그 (오른쪽 위 📋로 복사). 없으면 큐에 넣어 LLM이 쓴다."""
    text = result.get("upload")
    key = f"upload:{p.dir.name}:{result['id']}"
    item = next((q for q in jobqueue.items() if jobqueue.same_task(q, "upload", key)), None)
    with st.expander("📣 업로드 문구 (제목·설명·해시태그)", expanded=bool(text)):
        if text:
            st.caption("제목 후보 · 오른쪽 위 📋로 복사")
            for t in text["titles"]:
                st.code(t, language=None, wrap_lines=True)
            st.caption("설명")
            st.code(text["description"], language=None, wrap_lines=True)
            st.caption("해시태그")
            st.code(" ".join(text["hashtags"]), language=None, wrap_lines=True)
            st.caption("AI가 쓴 글이라 사실과 다를 수 있으니 올리기 전에 확인하세요.")
        if item is not None:
            st.caption(f"⏳ {jobqueue.STATUS_LABELS[item['status']]} · 위 「🚦 작업 큐」")
        elif st.button("🔄 다시 쓰기" if text else "✍️ 업로드 문구 쓰기", key=f"upload_{result['id']}",
                       disabled=llm_model is None, width="stretch",
                       help="이 쇼츠의 제목과 나레이션으로 업로드 제목 후보 3개, 설명, 해시태그를 씁니다. 공용 작업 큐에 들어갑니다."):
            enqueue("upload", f"{p.name} · '{result['title'][:20]}' 업로드 문구", {"result_id": result["id"]}, p, key=key)
            st.rerun()


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
                upload_box(proj, result)
                with st.expander("구간별 나레이션 길이"):
                    st.dataframe(result["report"], hide_index=True, width="stretch")
