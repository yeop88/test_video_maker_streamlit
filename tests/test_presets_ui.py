"""사이드바 프리셋 저장·불러오기·삭제와 설정 팝업을 Streamlit AppTest로 확인한다 (GPU 불필요, Ollama는 실행 중이면 사용).

세부 설정은 사이드바 버튼을 누르면 뜨는 팝업(st.dialog)에 있으므로, 값을 바꿀 때는 open_group으로 팝업을 연다.

사용법 (프로젝트 루트에서):
    .venv\\Scripts\\python tests\\test_presets_ui.py

실제 workdir를 건드리지 않도록 임시 폴더를 WORK_DIR로 쓴다.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

from shorts_maker import config, presets

config.WORK_DIR = Path(tempfile.mkdtemp(prefix="preset_test_"))
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def messages(at):
    return [m.value for m in [*at.sidebar.success, *at.sidebar.warning, *at.sidebar.error]]


def caption(at):
    return " ".join(c.value for c in at.sidebar.caption)


def open_group(at, label):
    """사이드바의 설정 분류 버튼을 눌러 팝업을 연다."""
    next(b for b in at.sidebar.button if b.label == label).click().run()
    return at


at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
check("첫 화면 예외 없음", not at.exception, at.exception)
options = at.sidebar.selectbox(key="preset_select").options
check("기본 제공 프리셋 4개 표시", len(options) == 4, options)
check("기본값 적용", at.session_state["font_size"] == 72 and at.session_state["engine"] == "edge")
check("설정 분류 버튼 5개", sum(b.label in ("🧠 대본·구성", "🎙️ 나레이션 목소리", "🔊 소리·배경음악", "🎬 화면·자막",
                                        "💻 컴퓨터·AI 모델") for b in at.sidebar.button) == 5)
open_group(at, "🎬 화면·자막")
check("팝업이 열리고 위젯이 보임", at.slider(key="font_size").value == 72, [s.key for s in at.slider])

# 저장 (팝업에서 바꾼 값은 팝업을 닫아도 남아 있어야 함)
at.slider(key="font_size").set_value(90).run()
open_group(at, "🧠 대본·구성")
at.slider(key="target_sec").set_value(60).run()
open_group(at, "🎙️ 나레이션 목소리")
at.selectbox(key="edge_voice").select("ko-KR-InJoonNeural").run()
at.run()  # 팝업 닫힘
check("팝업을 닫아도 바꾼 값 유지", at.session_state["font_size"] == 90 and at.session_state["target_sec"] == 60
      and at.session_state["edge_voice"] == "ko-KR-InJoonNeural")
at.sidebar.text_input(key="preset_new_name").input("테스트 프리셋")
button(at, "저장").click()
at.run()
saved = config.WORK_DIR / "presets" / "테스트 프리셋.json"
check("저장 파일 생성", saved.exists())
data = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
check("저장 내용", data.get("target_sec") == 60 and data.get("font_size") == 90
      and data.get("edge_voice") == "ko-KR-InJoonNeural", data)
check("저장 알림", any("저장했습니다" in m for m in messages(at)), messages(at))
check("목록에 추가 + 선택됨", at.sidebar.selectbox(key="preset_select").value == "테스트 프리셋")
check("현재 프리셋 표시", "현재: 테스트 프리셋" in caption(at) and "수정됨" not in caption(at), caption(at))

# 수정 표시
open_group(at, "🎬 화면·자막")
at.slider(key="font_size").set_value(50).run()
at.run()
check("바꾸면 (수정됨)", "(수정됨)" in caption(at), caption(at))

# 기본 불러오기
at.sidebar.selectbox(key="preset_select").select("기본").run()
button(at, "불러오기").click().run()
check("기본 불러오기", at.session_state["font_size"] == 72 and at.session_state["target_sec"] == 40,
      at.session_state["font_size"])

# 숨겨진 설정 유지: Chatterbox를 고르면 Edge 목소리 위젯이 사라지는데, 불러온 값이 지워지면 안 됨
at.sidebar.selectbox(key="preset_select").select("테스트 프리셋").run()
button(at, "불러오기").click().run()
# AppTest에서는 값을 바꾸면 스크립트 전체가 다시 돌아 팝업이 닫히므로 단계마다 다시 연다 (브라우저에서는 팝업이 그대로 열려 있음)
open_group(at, "🎙️ 나레이션 목소리")
at.radio(key="engine").set_value("chatterbox").run()
open_group(at, "🎙️ 나레이션 목소리")
check("Chatterbox를 고르면 Edge 목소리 칸이 사라짐", "edge_voice" not in [x.key for x in at.selectbox])
at.radio(key="engine").set_value("edge").run()
open_group(at, "🎙️ 나레이션 목소리")
check("엔진을 바꿨다 돌아와도 목소리 유지", at.selectbox(key="edge_voice").value == "ko-KR-InJoonNeural",
      at.selectbox(key="edge_voice").value)
at.run()
check("사용자 프리셋 값 복원", at.session_state["font_size"] == 90 and at.session_state["target_sec"] == 60)

# 기본 제공 프리셋 이름으로 저장 → 거부
at.sidebar.text_input(key="preset_new_name").input("기본")
button(at, "저장").click().run()
check("기본 제공 이름 덮어쓰기 거부", any("덮어쓸 수 없습니다" in m for m in messages(at)), messages(at))

# 기본 제공 프리셋 삭제 버튼 비활성
at.sidebar.selectbox(key="preset_select").select("감성 브이로그").run()
check("기본 제공 프리셋 삭제 불가", button(at, "삭제").disabled)
button(at, "불러오기").click().run()
check("기본 제공 프리셋 불러오기", at.session_state["style"] == "감성 스토리" and at.session_state["target_sec"] == 30)

# 손상된 파일, 범위 밖 값, 설치 안 된 모델
folder = config.WORK_DIR / "presets"
(folder / "손상.json").write_text("{bad json", encoding="utf-8")
(folder / "이상한 값.json").write_text(json.dumps(
    {"font_size": 999, "engine": "unknown", "llm_model": "없는모델:1b", "color": "#00ff00"}), encoding="utf-8")
at.run()
at.sidebar.selectbox(key="preset_select").select("손상").run()
button(at, "불러오기").click().run()
check("손상된 파일은 오류 표시", any("손상" in m for m in at.sidebar.error.values), messages(at))
model_before = at.session_state["llm_model"]
at.sidebar.selectbox(key="preset_select").select("이상한 값").run()
button(at, "불러오기").click().run()
check("범위 밖 값은 범위 안으로", at.session_state["font_size"] == 110)
check("잘못된 선택지는 기본값", at.session_state["engine"] == "edge")
check("색은 대문자로", at.session_state["color"] == "#00FF00")
if model_before:
    check("설치 안 된 모델은 지금 모델 유지", at.session_state["llm_model"] == model_before
          and any("설치되어 있지 않아" in m for m in messages(at)), messages(at))

# 추천 LLM이 설치되어 있지 않으면 팝업에 설치 버튼 (CPU 추천은 qwen3.5:4b)
from shorts_maker import llm

try:
    installed = llm.list_models()
except Exception:
    installed = []
if installed and "qwen3.5:4b" not in installed:
    at.session_state["device"] = "cpu"
    at.run()
    for group in ("🧠 대본·구성", "💻 컴퓨터·AI 모델"):
        open_group(at, group)
        check(f"추천 LLM 설치 버튼 ({group})", any(b.label == "⬇️ qwen3.5:4b 설치" for b in at.button),
              [b.label for b in at.button])
    at.session_state["device"] = "gpu"
    at.run()
    open_group(at, "🧠 대본·구성")
    check("설치된 추천 LLM이면 설치 버튼 없음 (GPU 추천 qwen3.5:9b)",
          "qwen3.5:9b" not in installed or not any(b.label.startswith("⬇️") for b in at.button))
    at.run()

# 삭제
at.sidebar.selectbox(key="preset_select").select("테스트 프리셋").run()
button(at, "삭제").click().run()
check("삭제", not saved.exists() and "테스트 프리셋" not in at.sidebar.selectbox(key="preset_select").options)
check("마지막까지 예외 없음", not at.exception, at.exception)

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} 통과" + (f" — 실패: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
