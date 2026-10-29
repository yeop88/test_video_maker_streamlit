"""사이드바 프리셋 저장·불러오기·삭제를 Streamlit AppTest로 확인한다 (GPU 불필요, Ollama는 실행 중이면 사용).

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


at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
check("첫 화면 예외 없음", not at.exception, at.exception)
options = at.sidebar.selectbox(key="preset_select").options
check("기본 제공 프리셋 4개 표시", len(options) == 4, options)
check("기본값 적용", at.sidebar.slider(key="font_size").value == 72 and at.session_state["engine"] == "edge")

# 저장
at.sidebar.slider(key="target_sec").set_value(60)
at.sidebar.slider(key="font_size").set_value(90)
at.sidebar.selectbox(key="edge_voice").select("ko-KR-InJoonNeural")
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
at.sidebar.slider(key="font_size").set_value(50).run()
check("바꾸면 (수정됨)", "(수정됨)" in caption(at), caption(at))

# 기본 불러오기
at.sidebar.selectbox(key="preset_select").select("기본").run()
button(at, "불러오기").click().run()
check("기본 불러오기", at.sidebar.slider(key="font_size").value == 72
      and at.sidebar.slider(key="target_sec").value == 40, at.sidebar.slider(key="font_size").value)

# 숨겨진 설정 유지: Chatterbox를 고르면 Edge 목소리 위젯이 사라지는데, 불러온 값이 지워지면 안 됨
at.sidebar.selectbox(key="preset_select").select("테스트 프리셋").run()
button(at, "불러오기").click().run()
at.sidebar.radio(key="engine").set_value("chatterbox").run()
at.sidebar.radio(key="engine").set_value("edge").run()
check("엔진을 바꿨다 돌아와도 목소리 유지", at.sidebar.selectbox(key="edge_voice").value == "ko-KR-InJoonNeural",
      at.sidebar.selectbox(key="edge_voice").value)
check("사용자 프리셋 값 복원", at.sidebar.slider(key="font_size").value == 90
      and at.sidebar.slider(key="target_sec").value == 60)

# 기본 제공 프리셋 이름으로 저장 → 거부
at.sidebar.text_input(key="preset_new_name").input("기본")
button(at, "저장").click().run()
check("기본 제공 이름 덮어쓰기 거부", any("덮어쓸 수 없습니다" in m for m in messages(at)), messages(at))

# 기본 제공 프리셋 삭제 버튼 비활성
at.sidebar.selectbox(key="preset_select").select("감성 브이로그").run()
check("기본 제공 프리셋 삭제 불가", button(at, "삭제").disabled)
button(at, "불러오기").click().run()
check("기본 제공 프리셋 불러오기", at.session_state["style"] == "감성 스토리"
      and at.sidebar.slider(key="target_sec").value == 30)

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
check("범위 밖 값은 범위 안으로", at.sidebar.slider(key="font_size").value == 110)
check("잘못된 선택지는 기본값", at.session_state["engine"] == "edge")
check("색은 대문자로", at.session_state["color"] == "#00FF00")
if at.sidebar.selectbox(key="llm_model") is not None and model_before:
    check("설치 안 된 모델은 지금 모델 유지", at.session_state["llm_model"] == model_before
          and any("설치되어 있지 않아" in m for m in messages(at)), messages(at))

# 삭제
at.sidebar.selectbox(key="preset_select").select("테스트 프리셋").run()
button(at, "삭제").click().run()
check("삭제", not saved.exists() and "테스트 프리셋" not in at.sidebar.selectbox(key="preset_select").options)
check("마지막까지 예외 없음", not at.exception, at.exception)

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} 통과" + (f" — 실패: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
