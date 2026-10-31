"""프로젝트 전역 설정."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 업로드 영상과 중간 결과물 저장 위치 (테스트는 SHORTS_WORK_DIR로 임시 폴더를 지정)
WORK_DIR = Path(os.environ.get("SHORTS_WORK_DIR") or ROOT / "workdir")

OUTPUT_SIZE = (1080, 1920)  # 쇼츠/릴스 세로 해상도 (기본)
# 출력 크기 설정(output_size)의 선택지. 둘 다 세로 화면이다.
OUTPUT_SIZES = {"1080x1920": (1080, 1920), "480x640": (480, 640)}
# 자막·제목 글자 크기는 너비 1080 기준 값이다. 작은 출력에서는 ASS의 PlayRes를 너비 1080(같은 비율)로 두고
# libass가 줄여 그리게 해서, 같은 설정이 어느 크기에서나 같은 비율로 보이게 한다.
ASS_WIDTH = 1080


def output_size(name: str | None) -> tuple[int, int]:
    return OUTPUT_SIZES.get(name or "", OUTPUT_SIZE)


def ass_size(size: tuple[int, int]) -> tuple[int, int]:
    """출력 크기 → 자막 좌표계 크기 (너비 ASS_WIDTH, 같은 가로세로 비율)."""
    w, h = size
    return ASS_WIDTH, int(round(ASS_WIDTH * h / w))
MAX_SOURCE_SECONDS = 900  # 분석할 원본 구간 최대 길이 (15분)

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"  # 이미지 이해(vision)를 지원하는 모델이어야 한다

FRAME_WIDTH = 512  # 장면 설명용 캡처 크기
MIN_FRAME_INTERVAL = 2.0  # 장면 캡처 간격 (초, 최소)
MAX_FRAMES = 48  # 긴 영상은 간격을 늘려서 이 개수 안으로 맞춘다
# 앞에서 설명한 장면과 거의 같은 캡처는 LLM에 다시 묻지 않고 그 설명을 쓴다 (슬라이드·정지 화면에서 분석이 몇 배 빨라짐).
# 96x54 흑백으로 줄여 밝기가 25 넘게 바뀐 픽셀 비율: 전체가 SAME_SCENE_CHANGED 미만이고 가장 많이 바뀐 블록(8x6 중 하나)도
# SAME_SCENE_BLOCK 미만이면 같은 장면. 측정: 같은 슬라이드 0.03/0.31~0.41, 슬라이드가 바뀌면 0.18~0.25/0.66~1.0.
SAME_SCENE_CHANGED = 0.08
SAME_SCENE_BLOCK = 0.5
MAX_REUSED_FRAMES = 2  # 글자 한 줄만 바뀐 화면을 놓쳐도 곧 다시 설명하도록, 연속으로 이만큼만 다시 쓴다

NARRATION_CHARS_PER_SECOND = 6.0  # 나레이션 대본 길이 기준 (문장 사이 쉼 포함)
NARRATION_GAP = 0.25  # 나레이션 문장 사이 쉼 (초)
MIX_SAMPLE_RATE = 44100
