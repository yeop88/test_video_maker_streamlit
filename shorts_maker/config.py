"""프로젝트 전역 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / "workdir"  # 업로드 영상과 중간 결과물 저장 위치

OUTPUT_SIZE = (1080, 1920)  # 쇼츠/릴스 세로 해상도
MAX_SOURCE_SECONDS = 900  # 분석할 원본 구간 최대 길이 (15분)

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"  # 이미지 이해(vision)를 지원하는 모델이어야 한다

FRAME_WIDTH = 512  # 장면 설명용 캡처 크기
MIN_FRAME_INTERVAL = 2.0  # 장면 캡처 간격 (초, 최소)
MAX_FRAMES = 48  # 긴 영상은 간격을 늘려서 이 개수 안으로 맞춘다

NARRATION_CHARS_PER_SECOND = 6.0  # 나레이션 대본 길이 기준 (문장 사이 쉼 포함)
NARRATION_GAP = 0.25  # 나레이션 문장 사이 쉼 (초)
MIX_SAMPLE_RATE = 44100
