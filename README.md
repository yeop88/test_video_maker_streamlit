# 🎬 AI 쇼츠 나레이션 스튜디오

영상을 올리면 AI가 **장면과 대사를 이해**해서 하이라이트 구간을 고르고, 구간마다 **한국어 나레이션**과 **한글 자막**을 입혀 **세로형 쇼츠/릴스(1080×1920)**로 만들어 주는 Streamlit 서비스입니다.

- 장면 이미지를 비전 LLM이 설명하고, 대사는 Whisper로 받아써서 영상 내용을 파악
- 목표 길이(15~90초)에 맞춰 쓸 구간을 고르고, 구간마다 그 장면에 맞는 나레이션을 작성
- 나레이션이 나올 때만 원본 소리를 줄여서(더킹) 섞음
- 구간·나레이션을 표에서 직접 고친 뒤 다시 만들기, 나레이션만 다시 쓰기
- 외국어 영상이어도 나레이션과 자막은 한국어

## 처리 과정

```
영상 업로드 → 분석할 구간 선택
   │
   ├─ 1단계 · 영상 이해
   │   ├─ Silero VAD   사람 목소리가 있는 구간 찾기 (음악만 있는 곳은 받아쓰지 않음)
   │   ├─ Whisper      대사 받아쓰기 + 언어 감지 (openai/whisper-large-v3-turbo)
   │   └─ 비전 LLM     2초 이상 간격으로 장면을 캡처해 한국어로 설명 (Ollama, 기본 qwen3.5:9b)
   │
   ├─ 1단계 · 쇼츠 구성 (LLM)
   │   ├─ 구간 고르기      타임라인을 보고 목표 길이에 맞는 구간 선택 (길이가 틀리면 알려 주고 다시)
   │   └─ 나레이션 쓰기    구간마다 그 시간의 장면·대사만 보여 주고 작성 (원본에 없는 숫자는 다시)
   │
   ├─ 2단계 · [사용자] 구간·나레이션 확인·수정
   │
   └─ 3단계 · 쇼츠 만들기
       ├─ Edge TTS / Chatterbox   문장마다 나레이션 음성 (자막 시간을 문장 단위로 정확히)
       ├─ 구간 맞추기              나레이션이 길면 구간을 대사 경계까지 늘림
       ├─ ffmpeg                  구간 자르고 잇기
       ├─ 믹싱                    나레이션 중에만 원본 소리 줄이기
       └─ ffmpeg                  세로 변환(블러 배경 / 크롭) + 제목 + 한글 자막 → MP4
```

## 사용 모델

| 역할 | 모델 | 라이선스 |
|---|---|---|
| 장면 이해, 구간 선택, 나레이션 작성 | Ollama 로컬 비전 모델 (기본 `qwen3.5:9b`, `gemma4` 등) | 모델별 라이선스 |
| 대사 인식 | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | MIT |
| 목소리 구간 감지 | [Silero VAD](https://github.com/snakers4/silero-vad) | MIT |
| 나레이션 음성 (기본) | Microsoft Edge TTS (`edge-tts`) | 온라인 서비스 |
| 나레이션 음성 (선택) | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) Multilingual, 기본 목소리 또는 목소리 샘플 | MIT |

## 실행 환경

- **NVIDIA GPU** (VRAM 12GB 이상 권장, RTX 4080 SUPER 16GB에서 테스트)
- Python 3.11
- [Ollama](https://ollama.com) 실행 중 + 비전 모델 설치 (`ollama pull qwen3.5:9b`)
- Edge TTS를 쓰려면 인터넷 연결
- ffmpeg: 시스템에 없으면 처음 실행할 때 `static-ffmpeg`가 자동으로 내려받습니다.

## 설치와 실행

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

처음 실행할 때는 Whisper(와 Chatterbox를 고르면 Chatterbox) 모델을 내려받느라 몇 분 걸립니다.

## 사용 방법

1. **업로드**: 영상을 올리고 분석할 구간을 고릅니다 (15분 이하).
2. **1단계 · 영상 분석 + 쇼츠 구성**: 대사 인식, 장면 설명 후 AI가 구간과 나레이션을 정합니다. 「AI가 이해한 내용」을 펼치면 타임라인을 볼 수 있습니다.
3. **2단계 · 확인·수정**: 표에서 구간 시간과 나레이션을 고치거나 행을 추가·삭제합니다.
   - 🔄 **구성 다시 만들기**: 구간과 나레이션을 처음부터 다시
   - ✍️ **나레이션만 다시 쓰기**: 표의 구간은 그대로 두고 나레이션만 새로
4. **3단계 · 나레이션 + 쇼츠 만들기**: 결과를 미리 보고 다운로드합니다. 나레이션이 구간보다 길어 다음 장면으로 넘어간 곳은 표로 알려 줍니다.

사이드바에서 LLM, 나레이션 말투(정보 전달 / 유쾌한 리뷰 / 감성 스토리), 쇼츠 길이, 목소리, 원본 소리 크기, 세로 변환 방식, 제목 표시, 자막 크기·위치·색을 바꿀 수 있습니다.

## 폴더 구조

```
app.py                  Streamlit 화면
shorts_maker/
  config.py             설정값 (해상도, 모델 이름, 캡처 간격, 나레이션 길이 기준 등)
  media.py              ffmpeg 래퍼 (자르기, 장면 캡처, 구간 잇기, 세로 렌더링)
  asr.py                Silero VAD + Whisper 대사 인식
  llm.py                Ollama: 장면 설명, 구간 고르기, 나레이션 쓰기 (JSON 구조화 출력 + 검사 후 재요청)
  tts.py                Edge TTS / Chatterbox 엔진 (문장별 캐시)
  korean_numbers.py     숫자 읽기("18도" → "십팔 도"), 글 속 숫자 값 찾기 (지어낸 숫자 검사용)
  mixing.py             나레이션 배치, 원본 소리 더킹 믹싱
  subtitles.py          쇼츠 스타일 ASS 자막 + 제목
  pipeline.py           단계별 작업 연결
tests/
  verify_pipeline.py    전체 파이프라인 검증 (3가지 조건, 수치 판정)
docs/test_reports/      검증 기록
workdir/                업로드 영상과 중간 결과물 (git 제외)
```

## 검증

```bash
.venv\Scripts\python tests\verify_pipeline.py
```

정답을 아는 테스트 영상 3개를 직접 만들어 전체 과정을 돌립니다.
- 한국어 호수 여행 브이로그
- 영어 수면 팁 슬라이드
- 대사 없는 추상 조형물 영상(목소리 샘플 사용)

출력 형식, 목표 길이, 장면 설명, 나레이션이 영상 내용에 근거하는지, 원본에 없는 숫자를 지어내지 않는지, 나레이션과 화면이 맞는지, 대사 중간에서 끊기지 않는지, 나레이션이 또렷하게 들리는지, 더킹이 되는지를 수치로 판정합니다. 결과는 [docs/test_reports/](docs/test_reports/)에, 변경 이력은 [CHANGELOG.md](CHANGELOG.md)에 있습니다.

## 배포에 대해

이 앱은 **GPU와 로컬 Ollama가 필요**해서 Streamlit Community Cloud(무료, CPU 전용)에서는 그대로 돌릴 수 없습니다.

- **내 PC에서 실행 + 터널**: `streamlit run app.py` 후 `cloudflared tunnel --url http://localhost:8501`로 임시 공개 주소를 만듭니다.
- **GPU 클라우드**: Hugging Face Spaces(GPU, Docker), RunPod 등에 올리고, LLM은 Ollama 대신 비전을 지원하는 HF Inference Providers API로 바꿉니다 (`shorts_maker/llm.py`의 `_chat`만 교체).

## 주의

- 본인이 권리를 가진 영상이나 허락받은 영상만 사용하세요.
- 목소리 샘플은 본인 또는 동의받은 사람의 목소리만 사용하세요. Chatterbox 출력에는 AI 생성 음성임을 알리는 워터마크(Perth)가 들어갑니다.
- AI가 쓴 나레이션은 틀릴 수 있습니다. 2단계 표에서 꼭 확인하세요.
- 비상업적 학습·포트폴리오용 프로젝트입니다.

## 라이선스

[MIT](LICENSE). 사용하는 모델과 서비스(Whisper, Chatterbox, Silero VAD, Ollama 모델, Edge TTS)는 각자의 라이선스·이용 약관을 따릅니다.
