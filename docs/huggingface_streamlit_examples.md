# Streamlit으로 배포하기 좋은 Hugging Face 예제 30선

> 작성일: 2026-10-28
> 기준: 2026년 상반기까지 공개된 모델 중 Streamlit 데모로 만들기 쉬운 것 위주로 골랐습니다.
> 모델 ID와 라이선스는 배포 전에 Hugging Face Hub에서 다시 확인하세요.

## 실행 환경 표시

| 표시 | 의미 |
|---|---|
| 🟢 | CPU로 실행 가능 (Streamlit Community Cloud 무료 플랜 가능) |
| 🟡 | GPU 권장 |
| 🔵 | 직접 실행하기 어려움 → HF Inference Providers API로 호출 권장 |

---

## 1. 텍스트 / LLM

| # | 모델 | 데모 아이디어 | 환경 |
|---|---|---|---|
| 1 | `Qwen/Qwen3-0.6B` / `Qwen/Qwen3-1.7B` | 생각 모드를 켜고 끌 수 있는 미니 챗봇 | 🟢 |
| 2 | `HuggingFaceTB/SmolLM3-3B` | 영어 글쓰기 도우미, 요약기 | 🟡 |
| 3 | `google/gemma-3-1b-it` | 다국어 번역, 말투 바꾸기 앱 | 🟢 |
| 4 | `openai/gpt-oss-20b` | 추론 과정을 보여주는 문제 풀이 챗봇 | 🔵 |
| 5 | `LGAI-EXAONE/EXAONE-4.0-1.2B` | 한국어 Q&A, 자기소개서 첨삭 | 🟢 |
| 6 | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B` | 한국어 초경량 챗봇 | 🟢 |

## 2. 임베딩 / RAG

| # | 모델 | 데모 아이디어 | 환경 |
|---|---|---|---|
| 7 | `google/embeddinggemma-300m` | PDF를 업로드하면 의미 기반으로 검색 | 🟢 |
| 8 | `Qwen/Qwen3-Embedding-0.6B` | 비슷한 뉴스·영화 추천기 | 🟢 |
| 9 | `BAAI/bge-m3` | 한국어·영어를 섞어 검색하는 RAG 챗봇 | 🟢 |

## 3. 비전 (이해 / 분석)

| # | 모델 | 데모 아이디어 | 환경 |
|---|---|---|---|
| 10 | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 사진·짧은 영상에 대해 질문하기 | 🟡 |
| 11 | `Qwen/Qwen2.5-VL-3B-Instruct` (또는 Qwen3-VL) | 영수증·차트를 읽고 JSON으로 추출 | 🟡 |
| 12 | `microsoft/Florence-2-base` | 캡션, 객체 탐지, OCR을 한 앱에서 | 🟢 |
| 13 | `depth-anything/Depth-Anything-V2-Small-hf` | 사진 → 깊이 맵 시각화 | 🟢 |
| 14 | `facebook/sam2.1-hiera-small` (또는 SAM 3) | 클릭한 물체만 잘라내기 | 🟡 |
| 15 | `IDEA-Research/grounding-dino-tiny` | "빨간 컵 찾아줘"처럼 텍스트로 객체 탐지 | 🟢 |
| 16 | `google/siglip2-base-patch16-224` | 라벨을 직접 입력하는 제로샷 이미지 분류 | 🟢 |
| 17 | `ZhengPeng7/BiRefNet` | 배경 제거 → 증명사진 만들기 | 🟢 |
| 18 | `ibm-granite/granite-docling-258M` | PDF·문서를 Markdown으로 변환 | 🟢 |
| 19 | `deepseek-ai/DeepSeek-OCR` | 복잡한 문서 OCR | 🟡 |

## 4. 이미지 생성 / 편집

| # | 모델 | 데모 아이디어 | 환경 |
|---|---|---|---|
| 20 | `black-forest-labs/FLUX.1-schnell` | 프롬프트로 이미지 생성 | 🔵 |
| 21 | `black-forest-labs/FLUX.1-Kontext-dev` | "이 사진을 지브리 스타일로"처럼 말로 편집 | 🔵 |
| 22 | `Qwen/Qwen-Image-Edit` | 이미지 속 글자 바꾸기, 편집 | 🔵 |
| 23 | `stabilityai/sdxl-turbo` | 1스텝 실시간 이미지 생성 | 🟡 |

## 5. 오디오 / 음성

| # | 모델 | 데모 아이디어 | 환경 |
|---|---|---|---|
| 24 | `openai/whisper-large-v3-turbo` | 회의 녹음 → 받아쓰기 → LLM 요약 | 🟡 (`whisper-small`이면 🟢) |
| 25 | `nvidia/parakeet-tdt-0.6b-v3` | 타임스탬프가 붙은 빠른 자막 생성 | 🟡 |
| 26 | `hexgrad/Kokoro-82M` | 초경량 TTS, 목소리 고르기 | 🟢 |
| 27 | `ResembleAI/chatterbox` | 목소리 복제, 감정 조절 TTS | 🟡 |
| 28 | `facebook/musicgen-small` | "lofi 공부 음악" 같은 텍스트로 음악 생성 | 🟡 |
| 29 | `MIT/ast-finetuned-audioset-10-10-0.4593` | 소리 분류 (개 짖는 소리, 박수 등) | 🟢 |

## 6. 3D

| # | 모델 | 데모 아이디어 | 환경 |
|---|---|---|---|
| 30 | `stabilityai/stable-fast-3d` | 사진 한 장 → 3D 모델(.glb) 생성 | 🟡 |

---

## 고르는 방법

- **처음이라면**: 13번(깊이 맵), 17번(배경 제거), 26번(TTS), 16번(제로샷 분류)부터 해보세요. 가볍고 결과가 눈에 바로 보입니다.
- **포트폴리오용 조합**
  - 회의록 자동 생성기: 24번 Whisper(받아쓰기) + 5번 EXAONE(요약)
  - 내 PDF 챗봇: 9번 bge-m3(검색) + 1번 Qwen3(답변)
- **무거운 모델(🔵)**: `huggingface_hub`의 `InferenceClient`로 호출하면 Streamlit 앱은 가볍게 유지됩니다.

```python
from huggingface_hub import InferenceClient
import streamlit as st

client = InferenceClient(token=st.secrets["HF_TOKEN"])
image = client.text_to_image("a cat astronaut", model="black-forest-labs/FLUX.1-schnell")
st.image(image)
```

## Streamlit 배포 팁

- 모델 로딩 함수에는 `@st.cache_resource`를 붙이세요. 붙이지 않으면 버튼을 누를 때마다 모델을 다시 불러옵니다.

  ```python
  @st.cache_resource
  def load_model():
      from transformers import pipeline
      return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
  ```

- Streamlit Community Cloud는 메모리가 작아서 파라미터 1B 이하 모델이 안전합니다.
- `HF_TOKEN`은 코드에 넣지 말고 `.streamlit/secrets.toml`(로컬) 또는 배포 설정의 Secrets에 넣고 `st.secrets`로 읽으세요. `secrets.toml`은 `.gitignore`에 추가하세요.
- Gemma, Llama 같은 gated 모델은 Hub에서 라이선스 동의를 먼저 해야 합니다.
- 라이선스를 확인하세요. 예를 들어 `FLUX.1-Kontext-dev`는 비상업용 라이선스입니다.

## 참고 자료

- [Hugging Face Models (Trending)](https://huggingface.co/models?sort=trending)
- [10 Hugging Face Models Worth Knowing in 2026 (KDnuggets)](https://www.kdnuggets.com/10-hugging-face-models-worth-knowing-in-2026)
- [Hugging Face new models announcements May 2026](https://fazm.ai/t/hugging-face-new-models-announcements-may-2026)
- [HF Hub Release Report Jun–Jul 2026](https://lvwerra-agent-artifacts.static.hf.space/hf-model-releases-2026-07.html)
- [See which AI models are trending (Firecrawl)](https://www.firecrawl.dev/prompts/trending-ai-models)
