# 🎬 AI 쇼츠 나레이션 스튜디오

영상을 올리면 AI가 **장면과 대사를 이해**해서 하이라이트 구간을 고르고, 구간마다 **한국어 나레이션**과 **한글 자막**을 입혀 **세로형 쇼츠/릴스(1080×1920)**로 만들어 주는 Streamlit 서비스입니다. 영상이 없어도 **주제만 쓰면 AI가 대본과 장면(이미지·영상)을 만들어** 쇼츠를 만듭니다.

- **📼 영상으로 만들기**: 장면은 비전 LLM이 설명하고 대사는 Whisper로 받아써서 내용을 파악한 뒤, 목표 길이(15~90초)에 맞는 구간과 구간별 나레이션을 고릅니다. 긴 영상은 주제별로 **여러 편**을 한 번에 만들 수 있습니다.
- **✨ 영상 없이 AI로 만들기**: 주제 → AI 대본(장면별 나레이션·그림 설명) → AI 장면(SDXL-Turbo 이미지 + 움직임 / LTX-Video 동영상).
- 원본 구간과 AI 장면을 카드로 섞고, 순서·시간·나레이션을 직접 고칩니다. 작업은 자동 저장되어 새로고침해도 이어집니다.
- 나레이션 목소리: Edge TTS, 또는 Chatterbox **제로샷 음성 복제**(올린 목소리·녹음·가상 목소리).
- 배경음악(올린 음악·**AI 음악**), 단어 강조 자막, 피사체 따라 크롭, SRT 자막·썸네일 내보내기.

## 처리 과정

```
📼 영상 업로드 → 분석할 구간 선택                       ✨ 주제 입력
   │                                                    │
   ├─ 1단계 · 영상 이해                                 └─ 1단계 · AI 대본 (LLM)
   │   ├─ Silero VAD   목소리 구간 (음악은 받아쓰지 않음)      장면마다 나레이션, 그림 설명(한글),
   │   ├─ Whisper      대사 받아쓰기 + 언어 감지                생성용 영어 프롬프트, 길이
   │   └─ 비전 LLM     장면 캡처를 한국어로 설명                (주제에 없는 숫자는 다시 요청)
   │
   ├─ 1단계 · 쇼츠 구성 (LLM, 여러 편이면 주제별 범위부터)
   │   ├─ 구간 고르기      목표 길이·핵심 내용 검사 후 다시 요청, 빠진 번호 보충, 긴 구간 나누기
   │   └─ 나레이션 쓰기    구간마다 그 시간의 장면·대사만 보고 작성, 문제 남은 구간만 다시
   │
   ├─ 2단계 · [사용자] 카드 확인·수정 (원본 구간 / ✨ AI 장면 추가, 순서·시간·나레이션)
   │
   └─ 3단계 · 쇼츠 만들기
       ├─ Edge TTS / Chatterbox   문장마다 나레이션 (단어별 시간으로 단어 강조 자막)
       ├─ 길이 맞추기              나레이션이 길면 구간·장면을 늘림 (대사 중간에서 끊지 않음)
       ├─ AI 장면 만들기           .venv-gen에서 SDXL-Turbo / LTX-Video
       ├─ 카드별 세로 장면         블러 / 크롭 / 피사체 따라 크롭, AI 장면 길이 맞춤 → 잇기
       ├─ 믹싱                    원본 소리·배경음악은 나레이션 중에만 줄이기
       └─ ffmpeg                  제목 + 한글 자막 → MP4, SRT, 썸네일
```

## 사용 모델

| 역할 | 모델 | 라이선스 |
|---|---|---|
| 장면 이해, 구간 선택, 나레이션 작성 | Ollama 로컬 비전 모델 (기본 `qwen3.5:9b`, `gemma4` 등) | 모델별 라이선스 |
| 대사 인식 | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | MIT |
| 목소리 구간 감지 | [Silero VAD](https://github.com/snakers4/silero-vad) | MIT |
| 나레이션 음성 (기본) | Microsoft Edge TTS (`edge-tts`) | 온라인 서비스 |
| 나레이션 음성 (선택) | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) Multilingual, 기본 목소리 또는 **목소리 보관함**의 목소리를 제로샷 복제 | MIT |
| 장면 이미지 (선택, 하나 고름) | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo), [Lykon/dreamshaper-xl-lightning](https://huggingface.co/Lykon/dreamshaper-xl-lightning), [Lykon/dreamshaper-xl-v2-turbo](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo), [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [FLUX.2 [klein] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | 비상업 연구용 / OpenRAIL++ / Apache 2.0 / FLUX 비상업 |
| 장면 영상 (선택, 하나 고름) | [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) 2B, [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) 22B (4bit/8bit 양자화) | LTX 오픈 가중치 / LTX-2 Community (연 매출 1천만 달러 미만 무료) |
| 배경음악 (선택) | [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) | CC-BY-NC 4.0 |

## 실행 환경

- **NVIDIA GPU** (VRAM 12GB 이상 권장, RTX 4080 SUPER 16GB에서 테스트). GPU가 작거나 없으면 사이드바 「🛠️ 세부 설정 → 💻 컴퓨터·AI 모델」에서 **GPU + CPU 함께** 또는 **CPU만**을 고릅니다 (느려짐). **작업별로 나누기**를 고르면 LLM·대사 인식·음성·이미지·동영상·음악마다 GPU/CPU를 따로 정할 수 있습니다 (기본: AI 동영상만 GPU, 나머지는 CPU). 추천 모델은 작업마다 고른 장치에 맞게 바뀌고, 「✨ 추천 모델로 맞추기」로 한 번에 적용합니다. 앱 위쪽 「💻 컴퓨터 사양과 추천 모델」에서 이 PC의 GPU·RAM·CPU와 작업별 GPU용·CPU용 추천 모델을 볼 수 있습니다.
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

### AI 생성 기능 설치 (장면 이미지·영상, 배경음악)

AI 장면·음악 생성은 앱과 **다른 가상환경**(`.venv-gen`)에서 별도 프로세스로 돌립니다.
chatterbox-tts가 `diffusers==0.29.0`에 고정되어 있는데, 그 버전은 앱의 transformers 5.x와 함께 쓸 수 없기 때문입니다.
이렇게 하면 생성이 끝날 때 GPU 메모리도 바로 돌려받습니다. 설치하지 않으면 AI 생성 메뉴만 꺼지고 나머지는 그대로 쓸 수 있습니다.

```bash
python -m venv .venv-gen
.venv-gen\Scripts\python -m pip install -r requirements-gen.txt
.venv\Scripts\python tests\test_generate.py     # 확인 (처음에는 모델 내려받기로 오래 걸림)
```

| 기능 | 모델 | 내려받기 | 걸리는 시간 (RTX 4080 SUPER) | 라이선스 |
|---|---|---|---|---|
| 장면 이미지 → 움직임 클립 | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) (fp16, 576×1024) | 약 7GB | 장당 약 2초 | 비상업 연구용 |
| 〃 | [DreamShaper XL Lightning](https://huggingface.co/Lykon/dreamshaper-xl-lightning) / [v2 Turbo](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo) (832×1216) | 각 약 7GB | 장당 1.2~1.5초 | OpenRAIL++ |
| 장면 영상 (약 4초) | [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) 2B | 약 28GB | 장면당 1분 안팎 | LTX 오픈 가중치 (비상업 무료) |
| 배경음악 (최대 30초) | [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) | 약 2GB | 10초 곡에 약 25초 | CC-BY-NC 4.0 |

### LTX-2.5·FLUX.2 설치 (최신 동영상·이미지 모델, 선택)

LTX-2.5와 FLUX.2 [klein]은 아직 diffusers 정식 버전에 없어 개발판이 필요하므로 **또 다른 가상환경**(`.venv-ltx`)을 씁니다.
LTX-2.5와 FLUX.2 [klein] 9B는 게이트 모델이라 Hugging Face 로그인과 모델 페이지에서 라이선스 동의가 필요합니다 (저장소마다 따로).

```bash
python -m venv .venv-ltx
.venv-ltx\Scripts\python -m pip install -r requirements-ltx.txt
.venv-ltx\Scripts\hf auth login        # Read 토큰 (https://huggingface.co/settings/tokens)
# https://huggingface.co/Lightricks/LTX-2.5-Diffusers 와 https://huggingface.co/black-forest-labs/FLUX.2-klein-9B 에서 동의
.venv-ltx\Scripts\python tests\test_ltx25.py nf4     # 확인 (처음에는 약 71GB 내려받기 + 4bit 양자화)
```

| 기능 | 모델 | 내려받기 | 걸리는 시간 (RTX 4080 SUPER) | 라이선스 |
|---|---|---|---|---|
| 장면 영상 (5초, 704×1280) | [LTX-2.5 22B](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) **4bit** | 약 71GB (+ 양자화 저장 18GB) | 장면당 약 1분 15초, GPU 최대 12.3GB | LTX-2 Community |
| 〃 | LTX-2.5 22B **8bit** (fp8, RAM에 두고 나눠 올림) | 〃 | 장면당 약 10분 | 〃 |
| 장면 이미지 (768×1344) | [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 약 16GB | 장당 약 7초 (불러오기 약 30초) | Apache 2.0 |
| 〃 | [FLUX.2 [klein] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) **4bit** | 약 35GB | 장당 약 7초 (처음 양자화 포함 불러오기 약 30초) | FLUX 비상업 |

- **양자화**: 22B·9B 모델은 원본(bf16)이 16GB GPU에 들어가지 않아 bitsandbytes 4bit로 줄여 올립니다. 처음 한 번 양자화해서 `workdir/models/`에 저장하고(LTX-2.5 약 2분), 다음부터는 저장본을 바로 불러옵니다 (약 20초).
- **LTX-2.5 설정**은 ComfyUI 공식 워크플로를 따릅니다: distilled, 절반 해상도에서 8단계 → 잠재 공간 2배 업스케일 → 3단계, 121프레임 24fps, 가이던스 1.0.
- 사이드바 「🛠️ 세부 설정 → 💻 컴퓨터·AI 모델」 팝업의 「AI 이미지·동영상 모델」 탭에서 고릅니다. 이 설정은 컴퓨터마다 다른 값이라 프리셋이 아니라 `workdir/device.json`에 저장됩니다. 바꾸면 새로 만드는 장면부터 적용됩니다.

생성 전에 Ollama 모델을 GPU에서 내리고(다음 요청 때 다시 올라옴), 결과는 `workdir/gen_cache/`에 저장해 같은 요청은 다시 만들지 않습니다.
생성 프로세스는 한 번 켜면 모델을 올려 둔 채 다음 장면에 다시 쓰고, 90초 동안 쓰지 않으면 꺼서 GPU 메모리를 돌려줍니다.

**🚦 공용 작업 큐**: 영상 분석, 구성·나레이션 다시 만들기, 쇼츠 만들기, AI 장면·배경음악, 목소리 미리 듣기는 버튼을 누르면 **공용 작업 큐**에 들어가 한 번에 하나씩 차례로 처리됩니다. 여러 사람이 접속해도 모두 같은 큐를 써서 GPU를 나눠 쓰지 않습니다. 화면 위쪽 「🚦 작업 큐」에 진행 중인 작업(단계·진행 막대)과 대기 순서·예상 시작 시각이 모든 사람에게 보이고, 자기 작업만 ⬆⬇로 순서를 바꾸거나 🗑로 빼고 ⏹로 취소하고 🔁로 다시 시도할 수 있습니다. 큐는 `workdir/gen_queue.json`에 저장되어 창을 닫거나 앱을 다시 켜도 이어서 처리합니다.

**📈 실시간 사용량**: 「💻 컴퓨터·AI 모델」 팝업에서 CPU·RAM·GPU·GPU 메모리 사용량을 1초마다 그래프와 프로그램별 표로 볼 수 있습니다 (계산 장치를 고를 때 참고). 사이드바에도 한 줄 요약이 나옵니다.

## 사용 방법

화면 위에서 **「📼 영상으로 만들기」**와 **「✨ 영상 없이 AI로 만들기」** 중 하나를 고릅니다.

### ✨ 영상 없이 AI로 만들기

1. 「어떤 쇼츠를 만들까요?」에 주제를 쓰고, 장면 만드는 방식(AI 이미지 + 움직임 / AI 동영상 / 섞기)을 고릅니다.
2. **1단계 · AI 대본 + 장면 구성**: 장면마다 나레이션과 그림 설명이 카드로 나옵니다. 설명을 고치고 「🎬 모두 큐에 넣기」로 장면을 한꺼번에 만듭니다. 시작 화면의 「🤖 사용하는 AI 모델」에서 단계마다 어떤 모델을 쓰는지 볼 수 있습니다.
3. **3단계**는 아래 영상 모드와 같습니다. AI가 쓴 대본은 사실과 다를 수 있으니 꼭 확인하세요.

### 📼 영상으로 만들기


1. **업로드**: 영상 파일을 **화면 아무 곳에나 끌어다 놓거나** Upload 버튼으로 올리고, 분석할 구간을 고릅니다 (15분 이하, MP4·MOV·WEBM·MKV·M4V·AVI). 작업은 자동으로 저장되어, 새로고침해도 「📂 최근 작업」에서 이어서 할 수 있습니다. 같은 영상을 다시 올려도 이전 작업을 불러옵니다.
2. **1단계 · 영상 분석 + 쇼츠 구성**: 대사 인식, 장면 설명 후 AI가 구간과 나레이션을 정합니다. 「AI가 이해한 내용」을 펼치면 타임라인을 볼 수 있습니다. 사이드바 「쇼츠 개수」를 2 이상으로 하면 긴 영상을 서로 다른 주제로 나눠 **쇼츠 여러 편**을 한 번에 구성합니다 (편마다 탭).
3. **2단계 · 확인·수정**: 구간마다 카드가 나옵니다. 카드에서 구간을 재생해 보고, 시작·끝 시간과 나레이션을 고치고, ⬆⬇로 순서를 바꾸거나 🗑로 빼고, ➕로 구간을 추가합니다. 카드 위 타임라인에서 쇼츠 순서와 원본 어디를 골랐는지(목소리·컷 전환 위치와 함께) 한눈에 보고, 「📱 세로 화면 미리보기」로 카드가 쇼츠에서 어떻게 보일지 확인합니다. 화면 위 단계 표시줄이 지금 단계를 알려 줍니다. 고친 것은 「↩️ 되돌리기」로 한 단계씩 되돌립니다.
   - 🔄 **구성 다시 만들기**: 구간과 나레이션을 처음부터 다시
   - ✍️ **나레이션만 다시 쓰기**: 카드의 구간은 그대로 두고 나레이션만 새로
   - ✨ **AI 장면 추가**: 원본에 없는 장면을 AI로 만들어 넣습니다. 장면 설명을 한글로 쓰고 「➕ 큐에 넣기」로 만듭니다 (AI 생성 기능 설치 필요).
4. **3단계 · 나레이션 + 쇼츠 만들기**: 만든 쇼츠는 「결과 기록」에 최신 순으로 쌓입니다. 설정을 바꿔 여러 번 만들고 나란히 비교한 뒤 마음에 드는 것을 다운로드하세요. 결과마다 설정 요약이 붙습니다. 결과마다 **SRT 자막 파일**과 **썸네일 이미지**(가장 좋은 장면 + 큰 제목)도 함께 만들어져 따로 다운로드할 수 있습니다. 「📣 업로드 문구」를 누르면 업로드 제목 후보 3개·설명·해시태그를 써 주고 📋로 복사할 수 있습니다.

**설정**은 사이드바에서 합니다. 맨 위 **💾 프리셋** 아래 「🛠️ 세부 설정」의 버튼을 누르면 분류별 **팝업 창**이 뜨고, 버튼 아래에는 지금 설정이 한 줄로 요약됩니다.

| 버튼 | 팝업에서 바꾸는 것 |
|---|---|
| 🧠 대본·구성 | LLM, 나레이션 말투(정보 전달 / 유쾌한 리뷰 / 감성 스토리), 쇼츠 길이, 쇼츠 개수 |
| 🎙️ 나레이션 목소리 | Edge TTS 목소리·속도, Chatterbox 목소리 보관함·감정 강도·미리 듣기 |
| 🔊 소리·배경음악 | 원본 소리 크기, **배경음악**(올린 음악이나 **AI로 만든 음악**을 보관함에서 골라 쓰기, 반복·페이드·나레이션 중 줄이기) |
| 🎬 화면·자막 | **출력 크기**(1080×1920 9:16 / 480×640 3:4, 자막은 같은 비율로 줄어듦), 세로 변환 방식(블러 배경 / 가운데 크롭 / **피사체 따라 크롭**), 제목 표시·**제목 모양**(크고 굵게 / 반투명 상자 / 위쪽 띠), 자막 스타일(기본 / **단어 강조**), **글꼴**(이 컴퓨터의 한글 글꼴 또는 올린 글꼴), 자막 크기·위치·색. 오른쪽에 **미리보기**(앱 UI가 가리는 곳 표시) |
| 💻 컴퓨터·AI 모델 | 컴퓨터 사양, 계산 장치(작업별 GPU/CPU), 장치에 맞는 추천 모델, AI 이미지·동영상 모델, LTX-2.5 양자화. 추천 LLM이 없으면 「⬇️ 설치」 버튼 |

배경음악은 권리가 있는 음악만 쓰세요.

**📂 최근 작업**에서 작업을 다시 열거나 🗑로 지울 수 있습니다 (확인 팝업 뒤 폴더째 삭제, 되돌릴 수 없음). 작업 화면의 「📦」로 작업 하나를 zip으로 내보내고, 「📦 작업 가져오기」로 다른 PC에서 이어서 작업할 수 있습니다. 「🧹 저장 공간 보기」에서 작업·캐시·모델이 차지하는 크기를 보고 캐시를 지웁니다.

### 목소리 보관함 (제로샷 음성 복제)

음성 엔진에서 Chatterbox를 고르면 「복제할 목소리」에서 보관함의 목소리를 고를 수 있습니다. Chatterbox가 목소리 샘플(5~15초)을 듣고 그 목소리로 나레이션을 읽습니다.

- **파일**: 목소리 파일을 올립니다.
- **녹음**: 브라우저에서 바로 녹음합니다 (읽을 문장을 보여 줍니다).
- **가상 목소리**: Edge TTS 목소리의 음높이와 음색을 바꿔 실존 인물이 아닌 새 목소리를 만듭니다.
- 파일·녹음은 **본인 목소리이거나 목소리 주인의 동의를 받았을 때만** 저장됩니다 (확인란).
- 「▶ 미리 듣기」로 예시 문장을 들어 보고 고르세요.

### 설정 프리셋

자주 쓰는 설정 조합을 사이드바 맨 위 **💾 프리셋**에 저장해 두고 불러올 수 있습니다.

- **불러오기**: 목록에서 고르고 「불러오기」. 그 뒤로 설정을 바꾸면 「현재: 이름 (수정됨)」으로 표시됩니다.
- **저장**: 「현재 설정을 프리셋으로 저장」에 이름을 넣고 「저장」. 같은 이름이 있으면 덮어씁니다.
- **삭제**: 목록에서 고르고 「삭제」.
- ⭐ 표시는 기본 제공 프리셋입니다(기본, 감성 브이로그, 정보 요약, 유쾌한 리뷰). 덮어쓰거나 지울 수 없습니다.
- 사용자 프리셋은 `workdir/presets/<이름>.json`에 저장됩니다. 파일을 복사하면 다른 PC로 옮길 수 있습니다.
- 목소리 샘플 파일은 프리셋에 저장되지 않습니다.
- 설치되지 않은 LLM이 들어 있는 프리셋을 불러오면 지금 모델을 그대로 둡니다.

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
  presets.py            설정 프리셋 저장·불러오기·검증
  project.py            작업 저장·불러오기 (state.json)
  music.py              배경음악 보관함, 반복·페이드
  reframe.py            피사체 따라 크롭 (얼굴·saliency·움직임)
  extras.py             SRT 자막, 썸네일 이미지
  voices.py             목소리 보관함 (올리기·녹음·가상 목소리)
  generate.py           AI 장면·음악 생성 (켜 둔 생성 프로세스에 요청)
  jobqueue.py           공용 작업 큐 (모든 접속자가 같은 큐, 한 번에 하나, 저장·순서·취소·예상 시간)
  tasks.py              큐가 처리하는 작업 (분석, 구성, 나레이션, 쇼츠 만들기, AI 장면·음악, 목소리 미리 듣기, 업로드 문구)
  storage.py            저장 공간 (항목별 크기, 다시 만들 수 있는 캐시 지우기)
  fonts.py              자막 글꼴 (이 컴퓨터의 한글 글꼴 + 올린 글꼴 보관함, 가변 글꼴을 굵게 고정)
  system.py             컴퓨터 사양 확인, 계산 장치(CPU/GPU)별 추천 모델
gen/
  worker.py             생성 작업 프로세스 (.venv-gen에서 실행)
ui/
  dropzone.css, .js     드래그 앤 드롭 업로드 (큰 업로드 칸, 화면 어디에 놓아도 업로드)
tests/
  verify_pipeline.py    전체 파이프라인 검증 (3가지 조건, 수치 판정)
  test_presets_ui.py    프리셋 화면 테스트 (Streamlit AppTest)
  test_ui_browser.py    실제 브라우저(Edge) 화면 테스트: 드래그 업로드, 1~3단계 전체
  test_queue_single_runner.py  공용 큐: 모듈을 다시 불러와도·서버가 둘이어도 한 번에 한 작업만
  test_ai_intent.py     AI로 만든 쇼츠가 주제 의도와 맞는지 (비전 LLM 판정, 출력 크기)
  test_smart_crop.py    피사체 따라 크롭 테스트 (움직이는 피사체, 흔들림)
docs/test_reports/      검증 기록
workdir/                업로드 영상과 중간 결과물 (git 제외)
```

## 검증

```bash
.venv\Scripts\python tests\verify_pipeline.py    # 전체 파이프라인 (GPU, Ollama, 인터넷 필요, 약 3분)
.venv\Scripts\python tests\test_presets_ui.py    # 프리셋 화면 (몇 초)

# 실제 브라우저 테스트 (개발용 패키지 필요: pip install -r requirements-dev.txt, Microsoft Edge 사용)
.venv\Scripts\python tests\test_ui_browser.py --quick   # 드래그 업로드만 (몇십 초)
.venv\Scripts\python tests\test_ui_browser.py           # 업로드부터 다운로드까지 (약 2분)
.venv\Scripts\python tests\test_queue_single_runner.py   # 공용 큐가 한 번에 한 작업만 처리하는지 (GPU 불필요, 십여 초)
.venv\Scripts\python tests\test_ai_intent.py 결과폴더 --size 480x640   # AI 쇼츠가 주제 의도와 맞는지 (GPU, 약 12분)
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

## 버전과 릴리즈

[Semantic Versioning](https://semver.org/lang/ko/)을 따릅니다. 버전은 `MAJOR.MINOR.PATCH`이고, [Releases](https://github.com/yeop88/test_video_maker_streamlit/releases)와 Git 태그 `vX.Y.Z`로 배포합니다. 변경 내용은 [CHANGELOG.md](CHANGELOG.md)에 있습니다.

이 프로젝트에서 "공개 인터페이스"는 다음을 말합니다.
- 화면의 설정 항목
- 프리셋 파일 형식(`workdir/presets/*.json`)
- 실행 방법과 요구 환경(Python, GPU, Ollama, `requirements.txt`)

| 올리는 자리 | 언제 |
|---|---|
| **MAJOR** | 기존 프리셋을 못 읽게 되거나, 설정 항목을 없애거나 뜻을 바꾸거나, 요구 환경이 바뀌어 다시 설치해야 할 때 |
| **MINOR** | 이전과 호환되는 기능 추가 (새 설정, 새 프리셋, 새 처리 단계, 품질 개선) |
| **PATCH** | 호환되는 버그 수정, 문서·테스트만 바꾼 경우 |

**1.0.0 전(0.y.z)** 에는 아직 형태가 자리 잡는 중이라, 호환되지 않는 변경도 MINOR를 올리고 CHANGELOG에 **Breaking**으로 표시합니다.

## 라이선스

[MIT](LICENSE). 사용하는 모델과 서비스(Whisper, Chatterbox, Silero VAD, Ollama 모델, Edge TTS)는 각자의 라이선스·이용 약관을 따릅니다.
