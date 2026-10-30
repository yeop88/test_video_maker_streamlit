"""AI 생성 작업 프로세스. 앱 본체와 다른 가상환경(.venv-gen)에서 실행한다.

chatterbox-tts가 diffusers==0.29.0에 고정되어 있고 그 버전은 앱의 transformers 5.x와 함께 쓸 수 없어서,
이미지·영상·음악 생성은 이 스크립트를 별도 프로세스로 돌린다.

사용법: .venv-gen\\Scripts\\python gen\\worker.py --serve [--device gpu|hybrid|cpu]
  --device: gpu = 모델을 GPU에 올림, hybrid = RAM에 두고 쓸 부분만 GPU로 (GPU 메모리 절약), cpu = GPU를 쓰지 않음 (느림)
LTX-2.5 (kind "ltx25")는 .venv-ltx에서 실행한다: .venv-ltx\\Scripts\\python gen\\worker.py --serve --quant nf4|fp8 --models-dir <폴더>
켜 둔 채로 표준 입력에서 작업을 한 줄(JSON)씩 받아 차례로 처리한다. 모델은 올려 둔 채 다음 작업에 다시 쓰고,
다른 종류의 작업이 오면 앞 모델을 내리고 바꾼다 (16GB GPU에 둘을 함께 올리지 않음).
  입력:  {"id": "...", "kind": "image" | "video" | "ltx25" | "flux2" | "music", "task": {...}}
    image: {model, prompt, negative, seed, width, height, steps, guidance, scheduler, out}  (SDXL 계열: SDXL-Turbo, DreamShaper XL)
    flux2: {model, quant, prompt, seed, width, height, steps, guidance, out}  (FLUX.2 [klein], .venv-ltx에서)
    video: {prompt, negative, seed, width, height, frames, fps, steps, out}
    ltx25: {prompt, negative, seed, width, height, frames, fps, out}  (width·height는 최종 해상도)
    music: {prompt, seconds, out}
  출력: "@@" 뒤에 JSON 한 줄 (라이브러리가 표준 출력에 쓰는 글과 섞이지 않게)
    {"event": "ready"} / {"event": "loading", "id", "model"} / {"event": "step", "id", "step", "total"}
    {"event": "done", "id", "seconds"} / {"event": "error", "id", "error"}
"""
import gc
import json
import sys
import time
import traceback

import truststore

truststore.inject_into_ssl()  # 사내 프록시 환경에서도 모델 다운로드가 되도록
import torch

MODELS = {
    "image": "stabilityai/sdxl-turbo",
    "video": "Lightricks/LTX-Video",
    "music": "facebook/musicgen-small",
    "ltx25": "Lightricks/LTX-2.5-Diffusers",
}
DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "gpu"
TORCH_DEVICE = "cpu" if DEVICE == "cpu" else "cuda"
HALF = torch.float32 if DEVICE == "cpu" else torch.float16  # CPU는 fp16 연산이 느리거나 안 됨
QUANT = sys.argv[sys.argv.index("--quant") + 1] if "--quant" in sys.argv else "nf4"
MODELS_DIR = sys.argv[sys.argv.index("--models-dir") + 1] if "--models-dir" in sys.argv else "models"
current_id = ""  # 지금 처리 중인 요청 (모델을 불러오는 동안 진행 상황을 알릴 때 씀)


def send(event: dict) -> None:
    print("@@" + json.dumps(event, ensure_ascii=False), flush=True)


class Loaded:
    """지금 GPU에 올린 모델 하나. (종류, 모델 저장소)가 바뀌면 앞 모델을 내리고 새로 불러온다."""

    def __init__(self):
        self.key = None
        self.model = None

    def get(self, kind: str, request_id: str, task: dict):
        key = (kind, task.get("model"), task.get("quant"))
        if self.key == key:
            return self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        send({"event": "loading", "id": request_id, "model": task.get("model") or MODELS[kind]})
        self.model = LOADERS[kind](task)
        self.key = key
        return self.model


def load_image(task: dict):
    """SDXL 계열 (SDXL-Turbo, DreamShaper XL Lightning/Turbo). 스케줄러는 모델 카드 권장대로."""
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

    pipe = AutoPipelineForText2Image.from_pretrained(task.get("model") or MODELS["image"], torch_dtype=HALF, variant="fp16")
    if task.get("scheduler") == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if DEVICE == "hybrid":
        pipe.enable_model_cpu_offload()  # 텍스트 인코더·UNet·VAE를 번갈아 GPU에
    else:
        pipe.to(TORCH_DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_video(task: dict):
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(MODELS["video"], torch_dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16)
    if DEVICE == "hybrid" or (DEVICE == "gpu" and torch.cuda.get_device_properties(0).total_memory < 24 * 1024 ** 3):
        pipe.enable_model_cpu_offload()  # 텍스트 인코더(T5)와 본체를 번갈아 GPU에 올려 16GB 안에서 돌린다
    else:
        pipe.to(TORCH_DEVICE)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_music(task: dict):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODELS["music"])
    model = MusicgenForConditionalGeneration.from_pretrained(MODELS["music"], torch_dtype=HALF).to(TORCH_DEVICE)
    return processor, model


def _note(text: str) -> None:
    send({"event": "loading", "id": current_id, "model": text})


def load_ltx25(task: dict):
    from ltx25 import LTX25

    return LTX25(QUANT, MODELS_DIR, _note)


def load_flux2(task: dict):
    from flux2 import Flux2

    return Flux2(task["model"], task.get("quant", ""), MODELS_DIR, _note)


LOADERS = {"image": load_image, "video": load_video, "music": load_music, "ltx25": load_ltx25, "flux2": load_flux2}


def run_image(pipe, task: dict, request_id: str) -> None:
    image = pipe(
        prompt=task["prompt"], negative_prompt=task.get("negative") or None,
        width=task.get("width", 576), height=task.get("height", 1024),
        num_inference_steps=task.get("steps", 4),
        guidance_scale=task.get("guidance", 0.0),  # SDXL-Turbo는 가이던스 없이, DreamShaper XL은 2
        generator=torch.Generator(TORCH_DEVICE).manual_seed(int(task.get("seed", 0))),
    ).images[0]
    image.save(task["out"])


def run_video(pipe, task: dict, request_id: str) -> None:
    from diffusers.utils import export_to_video

    steps = task.get("steps", 30)

    def on_step(pipeline, i, timestep, kwargs):
        send({"event": "step", "id": request_id, "step": i + 1, "total": steps})
        return kwargs

    frames = pipe(
        prompt=task["prompt"],
        negative_prompt=task.get("negative") or "worst quality, inconsistent motion, blurry, jittery, distorted",
        width=task.get("width", 512), height=task.get("height", 768),
        num_frames=task.get("frames", 97), num_inference_steps=steps, guidance_scale=3.0,
        generator=torch.Generator(TORCH_DEVICE).manual_seed(int(task.get("seed", 0))),
        callback_on_step_end=on_step,
    ).frames[0]
    export_to_video(frames, task["out"], fps=task.get("fps", 24))


def run_music(loaded, task: dict, request_id: str) -> None:
    import numpy as np
    import scipy.io.wavfile

    processor, model = loaded
    rate = model.config.audio_encoder.sampling_rate
    inputs = processor(text=[task["prompt"]], padding=True, return_tensors="pt").to(TORCH_DEVICE)
    tokens = int(min(30.0, float(task.get("seconds", 20))) * 50)  # 초당 50 토큰, 최대 30초
    audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=tokens)
    wav = audio[0, 0].float().cpu().numpy()
    wav = wav / max(1e-6, float(np.abs(wav).max())) * 0.9
    scipy.io.wavfile.write(task["out"], rate, (wav * 32767).astype(np.int16))


def run_ltx25(model, task: dict, request_id: str) -> None:
    model.run(task, lambda step, total: send({"event": "step", "id": request_id, "step": step, "total": total}))


def run_flux2(model, task: dict, request_id: str) -> None:
    model.run(task, lambda step, total: send({"event": "step", "id": request_id, "step": step, "total": total}))


RUNNERS = {"image": run_image, "video": run_video, "music": run_music, "ltx25": run_ltx25, "flux2": run_flux2}


def serve() -> None:
    global current_id
    loaded = Loaded()
    send({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        request_id = request.get("id", "")
        started = time.time()
        try:
            current_id = request_id
            model = loaded.get(request["kind"], request_id, request["task"])
            RUNNERS[request["kind"]](model, request["task"], request_id)
            send({"event": "done", "id": request_id, "seconds": round(time.time() - started, 1)})
        except Exception:
            send({"event": "error", "id": request_id, "error": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    if "--serve" not in sys.argv:
        sys.exit("사용법: python gen/worker.py --serve")
    serve()
