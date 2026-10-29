"""AI 생성 작업 프로세스. 앱 본체와 다른 가상환경(.venv-gen)에서 실행한다.

chatterbox-tts가 diffusers==0.29.0에 고정되어 있고 그 버전은 앱의 transformers 5.x와 함께 쓸 수 없어서,
이미지·영상·음악 생성은 이 스크립트를 별도 프로세스로 돌린다 (끝나면 GPU 메모리도 바로 돌려받음).

사용법: .venv-gen\\Scripts\\python gen\\worker.py <작업.json>
작업 파일: {"kind": "image" | "video" | "music", "tasks": [{...}, ...]}
  image: {prompt, negative, seed, width, height, steps, out}
  video: {prompt, negative, seed, width, height, frames, fps, steps, out}
  music: {prompt, seconds, out}
진행 상황은 표준 출력에 "PROGRESS 완료수 전체수" 줄로 알린다.
"""
import json
import sys
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # 사내 프록시 환경에서도 모델 다운로드가 되도록
import torch

IMAGE_MODEL = "stabilityai/sdxl-turbo"
VIDEO_MODEL = "Lightricks/LTX-Video"
MUSIC_MODEL = "facebook/musicgen-small"


def progress(done: int, total: int) -> None:
    print(f"PROGRESS {done} {total}", flush=True)


def run_image(tasks: list[dict]) -> None:
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for i, task in enumerate(tasks):
        image = pipe(
            prompt=task["prompt"], negative_prompt=task.get("negative") or None,
            width=task.get("width", 576), height=task.get("height", 1024),
            num_inference_steps=task.get("steps", 4), guidance_scale=0.0,  # Turbo는 가이던스 없이 1~4단계
            generator=torch.Generator("cuda").manual_seed(int(task.get("seed", 0))),
        ).images[0]
        image.save(task["out"])
        progress(i + 1, len(tasks))


def run_video(tasks: list[dict]) -> None:
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video

    pipe = LTXPipeline.from_pretrained(VIDEO_MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()  # 텍스트 인코더(T5)와 본체를 번갈아 GPU에 올려 16GB 안에서 돌린다
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    for i, task in enumerate(tasks):
        frames = pipe(
            prompt=task["prompt"],
            negative_prompt=task.get("negative") or "worst quality, inconsistent motion, blurry, jittery, distorted",
            width=task.get("width", 512), height=task.get("height", 768),
            num_frames=task.get("frames", 97), num_inference_steps=task.get("steps", 30), guidance_scale=3.0,
            generator=torch.Generator("cuda").manual_seed(int(task.get("seed", 0))),
        ).frames[0]
        export_to_video(frames, task["out"], fps=task.get("fps", 24))
        progress(i + 1, len(tasks))


def run_music(tasks: list[dict]) -> None:
    import numpy as np
    import scipy.io.wavfile
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MUSIC_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(MUSIC_MODEL, torch_dtype=torch.float16).to("cuda")
    rate = model.config.audio_encoder.sampling_rate
    for i, task in enumerate(tasks):
        inputs = processor(text=[task["prompt"]], padding=True, return_tensors="pt").to("cuda")
        tokens = int(min(30.0, float(task.get("seconds", 20))) * 50)  # 초당 50 토큰, 최대 30초
        audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=tokens)
        wav = audio[0, 0].float().cpu().numpy()
        wav = wav / max(1e-6, float(np.abs(wav).max())) * 0.9
        scipy.io.wavfile.write(task["out"], rate, (wav * 32767).astype(np.int16))
        progress(i + 1, len(tasks))


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    started = time.time()
    {"image": run_image, "video": run_video, "music": run_music}[spec["kind"]](spec["tasks"])
    print(f"DONE {time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
