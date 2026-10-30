"""LTX-2.5 (22B) 동영상 생성. gen/worker.py가 LTX-2.5 전용 가상환경(.venv-ltx)에서 불러 쓴다.

설정은 ComfyUI 공식 워크플로(video_ltx2_5_t2v)를 따른다.
- distilled 트랜스포머, 가이던스 없음 (guidance 1.0, STG 0, modality 1.0)
- 2단계: 절반 해상도에서 시그마 8개 → 잠재 공간 x2 업스케일 → 원래 해상도에서 시그마 3개
- 시그마 1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875 / 0.909375, 0.725, 0.421875
- 121프레임(24fps 5초), 프레임 수는 8의 배수 + 1, 가로·세로는 32의 배수

16GB GPU에 22B 트랜스포머(bf16 42GB)와 12B 텍스트 인코더(Gemma 4, 24GB)가 들어가지 않아서 양자화한다.
- nf4: 트랜스포머·텍스트 인코더를 bitsandbytes 4bit로 (약 12GB, 7GB). 둘은 번갈아 GPU에 올린다. 빠름.
- fp8: 트랜스포머를 fp8로 저장해 RAM에 두고 계산할 블록만 GPU로 옮긴다 (그룹 오프로딩). 텍스트 인코더는 4bit. 느리지만 화질이 낫다.
양자화한 가중치는 models_dir에 저장해 두고 다음부터는 그것을 바로 불러온다 (원본을 다시 읽고 양자화하지 않음).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import torch

REPO = "Lightricks/LTX-2.5-Diffusers"
# 쓰는 부분만 내려받는다 (SFT 트랜스포머, 프롬프트 개선 모델, 확산 디코더, LoRA는 받지 않음. 약 71GB)
ALLOW = ["model_index.json", "transformer/*", "text_encoder/*", "tokenizer/*", "connectors/*", "vae/*",
         "audio_vae/*", "vocoder/*", "scheduler/*", "latent_upsampler/*", "duration_head/*", "processor/*",
         "*.json", "*.txt", "*.model"]
# transformer/에는 4개·8개로 나눈 같은 가중치가 둘 다 있고, connectors/에는 한 파일짜리와 나눈 것이 둘 다 있다.
# index.json이 가리키는 쪽(4개, 2개)만 받는다 (중복 약 48GB를 받지 않음).
IGNORE = ["transformer_full/*", "prompt_enhancer/*", "diffusion_decoder/*", "*lora*",
          "transformer/*-of-00008.safetensors", "connectors/diffusion_pytorch_model.safetensors"]
BF16 = torch.bfloat16


def _folder_gb(path: Path) -> float:
    """내려받은 크기. snapshots/는 blobs/를 가리키는 링크라 blobs/만 센다 (받는 중인 .incomplete 포함)."""
    blobs = path / "blobs"
    return sum(f.stat().st_size for f in blobs.iterdir() if f.is_file()) / 1024 ** 3 if blobs.exists() else 0.0


def download(send) -> Path:
    """필요한 파일을 내려받고(이미 있으면 건너뜀) 로컬 경로를 돌려준다. 받는 동안 받은 크기를 알린다."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(HF_HUB_CACHE) / ("models--" + REPO.replace("/", "--"))
    done = threading.Event()

    def report():
        while not done.wait(5):
            send(f"LTX-2.5 내려받는 중 {_folder_gb(cache):.1f}GB / 약 71GB (처음 한 번)")

    threading.Thread(target=report, daemon=True).start()
    try:
        return Path(snapshot_download(REPO, allow_patterns=ALLOW, ignore_patterns=IGNORE))
    finally:
        done.set()


def _text_encoder_class(root: Path):
    """model_index.json에 적힌 텍스트 인코더 클래스 (Gemma 3/4 중 무엇인지)."""
    import transformers

    library, name = json.loads((root / "model_index.json").read_text(encoding="utf-8"))["text_encoder"]
    return getattr(transformers, name)


def _nf4_text_encoder(root: Path, saved: Path, send):
    from transformers import BitsAndBytesConfig

    cls = _text_encoder_class(root)
    if (saved / "config.json").exists():
        return cls.from_pretrained(saved, dtype=BF16)
    send("텍스트 인코더(Gemma 4 12B)를 4bit로 양자화하는 중 (처음 한 번)")
    model = cls.from_pretrained(root / "text_encoder", dtype=BF16, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=BF16))
    model.save_pretrained(saved)
    return model


def _nf4_transformer(root: Path, saved: Path, send):
    from diffusers import BitsAndBytesConfig, LTX2VideoTransformer3DModel

    if (saved / "config.json").exists():
        return LTX2VideoTransformer3DModel.from_pretrained(saved, torch_dtype=BF16)
    send("트랜스포머(22B)를 4bit로 양자화하는 중 (처음 한 번, 몇 분)")
    model = LTX2VideoTransformer3DModel.from_pretrained(
        root / "transformer", torch_dtype=BF16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=BF16))
    model.save_pretrained(saved)
    return model


class LTX25:
    """불러온 LTX-2.5 파이프라인 (2단계 생성)."""

    def __init__(self, quant: str, models_dir: str, send_note):
        from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        started = time.time()
        root = download(send_note)
        saved = Path(models_dir) / "ltx25_nf4"
        # 텍스트 인코더는 두 방식 모두 4bit (12B를 GPU에 올려야 해서). 다 쓰면 CPU로 내려 트랜스포머 자리를 비운다.
        text_encoder = _nf4_text_encoder(root, saved / "text_encoder", send_note).to("cpu")
        torch.cuda.empty_cache()
        if quant == "nf4":
            transformer = _nf4_transformer(root, saved / "transformer", send_note).to("cpu")
        else:
            send_note("트랜스포머(22B)를 RAM에 불러오는 중 (fp8로 저장, 약 42GB를 읽음)")
            from diffusers import LTX2VideoTransformer3DModel

            transformer = LTX2VideoTransformer3DModel.from_pretrained(root / "transformer", torch_dtype=BF16)
        torch.cuda.empty_cache()
        send_note("LTX-2.5 나머지 부분 불러오는 중")
        pipe = LTX2Pipeline.from_pretrained(root, transformer=transformer, text_encoder=text_encoder,
                                            prompt_enhancer=None, processor=None, torch_dtype=BF16)
        if quant == "nf4":
            # 텍스트 인코더 → 커넥터 → 트랜스포머 → VAE 순서로 하나씩 GPU에. 기본 순서에는 커넥터와 트랜스포머 사이에
            # duration_head가 있는데, 프레임 수를 직접 넘기면 쓰이지 않아 커넥터(5.9GB)가 GPU에서 내려가지 않았다
            # (트랜스포머와 합쳐 16GB를 넘어 단계당 70초 → 시스템 RAM으로 넘침). 순서에서 빼서 바로 이어지게 한다.
            pipe.model_cpu_offload_seq = "text_encoder->connectors->transformer->vae->audio_vae->vocoder"
            pipe.enable_model_cpu_offload()
        else:
            transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=BF16)
            # 트랜스포머와 커넥터(5.9GB)는 RAM에 두고 계산할 블록만 GPU로. 커넥터까지 GPU에 두면 텍스트 인코더(4bit 7GB)와
            # 합쳐 16GB가 차서 시스템 RAM으로 넘쳤다 (한 장면에 15분 넘게 걸림).
            for module in (transformer, pipe.connectors):
                module.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                            offload_type="block_level", num_blocks_per_group=2, use_stream=True)
            for name in ("text_encoder", "vae", "audio_vae", "vocoder"):
                getattr(pipe, name).to("cuda")
        pipe.vae.enable_tiling()  # 2단계는 원래 해상도로 디코딩
        pipe.set_progress_bar_config(disable=True)
        upsampler = LTX2LatentUpsamplerModel.from_pretrained(root / "latent_upsampler", torch_dtype=BF16)
        self.pipe = pipe
        self.upsample = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)
        self.quant = quant
        send_note(f"LTX-2.5 준비 완료 ({time.time() - started:.0f}초)")

    def run(self, task: dict, on_step) -> None:
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.utils import export_to_video

        width, height = int(task.get("width", 704)), int(task.get("height", 1280))  # 최종 해상도 (2단계)
        frames, fps = int(task.get("frames", 121)), float(task.get("fps", 24))
        total = len(DISTILLED_SIGMA_VALUES) + len(STAGE_2_DISTILLED_SIGMA_VALUES)
        generator = torch.Generator("cuda").manual_seed(int(task.get("seed", 0)))  # 두 단계가 같은 노이즈 흐름을 이어 씀
        shared = dict(
            prompt=task["prompt"], negative_prompt=task.get("negative") or None, frame_rate=fps,
            guidance_scale=1.0, audio_guidance_scale=1.0, stg_scale=0.0, audio_stg_scale=0.0,
            modality_scale=1.0, audio_modality_scale=1.0, generator=generator, return_dict=False,
        )

        def stage_callback(offset):
            def callback(pipeline, i, timestep, kwargs):
                on_step(offset + i + 1, total)
                return kwargs
            return callback

        latents, audio_latents = self.pipe(
            width=width // 2, height=height // 2, num_frames=frames, sigmas=DISTILLED_SIGMA_VALUES,
            output_type="latent", callback_on_step_end=stage_callback(0), **shared)
        self.upsample.latent_upsampler.to("cuda")
        latents = self.upsample(latents=latents, output_type="latent", return_dict=False)[0]
        self.upsample.latent_upsampler.to("cpu")
        video, _audio = self.pipe(
            num_frames=frames, sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, latents=latents, audio_latents=audio_latents,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], output_type="np",
            callback_on_step_end=stage_callback(len(DISTILLED_SIGMA_VALUES)), **shared)
        # 함께 만들어지는 소리는 쓰지 않는다 (쇼츠에는 나레이션·배경음악을 따로 넣음)
        export_to_video(list(video[0]), task["out"], fps=int(fps))
