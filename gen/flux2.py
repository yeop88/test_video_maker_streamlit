"""FLUX.2 [klein] 이미지 생성. gen/worker.py가 .venv-ltx(diffusers 개발판)에서 불러 쓴다.

- 4B (Apache 2.0): 트랜스포머 bf16 약 8GB + 텍스트 인코더 Qwen3-4B 약 8GB. 번갈아 GPU에 올리면 16GB에서 돈다.
- 9B (비상업 라이선스, 게이트): 트랜스포머 약 18GB + Qwen3-8B 약 16GB (bf16 29GB 필요).
  16GB GPU에 맞추려고 둘 다 bitsandbytes 4bit(NF4)로 양자화하고 models_dir에 저장해 다음부터 바로 불러온다.
설정은 모델 카드 권장값: distilled 4단계, guidance 1.0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

BF16 = torch.bfloat16


def _nf4(cls, source: str, subfolder: str, saved: Path, note, library: str):
    """구성 요소 하나를 4bit로 불러온다. 처음에는 원본을 양자화해 saved에 저장하고, 다음부터는 저장한 것을 바로 읽는다."""
    if (saved / "config.json").exists():
        return cls.from_pretrained(saved, **({"dtype": BF16} if library == "transformers" else {"torch_dtype": BF16}))
    note(f"{subfolder}를 4bit로 양자화하는 중 (처음 한 번)")
    if library == "transformers":
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=BF16)
        model = cls.from_pretrained(source, subfolder=subfolder, dtype=BF16, quantization_config=config)
    else:
        from diffusers import BitsAndBytesConfig

        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=BF16)
        model = cls.from_pretrained(source, subfolder=subfolder, torch_dtype=BF16, quantization_config=config)
    model.save_pretrained(saved)
    return model


class Flux2:
    def __init__(self, repo: str, quant: str, models_dir: str, note):
        from diffusers import Flux2KleinPipeline

        started = time.time()
        kwargs = {}
        if quant == "nf4":
            import transformers
            from diffusers import Flux2Transformer2DModel
            from huggingface_hub import hf_hub_download

            index = json.loads(Path(hf_hub_download(repo, "model_index.json")).read_text(encoding="utf-8"))
            text_cls = getattr(transformers, index["text_encoder"][1])
            saved = Path(models_dir) / (repo.split("/")[-1].lower() + "_nf4")
            # 둘을 한꺼번에 GPU에 두면 넘치므로 하나씩 양자화하고 CPU로 내린다 (쓸 때 번갈아 올림)
            kwargs["text_encoder"] = _nf4(text_cls, repo, "text_encoder", saved / "text_encoder", note, "transformers").to("cpu")
            torch.cuda.empty_cache()
            kwargs["transformer"] = _nf4(Flux2Transformer2DModel, repo, "transformer", saved / "transformer", note,
                                         "diffusers").to("cpu")
            torch.cuda.empty_cache()
        note(f"{repo.split('/')[-1]} 불러오는 중")
        pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=BF16, **kwargs)
        pipe.enable_model_cpu_offload()  # 텍스트 인코더 → 트랜스포머 → VAE 순서로 하나씩 GPU에
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        note(f"{repo.split('/')[-1]} 준비 완료 ({time.time() - started:.0f}초)")

    def run(self, task: dict, on_step) -> None:
        steps = int(task.get("steps", 4))

        def callback(pipeline, i, timestep, kwargs):
            on_step(i + 1, steps)
            return kwargs

        image = self.pipe(
            prompt=task["prompt"], width=int(task.get("width", 768)), height=int(task.get("height", 1344)),
            guidance_scale=float(task.get("guidance", 1.0)), num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(int(task.get("seed", 0))),
            callback_on_step_end=callback,
        ).images[0]
        image.save(task["out"])
