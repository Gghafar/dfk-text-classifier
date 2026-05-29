from typing import Any

import modal

APP_NAME = "dfk-text-classifier"
MODEL_ID = "aitf-komdigi/KomdigiITS-8B-DFK-TextClassification"
CACHE_DIR = "/cache/huggingface"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("dfk-classifier-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.9.0",
        "bitsandbytes",
        "accelerate",
        "sentencepiece",
        "mistral_common",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        "peft",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": CACHE_DIR,
            "TRANSFORMERS_CACHE": CACHE_DIR,
        }
    )
)

SYSTEM_PROMPT = (
    "Anda adalah sistem klasifikasi konten yang mendeteksi disinformasi, fitnah, "
    "dan ujaran kebencian dalam teks bahasa Indonesia. "
    "Klasifikasikan teks yang diberikan ke dalam salah satu dari lima kategori berikut:\n"
    "- Fakta\n"
    "- Disinformasi\n"
    "- Fitnah\n"
    "- Ujaran Kebencian\n"
    "- Non-DFK\n"
    "Jawab hanya dengan nama kategori, tanpa penjelasan tambahan."
)


@app.cls(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16 * 1024,
    timeout=600,
    scaledown_window=60,
    volumes={CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
@modal.concurrent(max_inputs=4)
class DFKClassifier:
    @modal.enter()
    def load_model(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            token=token,
            cache_dir=CACHE_DIR,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            token=token,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    @modal.method()
    def classify(
        self,
        text: str,
        max_new_tokens: int = 20,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        import torch

        # Build ChatML prompt manually to match training format
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for consistent classification
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean up: take only the first line as the label
        label = raw_output.split("\n")[0].split("<|im_end|>")[0].strip()

        return {
            "label": label,
            "raw_output": raw_output,
        }


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST", docs=True)
def classify(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "Field 'text' is required and cannot be empty."}

    return DFKClassifier().classify.remote(
        text=text,
        max_new_tokens=int(payload.get("max_new_tokens", 20)),
        temperature=float(payload.get("temperature", 0.1)),
    )


@app.local_entrypoint()
def main(text: str = "Pemerintah telah berhasil menurunkan angka kemiskinan secara signifikan."):
    result = DFKClassifier().classify.remote(text=text)
    print(result)
