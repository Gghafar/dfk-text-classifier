import json
import os
from typing import Any

import modal

APP_NAME = "dfk-text-classifier"
MODEL_ID = "aitf-komdigi/KomdigiITS-8B-DFK-TextClassification"
CACHE_DIR = "/cache/huggingface"
MODEL_LOCAL_DIR = "/cache/dfk_model"

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
        "mistral_common>=1.5.2",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        "peft",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": CACHE_DIR,
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


def _patch_configs(local_dir: str):
    """Fix config mismatches caused by the model being saved with an incompatible model_type.

    The model was saved with model_type='mistral3' which in transformers 5.x maps to the
    vision-language Mistral3 model, not the text-only causal LM. Patching to 'mistral'
    loads MistralForCausalLM which has the identical decoder architecture.
    """
    config_path = os.path.join(local_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("model_type") in ("mistral3", "ministral3"):
            cfg["model_type"] = "mistral"
            cfg["architectures"] = ["MistralForCausalLM"]
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print("Patched config.json: model_type=mistral, architectures=[MistralForCausalLM]")

    # tokenizer_config.json: "TokenizersBackend" is not a valid transformers class
    tok_path = os.path.join(local_dir, "tokenizer_config.json")
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            tok = json.load(f)
        if tok.get("tokenizer_class") == "TokenizersBackend":
            tok["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(tok_path, "w") as f:
                json.dump(tok, f, indent=2)
            print("Patched tokenizer_config.json: TokenizersBackend -> PreTrainedTokenizerFast")


@app.cls(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=24 * 1024,
    timeout=900,
    scaledown_window=60,
    volumes={CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=1)
class DFKClassifier:
    @modal.enter(snap=True)
    def load_to_cpu(self):
        """Download model and load to CPU RAM — this state is snapshotted for fast future cold starts."""
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN")

        if not os.path.exists(os.path.join(MODEL_LOCAL_DIR, "config.json")):
            print(f"Downloading {MODEL_ID} ...")
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_LOCAL_DIR,
                token=token,
            )
            try:
                hf_cache.commit()
                print("Volume committed.")
            except Exception as e:
                print(f"Volume commit warning (non-fatal): {e}")

        _patch_configs(MODEL_LOCAL_DIR)

        print("Loading tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_LOCAL_DIR,
            trust_remote_code=True,
        )

        print("Loading model to CPU ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_LOCAL_DIR,
            trust_remote_code=True,
            dtype="bfloat16",
            device_map="cpu",
        )
        self.model.eval()
        print("Model in CPU RAM — snapshot will be taken.")

    @modal.enter(snap=False)
    def move_to_gpu(self):
        """Runs after every snapshot restore — moves model from CPU RAM to GPU VRAM."""
        import torch
        print("Moving model to GPU ...")
        self.model = self.model.to("cuda", dtype=torch.bfloat16)
        print("Model ready on GPU.")

    @modal.method()
    def classify(
        self,
        text: str,
        max_new_tokens: int = 20,
    ) -> dict[str, Any]:
        import torch

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
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        label = raw_output.split("\n")[0].split("<|im_end|>")[0].strip()
        return {"label": label, "raw_output": raw_output}


@app.function(image=image, timeout=900)
@modal.fastapi_endpoint(method="POST", docs=True)
def classify(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "Field 'text' is required and cannot be empty."}
    return DFKClassifier().classify.remote(
        text=text,
        max_new_tokens=int(payload.get("max_new_tokens", 20)),
    )


@app.local_entrypoint()
def main(text: str = "Pemerintah telah berhasil menurunkan angka kemiskinan secara signifikan."):
    result = DFKClassifier().classify.remote(text=text)
    print(result)
