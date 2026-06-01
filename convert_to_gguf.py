"""
convert_to_gguf.py — Jalankan di mesin LOKAL (sekali saja)

Persyaratan:
  pip install llama-cpp-python huggingface_hub
  pip install gguf  # untuk conversion

Atau gunakan llama.cpp langsung:
  git clone https://github.com/ggerganov/llama.cpp
  pip install -r llama.cpp/requirements.txt

Langkah:
  1. python convert_to_gguf.py
  2. huggingface-cli login
  3. huggingface-cli upload ggapar/KomdigiITS-8B-DFK-GGUF model-q4_k_m.gguf
"""

import os
import subprocess
import sys
from pathlib import Path

HF_MODEL_ID = "aitf-komdigi/KomdigiITS-8B-DFK-TextClassification"
OUTPUT_FILE  = "model-q4_k_m.gguf"
LOCAL_DIR    = "./dfk_model_hf"

def download_model():
    """Download model dari HuggingFace ke folder lokal."""
    print(f"Downloading {HF_MODEL_ID} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to {LOCAL_DIR}")

def patch_config():
    """Patch config.json agar konversi berjalan benar."""
    import json
    config_path = os.path.join(LOCAL_DIR, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    changed = False
    if cfg.get("model_type") in ("mistral3", "ministral3"):
        cfg["model_type"] = "mistral"
        cfg["architectures"] = ["MistralForCausalLM"]
        changed = True
    if "generation_config" in cfg:
        del cfg["generation_config"]
        changed = True
    if changed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("Patched config.json")

def convert_to_gguf():
    """Konversi model ke GGUF Q4_K_M menggunakan llama.cpp."""
    llama_cpp_dir = "./llama.cpp"
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        print("Cloning llama.cpp ...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp"],
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", f"{llama_cpp_dir}/requirements.txt"],
            check=True
        )

    print("Converting to GGUF (float16 base) ...")
    subprocess.run(
        [sys.executable, convert_script, LOCAL_DIR, "--outfile", "model-f16.gguf"],
        check=True
    )

    print("Quantizing to Q4_K_M ...")
    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    if not os.path.exists(quantize_bin):
        print("Building llama.cpp quantize tool ...")
        os.makedirs(f"{llama_cpp_dir}/build", exist_ok=True)
        subprocess.run(
            ["cmake", "-B", f"{llama_cpp_dir}/build", f"{llama_cpp_dir}"],
            check=True
        )
        subprocess.run(
            ["cmake", "--build", f"{llama_cpp_dir}/build", "--target", "llama-quantize", "-j4"],
            check=True
        )

    subprocess.run(
        [quantize_bin, "model-f16.gguf", OUTPUT_FILE, "Q4_K_M"],
        check=True
    )
    os.remove("model-f16.gguf")
    print(f"\nDone! Output: {OUTPUT_FILE}")
    print(f"Size: {Path(OUTPUT_FILE).stat().st_size / 1e9:.1f} GB")
    print("\nSelanjutnya upload ke HuggingFace:")
    print("  huggingface-cli login")
    print(f"  huggingface-cli repo create ggapar/KomdigiITS-8B-DFK-GGUF --type model")
    print(f"  huggingface-cli upload ggapar/KomdigiITS-8B-DFK-GGUF {OUTPUT_FILE}")

if __name__ == "__main__":
    download_model()
    patch_config()
    convert_to_gguf()
