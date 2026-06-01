# ============================================================
# DFK Model → GGUF Converter
# Jalankan cell per cell di Google Colab (GPU/CPU tidak masalah)
# Runtime: ~30-45 menit (tergantung internet + CPU)
# Disk yang dibutuhkan: ~60 GB
# ============================================================

# ── CELL 1: Install semua dependency ──────────────────────────────────────────

# %% [cell 1]
import subprocess, sys

def run(cmd, **kwargs):
    """Jalankan shell command dengan output langsung."""
    result = subprocess.run(cmd, shell=True, text=True,
                            capture_output=False, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command gagal (exit {result.returncode}): {cmd}")
    return result

# Install dependency utama
run(f"{sys.executable} -m pip install -q huggingface_hub transformers sentencepiece gguf numpy")
print("✓ Dependencies installed")


# ── CELL 2: Clone llama.cpp & install requirements ───────────────────────────

# %% [cell 2]
import os

LLAMA_DIR  = "/content/llama.cpp"
MODEL_DIR  = "/content/dfk_model"
OUTPUT_F16 = "/content/model-f16.gguf"
OUTPUT_Q4  = "/content/model-q4_k_m.gguf"

if not os.path.exists(LLAMA_DIR):
    print("Cloning llama.cpp ...")
    run(f"git clone --depth=1 https://github.com/ggerganov/llama.cpp {LLAMA_DIR}")
    run(f"{sys.executable} -m pip install -q -r {LLAMA_DIR}/requirements.txt")
    print("✓ llama.cpp ready")
else:
    print("✓ llama.cpp already cloned")


# ── CELL 3: Download model dari HuggingFace ───────────────────────────────────

# %% [cell 3]
from huggingface_hub import snapshot_download

HF_TOKEN  = ""  # isi token HF kamu jika model private
MODEL_ID  = "aitf-komdigi/KomdigiITS-8B-DFK-TextClassification"

if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print(f"Downloading {MODEL_ID} (~35 GB) ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        token=HF_TOKEN or None,
    )
    print("✓ Download selesai")
else:
    print(f"✓ Model sudah ada di {MODEL_DIR}")

# Verifikasi file penting ada
for f in ["config.json", "tokenizer_config.json"]:
    path = os.path.join(MODEL_DIR, f)
    assert os.path.exists(path), f"File hilang: {path}"
print(f"Model files: {os.listdir(MODEL_DIR)}")


# ── CELL 4: Patch config.json & tokenizer_config.json ─────────────────────────

# %% [cell 4]
import json as jsonlib

# Patch config.json
config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path) as f:
    cfg = jsonlib.load(f)

changed = False
if cfg.get("model_type") in ("mistral3", "ministral3"):
    cfg["model_type"]    = "mistral"
    cfg["architectures"] = ["MistralForCausalLM"]
    changed = True
    print("Patched: model_type -> mistral")
if "generation_config" in cfg:
    del cfg["generation_config"]
    changed = True
    print("Patched: removed nested generation_config")
if changed:
    with open(config_path, "w") as f:
        jsonlib.dump(cfg, f, indent=2)

# Patch tokenizer_config.json
tok_path = os.path.join(MODEL_DIR, "tokenizer_config.json")
with open(tok_path) as f:
    tok = jsonlib.load(f)
if tok.get("tokenizer_class") == "TokenizersBackend":
    tok["tokenizer_class"] = "PreTrainedTokenizerFast"
    with open(tok_path, "w") as f:
        jsonlib.dump(tok, f, indent=2)
    print("Patched: tokenizer_class -> PreTrainedTokenizerFast")

print(f"\nconfig.json model_type: {cfg['model_type']}")
print(f"tokenizer_config.json tokenizer_class: {tok['tokenizer_class']}")
print("✓ Patches applied")


# ── CELL 5: Convert ke GGUF (f16) ────────────────────────────────────────────

# %% [cell 5]
convert_script = os.path.join(LLAMA_DIR, "convert_hf_to_gguf.py")
assert os.path.exists(convert_script), f"Script tidak ditemukan: {convert_script}"

print(f"Converting {MODEL_DIR} → {OUTPUT_F16} ...")
print("(ini membutuhkan ~10-15 menit)\n")

result = subprocess.run(
    [sys.executable, convert_script,
     MODEL_DIR,
     "--outfile", OUTPUT_F16,
     "--outtype", "f16"],
    text=True,
    capture_output=True,
)

# Tampilkan output lengkap untuk debugging
if result.stdout:
    print("STDOUT:", result.stdout[-3000:])
if result.stderr:
    print("STDERR:", result.stderr[-3000:])

if result.returncode != 0:
    print(f"\n❌ Conversion gagal (exit {result.returncode})")
    print("Coba jalankan manual untuk lihat error lengkap:")
    print(f"  !python {convert_script} {MODEL_DIR} --outfile {OUTPUT_F16} --outtype f16")
else:
    size_gb = os.path.getsize(OUTPUT_F16) / 1e9
    print(f"\n✓ f16 GGUF selesai: {OUTPUT_F16} ({size_gb:.1f} GB)")


# ── CELL 6: Build llama-quantize & quantize ke Q4_K_M ────────────────────────

# %% [cell 6]
assert os.path.exists(OUTPUT_F16), f"f16 GGUF belum ada, jalankan cell 5 dulu"

quantize_bin = os.path.join(LLAMA_DIR, "build", "bin", "llama-quantize")

if not os.path.exists(quantize_bin):
    print("Building llama-quantize ...")
    build_dir = os.path.join(LLAMA_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    run(f"cmake -B {build_dir} {LLAMA_DIR} -DLLAMA_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release")
    run(f"cmake --build {build_dir} --target llama-quantize -j$(nproc)")
    print("✓ llama-quantize built")

print(f"Quantizing f16 → Q4_K_M ...")
run(f"{quantize_bin} {OUTPUT_F16} {OUTPUT_Q4} Q4_K_M")

size_gb = os.path.getsize(OUTPUT_Q4) / 1e9
print(f"\n✓ Q4_K_M GGUF selesai: {OUTPUT_Q4} ({size_gb:.1f} GB)")

# Hapus f16 untuk hemat space
os.remove(OUTPUT_F16)
print(f"✓ f16 dihapus untuk menghemat disk")


# ── CELL 7: Upload ke HuggingFace Hub ────────────────────────────────────────

# %% [cell 7]
from huggingface_hub import HfApi

HF_TOKEN      = ""     # ISI token HF kamu
GGUF_REPO     = "ggapar/KomdigiITS-8B-DFK-GGUF"
GGUF_FILENAME = "model-q4_k_m.gguf"

assert HF_TOKEN, "Isi HF_TOKEN dulu!"
assert os.path.exists(OUTPUT_Q4), "GGUF belum dibuat"

api = HfApi(token=HF_TOKEN)

# Buat repo jika belum ada
try:
    api.create_repo(GGUF_REPO, repo_type="model", exist_ok=True)
    print(f"✓ Repo: https://huggingface.co/{GGUF_REPO}")
except Exception as e:
    print(f"Repo note: {e}")

print(f"Uploading {OUTPUT_Q4} ({os.path.getsize(OUTPUT_Q4)/1e9:.1f} GB) ...")
api.upload_file(
    path_or_fileobj=OUTPUT_Q4,
    path_in_repo=GGUF_FILENAME,
    repo_id=GGUF_REPO,
    repo_type="model",
    commit_message="Upload DFK GGUF Q4_K_M",
)
print(f"\n✓ Upload selesai!")
print(f"Model tersedia di: https://huggingface.co/{GGUF_REPO}")
print(f"\nSet HF Space secrets:")
print(f"  GGUF_REPO     = {GGUF_REPO}")
print(f"  GGUF_FILENAME = {GGUF_FILENAME}")
