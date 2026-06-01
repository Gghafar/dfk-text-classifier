"""
DFK Text Classifier — ZeroGPU Edition
Menggunakan HuggingFace ZeroGPU (A100 gratis, tidak butuh kartu kredit).
Model di-download dan di-patch saat startup, inference via @spaces.GPU.
"""

import json
import os
import time

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID  = "aitf-komdigi/KomdigiITS-8B-DFK-TextClassification"
HF_TOKEN  = os.environ.get("HF_TOKEN")
LOCAL_DIR = "/tmp/dfk_model"

SYSTEM_PROMPT = (
    "Anda adalah sistem klasifikasi konten yang mendeteksi disinformasi, fitnah, "
    "dan ujaran kebencian dalam teks bahasa Indonesia. "
    "Klasifikasikan teks ke dalam satu dari lima kategori: "
    "Fakta, Disinformasi, Fitnah, Ujaran Kebencian, Non-DFK. "
    "Jawab HANYA dengan nama kategori, tanpa penjelasan."
)

LABEL_COLORS = {
    "Fakta":            "#22c55e",
    "Disinformasi":     "#ef4444",
    "Fitnah":           "#f97316",
    "Ujaran Kebencian": "#dc2626",
    "Non-DFK":          "#6b7280",
}

EXAMPLES = [
    ["Pemerintah Indonesia berhasil menurunkan angka kemiskinan menjadi 9% pada 2024."],
    ["Vaksin COVID-19 mengandung chip 5G yang bisa dikendalikan dari jarak jauh."],
    ["Si A adalah koruptor meskipun belum terbukti di pengadilan."],
    ["Semua orang dari suku X itu malas dan tidak bisa dipercaya."],
    ["Hari ini cuaca di Jakarta cukup panas dengan suhu 32 derajat Celsius."],
]

# ── patch configs ───────────────────────────────────────────────────────────

def patch_configs(model_dir: str) -> None:
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        changed = False
        if cfg.get("model_type") in ("mistral3", "ministral3"):
            cfg["model_type"]    = "mistral"
            cfg["architectures"] = ["MistralForCausalLM"]
            changed = True
            print("Patched config.json: model_type -> mistral")
        if "generation_config" in cfg:
            del cfg["generation_config"]
            changed = True
            print("Patched config.json: removed nested generation_config")
        if changed:
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)

    tok_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            tok = json.load(f)
        if tok.get("tokenizer_class") == "TokenizersBackend":
            tok["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(tok_path, "w") as f:
                json.dump(tok, f, indent=2)
            print("Patched tokenizer_config.json")

# ── download model at startup (CPU, no GPU needed) ──────────────────────────

print("Checking model cache ...")
if not os.path.exists(os.path.join(LOCAL_DIR, "config.json")):
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL_ID} (~35 GB, sekali saja) ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    print("Download selesai.")

patch_configs(LOCAL_DIR)

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── model: lazy load di dalam @spaces.GPU ──────────────────────────────────
model = None

def _load_model_to_gpu():
    global model
    if model is not None:
        return
    print("Loading model ke GPU (4-bit NF4) ...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model siap di GPU.")

# ── inference ────────────────────────────────────────────────────────────────

@spaces.GPU(duration=120)
def classify_text(text: str):
    if not text.strip():
        return "—", "", ""

    _load_model_to_gpu()

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tok = out[0][enc["input_ids"].shape[-1]:]
    raw     = tokenizer.decode(new_tok, skip_special_tokens=True).strip()
    label   = raw.split("\n")[0].split("<|im_end|>")[0].strip()

    color = LABEL_COLORS.get(label, "#6b7280")
    badge = (
        f'<div style="padding:12px 24px;border-radius:8px;'
        f'background:{color}22;border:1px solid {color}66;display:inline-block">'
        f'<span style="color:{color};font-weight:600;font-size:1.15em">{label}</span></div>'
    )
    status = f"✓ {elapsed:.1f}s · ZeroGPU A100"
    return label, badge, status

# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="DFK Text Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # DFK Text Classifier
        Deteksi **Disinformasi, Fitnah, dan Kebencian** dalam teks bahasa Indonesia.

        Model: [`aitf-komdigi/KomdigiITS-8B-DFK-TextClassification`](https://huggingface.co/aitf-komdigi/KomdigiITS-8B-DFK-TextClassification)
        · Backend: **ZeroGPU (A100)**

        | Label | Keterangan |
        |---|---|
        | **Fakta** | Informasi benar dan dapat diverifikasi |
        | **Disinformasi** | Informasi menyesatkan atau salah |
        | **Fitnah** | Tuduhan tanpa dasar |
        | **Ujaran Kebencian** | Konten menarget kelompok tertentu |
        | **Non-DFK** | Konten netral |

        > Permintaan pertama membutuhkan ~60 detik (model loading ke GPU).
        > Permintaan berikutnya jauh lebih cepat.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Teks yang akan diklasifikasikan",
                placeholder="Masukkan teks bahasa Indonesia ...",
                lines=5,
            )
            with gr.Row():
                submit_btn = gr.Button("Klasifikasikan", variant="primary")
                clear_btn  = gr.Button("Bersihkan", variant="secondary")

        with gr.Column(scale=1):
            label_out  = gr.Textbox(label="Label", interactive=False)
            badge_html = gr.HTML(label="Hasil")
            status_out = gr.Textbox(label="Status", interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Contoh teks")

    submit_btn.click(classify_text, inputs=text_input, outputs=[label_out, badge_html, status_out])
    text_input.submit(classify_text, inputs=text_input, outputs=[label_out, badge_html, status_out])
    clear_btn.click(lambda: ("", "—", "", ""), outputs=[text_input, label_out, badge_html, status_out])

if __name__ == "__main__":
    demo.launch()
