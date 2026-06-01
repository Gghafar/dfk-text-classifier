"""
DFK Text Classifier — GGUF CPU Edition
Berjalan di HF Space CPU Basic (GRATIS, persistent, tidak butuh GPU).
Model: GGUF Q4_K_M ~5GB, inference ~3-5 detik/request.

Setup (sekali):
  1. Konversi model ke GGUF di mesin lokal (lihat convert_to_gguf.py)
  2. Upload GGUF ke HF Hub: huggingface-cli upload <repo> model.gguf
  3. Set GGUF_REPO di Space secrets (default: ggapar/KomdigiITS-8B-DFK-GGUF)
"""

import os
import time

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

GGUF_REPO     = os.environ.get("GGUF_REPO",     "ggapar/KomdigiITS-8B-DFK-GGUF")
GGUF_FILENAME = os.environ.get("GGUF_FILENAME", "model-q4_k_m.gguf")
HF_TOKEN      = os.environ.get("HF_TOKEN")

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

# ── load model ───────────────────────────────────────────────────────────────

print(f"Downloading GGUF from {GGUF_REPO}/{GGUF_FILENAME} ...")
model_path = hf_hub_download(
    repo_id=GGUF_REPO,
    filename=GGUF_FILENAME,
    token=HF_TOKEN,
)
print(f"Model path: {model_path}")

print("Loading model ...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=2,       # HF Space CPU Basic: 2 vCPU
    n_gpu_layers=0,    # CPU only
    verbose=False,
)
print("Model loaded!")

# ── inference ────────────────────────────────────────────────────────────────

def classify_text(text: str):
    if not text.strip():
        return "—", "", ""

    t0 = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text.strip()},
        ],
        max_tokens=20,
        temperature=0.0,
        stop=["\n", "<|im_end|>"],
    )
    elapsed = time.time() - t0

    raw   = response["choices"][0]["message"]["content"].strip()
    label = raw.split("\n")[0].split("<|im_end|>")[0].strip()

    color = LABEL_COLORS.get(label, "#6b7280")
    badge = (
        f'<div style="padding:12px 24px;border-radius:8px;'
        f'background:{color}22;border:1px solid {color}66;display:inline-block">'
        f'<span style="color:{color};font-weight:600;font-size:1.15em">{label}</span></div>'
    )
    status = f"✓ {elapsed:.1f}s · CPU (GGUF Q4)"
    return label, badge, status

# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="DFK Text Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # DFK Text Classifier
        Deteksi **Disinformasi, Fitnah, dan Kebencian** dalam teks bahasa Indonesia.

        Model: [`aitf-komdigi/KomdigiITS-8B-DFK-TextClassification`](https://huggingface.co/aitf-komdigi/KomdigiITS-8B-DFK-TextClassification)
        · Backend: **CPU (GGUF Q4_K_M, gratis)**

        | Label | Keterangan |
        |---|---|
        | **Fakta** | Informasi benar dan dapat diverifikasi |
        | **Disinformasi** | Informasi menyesatkan atau salah |
        | **Fitnah** | Tuduhan tanpa dasar |
        | **Ujaran Kebencian** | Konten menarget kelompok tertentu |
        | **Non-DFK** | Konten netral |
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
