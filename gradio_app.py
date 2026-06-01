"""
DFK Text Classifier — GGUF CPU Edition
Output: label + penalaran terstruktur (seperti contoh output asli model).
"""

import os
import re
import time

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

GGUF_REPO     = os.environ.get("GGUF_REPO",     "ggapar/KomdigiITS-8B-DFK-GGUF")
GGUF_FILENAME = os.environ.get("GGUF_FILENAME", "model-q4_k_m.gguf")
HF_TOKEN      = os.environ.get("HF_TOKEN")

SYSTEM_PROMPT = """Anda adalah sistem analisis konten yang mendeteksi disinformasi, fitnah, dan ujaran kebencian dalam teks bahasa Indonesia.

Untuk setiap teks yang diberikan, berikan:
1. Klasifikasi dalam satu dari lima kategori: Fakta, Disinformasi, Fitnah, Ujaran Kebencian, Non-DFK
2. Penalaran terstruktur yang menjelaskan alasan klasifikasi secara rinci

Format output WAJIB (gunakan tepat seperti ini):
[LABEL] {nama kategori}
[REASONING]
{penjelasan terstruktur dengan poin-poin bernama, setiap poin menjelaskan indikator spesifik yang ditemukan dalam teks}"""

LABEL_COLORS = {
    "Fakta":            "#22c55e",
    "Disinformasi":     "#ef4444",
    "Fitnah":           "#f97316",
    "Ujaran Kebencian": "#dc2626",
    "Non-DFK":          "#6b7280",
}

VALID_LABELS = set(LABEL_COLORS.keys())

EXAMPLES = [
    ["Pemerintah Indonesia berhasil menurunkan angka kemiskinan menjadi 9% pada 2024."],
    ["Vaksin COVID-19 mengandung chip 5G yang bisa dikendalikan dari jarak jauh oleh pemerintah asing."],
    ["Si A adalah koruptor yang mencuri miliaran uang rakyat meskipun kasusnya belum diputus pengadilan."],
    ["Semua warga suku X itu malas, tidak jujur, dan tidak layak dipercaya dalam pekerjaan apapun."],
    ["Hari ini cuaca di Jakarta cukup panas dengan suhu mencapai 32 derajat Celsius."],
]

# ── load model ───────────────────────────────────────────────────────────────

print(f"Downloading GGUF: {GGUF_REPO}/{GGUF_FILENAME} ...")
model_path = hf_hub_download(
    repo_id=GGUF_REPO,
    filename=GGUF_FILENAME,
    token=HF_TOKEN,
)
print(f"Loading model dari {model_path} ...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=2,
    n_gpu_layers=0,
    verbose=False,
)
print("Model siap!")

# ── parsing output ────────────────────────────────────────────────────────────

def parse_output(raw: str) -> tuple[str, str]:
    """Extract label dan reasoning dari output model."""
    label     = "—"
    reasoning = raw.strip()

    # Cari [LABEL] ...
    m = re.search(r"\[LABEL\]\s*(.+?)(?:
|$)", raw, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        # Cocokkan dengan label valid (toleran terhadap variasi kapitalisasi)
        for valid in VALID_LABELS:
            if valid.lower() in candidate.lower():
                label = valid
                break
        if label == "—":
            label = candidate  # tetap tampilkan meski tidak persis cocok

    # Cari [REASONING] ...
    m2 = re.search(r"\[REASONING\]\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
    if m2:
        reasoning = m2.group(1).strip()
    elif label != "—":
        # Hapus baris [LABEL] dari reasoning jika tidak ada marker [REASONING]
        reasoning = re.sub(r"\[LABEL\].*?(
|$)", "", raw, flags=re.IGNORECASE).strip()

    return label, reasoning

# ── inference ────────────────────────────────────────────────────────────────

def classify_text(text: str):
    if not text.strip():
        return "—", "", "", ""

    t0 = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text.strip()},
        ],
        max_tokens=768,        # cukup untuk reasoning panjang
        temperature=0.1,       # sedikit variasi agar reasoning lebih natural
        repeat_penalty=1.1,
        stop=["<|im_end|>", "</s>"],
    )
    elapsed = time.time() - t0

    raw            = response["choices"][0]["message"]["content"].strip()
    label, reasoning = parse_output(raw)

    color = LABEL_COLORS.get(label, "#6b7280")
    badge = (
        f'<div style="padding:10px 20px;border-radius:8px;'
        f'background:{color}22;border:1px solid {color}66;display:inline-block;margin-bottom:8px">'
        f'<span style="color:{color};font-weight:600;font-size:1.1em">{label}</span></div>'
    )
    status = f"✓ {elapsed:.1f}s · CPU (GGUF Q4)"
    return label, badge, reasoning, status

# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="DFK Text Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # DFK Text Classifier
        Deteksi dan analisis **Disinformasi, Fitnah, dan Kebencian** dalam teks bahasa Indonesia.

        Model: [`aitf-komdigi/KomdigiITS-8B-DFK-TextClassification`](https://huggingface.co/aitf-komdigi/KomdigiITS-8B-DFK-TextClassification)
        · Backend: **CPU (GGUF Q4_K_M, gratis)**

        | Label | Keterangan |
        |---|---|
        | **Fakta** | Informasi benar dan dapat diverifikasi |
        | **Disinformasi** | Informasi menyesatkan atau salah |
        | **Fitnah** | Tuduhan tanpa dasar yang merusak reputasi |
        | **Ujaran Kebencian** | Konten menarget kelompok tertentu |
        | **Non-DFK** | Konten netral |
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Teks yang akan diklasifikasikan",
                placeholder="Masukkan teks bahasa Indonesia ...",
                lines=7,
            )
            with gr.Row():
                submit_btn = gr.Button("Klasifikasikan", variant="primary", scale=3)
                clear_btn  = gr.Button("Bersihkan", variant="secondary", scale=1)

        with gr.Column(scale=1):
            label_out   = gr.Textbox(label="Label", interactive=False, max_lines=1)
            badge_html  = gr.HTML()
            status_out  = gr.Textbox(label="Status", interactive=False, max_lines=1)

    reasoning_out = gr.Textbox(
        label="Penalaran",
        interactive=False,
        lines=12,
        placeholder="Penalaran model akan muncul di sini setelah klasifikasi ...",
    )

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Contoh teks")

    submit_btn.click(
        classify_text,
        inputs=text_input,
        outputs=[label_out, badge_html, reasoning_out, status_out],
    )
    text_input.submit(
        classify_text,
        inputs=text_input,
        outputs=[label_out, badge_html, reasoning_out, status_out],
    )
    clear_btn.click(
        lambda: ("", "—", "", "", ""),
        outputs=[text_input, label_out, badge_html, reasoning_out, status_out],
    )

if __name__ == "__main__":
    demo.launch()
