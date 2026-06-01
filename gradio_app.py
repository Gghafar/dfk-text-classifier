import os
import time

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

GGUF_REPO     = os.environ.get("GGUF_REPO",     "ggapar/KomdigiITS-8B-DFK-GGUF")
GGUF_FILENAME = os.environ.get("GGUF_FILENAME", "model-q4_k_m.gguf")
HF_TOKEN      = os.environ.get("HF_TOKEN")

PROMPT_FAST = (
    "Anda adalah sistem klasifikasi konten bahasa Indonesia. "
    "Klasifikasikan teks ke dalam satu dari lima kategori: "
    "Fakta, Disinformasi, Fitnah, Ujaran Kebencian, Non-DFK. "
    "Jawab HANYA dengan nama kategori, tanpa penjelasan."
)

PROMPT_FULL = (
    "Anda adalah sistem analisis konten yang mendeteksi disinformasi, fitnah, "
    "dan ujaran kebencian dalam teks bahasa Indonesia. "
    "Untuk setiap teks, berikan:\n"
    "1. Klasifikasi: Fakta, Disinformasi, Fitnah, Ujaran Kebencian, atau Non-DFK\n"
    "2. Penalaran singkat dan terstruktur (maksimal 3-4 poin)\n\n"
    "Format output WAJIB:\n"
    "[LABEL] {nama kategori}\n"
    "[REASONING]\n"
    "{poin-poin penalaran}"
)

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
    ["Vaksin COVID-19 mengandung chip 5G yang bisa dikendalikan dari jarak jauh."],
    ["Si A adalah koruptor yang mencuri miliaran uang rakyat meskipun belum terbukti."],
    ["Semua warga suku X itu malas dan tidak bisa dipercaya dalam pekerjaan apapun."],
    ["Hari ini cuaca di Jakarta cukup panas dengan suhu mencapai 32 derajat Celsius."],
]

# ── load model ───────────────────────────────────────────────────────────────

print(f"Downloading GGUF: {GGUF_REPO}/{GGUF_FILENAME} ...")
model_path = hf_hub_download(
    repo_id=GGUF_REPO,
    filename=GGUF_FILENAME,
    token=HF_TOKEN,
)
print("Loading model ...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=2,
    n_batch=512,       # batch lebih besar = prefill lebih cepat
    n_gpu_layers=0,
    verbose=False,
)
print("Model siap!")


# ── parsing output ────────────────────────────────────────────────────────────

def parse_output(raw: str):
    label     = "—"
    reasoning = ""
    lines     = raw.strip().splitlines()

    # Coba parse format [LABEL] / [REASONING]
    reasoning_start = 0
    for i, line in enumerate(lines):
        if line.upper().strip().startswith("[LABEL]"):
            candidate = line[len("[LABEL]"):].strip()
            for valid in VALID_LABELS:
                if valid.lower() in candidate.lower():
                    label = valid
                    break
            if label == "—":
                label = candidate
            reasoning_start = i + 1
            break

    # Jika tidak ada [LABEL] marker (mode cepat), cari langsung
    if reasoning_start == 0:
        for valid in VALID_LABELS:
            if valid.lower() in raw.lower():
                label = valid
                break
        return label, ""

    for i, line in enumerate(lines[reasoning_start:], start=reasoning_start):
        if "[REASONING]" in line.upper():
            reasoning = "\n".join(lines[i + 1:]).strip()
            return label, reasoning

    reasoning = "\n".join(lines[reasoning_start:]).strip()
    return label, reasoning


# ── inference ────────────────────────────────────────────────────────────────

def classify_text(text: str, mode: str):
    if not text.strip():
        return "—", "", "", ""

    is_fast       = "Cepat" in mode
    system_prompt = PROMPT_FAST if is_fast else PROMPT_FULL
    max_tokens    = 15 if is_fast else 350

    t0 = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text.strip()},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
        repeat_penalty=1.1,
        stop=["<|im_end|>", "</s>"],
    )
    elapsed = time.time() - t0

    raw   = response["choices"][0]["message"]["content"].strip()
    label, reasoning = parse_output(raw)

    color = LABEL_COLORS.get(label, "#6b7280")
    badge = (
        f'<div style="padding:10px 20px;border-radius:8px;'
        f'background:{color}22;border:1px solid {color}66;'
        f'display:inline-block;margin-bottom:8px">'
        f'<span style="color:{color};font-weight:600;font-size:1.1em">{label}</span></div>'
    )
    mode_str = "cepat" if is_fast else "lengkap"
    status   = f"\u2713 {elapsed:.1f}s \u00b7 CPU (GGUF Q4) \u00b7 mode {mode_str}"
    return label, badge, reasoning, status


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="DFK Text Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # DFK Text Classifier
        Deteksi dan analisis **Disinformasi, Fitnah, dan Kebencian** dalam teks bahasa Indonesia.

        Model: [`aitf-komdigi/KomdigiITS-8B-DFK-TextClassification`](https://huggingface.co/aitf-komdigi/KomdigiITS-8B-DFK-TextClassification)
        \u00b7 Backend: **CPU (GGUF Q4\\_K\\_M)**

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
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Teks yang akan diklasifikasikan",
                placeholder="Masukkan teks bahasa Indonesia ...",
                lines=7,
            )
            mode_radio = gr.Radio(
                choices=[
                    "Cepat (~30 detik) — Label saja",
                    "Lengkap (~3-5 menit) — Label + Penalaran",
                ],
                value="Cepat (~30 detik) — Label saja",
                label="Mode inferensi",
            )
            with gr.Row():
                submit_btn = gr.Button("Klasifikasikan", variant="primary", scale=3)
                clear_btn  = gr.Button("Bersihkan", variant="secondary", scale=1)

        with gr.Column(scale=1):
            label_out  = gr.Textbox(label="Label", interactive=False, max_lines=1)
            badge_html = gr.HTML()
            status_out = gr.Textbox(label="Status", interactive=False, max_lines=1)

    reasoning_out = gr.Textbox(
        label="Penalaran (hanya tersedia di mode Lengkap)",
        interactive=False,
        lines=10,
        placeholder="Gunakan mode 'Lengkap' untuk melihat penalaran model ...",
    )
    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Contoh teks")

    submit_btn.click(
        classify_text,
        inputs=[text_input, mode_radio],
        outputs=[label_out, badge_html, reasoning_out, status_out],
    )
    text_input.submit(
        classify_text,
        inputs=[text_input, mode_radio],
        outputs=[label_out, badge_html, reasoning_out, status_out],
    )
    clear_btn.click(
        lambda: ("", "—", "", "", ""),
        outputs=[text_input, label_out, badge_html, reasoning_out, status_out],
    )

if __name__ == "__main__":
    demo.queue(max_size=5)
    demo.launch()
