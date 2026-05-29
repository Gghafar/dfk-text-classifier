import os

import gradio as gr
import requests

# Set this to your Modal endpoint URL after deploying
MODAL_ENDPOINT_URL = os.environ.get("MODAL_ENDPOINT_URL", "https://gghafar--dfk-text-classifier-classify.modal.run")

LABELS = ["Fakta", "Disinformasi", "Fitnah", "Ujaran Kebencian", "Non-DFK"]

LABEL_COLORS = {
    "Fakta": "#22c55e",
    "Disinformasi": "#ef4444",
    "Fitnah": "#f97316",
    "Ujaran Kebencian": "#dc2626",
    "Non-DFK": "#6b7280",
}

EXAMPLES = [
    ["Pemerintah Indonesia berhasil menurunkan angka kemiskinan menjadi 9% pada 2024."],
    ["Vaksin COVID-19 mengandung chip 5G yang bisa dikendalikan dari jarak jauh."],
    ["Si A adalah koruptor yang mencuri uang rakyat meskipun belum terbukti di pengadilan."],
    ["Semua orang dari suku X itu malas dan tidak bisa dipercaya."],
    ["Hari ini cuaca di Jakarta cukup panas dengan suhu mencapai 32 derajat Celsius."],
]


def classify_text(text: str) -> tuple[str, str]:
    if not text.strip():
        return "—", ""

    if not MODAL_ENDPOINT_URL:
        return "Error", "MODAL_ENDPOINT_URL belum dikonfigurasi. Tambahkan ke Secrets di HuggingFace Space."

    try:
        response = requests.post(
            MODAL_ENDPOINT_URL,
            json={"text": text},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return "Error", data["error"]

        label = data.get("label", "—")
        color = LABEL_COLORS.get(label, "#6b7280")
        label_html = f'<span style="color:{color}; font-weight:bold; font-size:1.2em;">{label}</span>'
        return label, label_html

    except requests.exceptions.Timeout:
        return "Error", "Request timeout. Coba lagi — container Modal mungkin sedang cold start (~30 detik)."
    except Exception as e:
        return "Error", f"Terjadi kesalahan: {str(e)}"


with gr.Blocks(title="DFK Text Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # DFK Text Classifier
        Deteksi **Disinformasi, Fitnah, dan Kebencian** dalam teks bahasa Indonesia.

        Model: [`aitf-komdigi/KomdigiITS-8B-DFK-TextClassification`](https://huggingface.co/aitf-komdigi/KomdigiITS-8B-DFK-TextClassification)
        Backend: Modal.com (GPU serverless)

        ### Kategori:
        - **Fakta** — Informasi yang benar dan dapat diverifikasi
        - **Disinformasi** — Informasi yang menyesatkan atau salah
        - **Fitnah** — Tuduhan tanpa dasar yang merusak reputasi
        - **Ujaran Kebencian** — Konten yang menarget kelompok tertentu
        - **Non-DFK** — Konten netral, tidak termasuk kategori di atas
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Teks yang akan diklasifikasikan",
                placeholder="Masukkan teks bahasa Indonesia di sini...",
                lines=6,
            )
            submit_btn = gr.Button("Klasifikasikan", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Textbox(label="Label", interactive=False)
            label_html = gr.HTML(label="Hasil")

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        label="Contoh teks",
    )

    gr.Markdown(
        """
        ---
        > **Catatan:** Inferensi pertama mungkin membutuhkan ~30 detik karena cold start container GPU.
        > Inferensi berikutnya akan lebih cepat selama container masih aktif.
        """
    )

    submit_btn.click(
        fn=classify_text,
        inputs=text_input,
        outputs=[label_output, label_html],
    )
    text_input.submit(
        fn=classify_text,
        inputs=text_input,
        outputs=[label_output, label_html],
    )

if __name__ == "__main__":
    demo.launch()
