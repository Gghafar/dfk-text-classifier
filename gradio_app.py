import os
import time

import gradio as gr
import requests

MODAL_ENDPOINT_URL = os.environ.get(
    "MODAL_ENDPOINT_URL",
    "https://gghafar--dfk-text-classifier-classify.modal.run",
)

LABELS = ["Fakta", "Disinformasi", "Fitnah", "Ujaran Kebencian", "Non-DFK"]

LABEL_COLORS = {
    "Fakta":           "#22c55e",
    "Disinformasi":    "#ef4444",
    "Fitnah":          "#f97316",
    "Ujaran Kebencian":"#dc2626",
    "Non-DFK":         "#6b7280",
}

EXAMPLES = [
    ["Pemerintah Indonesia berhasil menurunkan angka kemiskinan menjadi 9% pada 2024."],
    ["Vaksin COVID-19 mengandung chip 5G yang bisa dikendalikan dari jarak jauh."],
    ["Si A adalah koruptor yang mencuri uang rakyat meskipun belum terbukti di pengadilan."],
    ["Semua orang dari suku X itu malas dan tidak bisa dipercaya."],
    ["Hari ini cuaca di Jakarta cukup panas dengan suhu mencapai 32 derajat Celsius."],
]

# Timeout dinaikkan dari 120 → 300s untuk mengakomodasi cold start A10G (~2-3 menit)
REQUEST_TIMEOUT = 300


def classify_text(text: str) -> tuple[str, str, str]:
    """Kirim teks ke Modal endpoint dan kembalikan (label, html_badge, status_info)."""
    if not text.strip():
        return "—", "", ""

    if not MODAL_ENDPOINT_URL:
        return (
            "Error",
            "",
            "MODAL_ENDPOINT_URL belum dikonfigurasi.",
        )

    start = time.time()
    try:
        response = requests.post(
            MODAL_ENDPOINT_URL,
            json={"text": text.strip()},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return "Error", "", data["error"]

        elapsed = time.time() - start
        label   = data.get("label", "—")
        color   = LABEL_COLORS.get(label, "#6b7280")

        label_html = (
            f'<div style="padding:12px 20px;border-radius:8px;'
            f'background:{color}22;border:1px solid {color}66;display:inline-block">'
            f'<span style="color:{color};font-weight:600;font-size:1.1em">{label}</span>'
            f"</div>"
        )
        cold_note = " (cold start)" if elapsed > 30 else ""
        status    = f"✓ selesai dalam {elapsed:.1f}s{cold_note}"

        return label, label_html, status

    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        return (
            "Timeout",
            "",
            f"Request timeout setelah {elapsed:.0f}s. Modal container mungkin mengalami cold start "
            f"yang panjang. Coba lagi dalam 30 detik.",
        )
    except requests.exceptions.ConnectionError:
        return "Error", "", "Tidak dapat terhubung ke Modal endpoint. Pastikan app sudah di-deploy."
    except requests.exceptions.HTTPError as e:
        return "Error", "", f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return "Error", "", f"Terjadi kesalahan: {str(e)}"


# ─── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="DFK Text Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # DFK Text Classifier
        Deteksi **Disinformasi, Fitnah, dan Kebencian** dalam teks bahasa Indonesia.

        Model: [`aitf-komdigi/KomdigiITS-8B-DFK-TextClassification`](https://huggingface.co/aitf-komdigi/KomdigiITS-8B-DFK-TextClassification)  
        Backend: Modal.com (A10G GPU serverless)

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
                placeholder="Masukkan teks bahasa Indonesia di sini...",
                lines=6,
            )
            with gr.Row():
                submit_btn  = gr.Button("Klasifikasikan", variant="primary")
                clear_btn   = gr.Button("Bersihkan", variant="secondary")

        with gr.Column(scale=1):
            label_output = gr.Textbox(label="Label", interactive=False)
            label_html   = gr.HTML(label="Hasil")
            status_info  = gr.Textbox(label="Status", interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        label="Contoh teks",
    )

    gr.Markdown(
        """
        ---
        > **Catatan:** Inferensi pertama mungkin membutuhkan **30–120 detik** karena cold start container GPU.  
        > Inferensi berikutnya akan lebih cepat selama container masih aktif (timeout idle: 5 menit).
        """
    )

    # Event bindings
    submit_btn.click(
        fn=classify_text,
        inputs=text_input,
        outputs=[label_output, label_html, status_info],
    )
    text_input.submit(
        fn=classify_text,
        inputs=text_input,
        outputs=[label_output, label_html, status_info],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[text_input, label_output, label_html, status_info],
    )


if __name__ == "__main__":
    demo.launch()
