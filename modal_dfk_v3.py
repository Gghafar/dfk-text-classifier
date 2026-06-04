"""
DFK Text Classification & Summarization — Modal GPU Inference Endpoint (v3)
=============================================================================
Model    : aitf-komdigi/KomdigiITS-8B-DFK-TextClassification
Backend  : Unsloth FastLanguageModel (4-bit)
GPU      : L4 (24 GB VRAM)
Endpoints: GET  /          → info page
           POST /classify  → structured input (ringkasan, klaim, fakta, ...)
           POST /summarize → { "text": "...", "temperature": 0.3 }
           GET  /docs      → Swagger UI
"""

import math
import modal
import os
from typing import Optional
from pydantic import BaseModel, Field

APP_NAME  = "dfk-text-classification-v3"
MODEL_ID  = "aitf-komdigi/KomdigiITS-8B-DFK-TextClassification"
CACHE_DIR = "/cache/huggingface"

app      = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("dfk-8b-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "libz3-dev")
    .pip_install(
        "torch==2.8.0",
        "triton>=3.3.0",
        "torchvision",
        "bitsandbytes",
        "numpy",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
        "packaging",
        "ninja",
    )
    .run_commands(
        "pip install -q --no-deps 'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo'",
        "pip install -q --no-deps 'unsloth[base] @ git+https://github.com/unslothai/unsloth'",
        "pip install -q unsloth unsloth_zoo",
    )
)

VALID_LABELS = {"Fakta", "Disinformasi", "Fitnah", "Ujaran Kebencian", "Non-DFK"}

LABEL_DESC = {
    "fakta":            "Konten yang sesuai dengan fakta",
    "disinformasi":     "Informasi yang menyesatkan",
    "fitnah":           "Tuduhan tanpa bukti",
    "ujaran kebencian": "Konten menyerang kelompok tertentu",
    "non-dfk":          "Konten di luar kategori DFK",
    "—":                "Label tidak terdeteksi",
}

CLASSIFY_SYSTEM = (
    "Anda adalah sistem analisis konten yang mendeteksi disinformasi, fitnah, "
    "dan ujaran kebencian dalam teks bahasa Indonesia.\n\n"
    "Input diberikan dalam format terstruktur:\n"
    "- Ringkasan: konteks singkat postingan media sosial\n"
    "- Klaim: pernyataan atau tuduhan yang dibuat dalam postingan — INI YANG HARUS DIKLASIFIKASI\n"
    "- Fakta: informasi terverifikasi sebagai pembanding\n\n"
    "TUGAS UTAMA: Klasifikasi KLAIM berdasarkan perbandingan dengan FAKTA yang tersedia.\n"
    "Jangan mengklasifikasi FAKTA itu sendiri — fokus pada apakah KLAIM tersebut:\n"
    "- Sesuai fakta terverifikasi → Fakta\n"
    "- Menyesatkan atau tidak akurat → Disinformasi\n"
    "- Tuduhan serius tanpa bukti yang dapat diverifikasi → Fitnah\n"
    "- Menyerang atau merendahkan kelompok/individu → Ujaran Kebencian\n"
    "- Di luar kategori di atas → Non-DFK\n\n"
    "Format output WAJIB:\n"
    "[LABEL] {nama kategori}\n"
    "[CONFIDENCE] {skor_persentase_keyakinan_anda}%\n"
    "[REASONING]\n"
    "{poin-poin penalaran yang membandingkan klaim dengan fakta}"
)

SUMMARIZE_SYSTEM = (
    "Anda adalah sistem ringkasan teks bahasa Indonesia. "
    "Buat ringkasan yang jelas, padat, dan akurat dari teks yang diberikan. "
    "Gunakan bahasa yang mudah dipahami dan pertahankan informasi penting. "
    "Ringkasan maksimal 3-5 kalimat."
)

INFO_HTML = """<!DOCTYPE html><html><head><title>DFK API v3</title>
<style>body{font-family:sans-serif;max-width:680px;margin:48px auto;color:#1e293b}
pre{background:#f1f5f9;padding:14px;border-radius:8px;overflow-x:auto}
a{color:#2563eb}</style></head><body>
<h2>DFK Text Classification &amp; Summarization API</h2>
<p><b>Model:</b> aitf-komdigi/KomdigiITS-8B-DFK-TextClassification</p>
<p><b>Backend:</b> Unsloth &mdash; <b>GPU:</b> L4 24GB (bfloat16)</p>
<h3>POST /classify</h3>
<pre>{{
  "ringkasan": "Summary of the social media post",
  "klaim": "Claim made in the post",
  "fakta": "Verified fact for comparison",
  "image_url": "https://...",
  "max_new_tokens": 128,
  "temperature": 0.0
}}</pre>
<h3>POST /summarize</h3>
<pre>{{
  "text": "teks yang ingin diringkas",
  "temperature": 0.3
}}</pre>
<p><a href="/docs">Open Swagger UI</a></p>
</body></html>"""


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    ringkasan:      str             = Field(..., description="Summary of the social media post")
    klaim:          str             = Field(..., description="Claim made in the post")
    fakta:          str             = Field(..., description="Verified fact for comparison")
    image_url:      Optional[str]   = Field(None, description="Optional image URL for visual context")
    max_new_tokens: Optional[int]   = Field(128, ge=32, le=1024)
    temperature:    Optional[float] = Field(0.0, ge=0.0, le=1.0)
    num_trials:     Optional[int]   = Field(3, ge=1, le=10)


class TrialDetail(BaseModel):
    trial:      int
    label:      str
    confidence: float
    reasoning:  str


class ClassifyResponse(BaseModel):
    label:       str
    label_key:   str
    description: str
    confidence:  float
    consistency: str
    ambiguous:   bool
    reasoning:   str
    method:      str
    trials:      list[TrialDetail]


class SummarizeRequest(BaseModel):
    text:        str             = Field(..., description="Text to summarize")
    temperature: Optional[float] = Field(0.3, ge=0.1, le=1.0)


class SummarizeResponse(BaseModel):
    summary:         str
    original_length: int
    summary_length:  int


# ── JSON sanitizer ────────────────────────────────────────────────────────────

def _sanitize_body(raw: bytes) -> bytes:
    import json
    text = raw.decode("utf-8", errors="replace")
    try:
        json.loads(text)
        return raw
    except json.JSONDecodeError:
        out, in_str, esc = [], False, False
        _map = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}
        for ch in text:
            if esc:
                out.append(ch); esc = False
            elif ch == "\\" and in_str:
                out.append(ch); esc = True
            elif ch == '"':
                in_str = not in_str; out.append(ch)
            elif in_str and ord(ch) < 0x20:
                out.append(_map.get(ch, f"\\u{ord(ch):04x}"))
            else:
                out.append(ch)
        return "".join(out).encode("utf-8")


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _parse_output(raw: str):
    label, confidence, reasoning = "—", "—", ""
    clean = raw.split("<|im_end|>")[0].split("<|im_start|>")[0].split("</s>")[0].strip()
    lines = clean.splitlines()
    reasoning_start = 0

    for i, line in enumerate(lines):
        lc = line.strip()
        if lc.upper().startswith("[LABEL]"):
            candidate = lc[len("[LABEL]"):].strip()
            for v in VALID_LABELS:
                if v.lower() in candidate.lower():
                    label = v; break
            if label == "—":
                label = candidate
        elif lc.upper().startswith("[CONFIDENCE]"):
            confidence = lc[len("[CONFIDENCE]"):].strip()
        elif lc.upper().startswith("[REASONING]"):
            reasoning_start = i + 1; break

    if label == "—":
        cl = clean.lower()
        for v in VALID_LABELS:
            if v.lower() in cl:
                label = v; break

    reasoning = "\n".join(lines[reasoning_start:]).strip() if reasoning_start else clean
    return label, confidence, reasoning


def _mtla_confidence(scores_list, gen_ids, K: int = 10) -> float:
    import numpy as np
    import torch
    K_act = min(K, len(scores_list), len(gen_ids))
    log_probs = [
        math.log(max(torch.softmax(scores_list[t], dim=-1)[0, gen_ids[t].item()].item(), 1e-10))
        for t in range(K_act)
    ]
    avg_lp = float(np.mean(log_probs))
    return round(1.0 / (1.0 + math.exp(-(avg_lp + 2.5) * 1.5)), 4)


# ── Modal class ───────────────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24 * 1024,
    timeout=600,
    volumes={CACHE_DIR: hf_cache},
    scaledown_window=300,
    enable_memory_snapshot=True,
)
class DFKModel:

    @modal.enter(snap=True)
    def load_model(self):
        # Only load weights here — this is snapshotted for fast restore.
        # Do NOT build the FastAPI app here; endpoint closures must be rebuilt
        # fresh on every cold start so code changes take effect immediately.
        from unsloth import FastLanguageModel

        token = os.environ.get("HF_TOKEN")
        kwargs = dict(model_name=MODEL_ID, max_seq_length=2048, load_in_4bit=False, device_map="auto")
        if token:
            kwargs["token"] = token

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**kwargs)
        FastLanguageModel.for_inference(self.model)
        print("Model ready.")

    @modal.enter()
    def build_app(self):
        # Runs after snapshot restore — always uses latest code.
        self._app = self._build_app()

    def _build_prompt(self, system: str, user_text: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_app(self):
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import HTMLResponse

        web = FastAPI(
            title="DFK Text Classifier & Summarizer",
            version="3.0",
            docs_url="/docs",
            redirect_slashes=False,
        )

        @web.middleware("http")
        async def sanitize_json_body(request: Request, call_next):
            if request.method == "POST":
                raw = await request.body()
                request._body = _sanitize_body(raw)
            return await call_next(request)

        @web.get("/", response_class=HTMLResponse)
        def index(request: Request):
            base = str(request.base_url).rstrip("/")
            return INFO_HTML.replace("{base_url}", base)

        # ── POST /classify ────────────────────────────────────────────────────
        @web.post("/classify", response_model=ClassifyResponse)
        def classify(body: ClassifyRequest):
            import torch
            import numpy as np
            from collections import Counter

            ringkasan   = (body.ringkasan or "").strip()
            klaim       = (body.klaim or "").strip()
            fakta       = (body.fakta or "").strip()
            temperature = float(body.temperature if body.temperature is not None else 0.0)
            max_tokens  = int(body.max_new_tokens or 128)

            if not ringkasan or not klaim or not fakta:
                raise HTTPException(status_code=400, detail="ringkasan, klaim, dan fakta tidak boleh kosong.")

            user_msg = f"Ringkasan: {ringkasan}\nKlaim: {klaim}\nFakta: {fakta}"
            if body.image_url:
                user_msg += f"\nImage URL: {body.image_url}"

            prompt = self._build_prompt(CLASSIFY_SYSTEM, user_msg)

            num_trials = max(1, min(body.num_trials or 3, 10))
            # Greedy (temp=0) with multiple trials produces identical outputs — use 0.3 instead
            if temperature == 0.0 and num_trials > 1:
                temperature = 0.3
            do_sample = num_trials > 1 or temperature > 0.0

            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                text=[prompt] * num_trials,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048 - max_tokens,
                add_special_tokens=False,
            ).to(device)

            pad_id    = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            eos_id    = self.tokenizer.eos_token_id
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
                output_scores=True,
                images=None,
                pixel_values=None,
                image_sizes=None,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature

            with torch.inference_mode():
                out = self.model.generate(**gen_kwargs)

            trials = []
            for i in range(num_trials):
                gen_ids  = out.sequences[i][input_len:]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                scores_i = [s[i:i+1] for s in out.scores]
                conf     = _mtla_confidence(scores_i, gen_ids, K=10)
                label, _, reasoning = _parse_output(gen_text)
                trials.append({"label": label, "reasoning": reasoning, "confidence": conf})

            vote              = Counter(t["label"] for t in trials)
            best_label, count = vote.most_common(1)[0]
            winners           = [t for t in trials if t["label"] == best_label]
            avg_conf          = float(np.mean([t["confidence"] for t in winners]))
            best_reason       = max(winners, key=lambda x: x["confidence"])["reasoning"]
            is_ambiguous      = count == 1 or avg_conf < 0.45

            return ClassifyResponse(
                label       = best_label.upper(),
                label_key   = best_label.lower().replace(" ", "_"),
                description = LABEL_DESC.get(best_label.lower(), ""),
                confidence  = round(avg_conf * 100, 1),
                consistency = f"{count}/{num_trials}",
                ambiguous   = is_ambiguous,
                reasoning   = best_reason,
                method      = f"Unsloth {'greedy' if not do_sample else f'LogitsScore K=10, N={num_trials}'}",
                trials      = [
                    TrialDetail(
                        trial      = i + 1,
                        label      = t["label"].upper(),
                        confidence = round(t["confidence"] * 100, 1),
                        reasoning  = t["reasoning"],
                    )
                    for i, t in enumerate(trials)
                ],
            )

        # ── POST /summarize ───────────────────────────────────────────────────
        @web.post("/summarize", response_model=SummarizeResponse)
        def summarize(body: SummarizeRequest):
            import torch

            text        = (body.text or "").strip()
            temperature = float(body.temperature or 0.3)

            if not text:
                raise HTTPException(status_code=400, detail="Teks tidak boleh kosong.")

            prompt = self._build_prompt(SUMMARIZE_SYSTEM, text)

            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - 512,
                add_special_tokens=False,
            ).to(device)

            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            eos_id = self.tokenizer.eos_token_id
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    repetition_penalty=1.05,
                    images=None,
                    pixel_values=None,
                    image_sizes=None,
                )

            gen_ids = out[0][input_len:]
            summary = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            summary = summary.split("<|im_end|>")[0].split("</s>")[0].strip()

            return SummarizeResponse(
                summary         = summary,
                original_length = len(text),
                summary_length  = len(summary),
            )

        return web

    @modal.asgi_app()
    def serve(self):
        return self._app
