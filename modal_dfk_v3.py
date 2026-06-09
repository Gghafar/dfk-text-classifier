"""
DFK Text Classification & Summarization — Modal GPU Inference Endpoint (v3)
=============================================================================
Model    : hnuka/BEST-Ministral-8B-DFK-Final
Backend  : Unsloth FastLanguageModel (bfloat16, full size)
GPU      : H100
Endpoints: GET  /          → info page
          POST /classify  → structured input (klaim required; ringkasan/fakta optional)
           POST /summarize → { "text": "...", "temperature": 0.3 }
           GET  /docs      → Swagger UI
"""

import math
import modal
import os
from typing import Optional
from pydantic import BaseModel, Field

APP_NAME  = "dfk-text-classification-v3"
MODEL_ID  = "hnuka/BEST-Ministral-8B-DFK-Final"
CACHE_DIR = "/cache/huggingface"
LOG_DIR   = f"{CACHE_DIR}/api_logs"
LOG_FILE  = f"{LOG_DIR}/api_calls.jsonl"
WEAVE_LOG_FILE = f"{LOG_DIR}/weave_traces.jsonl"
WEAVE_PROJECT_ID = "Aitf-dfk-3/ministral-cpt"
WEAVE_OP_BASE = "dfk-text-classification-v3-generate"

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

VALID_LABELS = {"Fakta", "Netral", "Disinformasi", "Fitnah", "Ujaran Kebencian", "Non-DFK"}

LABEL_DESC = {
    "fakta":            "Konten yang sesuai dengan fakta",
    "disinformasi":     "Informasi yang menyesatkan",
    "fitnah":           "Tuduhan tanpa bukti",
    "ujaran kebencian": "Konten menyerang kelompok tertentu",
    "netral":           "Konten netral atau tidak termasuk pelanggaran DFK",
    "non-dfk":          "Konten di luar kategori DFK",
    "—":                "Label tidak terdeteksi",
}

CLASSIFY_SYSTEM = (
    "Anda adalah sistem analisis konten yang mendeteksi disinformasi, fitnah, "
    "dan ujaran kebencian dalam teks bahasa Indonesia.\n\n"
    "Input diberikan dalam format terstruktur:\n"
    "- Ringkasan: konteks singkat postingan media sosial jika tersedia\n"
    "- Klaim: pernyataan atau tuduhan yang dibuat dalam postingan — INI YANG HARUS DIKLASIFIKASI\n"
    "- Fakta: informasi terverifikasi sebagai pembanding jika tersedia\n\n"
    "TUGAS UTAMA: Klasifikasi KLAIM berdasarkan isi klaim itu sendiri.\n"
    "Jika FAKTA tersedia dan relevan langsung dengan KLAIM, gunakan untuk membantu verifikasi.\n"
    "Jika FAKTA tidak tersedia, kosong, membahas topik berbeda, membahas entitas berbeda, "
    "atau tidak berkorelasi langsung dengan KLAIM, maka FAKTA TIDAK RELEVAN dan HARUS DIABAIKAN.\n"
    "Jangan memaksa hubungan antara KLAIM dan FAKTA yang tidak relevan.\n"
    "Jangan mengubah, mengarang, atau memperluas FAKTA agar terlihat relevan.\n"
    "Jika FAKTA tidak relevan, cukup sebutkan singkat bahwa fakta pembanding tidak relevan lalu analisis KLAIM berdasarkan isi klaim.\n"
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
    "{poin-poin penalaran; bandingkan klaim dengan fakta jika fakta tersedia}\n\n"
    "Jangan membuat bagian 'fakta sebenarnya' kecuali fakta tersebut diberikan pada input. "
    "Jangan menyebut sumber, lembaga, atau detail yang tidak muncul pada input. "
    "Jangan menambahkan slogan, ajakan, call-to-action, atau kalimat penutup seperti 'mari bersama'."
)

SUMMARIZE_SYSTEM = (
    "Anda adalah sistem ringkasan teks bahasa Indonesia, bukan sistem klasifikasi atau fact-checking. "
    "Ringkas HANYA teks yang diberikan oleh user. Jangan menambah informasi baru. "
    "Output WAJIB satu paragraf berisi 2-4 kalimat. "
    "Jangan memakai bullet/daftar. Jangan memberi label, analisis, alasan, fakta sebenarnya, klaim, "
    "saran verifikasi, ajakan, slogan, atau call-to-action. "
    "Jika model mulai menulis selain ringkasan, hentikan output."
)

INFO_HTML = """<!DOCTYPE html><html><head><title>DFK API v3</title>
<style>body{font-family:sans-serif;max-width:680px;margin:48px auto;color:#1e293b}
pre{background:#f1f5f9;padding:14px;border-radius:8px;overflow-x:auto}
a{color:#2563eb}</style></head><body>
<h2>DFK Text Classification &amp; Summarization API</h2>
<p><b>Model:</b> hnuka/BEST-Ministral-8B-DFK-Final (bfloat16)</p>
<p><b>Backend:</b> Unsloth &mdash; <b>GPU:</b> H100 80GB</p>
<h3>POST /classify</h3>
<pre>{{
  "klaim": "Claim made in the post",
  "ringkasan": "Optional summary of the social media post",
  "fakta": "Optional verified fact for comparison",
  "image_url": "https://...",
  "max_new_tokens": 512,
  "temperature": 0.0,
  "num_trials": 3
}}</pre>
<h3>POST /summarize</h3>
<pre>{{
	  "text": "teks yang ingin diringkas",
	  "temperature": 0.0
	}}</pre>
<p><a href="/docs">Open Swagger UI</a></p>
</body></html>"""


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    ringkasan:      Optional[str]   = Field(None, description="Optional summary of the social media post")
    klaim:          str             = Field(..., description="Claim made in the post")
    fakta:          Optional[str]   = Field(None, description="Optional verified fact for comparison")
    image_url:      Optional[str]   = Field(None, description="Optional image URL for visual context")
    max_new_tokens: Optional[int]   = Field(512, ge=32, le=2048)
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
    temperature: Optional[float] = Field(0.0, ge=0.0, le=1.0)


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
    import re

    label, reasoning = "—", ""
    clean = raw.split("<|im_end")[0].split("<|im_start")[0].split("</s>")[0].strip()

    inline = re.search(
        r"\[LABEL\]\s*(.*?)\s*(?:\[CONFIDENCE\]\s*.*?\s*)?\[REASONING\]\s*(.*)",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if inline:
        candidate = inline.group(1).strip()
        for v in VALID_LABELS:
            if v.lower() in candidate.lower():
                label = v
                break
        if label == "—":
            label = candidate
        return label, inline.group(2).strip()

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
        elif lc.lower().startswith("label:"):
            candidate = lc[len("label:"):].strip()
            for v in VALID_LABELS:
                if v.lower() in candidate.lower():
                    label = v; break
            if label == "—":
                label = candidate
        elif lc.upper().startswith("[REASONING]"):
            reasoning_start = i + 1; break
        elif lc.lower().startswith("analisis:") or lc.lower().startswith("analysis:"):
            reasoning_start = i
            break

    if label == "—":
        for v in VALID_LABELS:
            if v.lower() in clean.lower():
                label = v; break

    reasoning = "\n".join(lines[reasoning_start:]).strip() if reasoning_start else clean
    if reasoning.lower().startswith("analisis:"):
        reasoning = reasoning[len("analisis:"):].strip()
    elif reasoning.lower().startswith("analysis:"):
        reasoning = reasoning[len("analysis:"):].strip()
    return label, reasoning


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


def _is_fact_relevant(klaim: str, fakta: str) -> bool:
    import re

    if not klaim or not fakta:
        return False

    stopwords = {
        "yang", "dan", "atau", "dari", "dengan", "untuk", "pada", "dalam",
        "adalah", "tidak", "bukan", "telah", "akan", "ini", "itu", "karena",
        "the", "and", "for", "with", "from", "that", "this", "not",
    }

    def terms(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(token) >= 4 and token not in stopwords
        }

    klaim_terms = terms(klaim)
    fakta_terms = terms(fakta)
    overlap = klaim_terms & fakta_terms
    return len(overlap) >= 2


def _clean_summary_output(raw: str) -> str:
    import re

    clean = raw.split("<|im_end")[0].split("<|im_start")[0].split("</s>")[0].strip()
    lower = clean.lower()
    cut_markers = [
        "\nberikut adalah",
        "\nberikut beberapa",
        "\nfakta sebenarnya",
        "\nbeberapa fakta",
        "\nalasan:",
        "\nanalisis:",
        "\nlabel:",
        "\nklaim:",
        "\njangan mudah",
        "\nmari ",
        "\n**jangan",
    ]
    positions = [lower.find(marker) for marker in cut_markers if lower.find(marker) >= 0]
    if positions:
        clean = clean[:min(positions)]

    lines = []
    stop_prefixes = (
        "berikut adalah",
        "berikut beberapa",
        "fakta sebenarnya",
        "beberapa fakta",
        "alasan:",
        "analisis:",
        "label:",
        "klaim:",
        "jangan mudah",
        "mari ",
        "pastikan ",
    )
    for line in clean.splitlines():
        item = line.strip().strip("*_`")
        if not item:
            continue
        item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", item).strip()
        if item.lower().startswith("ringkasan:"):
            item = item.split(":", 1)[1].strip()
        if item.lower().startswith(stop_prefixes):
            break
        if item:
            lines.append(item)

    summary = " ".join(lines).strip()
    summary = re.sub(r"\s+", " ", summary)
    summary = re.sub(
        r"^(?:berikut\s+(?:merupakan|adalah)\s+)?(?:beberapa\s+)?fakta\s+sebenarnya\s*:\s*",
        "",
        summary,
        flags=re.IGNORECASE,
    ).strip()
    summary = re.sub(
        r"^berikut\s+(?:merupakan|adalah)\s+(?:beberapa\s+)?",
        "",
        summary,
        flags=re.IGNORECASE,
    ).strip()
    sentences = re.split(r"(?<=[.!?])\s+", summary)
    if len(sentences) > 4:
        summary = " ".join(sentences[:4]).strip()
    return summary


def _clean_reasoning_output(raw: str) -> str:
    import re

    clean = raw.split("<|im_end")[0].split("<|im_start")[0].split("</s>")[0].strip()
    cut_markers = [
        "\nberikut adalah",
        "\nberikut merupakan",
        "\nfakta sebenarnya",
        "\n**mari",
        "\nmari ",
        "\n**jangan",
        "\njangan mudah",
        "\npastikan ",
    ]
    lower = clean.lower()
    positions = [lower.find(marker) for marker in cut_markers if lower.find(marker) >= 0]
    if positions:
        clean = clean[:min(positions)]

    clean = re.sub(
        r"^(?:berikut\s+(?:adalah|merupakan)\s+)?(?:beberapa\s+)?fakta\s+sebenarnya\s*:\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()
    clean = re.sub(
        r"^berikut\s+(?:adalah|merupakan)\s+(?:beberapa\s+)?",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()
    return clean.strip("*_` \n")


def _append_jsonl(record: dict, path: str = LOG_FILE) -> None:
    import json

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())

    # Persist the append to the Modal Volume so logs survive container shutdown.
    try:
        hf_cache.commit()
    except Exception as exc:
        print(f"API log volume commit failed: {exc}")


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _weave_trace_name() -> str:
    from datetime import datetime, timezone, timedelta

    jakarta_tz = timezone(timedelta(hours=7))
    stamp = datetime.now(jakarta_tz).strftime("%Y-%m-%d-%H-%M-WIB")
    return f"{WEAVE_OP_BASE}-{stamp}"


def _weave_attributes() -> dict:
    import platform
    import sys

    return {
        "weave": {
            "client_version": "local-jsonl",
            "source": "modal-api-jsonl",
            "sys_version": sys.version,
            "os_name": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
        }
    }


# ── Modal class ───────────────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32 * 1024,
    timeout=600,
    volumes={CACHE_DIR: hf_cache},
    scaledown_window=300,
)
class DFKModel:

    @modal.enter()
    def load_model(self):
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
        self._app = self._build_app()

    def _build_prompt(self, system: str, user_text: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _log_api_call(
        self,
        *,
        request_id: str,
        endpoint: str,
        started_at: float,
        input_payload: dict,
        output_payload: Optional[dict] = None,
        status: str = "success",
        error: Optional[str] = None,
        started_at_iso: Optional[str] = None,
        weave_inputs: Optional[dict] = None,
        weave_output: Optional[dict] = None,
    ) -> None:
        import time
        import uuid

        try:
            ended_at_iso = _utc_now_iso()
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            record = {
                "request_id": request_id,
                "timestamp": ended_at_iso,
                "endpoint": endpoint,
                "status": status,
                "latency_ms": latency_ms,
                "input": input_payload,
                "output": output_payload,
                "error": error,
            }
            _append_jsonl(record)

            trace_name = _weave_trace_name()
            trace_id = request_id
            weave_record = {
                "id": str(uuid.uuid4()),
                "project_id": WEAVE_PROJECT_ID,
                "op_name": f"weave:///{WEAVE_PROJECT_ID}/op/{trace_name}:local-jsonl",
                "display_name": None,
                "trace_id": trace_id,
                "parent_id": None,
                "thread_id": None,
                "turn_id": None,
                "started_at": started_at_iso or ended_at_iso,
                "attributes": _weave_attributes(),
                "inputs": weave_inputs or input_payload,
                "ended_at": ended_at_iso,
                "exception": error,
                "output": weave_output if weave_output is not None else output_payload,
                "summary": {
                    "status_counts": {
                        "success": 1 if status == "success" else 0,
                        "error": 1 if status != "success" else 0,
                    },
                    "weave": {
                        "status": status,
                        "trace_name": trace_name,
                        "latency_ms": int(latency_ms),
                    },
                },
                "wb_user_id": None,
                "wb_run_id": None,
                "wb_run_step": 0,
                "wb_run_step_end": 0,
                "deleted_at": None,
                "expire_at": None,
                "storage_size_bytes": None,
                "total_storage_size_bytes": None,
            }
            _append_jsonl(weave_record, WEAVE_LOG_FILE)
        except Exception as exc:
            print(f"API logging failed: {exc}")

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
            return INFO_HTML.replace("{base_url}", str(request.base_url).rstrip("/"))

        @web.post("/classify", response_model=ClassifyResponse)
        def classify(body: ClassifyRequest):
            import torch
            import numpy as np
            import time
            import uuid
            from collections import Counter

            started_at = time.perf_counter()
            started_at_iso = _utc_now_iso()
            request_id = str(uuid.uuid4())
            input_payload = body.model_dump()

            ringkasan   = (body.ringkasan or "").strip()
            klaim       = (body.klaim or "").strip()
            fakta       = (body.fakta or "").strip()
            temperature = float(body.temperature if body.temperature is not None else 0.0)
            max_tokens  = int(body.max_new_tokens or 512)
            effective_fakta = fakta if _is_fact_relevant(klaim, fakta) else ""

            if not klaim:
                self._log_api_call(
                    request_id=request_id,
                    endpoint="/classify",
                    started_at=started_at,
                    input_payload=input_payload,
                    status="error",
                    error="klaim tidak boleh kosong.",
                    started_at_iso=started_at_iso,
                    weave_inputs={
                        "mode": "dfk",
                        "image": body.image_url,
                        "image_preview": None,
                        "ringkasan": ringkasan,
                        "klaim": klaim,
                        "fakta": fakta,
                        "prompt": None,
                        "dfk_prompt": CLASSIFY_SYSTEM,
                        "caption_prompt": None,
                        "model_prompt": None,
                        "messages_input": None,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                raise HTTPException(status_code=400, detail="klaim tidak boleh kosong.")

            context_parts = []
            if ringkasan:
                context_parts.append(f"Ringkasan: {ringkasan}")
            context_parts.append(f"Klaim: {klaim}")
            if effective_fakta:
                context_parts.append(f"Fakta: {effective_fakta}")
            else:
                context_parts.append(
                    "Fakta: TIDAK TERSEDIA ATAU TIDAK RELEVAN. "
                    "Jangan gunakan fakta pembanding dan jangan menyebut sumber/detail eksternal yang tidak diberikan."
                )
            user_msg = "\n".join(context_parts)
            if body.image_url and body.image_url not in ("string", ""):
                user_msg += f"\nImage URL: {body.image_url}"

            num_trials = max(1, min(body.num_trials or 3, 10))
            if temperature == 0.0 and num_trials > 1:
                temperature = 0.3
            do_sample = num_trials > 1 or temperature > 0.0
            messages_input = [
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            prompt = self._build_prompt(CLASSIFY_SYSTEM, user_msg)
            weave_inputs = {
                "mode": "dfk",
                "image": body.image_url,
                "image_preview": None,
                "ringkasan": ringkasan,
                "klaim": klaim,
                "fakta": fakta,
                "prompt": None,
                "dfk_prompt": CLASSIFY_SYSTEM,
                "caption_prompt": None,
                "model_prompt": prompt,
                "messages_input": messages_input,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }

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
                repetition_penalty=1.15,
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
                label, reasoning = _parse_output(gen_text)
                reasoning = _clean_reasoning_output(reasoning)
                trials.append({
                    "label": label,
                    "reasoning": reasoning,
                    "confidence": conf,
                    "raw_text": gen_text,
                    "tokens_generated": int(len(gen_ids)),
                })

            vote              = Counter(t["label"] for t in trials)
            best_label, count = vote.most_common(1)[0]
            winners           = [t for t in trials if t["label"] == best_label]
            avg_conf          = float(np.mean([t["confidence"] for t in winners]))
            best_reason       = max(winners, key=lambda x: x["confidence"])["reasoning"]
            is_ambiguous      = count == 1 or avg_conf < 0.45

            response = ClassifyResponse(
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
            self._log_api_call(
                request_id=request_id,
                endpoint="/classify",
                started_at=started_at,
                input_payload=input_payload,
                output_payload=response.model_dump(),
                started_at_iso=started_at_iso,
                weave_inputs=weave_inputs,
                weave_output={
                    "text": f"Label: {response.label}\n\nAnalisis: {response.reasoning}",
                    "tokens_generated": max((t.get("tokens_generated", 0) for t in trials), default=0),
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
                    "total_request_ms": round((time.perf_counter() - started_at) * 1000),
                    "logits_label": response.label,
                    "logits_scores": {
                        response.label: response.confidence,
                    },
                    "api_response": response.model_dump(),
                },
            )
            return response

        @web.post("/summarize", response_model=SummarizeResponse)
        def summarize(body: SummarizeRequest):
            import torch
            import time
            import uuid

            started_at = time.perf_counter()
            started_at_iso = _utc_now_iso()
            request_id = str(uuid.uuid4())
            input_payload = body.model_dump()

            text        = (body.text or "").strip()
            temperature = float(body.temperature if body.temperature is not None else 0.0)

            if not text:
                self._log_api_call(
                    request_id=request_id,
                    endpoint="/summarize",
                    started_at=started_at,
                    input_payload=input_payload,
                    status="error",
                    error="Teks tidak boleh kosong.",
                    started_at_iso=started_at_iso,
                    weave_inputs={
                        "mode": "summarize",
                        "image": None,
                        "image_preview": None,
                        "ringkasan": "",
                        "klaim": "",
                        "fakta": "",
                        "prompt": text,
                        "dfk_prompt": None,
                        "caption_prompt": None,
                        "model_prompt": None,
                        "messages_input": None,
                        "max_new_tokens": 512,
                        "temperature": temperature,
                    },
                )
                raise HTTPException(status_code=400, detail="Teks tidak boleh kosong.")

            prompt = self._build_prompt(SUMMARIZE_SYSTEM, text)
            messages_input = [
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": text},
            ]

            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - 512,
                add_special_tokens=False,
            ).to(device)

            pad_id    = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            eos_id    = self.tokenizer.eos_token_id
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                gen_kwargs = dict(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    do_sample=temperature > 0.0,
                    use_cache=True,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    repetition_penalty=1.2,
                    images=None,
                    pixel_values=None,
                    image_sizes=None,
                )
                if temperature > 0.0:
                    gen_kwargs["temperature"] = temperature
                out = self.model.generate(**gen_kwargs)

            gen_ids = out[0][input_len:]
            summary = _clean_summary_output(self.tokenizer.decode(gen_ids, skip_special_tokens=True))

            response = SummarizeResponse(
                summary         = summary,
                original_length = len(text),
                summary_length  = len(summary),
            )
            self._log_api_call(
                request_id=request_id,
                endpoint="/summarize",
                started_at=started_at,
                input_payload=input_payload,
                output_payload=response.model_dump(),
                started_at_iso=started_at_iso,
                weave_inputs={
                    "mode": "summarize",
                    "image": None,
                    "image_preview": None,
                    "ringkasan": "",
                    "klaim": "",
                    "fakta": "",
                    "prompt": text,
                    "dfk_prompt": None,
                    "caption_prompt": None,
                    "model_prompt": prompt,
                    "messages_input": messages_input,
                    "max_new_tokens": 512,
                    "temperature": temperature,
                },
                weave_output={
                    "text": summary,
                    "tokens_generated": int(len(gen_ids)),
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
                    "total_request_ms": round((time.perf_counter() - started_at) * 1000),
                    "api_response": response.model_dump(),
                },
            )
            return response

        return web

    @modal.asgi_app()
    def serve(self):
        return self._app
