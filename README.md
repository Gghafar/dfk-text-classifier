# dfk-text-classifier

A [Modal](https://modal.com) deployment of [`hnuka/BEST-Ministral-8B-DFK-Final`](https://huggingface.co/hnuka/BEST-Ministral-8B-DFK-Final) — a merged/full 8B Mistral3 DFK model served as a GPU-backed FastAPI endpoint for Indonesian social media content analysis.

The model classifies content into DFK categories (Disinformasi, Fitnah, Kebencian) and can also summarize text in Bahasa Indonesia.

## Features

- **DFK classification** — detects disinformation, slander, and hate speech from a claim; only `klaim` is required
- **Optional context** — `ringkasan` and `fakta` are optional; unrelated `fakta` is ignored before the prompt is sent to the model
- **Summarization mode** — summarizes Indonesian text with a dedicated summarization prompt and response cleanup
- **Multi-trial MTLA voting** — runs N generation trials, scores each via logit-based confidence (K=10 tokens), then majority-votes the result
- **Greedy mode** — `temperature: 0` with single trial for fastest deterministic inference
- **Weave-style JSONL logs** — stores API calls in a Modal Volume using the same top-level trace shape as the exported W&B Weave JSONL
- **JSON sanitizer** — middleware that fixes copy-pasted text containing literal newline characters inside JSON strings

## Setup

```bash
pip install modal
modal setup
```

If the HF repo is private, create a Modal secret:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

Then add `secrets=[modal.Secret.from_name("huggingface-secret")]` to `@app.cls(...)` in `modal_dfk_v3.py`.

## Commands

```bash
# Dev server (hot-reload, temporary endpoint URL printed to console)
modal serve modal_dfk_v3.py

# Production deploy
modal deploy modal_dfk_v3.py

# Stream logs
modal app logs dfk-text-classification-v3
```

## API Endpoints

**Base URL:** `https://gghafar--dfk-text-classification-v3-dfkmodel-serve.modal.run`

**Swagger UI:** `https://gghafar--dfk-text-classification-v3-dfkmodel-serve.modal.run/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Basic API information page |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/classify` | DFK classification from structured claim/fact input |
| `POST` | `/summarize` | Indonesian text summarization |

### DFK Classification

**Endpoint:** `POST /classify`

**Full URL:** `https://gghafar--dfk-text-classification-v3-dfkmodel-serve.modal.run/classify`

**Request:**

```json
{
  "klaim": "Claim made in the post",
  "ringkasan": "Optional summary of the social media post",
  "fakta": "Optional verified fact for comparison",
  "image_url": "https://...",
  "max_new_tokens": 128,
  "temperature": 0.0,
  "num_trials": 3
}
```

**Example curl:**
```bash
curl -X POST "https://gghafar--dfk-text-classification-v3-dfkmodel-serve.modal.run/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "klaim": "Vaksin mengandung chip untuk memata-matai warga negara",
    "max_new_tokens": 256,
    "temperature": 0.0,
    "num_trials": 1
  }'
```

**Response:**
```json
{
  "label": "DISINFORMASI",
  "label_key": "disinformasi",
  "description": "Informasi yang menyesatkan",
  "confidence": 97.7,
  "consistency": "3/3",
  "ambiguous": false,
  "reasoning": "...",
  "method": "Unsloth LogitsScore K=10, N=3",
  "trials": [
    { "trial": 1, "label": "DISINFORMASI", "confidence": 97.7, "reasoning": "..." },
    { "trial": 2, "label": "DISINFORMASI", "confidence": 97.7, "reasoning": "..." },
    { "trial": 3, "label": "DISINFORMASI", "confidence": 97.7, "reasoning": "..." }
  ]
}
```

### Summarization

**Endpoint:** `POST /summarize`

**Full URL:** `https://gghafar--dfk-text-classification-v3-dfkmodel-serve.modal.run/summarize`

**Request:**

```json
{
  "text": "Teks yang ingin diringkas...",
  "temperature": 0.0
}
```

**Example curl:**
```bash
curl -X POST "https://gghafar--dfk-text-classification-v3-dfkmodel-serve.modal.run/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Pemerintah Indonesia mengesahkan regulasi baru tentang penggunaan kecerdasan buatan di sektor publik.",
    "temperature": 0.0
  }'
```

**Response:**
```json
{
  "summary": "Ringkasan teks...",
  "original_length": 668,
  "summary_length": 312
}
```

### Classification Input Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `ringkasan` | string | no | null | Optional summary/context of the social media post |
| `klaim` | string | yes | — | The specific claim or statement to classify |
| `fakta` | string | no | null | Optional verified fact(s) to compare the claim against |
| `image_url` | string | no | null | Optional image URL for additional context |
| `max_new_tokens` | int | no | 512 | Max tokens to generate (32–2048) |
| `temperature` | float | no | 0.0 | Sampling temperature. `0` = greedy. If `> 0` and `num_trials > 1`, enables MTLA voting |
| `num_trials` | int | no | 3 | Number of generation trials for voting (1–10). Auto-raises temperature to `0.3` when `temperature: 0` and `num_trials > 1` |

### Summarization Input Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Indonesian text to summarize |
| `temperature` | float | no | 0.0 | Sampling temperature for summary generation. `0` uses deterministic generation |

### DFK Labels

| Label | Description |
|-------|-------------|
| `Fakta` | Content consistent with verified facts |
| `Disinformasi` | Misleading or inaccurate information |
| `Fitnah` | Serious accusations without verifiable evidence |
| `Ujaran Kebencian` | Content attacking or degrading individuals/groups |
| `Netral` | Neutral content that does not fall into DFK violations |
| `Non-DFK` | Content outside DFK categories |

## Architecture

**File:** `modal_dfk_v3.py`

| Component | Description |
|-----------|-------------|
| `DFKModel` | Modal class on H100 GPU. Loads the model via `@modal.enter()` and serves FastAPI routes through `@modal.asgi_app()`. |
| `POST /classify` | Structured DFK classification with MTLA multi-trial voting. |
| `POST /summarize` | Indonesian text summarization using a summarization system prompt. |
| `_mtla_confidence` | Computes logit-based confidence score from first K generated token probabilities. |
| `_parse_output` | Extracts `[LABEL]`, `[CONFIDENCE]`, and `[REASONING]` blocks from raw model output. |
| `_is_fact_relevant` | Prevents unrelated optional facts from being included in the model prompt. |
| `_clean_reasoning_output` / `_clean_summary_output` | Removes model-style CTA text, duplicate fact-check sections, and leaked prompt tags from API responses. |
| HF Volume cache | `modal.Volume` named `dfk-8b-cache` persists downloaded weights across cold starts. |

**Model loading flow:**
1. Modal starts an H100 container.
2. `@modal.enter()` loads model + tokenizer via Unsloth `FastLanguageModel.from_pretrained` in full bfloat16.
3. `FastLanguageModel.for_inference(...)` applies Unsloth inference optimizations.
4. FastAPI routes are built and served through `@modal.asgi_app()`.

Memory snapshots are intentionally disabled because Unsloth checks for a visible GPU during import. Modal CPU snapshot initialization does not expose the GPU, which causes Unsloth startup failures.

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 512 | Maximum tokens to generate for classification |
| `temperature` | 0.0 | `0` = greedy (single trial). `> 0` = sampling with MTLA voting |
| `num_trials` | 3 | Number of parallel generation trials for majority voting |
| `repetition_penalty` | 1.15 | Penalizes repeated tokens (hardcoded) |

## API Call Logging

Every `/classify` and `/summarize` call is appended as one JSON object per line in the Modal Volume:

```text
dfk-8b-cache:/api_logs/api_calls.jsonl
```

A second Weave-compatible trace log is written to:

```text
dfk-8b-cache:/api_logs/weave_traces.jsonl
```

The Weave-style records use these top-level keys, matching the exported JSONL structure:

```text
id, project_id, op_name, display_name, trace_id, parent_id, thread_id,
turn_id, started_at, attributes, inputs, ended_at, exception, output,
summary, wb_user_id, wb_run_id, wb_run_step, wb_run_step_end, deleted_at,
expire_at, storage_size_bytes, total_storage_size_bytes
```

The `inputs` object uses the same columns visible in the Weave trace table:

```text
mode, image, image_preview, ringkasan, klaim, fakta, prompt, dfk_prompt,
caption_prompt, model_prompt, messages_input, max_new_tokens, temperature
```

Each record contains:

| Field | Description |
|-------|-------------|
| `request_id` | Server-generated UUID for the API call |
| `timestamp` | Server-generated UTC timestamp |
| `endpoint` | `/classify` or `/summarize` |
| `status` | `success` or `error` |
| `latency_ms` | End-to-end endpoint latency in milliseconds |
| `input` | Parsed request body sent by the user |
| `output` | Response body returned by the API |
| `error` | Error message, if logged |

Download the live API logs:

```bash
modal volume get dfk-8b-cache api_logs/api_calls.jsonl ./api_calls.jsonl --force
modal volume get dfk-8b-cache api_logs/weave_traces.jsonl ./weave_traces.jsonl --force
```

A metadata-only historical backfill from Modal logs is also stored at:

```text
dfk-8b-cache:/api_logs/backfill_modal_metadata_last_7d.jsonl
```

That backfill includes endpoint, timestamp, status, duration, and execution time from Modal logs. It does not include historical request bodies or model outputs because those were not logged before JSONL logging was added.

## Infrastructure

| Setting | Value |
|---------|-------|
| GPU | NVIDIA H100 requested; Modal may assign an H100-compatible H200 when available |
| CPU | 4 vCPU |
| Memory | 32 GB RAM |
| Timeout | 600s |
| Scale-down | 300s idle |
| Precision | bfloat16 (full size, no quantization) |
| Snapshot | Disabled for Unsloth GPU startup compatibility |
| Model | hnuka/BEST-Ministral-8B-DFK-Final |
| Parameters | 8B |

The Hugging Face repository contains full model weights (`model.safetensors`) and does not expose LoRA adapter files such as `adapter_config.json` or `adapter_model.safetensors`, so it is loaded directly as a merged/full model.
