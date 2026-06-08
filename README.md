# dfk-text-classifier

A [Modal](https://modal.com) deployment of [`aitf-komdigi/KomdigiITS-8B-DFK-TextClassification`](https://huggingface.co/aitf-komdigi/KomdigiITS-8B-DFK-TextClassification) — a fine-tuned 8.9B Mistral3 model served as a GPU-backed FastAPI endpoint for Indonesian social media content analysis.

The model classifies content into DFK categories (Disinformasi, Fitnah, Kebencian) and can also summarize text in Bahasa Indonesia.

## Features

- **DFK classification** — detects disinformation, slander, and hate speech from structured social media post metadata
- **Summarization mode** — summarizes Indonesian text with a dedicated summarization prompt
- **Multi-trial MTLA voting** — runs N generation trials, scores each via logit-based confidence (K=10 tokens), then majority-votes the result
- **Greedy mode** — `temperature: 0` with single trial for fastest deterministic inference
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
  "ringkasan": "Summary of the social media post",
  "klaim": "Claim made in the post",
  "fakta": "Verified fact for comparison",
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
    "ringkasan": "Vaksin COVID-19 diklaim mengandung chip mikro",
    "klaim": "Vaksin mengandung chip untuk memata-matai warga negara",
    "fakta": "WHO dan Kemenkes menyatakan vaksin tidak mengandung chip apapun",
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
  "temperature": 0.3
}
```

**Example curl:**
```bash
curl -X POST "https://gghafar--dfk-text-classification-v3-dfkmodel-serve.modal.run/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Pemerintah Indonesia mengesahkan regulasi baru tentang penggunaan kecerdasan buatan di sektor publik.",
    "temperature": 0.3
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
| `ringkasan` | string | yes | — | Summary/context of the social media post |
| `klaim` | string | yes | — | The specific claim or statement to classify |
| `fakta` | string | yes | — | Verified fact(s) to compare the claim against |
| `image_url` | string | no | null | Optional image URL for additional context |
| `max_new_tokens` | int | no | 512 | Max tokens to generate (32–2048) |
| `temperature` | float | no | 0.0 | Sampling temperature. `0` = greedy. If `> 0` and `num_trials > 1`, enables MTLA voting |
| `num_trials` | int | no | 3 | Number of generation trials for voting (1–10). Auto-raises temperature to `0.3` when `temperature: 0` and `num_trials > 1` |

### Summarization Input Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Indonesian text to summarize |
| `temperature` | float | no | 0.3 | Sampling temperature for summary generation |

### DFK Labels

| Label | Description |
|-------|-------------|
| `Fakta` | Content consistent with verified facts |
| `Disinformasi` | Misleading or inaccurate information |
| `Fitnah` | Serious accusations without verifiable evidence |
| `Ujaran Kebencian` | Content attacking or degrading individuals/groups |
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

## Infrastructure

| Setting | Value |
|---------|-------|
| GPU | NVIDIA H100 (80 GB VRAM) |
| CPU | 4 vCPU |
| Memory | 32 GB RAM |
| Timeout | 600s |
| Scale-down | 300s idle |
| Precision | bfloat16 (full size, no quantization) |
| Snapshot | Disabled for Unsloth GPU startup compatibility |
| Model | aitf-komdigi/KomdigiITS-8B-DFK-TextClassification |
| Parameters | 8.9B |
