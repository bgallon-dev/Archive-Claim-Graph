# Open-source claim extraction (self-hosted / vLLM)

gemynd's claim-extraction LLM path is pluggable. Besides the Anthropic adapter it
ships an **OpenAI-compatible** adapter that talks to any server exposing the
`/v1/chat/completions` API — **vLLM**, Ollama, TGI, LM Studio, llama.cpp's server,
or OpenAI itself. This lets you run extraction on an open-weights model
(e.g. Qwen2.5-32B-Instruct) on your own GPU instead of the metered Anthropic API.

## How the backend is selected

`gemynd/ingest/extractors/llm_adapter_factory.py` dispatches on environment:

| Condition | Backend |
|---|---|
| `CLAIM_LLM_PROVIDER` set to `openai` / `openai_compatible` / `vllm` / `ollama` / `tgi` / `local` | OpenAI-compatible adapter |
| `CLAIM_LLM_PROVIDER=anthropic` | Anthropic adapter |
| unset **and** `CLAIM_LLM_BASE_URL` set | OpenAI-compatible (auto) |
| unset and no base URL | Anthropic (default) |

Both adapters degrade to `NullLLMAdapter` (LLM layer off, rule-based only) when their
backend is unconfigured — an unset environment never raises.

## Environment variables

| Variable | Meaning | Example |
|---|---|---|
| `CLAIM_LLM_BASE_URL` | server base URL (with or without `/v1`) | `http://localhost:8000/v1` |
| `CLAIM_LLM_MODEL` | served model id | `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| `CLAIM_LLM_API_KEY` | bearer token (optional; local servers ignore it) | `not-needed` |
| `CLAIM_LLM_PROVIDER` | force a backend (optional) | `vllm` |
| `CLAIM_CACHE_PATH` | SQLite response cache (shared across backends, keyed by model + prompt hash) | default under repo |

The same prompt builder and JSON parser back both adapters, so claim rows are
identical in shape. Responses are cached per `(model, system_prompt_hash, paragraph)`,
so re-runs of an already-processed corpus on the same model are free.

## Quick start — local server

```bash
# 1. serve (example: vLLM on a 40GB GPU)
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --quantization awq_marlin \
    --max-model-len 8192 --gpu-memory-utilization 0.92 --port 8000
```

```powershell
# 2. point gemynd at it and ingest
$env:CLAIM_LLM_BASE_URL = "http://localhost:8000/v1"
$env:CLAIM_LLM_MODEL    = "Qwen/Qwen2.5-32B-Instruct-AWQ"
py scripts/ingest_tutorstructure.py --mode articles --limit 300 `
    --out-dir out/newspapers_oss_smoke `
    --corpus C:\Users\Benjamin\Desktop\Tutorstructure\corpus
```

## Colab (A100)

See `notebooks/colab_oss_claim_extraction.ipynb`. It installs vLLM, serves the model,
and offers two modes: a **cloudflared tunnel** (serve on Colab, run the pipeline on
your laptop) or an **inline full run** (upload repo + corpus to Drive, run everything
on the A100 — recommended for the whole corpus, no per-paragraph network hop).

## Model sizing on a 40GB A100

- `Qwen/Qwen2.5-32B-Instruct-AWQ` — ~20GB weights, fits with KV-cache headroom. Good default.
- `Qwen/Qwen2.5-14B-Instruct-AWQ` — smaller/faster fallback if you hit OOM.
- 70B-class models do **not** fit in 40GB even at 4-bit; use a smaller model or a bigger GPU.

## Interaction with entity minting

A stronger extractor proposes more — and cleaner — claim-link surfaces. With entity
minting enabled for the newspaper domain (`resources_newspaper/domain_schema.yaml`),
those surfaces become first-class `Person` / `Place` / `Organization` entities, so the
extra links the OSS model proposes actually land as graph edges instead of being
dropped as `NO_RESOLVED_MENTION`. The two changes compound.
