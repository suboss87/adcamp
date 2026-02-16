# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**AdCamp** is an enterprise-ready AI video generation pipeline for D2C e-commerce, built on BytePlus ModelArk APIs. It generates platform-optimized product videos at scale with intelligent cost optimization through smart model routing.

### Key Architecture Concept: Smart Model Routing

The pipeline's core differentiator is **automatic tier-based routing**:
- **Hero SKUs** (top 20% products) → `Seedance 1.5 Pro` ($1.20/M tokens, premium quality)
- **Catalog SKUs** (80% inventory) → `Seedance 1.0 Pro Fast` ($0.70/M tokens, cost-optimized)

This achieves **$0.08/video** average cost (50% under target), enabling 34,500+ videos/year for ~$2,760.

### Pipeline Flow (6 Steps)

1. **Input**: Campaign brief + optional product image + SKU tier
2. **Script Generation**: `Seed 1.8` generates ad copy, scene description, and video prompt (~5s, $0.002)
3. **Smart Router**: Routes to appropriate Seedance model based on SKU tier
4. **Video Generation**: Async task creation via ModelArk API (~30s, $0.076-0.132)
5. **Post-Processing**: FFmpeg generates platform variants (TikTok 9:16, Instagram 1:1, YouTube 16:9)
6. **Output**: Platform-ready MP4 files

## Development Commands

### Quick Start
```bash
make install    # Setup Python venv, install dependencies
make dev        # Run FastAPI (8000) + Streamlit dashboard (8501)
```

### Testing
```bash
make test       # Run pytest with coverage
pytest tests/unit/test_model_router.py -v    # Run specific test file
pytest -k "test_hero" -v                     # Run tests matching pattern
```

### Code Quality
```bash
make lint       # Run ruff + black checks
black app/ dashboard/    # Format code
ruff check app/ --fix    # Auto-fix linting issues
```

### Docker
```bash
make docker-build       # Build image (multi-stage, includes FFmpeg)
make docker-up          # Start services via docker-compose
make docker-down        # Stop services
```

### Deployment
```bash
# GCP Cloud Run (requires GCP_PROJECT_ID env var)
export GCP_PROJECT_ID=adcamp-487609
make deploy-gcp

# Build dashboard separately
gcloud builds submit --config=cloudbuild-dashboard.yaml . --project=adcamp-487609

# Check deployment logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50 --project=adcamp-487609
```

## Architecture & Code Structure

### Service Layer (`app/services/`)

**Critical Design Pattern**: All services implement async/await and use `@retry_with_backoff` decorator for resilience.

1. **`script_writer.py`** - Seed 1.8 integration
   - Uses OpenAI client (ModelArk is OpenAI-compatible)
   - Returns `(AdScript, input_tokens, output_tokens)` for cost tracking
   - System prompt enforces JSON output with motion-focused video prompts

2. **`model_router.py`** - Smart routing logic
   - Pure function: `route(sku_tier) -> (model_id, cost_per_m)`
   - Decision point: `SKUTier.hero` vs `SKUTier.catalog`
   - Returns both model ID and cost for downstream tracking

3. **`video_gen.py`** - Seedance video generation
   - **`create_video_task()`**: POST to `/contents/generations/tasks`, returns `task_id`
   - **`get_video_status()`**: GET task status, parses `succeeded`/`failed`/`running`
   - **`wait_for_video()`**: Polls until completion (timeout: 300s, interval: 5s)
   - **Image requirement**: Min 300px height (ModelArk enforces this)

4. **`cost_tracker.py`** - Cost calculation
   - Combines script tokens (Seed 1.8) + estimated video tokens
   - Returns `CostSummary` with per-component breakdown

5. **`post_process.py`** - FFmpeg platform variants
   - Converts single video to multiple aspect ratios
   - Uses subprocess to call FFmpeg (must be in PATH/Docker image)

### Retry Logic (`app/utils/retry.py`)

**Critical for production**: All ModelArk API calls wrapped in `@retry_with_backoff`.

- **Retryable errors**: Network failures, 5xx, 429 (rate limits)
- **Non-retryable**: 4xx (except 429), invalid API keys, image validation errors
- **Honors `Retry-After` header** for rate limit recovery
- **Custom exceptions**: `InvalidAPIKeyError`, `RateLimitError`, `QuotaExceededError`

### API Endpoints (`app/main.py`)

- **`POST /api/generate`**: Full pipeline (steps 1-4), returns immediately with `task_id`
- **`GET /api/status/{task_id}`**: Poll video generation status
- **`GET /api/wait/{task_id}`**: Blocking call (waits for completion)
- **`GET /health`**: Health check with model config + metrics
- **`GET /metrics`**: Prometheus text format
- **`GET /health/detailed`**: Extended health with dependency status

### Environment Configuration (`app/config.py`)

Uses Pydantic Settings for type-safe config:

```python
ARK_API_KEY           # Required: BytePlus ModelArk API key
ARK_BASE_URL          # Default: https://ark.ap-southeast.bytepluses.com/api/v3
script_model          # Default: seed-1-8-251228
video_model_pro       # Default: seedance-1-5-pro-251215 (Hero)
video_model_fast      # Default: seedance-1-0-pro-fast-251015 (Catalog)
```

### Monitoring (`app/monitoring.py`)

In-memory metrics store (Prometheus-compatible):
- **Counters**: `videos_generated_total`, `videos_failed_total`, `hero_videos`, `catalog_videos`
- **Gauges**: `script_generation_avg_seconds`, `video_generation_avg_seconds`, `total_cost_usd`
- **Histogram**: Duration tracking for script/video generation

Metrics are **not persisted** across restarts (extend to Redis/Prometheus for production).

## Deployment Architecture

### Platform Organization (`deploy/`)

- **`deploy/docker/`**: Dockerfile (API + FFmpeg), docker-compose.yml, Dockerfile.dashboard (Streamlit)
- **`deploy/gcp/`**: Cloud Run configs, Terraform templates, deployment scripts
- **`deploy/byteplus/`**: VKE Kubernetes manifests (namespace, deployment, service, ingress)
- **`deploy/kubernetes/`**: Generic K8s with Kustomize (base + overlays for dev/staging/prod)

### Docker Image Details

**Multi-stage build** (`deploy/docker/Dockerfile`):
1. **Builder stage**: Compile Python dependencies
2. **Runtime stage**: Copy deps + FFmpeg + app code
3. **Key**: Uses `ENV PORT=${PORT}` for Cloud Run compatibility (defaults to 8000, Cloud Run overrides to 8080)
4. **CMD**: `uvicorn app.main:app --host 0.0.0.0 --port ${PORT}`

### Current Deployment

- **Project**: `adcamp-487609` (GCP)
- **API**: https://adcamp-api-309502792454.asia-southeast1.run.app
- **Dashboard**: https://adcamp-dashboard-309502792454.asia-southeast1.run.app
- **Secret**: `adcamp-ark-api-key` in GCP Secret Manager
- **Images**: `gcr.io/adcamp-487609/adcamp:latest`, `gcr.io/adcamp-487609/adcamp-dashboard:latest`

## Common Patterns & Conventions

### Error Handling

1. **Always use `@retry_with_backoff`** for external API calls
2. **Parse ModelArk errors** with `parse_modelark_error()` for user-friendly messages
3. **Log errors with context**: Include SKU ID, task ID, model used
4. **Return detailed responses**: Include error message in API response, not just HTTP status

### Async Programming

- **All services are async** (script_writer, video_gen)
- **Use `httpx.AsyncClient`** for HTTP calls (not `requests`)
- **Use `asyncio.sleep()`** for polling delays
- **FastAPI endpoints** should be `async def` when calling services

### Cost Tracking

- **Track tokens at every step**: Script input/output, video tokens
- **Estimate video tokens** from duration + resolution (actual from API when available)
- **Store per-tier costs**: Separate hero vs catalog in metrics
- **Include cost in responses**: Users should always see cost breakdown

### Image Validation

**Critical**: ModelArk requires images ≥300px height. Validate before sending to API to avoid cryptic 400 errors.

### Testing Against Real API

- **Use `.env` file** for local API key (never commit)
- **Test with catalog tier first** (cheaper for development)
- **Test without images initially** (simplifies debugging)
- **Monitor costs**: Each test video costs $0.08-0.13

## Security & Secrets

- **API keys**: Stored in GCP Secret Manager (prod), `.env` (local)
- **Never commit** `.env`, API keys, or credentials
- **Service accounts**: Cloud Run default compute SA needs `roles/secretmanager.secretAccessor`
- **CORS**: Currently allows all origins (restrict in production if needed)

## Known Limitations & Gotchas

1. **Cold start**: First API request after idle takes ~10s (Cloud Run scales to zero)
2. **Video timeout**: Default 300s, can be increased but costs more
3. **Image size**: ModelArk requires min 300px height (267px will fail with cryptic error)
4. **Metrics persistence**: In-memory only, resets on container restart
5. **No video download**: Videos returned as URLs (hosted by ModelArk), not stored locally
6. **Post-processing**: Currently not implemented in API endpoints (only in services layer)
7. **Dashboard API URL**: Must be set via `API_URL` env var (defaults to localhost)

## BytePlus ModelArk Specifics

### API Authentication
```python
headers = {
    "Authorization": f"Bearer {ARK_API_KEY}",
    "Content-Type": "application/json"
}
```

### Endpoint Structure
- Base: `https://ark.ap-southeast.bytepluses.com/api/v3`
- Script: `POST /chat/completions` (OpenAI-compatible)
- Video task: `POST /contents/generations/tasks`
- Video status: `GET /contents/generations/tasks/{task_id}`

### Response Formats
- **Script**: Standard OpenAI format with `usage.prompt_tokens`, `usage.completion_tokens`
- **Video task**: Returns `{"id": "task_id", ...}`
- **Video status**: Returns `{"status": "succeeded|failed|processing", "content": {"video_url": "..."}}`

### Rate Limits
- Not explicitly documented, but honors `Retry-After` header
- Recommended: 2-10 concurrent video generations
- Script generation: Higher throughput than video

## Repository Organization Note

This repo follows **Google/Kubernetes/Terraform standards**:
- Platform-separated deployment configs in `deploy/`
- Layered documentation in `docs/` (architecture, guides, API)
- Working examples in `examples/`
- Test structure in `tests/` (unit, integration, fixtures)
- Single-entry Makefile for all common tasks
- Community files: CONTRIBUTING.md, SECURITY.md, CHANGELOG.md
