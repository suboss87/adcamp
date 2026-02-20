# AdCamp: AI Video Ad Pipeline for D2C E-Commerce

[![BytePlus ModelArk](https://img.shields.io/badge/BytePlus-ModelArk-blue)](https://www.byteplus.com/en/product/modelark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-13%20passing-brightgreen)]()

> **Built by [Subash Natarajan](https://www.linkedin.com/in/subashn/)** — Reference architecture for cost-optimized AI video generation at scale.

---

## The Problem

D2C e-commerce brands manage **thousands of SKUs** across TikTok, Instagram, and YouTube — each platform demanding its own video format. Manually producing product videos costs **$50-500 per video** through agencies, or **$5-15 per video** using tools like Runway/Sora that don't understand SKU economics. A brand with 2,500 products refreshing content quarterly would spend **$35,000-125,000/year** on video production alone.

**The real insight**: not every product deserves the same video quality. Your top 20% hero products (driving 80% of revenue) need cinematic quality. The other 80% of your catalog just needs good-enough video, fast and cheap.

## The Solution

AdCamp is a **production-ready pipeline** that generates platform-optimized product videos using AI, with intelligent cost routing:

- **Hero SKUs** (top 20% products) route to **Seedance 1.5 Pro** — cinematic quality
- **Catalog SKUs** (bottom 80%) route to **Seedance 1.0 Pro Fast** — 3x faster, 72% cheaper
- **AI scriptwriting** (Seed 1.8) generates ad copy automatically from a campaign brief
- **Multi-platform output** — TikTok (9:16), Instagram (1:1), YouTube (16:9)

**Cost per video: ~$0.33 blended average** (standard BytePlus pricing, 5s 720p with audio)

## Who This Is For

| Audience | Use Case |
|---|---|
| **D2C brands** | Automate product video creation for 1,000+ SKU catalogs |
| **Performance marketing teams** | Generate A/B test variants at near-zero marginal cost |
| **E-commerce agencies** | White-label video production pipeline for multiple clients |
| **AI/ML engineers** | Reference architecture for cost-optimized model routing |
| **Platform teams** | Blueprint for async task pipelines with polling, retry, and observability |

## Architecture

<img width="1166" alt="AdCamp Pipeline Architecture" src="https://github.com/user-attachments/assets/b94bff18-f54d-4962-858e-b4d866afa79a" />

### Pipeline Flow (5 Steps)

| Step | Component | What It Does | Model | Cost |
|------|-----------|-------------|-------|------|
| 1 | **Input** | Campaign brief + product image + SKU tier | — | — |
| 2 | **Script Gen** | AI generates ad copy + video prompt | Seed 1.8 | ~$0.002 |
| 3 | **Smart Router** | Routes hero vs catalog to different models | — | — |
| 4 | **Video Gen** | Async video generation with polling | Seedance Pro/Fast | $0.29-0.49 |
| 5 | **Output** | Platform-ready MP4 via FFmpeg transcoding | — | — |

### Smart Routing — The Core Differentiator

```
Hero SKUs (20%)  ──▶  Seedance 1.5 Pro     ($1.20/M tokens, cinematic quality)
                                              ~$0.49/video (5s, 720p, audio)

Catalog SKUs (80%) ──▶  Seedance 1.0 Pro Fast ($0.70/M tokens, 3x faster)
                                              ~$0.29/video (5s, 720p, audio)

Blended average (20/80 mix): ~$0.33/video
```

> Token formula: `(Width x Height x FPS x Duration) / 1024 x Coefficient` — [BytePlus docs](https://docs.byteplus.com/en/docs/ModelArk/1544106)

### Cost at Scale

| Scale | Videos/Year | Annual Cost | vs Manual ($50/video) |
|-------|-------------|-------------|----------------------|
| Small catalog (500 SKUs) | ~6,900 | ~$2,277 | Save $342,723 |
| Medium catalog (2,500 SKUs) | ~34,500 | ~$11,385 | Save $1,713,615 |
| Large catalog (10,000 SKUs) | ~138,000 | ~$45,540 | Save $6,854,460 |

*Assumes 3 platforms x 30% monthly refresh x 12 months + 25% buffer.*

Enterprise/promotional pricing from BytePlus can reduce costs further — contact BytePlus for volume discounts.

---

## Quick Start

```bash
# Clone
git clone https://github.com/suboss87/adcamp.git
cd adcamp

# Install
make install

# Configure — add your BytePlus ModelArk API key
cp .env.example .env
# Edit .env: ARK_API_KEY=your_key_here

# Run (API on :8000, Dashboard on :8501)
make dev
```

**Generate your first video:**
```bash
python3 examples/generate_single_video.py
```

**Interactive API docs:** http://localhost:8000/docs

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **AI Models** | [BytePlus ModelArk](https://www.byteplus.com/en/product/modelark) | Seed 1.8 (scripts), Seedance Pro/Fast (video) |
| **Backend** | FastAPI + async/await | API server, SSE streaming, async polling |
| **Dashboard** | Streamlit | Campaign management, A/B comparison, analytics |
| **Persistence** | Google Firestore | Campaign and product data |
| **Resilience** | Custom `@retry_with_backoff` | Exponential backoff, rate-limit honoring, error classification |
| **Deployment** | Docker, GCP Cloud Run, Terraform | Multi-platform with IaC |
| **Monitoring** | Prometheus-compatible `/metrics` | Cost tracking, request counts, health checks |

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/generate` | POST | Full pipeline — returns task_id for polling |
| `/api/generate-stream` | POST | Full pipeline with SSE live progress |
| `/api/status/{task_id}` | GET | Poll video generation status |
| `/api/wait/{task_id}` | GET | Block until video ready |
| `/api/campaigns/` | POST | Create a campaign |
| `/api/campaigns/{id}/products/csv` | POST | Upload product CSV |
| `/api/campaigns/{id}/generate` | POST | Start batch generation |
| `/api/cost-summary` | GET | Aggregate cost tracking |
| `/health` | GET | Health + model config |
| `/metrics` | GET | Prometheus text format |

## Deployment

| Platform | Setup | Best For | Guide |
|----------|-------|----------|-------|
| **Docker Compose** | 5 min | Local development | [deploy/docker/](deploy/docker/) |
| **GCP Cloud Run** | 20 min | Production (serverless) | [deploy/gcp/](deploy/gcp/) |
| **Terraform (GCP)** | 30 min | Infrastructure as Code | [deploy/gcp/terraform/](deploy/gcp/terraform/) |
| **AWS ECS** | 30 min | AWS ecosystem | [deploy/aws/](deploy/aws/) |
| **Kubernetes** | 45 min | Full control | [deploy/byteplus/](deploy/byteplus/) |

**Production quickstart (GCP):**
```bash
export GCP_PROJECT_ID=your-gcp-project-id
make deploy-gcp
```

## Project Structure

```
adcamp/
├── app/
│   ├── main.py                 # FastAPI orchestrator
│   ├── config.py               # Pydantic Settings (.env)
│   ├── models/                 # Pydantic schemas
│   ├── services/
│   │   ├── script_writer.py    # Seed 1.8 integration (OpenAI-compatible)
│   │   ├── model_router.py     # Smart tier-based routing
│   │   ├── video_gen.py        # Seedance async task + polling
│   │   ├── cost_tracker.py     # Per-video cost aggregation
│   │   ├── pipeline.py         # Extracted pipeline logic
│   │   ├── batch_generator.py  # Semaphore-based batch processing
│   │   ├── brief_generator.py  # AI brief generation per product
│   │   ├── csv_parser.py       # Product catalog CSV import
│   │   └── firestore_client.py # Firestore persistence
│   ├── routes/
│   │   └── campaigns.py        # Campaign CRUD + batch endpoints
│   └── utils/
│       └── retry.py            # @retry_with_backoff decorator
├── dashboard/
│   ├── app.py                  # Streamlit single-page app
│   ├── sections.py             # Tab rendering functions
│   └── config.py               # Design tokens + API base
├── deploy/                     # Docker, GCP, AWS, K8s, monitoring
├── examples/                   # Runnable scripts
├── tests/                      # Unit tests (13 passing)
└── docs/                       # Architecture docs + guides
```

## Testing

```bash
make test                                    # All tests with coverage
pytest tests/unit/test_model_router.py -v    # Router tests only
pytest tests/unit/test_csv_parser.py -v      # CSV parser tests only
```

## Documentation

- **[DEPLOY.md](DEPLOY.md)** — GCP Cloud Run deployment guide
- **[AGENTS.md](AGENTS.md)** — Detailed architecture for AI coding agents
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines
- **[examples/](examples/)** — Runnable Python scripts
- **API Docs** — http://localhost:8000/docs (Swagger) / http://localhost:8000/redoc

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
make install    # Setup environment
make test       # Run tests (13 passing)
make lint       # Check code style
```

## License

MIT License — see [LICENSE](LICENSE).

---

**Built by [Subash Natarajan](https://www.linkedin.com/in/subashn/)** | Powered by [BytePlus ModelArk](https://www.byteplus.com/en/product/modelark) | [View Source](https://github.com/suboss87/adcamp)
