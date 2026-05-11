# 🍽️ Yelp Hybrid Recommendation System

A **Hybrid Restaurant Recommendation System** built on the [Yelp Open Dataset](https://www.yelp.com/dataset), combining **Content-Based Filtering** (TF-IDF) and **Collaborative Filtering** (SVD) with **adaptive alpha weighting** for cold-start handling.

> **Graduation Thesis Project** — Computer Science, Hung Yen University of Technology and Education (UTEHY)
> Student: Vu Quang Phuc · Supervisor: Ph.D Pham Minh Chuan

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [How It Works](#how-it-works)

---

## Overview

Choosing a restaurant from thousands of options is overwhelming. Traditional approaches each have limitations:

| Approach | Strength | Weakness |
|----------|----------|----------|
| **Content-Based** | Works for new users | Filter bubble (over-specialization) |
| **Collaborative** | Discovers non-obvious preferences | Cold-start problem |
| **This Project (Hybrid)** | Best of both | Adaptive weighting solves cold-start |

This system uses a **Weighted Hybrid** approach where the blending parameter α adapts automatically:
- **New users** (few interactions) → α ≈ 1.0 → content-dominant recommendations
- **Active users** (many interactions) → α → 0.2 → collaborative-dominant recommendations

**Scale**: 32,952 US restaurants · 346,716 users · 3M+ interactions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                               │
│  Raw Yelp JSON/Parquet ──► preprocess_yelp.py ──► .pkl files    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        ML Core (src/)                           │
│                                                                 │
│  ContentEngine          CFEngine           HybridRecommender    │
│  TF-IDF (80K feat.)     Bias-aware SVD     Adaptive α blending  │
│  + Cosine Similarity    (64 components)    + Popularity Prior   │
│                                                                 │
│  Evaluator ◄── Temporal Leave-Last-Out (P@K, R@K, NDCG@K)      │
│  Pipeline  ◄── End-to-end orchestration                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ model_bundle.pkl (~273MB)
┌──────────────────────────────▼──────────────────────────────────┐
│                     Serving Layer (api/)                        │
│                                                                 │
│  FastAPI ──► 5 Routers:                                        │
│    • public  → /health, /recommend, /business/search           │
│    • auth    → /auth/signup, /auth/signin, /me/rate            │
│    • ux      → /ux/login, /ux/recommend, /ux/search            │
│    • admin   → /admin/config, /admin/stats, /admin/retrain     │
│    • pages   → /, /basic, /experience, /management             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       Frontend (HTML)                           │
│                                                                 │
│  /basic       → Demo with explainability + interactive map     │
│  /experience  → UX login → paginated feed + search             │
│  /management  → Admin dashboard + retrain controls             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML | NumPy, Pandas, SciPy, scikit-learn |
| Data | Parquet (PyArrow), Pickle, JSON |
| API | FastAPI, Uvicorn |
| Frontend | Jinja2 templates (HTML/CSS/JS) |
| Data Pipeline | Chunked I/O, Spark (EDA notebooks) |

---

## Project Structure

```
├── src/                        # ML core modules
│   ├── content_base.py         # TF-IDF content engine
│   ├── collab_filter.py        # Bias-aware SVD CF engine
│   ├── hybrid_engine.py        # Adaptive α hybrid blending
│   ├── evaluator.py            # Offline evaluation metrics
│   ├── pipeline.py             # End-to-end pipeline orchestration
│   ├── model_bundle.py         # Model serialization & enriched output
│   ├── preprocess.py           # Data loading & feature engineering
│   └── local_user_store.py     # Local user interaction storage
│
├── api/                        # FastAPI serving layer
│   ├── main.py                 # App entrypoint
│   ├── runtime.py              # Core business logic
│   ├── schemas.py              # Pydantic models
│   ├── routers/                # Route handlers
│   │   ├── public.py           # Health, search, recommend
│   │   ├── auth.py             # Signup, signin, rate
│   │   ├── ux.py               # UX experience endpoints
│   │   ├── admin.py            # Admin operations
│   │   └── pages.py            # HTML page serving
│   └── templates/              # Jinja2 HTML templates
│       ├── index.html          # Basic demo page
│       ├── login.html          # UX login page
│       ├── app.html            # UX feed/recommendation app
│       └── management.html     # Admin dashboard
│
├── scripts/                    # CLI scripts
│   ├── preprocess_yelp.py      # Preprocess raw data
│   ├── train_recommender.py    # Train & evaluate model
│   ├── run_pipeline.py         # Full pipeline shortcut
│   ├── run_full_usa_retrain.py # Full US retrain shortcut
│   └── checkpoint.py           # Deployment health check
│
├── notebooks/                  # Jupyter notebooks (EDA + workflows)
├── data/                       # Parquet exports (gitignored)
├── raw_data/                   # Raw Yelp JSON files (gitignored)
├── processed/                  # Preprocessed .pkl files
├── artifacts/                  # Model bundle (gitignored)
├── local_data/                 # Local user data
├── docs/                       # Reports, references, notes
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── scripts.md                  # Detailed runbook
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Yelp Open Dataset](https://www.yelp.com/dataset) (download separately)

### Installation

```bash
# Clone the repository
git clone https://github.com/lovecrushfamily/Yelp_Recomendation_System_FinUnipro.git
cd Yelp_Recomendation_System_FinUnipro

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place the Yelp dataset files in `raw_data/`:
```
raw_data/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
├── yelp_academic_dataset_user.json
├── yelp_academic_dataset_tip.json
└── yelp_academic_dataset_checkin.json
```

### Quick Start (Partial Data)

```bash
# 1. Preprocess (partial data for fast testing)
python3 scripts/run_pipeline.py \
  --dataset-mode yelp_only \
  --max-review-chunks 8 \
  --max-aux-chunks 8 \
  --max-eval-users 200

# 2. Start the API
uvicorn api.main:app --reload
```

### Full Data Pipeline

```bash
# Full US restaurants preprocessing + training
python3 scripts/run_full_usa_retrain.py \
  --input-format parquet \
  --data-dir data \
  --dataset-mode yelp_only \
  --max-eval-users 300

# Start the API
uvicorn api.main:app --reload
```

---

## Usage

Once the API is running at `http://127.0.0.1:8000`:

| URL | Description |
|-----|-------------|
| `/basic` | Basic demo with explainability breakdown and interactive map |
| `/experience` | UX login → personalized recommendation feed |
| `/management` | Admin dashboard with stats, retrain controls |
| `/docs` | Swagger API documentation |
| `/health` | Health check endpoint |
| `/checkpoint` | Pipeline health status |

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/checkpoint` | Pipeline deployment status |
| `GET` | `/users` | List available users |
| `POST` | `/recommend` | Get recommendations for a user |
| `GET` | `/business/search` | Search businesses by keyword/category |

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/signup` | Register a new local user |
| `POST` | `/auth/signin` | Sign in and get auth token |
| `POST` | `/me/rate` | Submit a business rating |
| `GET` | `/me/history` | Get rating history |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/config` | View runtime configuration |
| `PATCH` | `/admin/config` | Update runtime configuration |
| `GET` | `/admin/stats` | System statistics |
| `POST` | `/admin/retrain` | Trigger model retraining |

---

## Evaluation

### Methodology

- **Protocol**: Temporal Leave-Last-Out — hold out each user's most recent interaction as the test item
- **Metrics**: Precision@K, Recall@K, NDCG@K, Hit Rate@K, Coverage
- **Configuration**: K=10, relevance threshold ≥ 4.0 stars, evaluated on 300 users

### Results

| Metric | Full-Catalog Eval | Sampled Eval (100 negatives) |
|--------|-------------------|------------------------------|
| Precision@10 | 0.03% | — |
| Recall@10 | 0.33% | — |
| NDCG@10 | 0.26% | — |
| Hit Rate@10 | — | — |
| Coverage | 1.37% | — |

> **Note on Full-Catalog Metrics**: With 33K candidate items and only 1 held-out test item per user, the theoretical random baseline for Precision@10 is ~0.03%. Low absolute values are expected and are a well-known challenge in recommendation evaluation on large catalogs ("needle in a haystack" problem). Sampled evaluation with negative candidates provides more interpretable metrics.

---

## How It Works

### 1. Content-Based Filtering

Builds a text profile ("soup") for each business from its name, categories, and attributes. Uses **TF-IDF** (80K features, bigrams) to vectorize, then computes **cosine similarity** between a user's weighted history profile and candidate businesses. Recent interactions are weighted higher via exponential decay.

### 2. Collaborative Filtering

Uses **Bias-Aware SVD** to decompose the user-item rating matrix:

```
r̂(u,i) = μ + bᵤ + bᵢ + pᵤ · qᵢ
```

Where μ = global mean, bᵤ = user bias, bᵢ = item bias, pᵤ/qᵢ = 64-dimensional latent factors learned via TruncatedSVD on rating residuals.

### 3. Hybrid Blending

Combines both signals with an **adaptive weight α** that adapts to user cold-start:

```
α(n) = 0.2 + 0.8 × exp(-n / 12)

score = (1 - 0.1) × [α × content + (1-α) × cf_normalized] + 0.1 × popularity_prior
```

- **n=0** (new user): α ≈ 1.0 → pure content recommendations
- **n=12**: α ≈ 0.49 → balanced
- **n=50**: α ≈ 0.21 → collaborative-dominant

### 4. Candidate Generation

Union of top-800 CF candidates + top-800 content candidates + top-400 popularity candidates → score all → return top-K.

---

## License

This project uses the [Yelp Open Dataset](https://www.yelp.com/dataset) for academic purposes under the Yelp Dataset License Agreement.

---

## Acknowledgments

- [Yelp Open Dataset](https://www.yelp.com/dataset) for providing rich, real-world restaurant data
- [scikit-learn](https://scikit-learn.org/) for TF-IDF and TruncatedSVD implementations
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance API framework
