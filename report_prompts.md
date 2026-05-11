# Prompts Viết Báo Cáo — Yelp Hybrid RecSys

> Mỗi prompt dưới đây là **self-contained** — copy nguyên block rồi paste vào Gemini/Claude.
> Output sẽ là nội dung tiếng Anh, bạn paste thẳng vào báo cáo.

---

## PROMPT 1 — GLOSSARY OF TERMS

```
You are writing the Glossary of Terms table for a graduation thesis about a Yelp Hybrid Recommendation System.

The project uses these technologies and concepts:
- TF-IDF (Term Frequency–Inverse Document Frequency) for content-based filtering
- SVD (Singular Value Decomposition) for collaborative filtering — specifically TruncatedSVD on rating residuals
- Cosine Similarity for measuring item similarity
- Hybrid Recommender combining content-based + collaborative filtering with adaptive alpha weighting
- Cold-start problem
- Popularity Prior (normalized score combining tip count, checkin count, review count)
- FastAPI for serving the recommendation API
- Parquet format for columnar data storage
- NDCG (Normalized Discounted Cumulative Gain)
- Precision@K, Recall@K
- Temporal Leave-Last-Out split for evaluation
- F&B (Food and Beverage)
- RecSys (Recommendation System)
- API (Application Programming Interface)
- EDA (Exploratory Data Analysis)

Generate a table with columns: Index | Term | Full Form | Meaning
Include 12-15 most important terms. Keep meanings concise (1 sentence each). Order alphabetically by Term.
Output ONLY the table content rows, no headers or formatting instructions.
```

---

## PROMPT 2 — PREFACE

```
Write a PREFACE section (150-200 words) for a graduation thesis titled "Yelp Recommendation System".

Context:
- Student: Vu Quang Phuc, Computer Science major, Hung Yen University of Technology and Education
- Supervisor: Ph.D Pham Minh Chuan
- The project builds a Hybrid Recommendation System for restaurants on the Yelp platform
- It combines Content-Based Filtering (TF-IDF) and Collaborative Filtering (SVD) with an adaptive blending mechanism
- The system processes the Yelp Open Dataset (~33,000 US restaurants, ~347,000 users)
- A full-stack prototype was built: Python ML pipeline + FastAPI backend + web-based demo UI
- The project demonstrates practical application of machine learning to solve information overload in restaurant discovery

The preface should:
- Briefly state the motivation (information overload in dining choices)
- Mention the approach (hybrid recommendation combining content and collaborative methods)
- Note the deliverables (ML pipeline, API, demo application)
- Be written in formal academic English

Output ONLY the preface paragraphs, no title.
```

---

## PROMPT 3 — GENERAL INTRODUCTION: Problem Statement

```
Write the "Problem Statement" subsection (250-350 words) for a graduation thesis chapter called "GENERAL INTRODUCTION". The project is a Yelp Hybrid Recommendation System.

Key points to cover:
1. The explosion of online restaurant/F&B information on platforms like Yelp (150M+ reviews), Google Maps, Foody, TripAdvisor
2. Information overload: users struggle to find suitable restaurants among thousands of options
3. Limitations of existing approaches:
   - Simple star-rating sorting ignores personal preferences
   - Content-based filtering alone suffers from over-specialization (filter bubble)
   - Collaborative filtering alone suffers from cold-start problem for new users/items
   - Data sparsity: most users only rate a tiny fraction of available businesses
4. The need for a hybrid approach that combines multiple signals to overcome individual method weaknesses
5. Business value: better recommendations increase user satisfaction, restaurant visibility, and platform engagement

Write in formal academic English. Include 1-2 citations placeholders like [1], [2] where relevant academic references would go.
Output ONLY the content paragraphs.
```

---

## PROMPT 4 — GENERAL INTRODUCTION: Scope, Planning & My Work

```
Write two subsections for the "GENERAL INTRODUCTION" chapter of a graduation thesis.

### Subsection 1: "Scope and Planning" (150-200 words)
The project scope:
- Dataset: Yelp Open Dataset (publicly available academic dataset)
- Domain: Food & Beverage businesses (restaurants, cafes, bars) in the United States
- Scale: ~33,000 businesses after filtering, ~347,000 users, millions of reviews
- Technical focus: Weighted Hybrid method combining Content-Based (TF-IDF) and Collaborative Filtering (Bias-aware SVD) with dynamic alpha weighting
- Output: functional prototype with web-based demo UI, not production cloud deployment
- Four development phases:
  Phase 1: Data exploration and preprocessing (EDA, filtering, feature engineering)
  Phase 2: Model development (Content engine, CF engine, Hybrid blending)
  Phase 3: System integration (FastAPI API, model bundling, serving pipeline)
  Phase 4: Evaluation and demo (offline metrics, explainability, admin dashboard)

### Subsection 2: "My Work" (150-200 words)
Personal contributions (this is a solo project):
- Designed and implemented the full data preprocessing pipeline (JSON/Parquet ingestion, chunked loading, US-state filtering, popularity prior computation, TF-IDF soup feature engineering)
- Built the Content-Based Engine (TF-IDF vectorization with 80K features, cosine similarity scoring)
- Built the Collaborative Filtering Engine (bias-aware TruncatedSVD with 64 latent factors)
- Designed the Hybrid Recommender with adaptive alpha weighting (exponential decay based on user interaction count for cold-start handling)
- Implemented offline evaluation (temporal leave-last-out, Precision@K, Recall@K, NDCG@K, Coverage)
- Built the serving layer (FastAPI with 5 router modules, model bundle serialization, user session management)
- Created 4 web-based demo interfaces (Basic demo with explainability + map, UX login/feed, Admin management dashboard)
- Wrote orchestration scripts for end-to-end pipeline automation

Output in formal academic English. Output ONLY the content for both subsections, clearly separated.
```

---

## PROMPT 5 — THEORETICAL BACKGROUND: Recommendation Systems Overview

```
Write the "Recommendation Systems" section (500-700 words) for the THEORETICAL BACKGROUND chapter of a graduation thesis about a Yelp Hybrid Recommendation System.

Cover these topics with academic rigor:

1. **Definition and Role** (1 paragraph)
   - What recommendation systems are (information filtering systems that predict user preferences)
   - Why they matter (information overload, personalization, business value)
   - Types of feedback: explicit (ratings, likes) vs implicit (clicks, views, purchase history)

2. **Content-Based Filtering** (1-2 paragraphs)
   - Core idea: recommend items similar to what the user liked before
   - How it works: build item profiles from features, build user profiles from interaction history, compute similarity
   - Common techniques: TF-IDF for text features, cosine similarity
   - Advantages: no cold-start for new items (if features available), transparent/explainable
   - Limitations: over-specialization (filter bubble), cannot capture collaborative patterns, feature engineering dependent

3. **Collaborative Filtering** (1-2 paragraphs)
   - Core idea: users who agreed in the past will agree in the future
   - Memory-based: user-user or item-item similarity (KNN approaches)
   - Model-based: Matrix Factorization (SVD, ALS, NMF) — decompose user-item matrix into latent factors
   - Advantages: discovers non-obvious preferences, no need for item features
   - Limitations: cold-start problem (new users/items), data sparsity, scalability

4. **Hybrid Approaches** (1-2 paragraphs)
   - Motivation: combine strengths of content-based and collaborative to overcome individual weaknesses
   - Types: weighted, switching, mixed, feature combination, cascade, meta-level
   - Focus on Weighted Hybrid: blend scores from both systems using a weighting parameter
   - The concept of adaptive/dynamic weighting (adjusting weights based on user context — e.g., interaction count)

Include citation placeholders [1]-[6] where academic references would naturally appear.
Use formal academic English. Output ONLY the content paragraphs, organized under clear sub-headings.
```

---

## PROMPT 6 — THEORETICAL BACKGROUND: Key Algorithms Used

```
Write a section called "Key Algorithms and Techniques" (400-500 words) for the THEORETICAL BACKGROUND chapter. This section explains the specific algorithms used in the Yelp Hybrid Recommendation System project.

Cover:

1. **TF-IDF (Term Frequency–Inverse Document Frequency)**
   - Formula: TF-IDF(t,d) = TF(t,d) × IDF(t)
   - Purpose in this project: convert business metadata (name, categories, attributes) into numerical feature vectors
   - Configuration used: max_features=80,000, ngram_range=(1,2), min_df=2, English stop words removed

2. **Cosine Similarity**
   - Formula: cos(A,B) = (A·B) / (||A|| × ||B||)
   - Purpose: measure similarity between business TF-IDF vectors and user profile vectors
   - Range: [0, 1] for non-negative TF-IDF vectors

3. **Bias-Aware SVD (Singular Value Decomposition)**
   - Standard SVD decomposes rating matrix R ≈ U × Σ × V^T
   - Bias-aware approach: r̂(u,i) = μ + b_u + b_i + p_u · q_i
     where μ = global mean, b_u = user bias, b_i = item bias, p_u/q_i = latent factors
   - Implementation: compute residuals (ratings - baseline), then apply TruncatedSVD on residual matrix
   - Configuration: n_components=64, ratings clipped to [1.0, 5.0]

4. **Adaptive Alpha Weighting**
   - Formula: α(n) = α_min + (1 - α_min) × exp(-n / half_life)
   - Where n = number of user interactions, α_min = 0.2, half_life = 12
   - When n=0 (new user): α ≈ 1.0 → content-dominant
   - When n is large: α → 0.2 → collaborative-dominant
   - Final hybrid score: score = (1 - pw) × [α × content_score + (1-α) × cf_score] + pw × prior_score
   - pw (prior_weight) = 0.1 for popularity signal injection

Write in formal academic English with mathematical notation where appropriate.
Include citation placeholders where relevant.
Output ONLY the content.
```

---

## PROMPT 7 — IMPLEMENTATION: Data Collection & Preprocessing

```
Write the "Data Collection and Preprocessing" section (400-500 words) for the IMPLEMENTATION chapter of a graduation thesis.

The project uses the Yelp Open Dataset. Here are the exact details from the actual implementation:

**Raw Data Sources:**
- yelp_academic_dataset_business.json → business metadata (name, city, state, categories, attributes, stars, review_count, latitude, longitude, hours)
- yelp_academic_dataset_review.json → user reviews (user_id, business_id, stars, date, text)
- yelp_academic_dataset_tip.json → short tips from users
- yelp_academic_dataset_checkin.json → check-in timestamps
- Data also available as pre-exported Parquet files for faster loading

**Preprocessing Pipeline (implemented in src/preprocess.py):**
1. Business filtering:
   - Keep only businesses with categories containing "Restaurants"
   - Filter to US states only (51 state codes including DC)
   - Minimum review count threshold: 10 reviews
   - Result: ~33,000 qualifying businesses

2. Feature Engineering — "Soup" column:
   - Concatenate: business name + categories + attributes text
   - Lowercase normalization
   - This soup column feeds into TF-IDF vectorization

3. Popularity Prior computation:
   - Combine tip_count, checkin_count, review_count into a normalized [0,1] popularity score
   - Used as a business-level prior signal in the hybrid model

4. Review loading:
   - Chunked reading (200,000 rows per chunk) to handle memory constraints
   - Filter to only reviews for qualifying businesses
   - Parse dates for temporal ordering

5. Interaction table construction:
   - Aggregate multiple reviews per (user, business) pair → mean rating, latest date
   - Filter: users with ≥3 interactions, businesses with ≥5 interactions
   - Final interaction matrix: ~347,000 users

6. Output artifacts:
   - processed/businesses.pkl — business metadata with engineered features
   - processed/interactions.pkl — clean user-item interaction table
   - processed/preprocess_summary.json — pipeline statistics

Write in formal academic English. Describe the pipeline as a systematic data engineering process.
Output ONLY the content paragraphs.
```

---

## PROMPT 8 — IMPLEMENTATION: Model Architecture & Training

```
Write the "Model Architecture and Training" section (500-600 words) for the IMPLEMENTATION chapter.

Describe the three-component architecture and training process based on these exact implementation details:

**Component 1: Content-Based Engine (src/content_base.py)**
- Input: business DataFrame with "soup" text column
- Process: TfidfVectorizer(max_features=80000, ngram_range=(1,2), min_df=2, stop_words='english')
- Output: sparse TF-IDF matrix (33K businesses × up to 80K features)
- Recommendation method: build weighted user profile from history businesses, compute cosine similarity against candidates
- Supports recency-weighted history: recent interactions weighted higher via exponential decay

**Component 2: Collaborative Filtering Engine (src/collab_filter.py)**
- Input: interaction DataFrame (user_id, business_id, stars)
- Process:
  1. Compute global mean rating μ
  2. Compute user bias b_u = mean(user_ratings) - μ
  3. Compute item bias b_i = mean(item_ratings) - μ
  4. Compute residuals: r_residual = rating - (μ + b_u + b_i)
  5. Build sparse residual matrix (347K users × 33K items)
  6. Apply TruncatedSVD(n_components=64) → user_factors, item_factors
- Prediction: r̂(u,i) = μ + b_u + b_i + dot(user_factor_u, item_factor_i), clipped to [1.0, 5.0]
- Vectorized batch prediction for efficiency

**Component 3: Hybrid Recommender (src/hybrid_engine.py)**
- Adaptive alpha: α(n) = 0.2 + 0.8 × exp(-n/12)
  - n=0 → α=1.0 (pure content for cold-start users)
  - n=12 → α≈0.49 (balanced)
  - n=50 → α≈0.21 (collaborative-dominant for active users)
- CF score normalization: map predicted rating from [1,5] to [0,1]
- Final score: (1-0.1) × [α × content_score + (1-α) × cf_normalized] + 0.1 × popularity_prior
- Candidate generation: union of top-800 CF candidates + top-800 content candidates + top-400 popularity candidates

**Training Pipeline (src/pipeline.py + scripts/train_recommender.py):**
1. Load preprocessed businesses.pkl and interactions.pkl
2. Fit ContentEngine on business soup features
3. Fit CFEngine on interaction matrix
4. Wire popularity priors into HybridRecommender
5. Evaluate using temporal holdout (see next section)
6. Refit on full data for serving
7. Build user histories and activity logs
8. Serialize everything into model_bundle.pkl (~273MB)

**Model Bundle (src/model_bundle.py):**
- Contains: content_engine, cf_engine, hybrid_recommender, business_index, user_histories, user_events, evaluation metrics
- Enriched recommendation output includes: score, explanation text, reason_tags, content_score, cf_predicted_rating, hybrid_alpha, category_overlap with user preferences

Write in formal academic English. Use technical terminology precisely.
Output ONLY the content paragraphs with clear sub-headings.
```

---

## PROMPT 9 — IMPLEMENTATION: Evaluation

```
Write the "Model Evaluation" section (300-400 words) for the IMPLEMENTATION chapter.

Based on these exact implementation details:

**Evaluation Protocol (src/evaluator.py):**
- Method: Temporal Leave-Last-Out split
  - For each user with ≥ (min_history + n_test_items) interactions, hold out the last n_test_items by date
  - Train set: all interactions except the last one per user
  - Test set: the most recent interaction per user
- Configuration: min_history=3, n_test_items=1, relevance_threshold=4.0
- Only held-out items with rating ≥ 4.0 stars are considered "relevant"
- Evaluated on max 300 users

**Metrics:**
- Precision@K: fraction of top-K recommendations that are relevant
- Recall@K: fraction of relevant items found in top-K
- NDCG@K: position-aware ranking quality metric
- Coverage: fraction of total item catalog recommended across all users

**Results (from bundle_summary.json):**
- evaluated_users: 300
- Precision@10: 0.00033
- Recall@10: 0.0033
- NDCG@10: 0.0026
- Coverage: 0.0137

**Analysis of Results:**
The metrics appear very low but this is expected given the evaluation setup:
1. With 33,000 candidate businesses and only 1 held-out item per user, the random baseline for Precision@10 is approximately 10/33000 ≈ 0.03%, which is close to the observed result
2. The strict relevance threshold (4.0 stars) further reduces the number of valid test cases
3. This is a well-known challenge in recommendation evaluation on large item catalogs — the "needle in a haystack" problem
4. More informative evaluation would use sampled negative candidates (e.g., 100-1000 negatives per positive) or metrics like Hit Rate@K
5. The coverage of 1.37% indicates the model tends to recommend popular items — a known behavior of SVD-based approaches

The evaluation methodology follows established academic practice for offline recommendation evaluation, though the absolute metric values should be interpreted in context of the catalog size.

Write in formal academic English. Present the results honestly with proper analysis.
Output ONLY the content paragraphs.
```

---

## PROMPT 10 — IMPLEMENTATION: System Architecture & Demo

```
Write the "System Architecture and Demonstration" section (400-500 words) for the IMPLEMENTATION chapter.

Based on these exact implementation details:

**Technology Stack:**
- Backend: Python 3.10+, FastAPI 0.115+, Uvicorn ASGI server
- ML Libraries: NumPy, Pandas, SciPy, scikit-learn, PyArrow
- Data: Parquet columnar format for efficient I/O
- Frontend: Server-rendered HTML templates (Jinja2) with inline CSS/JavaScript
- No external database — file-based storage (pickle, JSON, JSONL, Parquet)

**API Architecture (api/ directory):**
FastAPI application with 5 router modules:
1. Public Router (/health, /checkpoint, /users, /business/search, /recommend)
   - Health checks and model status
   - User listing and profile retrieval
   - Business search with category filtering
   - Main recommendation endpoint (POST /recommend)
2. Auth Router (/auth/signup, /auth/signin, /me/rate, /me/history, /me/profile)
   - Local user registration and authentication
   - Rating submission and history tracking
3. UX Router (/ux/login, /ux/me, /ux/recommend, /ux/search, /ux/categories)
   - Experience-oriented endpoints with session management
   - Paginated recommendation feed
4. Admin Router (/admin/config, /admin/stats, /admin/retrain, /admin/reload)
   - Runtime configuration (default K, algorithm selection)
   - System statistics and monitoring
   - Model retraining trigger (background thread)
5. Pages Router (/, /basic, /experience, /management)
   - Serves 4 HTML template pages

**Demo Interfaces:**
1. Basic Demo Page (/basic): recommendation cards with explainability breakdown (content score, CF prediction, hybrid alpha, reason tags) and interactive map showing restaurant locations
2. UX Login Page (/experience): user selection from Yelp user pool with profile preview
3. UX Feed Page (/experience/app): paginated recommendation feed with infinite scroll, search, and category filters
4. Management Dashboard (/management): admin panel showing active users, model metrics, algorithm usage statistics, activity timeline chart, and retrain controls

**Key Runtime Features:**
- Model bundle hot-loading at startup (~273MB pickle)
- In-memory session management for both auth tokens and UX tokens
- Active user tracking with 20-minute activity window
- Business hours parsing with real-time "Open Now" status
- Geo-enrichment from business_main.parquet (latitude, longitude, address)
- Background model retraining without server restart
- User profile caching from Parquet for rich profile display

Write in formal academic English. Focus on the engineering decisions and system design.
Output ONLY the content paragraphs with clear sub-headings.
```

---

## PROMPT 11 — CONCLUSION

```
Write the CONCLUSION section (300-400 words) for a graduation thesis titled "Yelp Recommendation System".

The conclusion should cover:

**Achievements:**
1. Successfully designed and implemented a complete Hybrid Recommendation System combining Content-Based Filtering (TF-IDF with 80K features) and Collaborative Filtering (bias-aware SVD with 64 latent factors)
2. Processed the Yelp Open Dataset at scale: 33,000 US restaurants, 347,000 users with interaction histories
3. Implemented an adaptive alpha weighting mechanism that automatically adjusts the content-vs-collaborative balance based on user interaction count, effectively addressing the cold-start problem
4. Built a full-stack serving system with FastAPI providing RESTful API endpoints, user authentication, and real-time recommendation serving
5. Created 4 interactive web-based demo interfaces including recommendation explainability (showing why each restaurant was recommended) and geographic visualization
6. Implemented admin capabilities including runtime configuration, system monitoring, and model retraining without server downtime
7. Established a reproducible ML pipeline with automated preprocessing, training, evaluation, and deployment checkpoints

**Limitations:**
1. Offline evaluation metrics are difficult to interpret due to large catalog size (33K items with 1 held-out test item)
2. The system uses in-memory state management, which does not persist across server restarts
3. Content features rely on bag-of-words (TF-IDF) which cannot capture semantic meaning
4. The model bundle size (273MB) is large for deployment scenarios
5. Security features are basic (no rate limiting, simple password hashing)

**Future Enhancements:**
1. Upgrade evaluation protocol with sampled negative candidates for more interpretable metrics
2. Replace TF-IDF with semantic embeddings (e.g., sentence-transformers) for richer content understanding
3. Explore deep learning approaches: Neural Collaborative Filtering or Two-Tower models
4. Add location-aware recommendation using user-restaurant distance
5. Implement online A/B testing to measure real-world recommendation quality
6. Containerize with Docker and deploy to cloud platform for public access
7. Integrate review text analysis (sentiment) as additional recommendation signal

Write in formal academic English. Be honest about limitations while highlighting the technical depth of what was achieved.
Output ONLY the content paragraphs.
```

---

## PROMPT 12 — REFERENCES

```
Generate a list of 10-12 academic references for a graduation thesis about a Hybrid Recommendation System for restaurants using the Yelp dataset. The system combines Content-Based Filtering (TF-IDF) and Collaborative Filtering (SVD).

Include references for:
1. The Yelp Open Dataset (official source)
2. Foundational recommendation systems survey paper (e.g., Ricci et al. or Aggarwal)
3. Content-based filtering (TF-IDF approach)
4. Collaborative filtering and matrix factorization (Koren et al. 2009 — the Netflix Prize paper)
5. SVD/matrix factorization for recommendations
6. Hybrid recommendation systems (Burke 2002 or 2007)
7. Cold-start problem in recommendation systems
8. Evaluation metrics for recommendation systems (Precision, Recall, NDCG)
9. FastAPI or web-based ML serving
10. scikit-learn library reference
11. A recent survey on deep learning for recommendation (optional)

Format each reference in IEEE style:
[N] Author(s), "Title," Journal/Conference, vol. X, no. Y, pp. XX-YY, Year.

Use REAL references only — do not fabricate papers. If unsure of exact page numbers, omit them.
Output ONLY the numbered reference list.
```

---

## Checklist — Những gì cần làm sau khi có output

- [ ] Paste output từ mỗi prompt vào đúng section trong docx
- [ ] Cập nhật **Glossary of Terms** table (xóa CNN, ResNet, OKS cũ)
- [ ] Cập nhật **List of Figures** sau khi thêm screenshots
- [ ] Xóa dòng "tennis court keypoints using ResNet50" ở Conclusion cũ
- [ ] Xóa bảng References cũ (Git link tennis court)
- [ ] Thêm screenshots: Basic demo, UX feed, Management dashboard, API Swagger docs
- [ ] Thêm architecture diagram (có thể vẽ từ Mermaid trong analysis trước)
- [ ] Cross-check citation numbers [1]-[N] khớp với References list
- [ ] Kiểm tra tên "FAISS" ở mục Data Splitting — project thực tế **không dùng FAISS**, đổi thành "Temporal Leave-Last-Out"
