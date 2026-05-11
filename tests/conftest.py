"""Shared fixtures for Yelp Hybrid RecSys tests.

Uses tiny synthetic data so tests run in seconds without needing
the real Yelp dataset or pre-trained model bundle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.collab_filter import CFEngine
from src.content_base import ContentEngine
from src.hybrid_engine import HybridRecommender


# ---------------------------------------------------------------------------
# Tiny synthetic dataset
# ---------------------------------------------------------------------------

_BUSINESSES = [
    {"business_id": "biz_001", "name": "Pizza Palace", "categories": "Pizza, Italian, Restaurants", "soup": "pizza palace pizza italian restaurants delivery"},
    {"business_id": "biz_002", "name": "Sushi World", "categories": "Sushi, Japanese, Restaurants", "soup": "sushi world sushi japanese restaurants dine-in"},
    {"business_id": "biz_003", "name": "Burger Barn", "categories": "Burgers, American, Restaurants", "soup": "burger barn burgers american restaurants fast food"},
    {"business_id": "biz_004", "name": "Taco Town", "categories": "Mexican, Tacos, Restaurants", "soup": "taco town mexican tacos restaurants spicy"},
    {"business_id": "biz_005", "name": "Pasta Place", "categories": "Italian, Pasta, Restaurants", "soup": "pasta place italian pasta restaurants homemade"},
    {"business_id": "biz_006", "name": "Ramen House", "categories": "Japanese, Ramen, Restaurants", "soup": "ramen house japanese ramen restaurants noodles"},
    {"business_id": "biz_007", "name": "Steak House", "categories": "Steakhouse, American, Restaurants", "soup": "steak house steakhouse american restaurants grill"},
    {"business_id": "biz_008", "name": "Dim Sum", "categories": "Chinese, Dim Sum, Restaurants", "soup": "dim sum chinese dim sum restaurants tea"},
    {"business_id": "biz_009", "name": "Curry Corner", "categories": "Indian, Curry, Restaurants", "soup": "curry corner indian curry restaurants spicy naan"},
    {"business_id": "biz_010", "name": "Fish Fry", "categories": "Seafood, Fish, Restaurants", "soup": "fish fry seafood fish restaurants fried"},
]

_INTERACTIONS = [
    # user_001 loves Italian
    {"user_id": "user_001", "business_id": "biz_001", "stars": 5.0, "date": "2024-01-01"},
    {"user_id": "user_001", "business_id": "biz_005", "stars": 5.0, "date": "2024-02-01"},
    {"user_id": "user_001", "business_id": "biz_003", "stars": 3.0, "date": "2024-03-01"},
    {"user_id": "user_001", "business_id": "biz_007", "stars": 2.0, "date": "2024-04-01"},
    # user_002 loves Japanese
    {"user_id": "user_002", "business_id": "biz_002", "stars": 5.0, "date": "2024-01-15"},
    {"user_id": "user_002", "business_id": "biz_006", "stars": 5.0, "date": "2024-02-15"},
    {"user_id": "user_002", "business_id": "biz_008", "stars": 4.0, "date": "2024-03-15"},
    {"user_id": "user_002", "business_id": "biz_004", "stars": 3.0, "date": "2024-04-15"},
    # user_003 is a generalist
    {"user_id": "user_003", "business_id": "biz_003", "stars": 4.0, "date": "2024-01-20"},
    {"user_id": "user_003", "business_id": "biz_009", "stars": 5.0, "date": "2024-02-20"},
    {"user_id": "user_003", "business_id": "biz_010", "stars": 4.0, "date": "2024-03-20"},
    {"user_id": "user_003", "business_id": "biz_001", "stars": 4.0, "date": "2024-04-20"},
    # user_004 has minimal history (cold start)
    {"user_id": "user_004", "business_id": "biz_004", "stars": 4.0, "date": "2024-03-01"},
    {"user_id": "user_004", "business_id": "biz_009", "stars": 5.0, "date": "2024-04-01"},
]


@pytest.fixture
def businesses_df() -> pd.DataFrame:
    return pd.DataFrame(_BUSINESSES)


@pytest.fixture
def interactions_df() -> pd.DataFrame:
    return pd.DataFrame(_INTERACTIONS)


@pytest.fixture
def content_engine(businesses_df) -> ContentEngine:
    """A fitted content engine on synthetic data."""
    engine = ContentEngine(max_features=500, ngram_range=(1, 1), min_df=1)
    engine.fit(businesses_df)
    return engine


@pytest.fixture
def cf_engine(interactions_df) -> CFEngine:
    """A fitted CF engine on synthetic data."""
    engine = CFEngine(n_components=4, random_state=42)
    engine.fit(interactions_df)
    return engine


@pytest.fixture
def hybrid_recommender(content_engine, cf_engine) -> HybridRecommender:
    """A configured hybrid recommender."""
    priors = {f"biz_{i:03d}": float(i) / 10.0 for i in range(1, 11)}
    rec = HybridRecommender(
        content_engine=content_engine,
        cf_engine=cf_engine,
        min_alpha=0.2,
        half_life=12.0,
        prior_weight=0.1,
        cf_candidate_limit=50,
        content_candidate_limit=50,
        prior_candidate_limit=50,
    )
    rec.set_business_priors(priors)
    return rec
