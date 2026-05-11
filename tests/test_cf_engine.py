"""Tests for CFEngine (Bias-aware SVD collaborative filtering)."""
from __future__ import annotations

import numpy as np
import pytest

from src.collab_filter import CFEngine


class TestCFEngineFit:
    """Tests for the fit() method."""

    def test_fit_populates_users_and_items(self, cf_engine):
        assert len(cf_engine.users) == 4  # user_001..004
        assert len(cf_engine.items) >= 8  # at least 8 unique businesses

    def test_fit_computes_global_mean(self, cf_engine):
        assert 1.0 <= cf_engine.global_mean <= 5.0

    def test_fit_computes_biases(self, cf_engine):
        assert cf_engine.user_bias is not None
        assert cf_engine.item_bias is not None
        assert len(cf_engine.user_bias) == len(cf_engine.users)
        assert len(cf_engine.item_bias) == len(cf_engine.items)

    def test_fit_creates_factors(self, cf_engine):
        assert cf_engine.user_factors is not None
        assert cf_engine.item_factors is not None
        assert cf_engine.user_factors.shape[0] == len(cf_engine.users)
        assert cf_engine.item_factors.shape[0] == len(cf_engine.items)

    def test_fit_tracks_seen_items(self, cf_engine):
        seen = cf_engine.seen_items_by_user.get("user_001", set())
        assert "biz_001" in seen
        assert "biz_005" in seen

    def test_fit_raises_on_missing_columns(self, interactions_df):
        engine = CFEngine(n_components=2)
        with pytest.raises(ValueError, match="Missing required columns"):
            engine.fit(interactions_df, user_col="nonexistent")


class TestCFEnginePrediction:
    """Tests for rating prediction."""

    def test_predict_known_user_item_in_range(self, cf_engine):
        rating = cf_engine.predict_rating("user_001", "biz_001")
        assert 1.0 <= rating <= 5.0

    def test_predict_unknown_user_returns_baseline(self, cf_engine):
        rating = cf_engine.predict_rating("unknown_user", "biz_001")
        # Should still return a valid rating (global_mean + item_bias)
        assert 1.0 <= rating <= 5.0

    def test_predict_unknown_item_returns_baseline(self, cf_engine):
        rating = cf_engine.predict_rating("user_001", "unknown_biz")
        assert 1.0 <= rating <= 5.0

    def test_vectorized_prediction_matches_individual(self, cf_engine):
        candidates = ["biz_001", "biz_002", "biz_003"]
        batch = cf_engine.predict_ratings_for_user("user_001", candidates)
        batch_dict = dict(batch)
        for biz_id in candidates:
            individual = cf_engine.predict_rating("user_001", biz_id)
            assert batch_dict[biz_id] == pytest.approx(individual, abs=1e-6)


class TestCFEngineRecommend:
    """Tests for recommendation."""

    def test_recommend_returns_k_items(self, cf_engine):
        recs = cf_engine.recommend_for_user("user_001", k=3)
        assert len(recs) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)

    def test_recommend_excludes_seen(self, cf_engine):
        recs = cf_engine.recommend_for_user("user_001", k=10, exclude_seen=True)
        rec_ids = {bid for bid, _ in recs}
        seen = cf_engine.seen_items_by_user.get("user_001", set())
        assert rec_ids.isdisjoint(seen)

    def test_recommend_includes_seen_when_disabled(self, cf_engine):
        recs = cf_engine.recommend_for_user("user_001", k=10, exclude_seen=False)
        rec_ids = {bid for bid, _ in recs}
        seen = cf_engine.seen_items_by_user.get("user_001", set())
        # At least some seen items should appear
        assert len(rec_ids & seen) > 0

    def test_recommend_sorted_descending(self, cf_engine):
        recs = cf_engine.recommend_for_user("user_001", k=5)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)
