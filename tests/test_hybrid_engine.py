"""Tests for HybridRecommender (adaptive alpha hybrid blending)."""
from __future__ import annotations

import numpy as np
import pytest

from src.hybrid_engine import HybridRecommender


class TestAdaptiveAlpha:
    """Tests for the adaptive alpha weighting mechanism."""

    def test_cold_start_alpha_near_one(self, hybrid_recommender):
        alpha = hybrid_recommender.calculate_alpha(0)
        assert alpha == pytest.approx(1.0, abs=0.01)

    def test_active_user_alpha_near_min(self, hybrid_recommender):
        alpha = hybrid_recommender.calculate_alpha(100)
        assert alpha == pytest.approx(0.2, abs=0.02)

    def test_alpha_decreases_with_interactions(self, hybrid_recommender):
        alphas = [hybrid_recommender.calculate_alpha(n) for n in range(0, 50, 5)]
        # Alpha should be monotonically non-increasing
        for i in range(len(alphas) - 1):
            assert alphas[i] >= alphas[i + 1]

    def test_alpha_at_half_life(self, hybrid_recommender):
        # At n = half_life (12), decay factor = exp(-1) ≈ 0.368
        alpha = hybrid_recommender.calculate_alpha(12)
        expected = 0.2 + 0.8 * np.exp(-1)  # ≈ 0.494
        assert alpha == pytest.approx(expected, abs=0.01)

    def test_alpha_clipped_to_valid_range(self, hybrid_recommender):
        for n in [-5, 0, 10, 100, 10000]:
            alpha = hybrid_recommender.calculate_alpha(n)
            assert 0.2 <= alpha <= 1.0


class TestHybridRecommend:
    """Tests for the recommend() method."""

    def test_recommend_returns_k_items(self, hybrid_recommender):
        recs = hybrid_recommender.recommend(
            user_id="user_001",
            k=3,
            user_history_business_ids=["biz_001", "biz_005"],
            user_history_ratings=[5.0, 5.0],
        )
        assert len(recs) <= 3

    def test_recommend_excludes_history(self, hybrid_recommender):
        history = ["biz_001", "biz_005"]
        recs = hybrid_recommender.recommend(
            user_id="user_001",
            k=5,
            user_history_business_ids=history,
            user_history_ratings=[5.0, 5.0],
            exclude_history=True,
        )
        rec_ids = [bid for bid, _ in recs]
        for h in history:
            assert h not in rec_ids

    def test_recommend_sorted_descending(self, hybrid_recommender):
        recs = hybrid_recommender.recommend(
            user_id="user_001",
            k=5,
            user_history_business_ids=["biz_001", "biz_005"],
            user_history_ratings=[5.0, 5.0],
        )
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_with_candidate_list(self, hybrid_recommender):
        candidates = ["biz_002", "biz_006", "biz_008"]
        recs = hybrid_recommender.recommend(
            user_id="user_001",
            k=2,
            user_history_business_ids=["biz_001"],
            user_history_ratings=[5.0],
            candidate_business_ids=candidates,
        )
        rec_ids = [bid for bid, _ in recs]
        assert all(bid in candidates for bid in rec_ids)

    def test_recommend_empty_history(self, hybrid_recommender):
        recs = hybrid_recommender.recommend(
            user_id="user_001",
            k=3,
            user_history_business_ids=[],
            user_history_ratings=[],
        )
        # Should still return results (from prior/CF)
        assert isinstance(recs, list)


class TestHybridScoring:
    """Tests for individual scoring components."""

    def test_get_hybrid_score_returns_float(self, hybrid_recommender):
        score = hybrid_recommender.get_hybrid_score(
            user_id="user_001",
            biz_id="biz_002",
            user_history_business_ids=["biz_001"],
            user_history_ratings=[5.0],
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_prior_score_in_range(self, hybrid_recommender):
        score = hybrid_recommender._prior_score("biz_005")
        assert 0.0 <= score <= 1.0

    def test_prior_unknown_returns_zero(self, hybrid_recommender):
        score = hybrid_recommender._prior_score("biz_unknown")
        assert score == 0.0


class TestBusinessPriors:
    """Tests for business prior management."""

    def test_set_business_priors(self, hybrid_recommender):
        priors = {"biz_001": 0.9, "biz_002": 0.5}
        hybrid_recommender.set_business_priors(priors)
        assert hybrid_recommender.business_prior_scores == priors
        assert hybrid_recommender._sorted_prior_ids[0] == "biz_001"

    def test_set_none_clears_priors(self, hybrid_recommender):
        hybrid_recommender.set_business_priors(None)
        assert hybrid_recommender.business_prior_scores == {}
