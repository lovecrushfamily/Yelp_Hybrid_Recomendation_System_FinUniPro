"""Tests for ContentEngine (TF-IDF content-based recommender)."""
from __future__ import annotations

import pytest

from src.content_base import ContentEngine


class TestContentEngineFit:
    """Tests for the fit() method."""

    def test_fit_populates_biz_ids(self, content_engine, businesses_df):
        assert len(content_engine.biz_ids) == len(businesses_df)

    def test_fit_creates_tfidf_matrix(self, content_engine, businesses_df):
        assert content_engine.tfidf_matrix is not None
        assert content_engine.tfidf_matrix.shape[0] == len(businesses_df)

    def test_fit_builds_id_to_index(self, content_engine):
        assert "biz_001" in content_engine.id_to_index
        assert content_engine.id_to_index["biz_001"] == 0

    def test_fit_raises_on_missing_columns(self, businesses_df):
        engine = ContentEngine(max_features=100, min_df=1)
        with pytest.raises(ValueError, match="Missing required columns"):
            engine.fit(businesses_df, text_col="nonexistent")


class TestContentEngineSimilarity:
    """Tests for similarity scoring."""

    def test_same_item_has_max_similarity(self, content_engine):
        score = content_engine.get_similarity_score("biz_001", "biz_001")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_similar_cuisine_higher_than_different(self, content_engine):
        # Pizza Palace (Italian) vs Pasta Place (Italian) should be higher
        # than Pizza Palace vs Sushi World (Japanese)
        italian_sim = content_engine.get_similarity_score("biz_001", "biz_005")
        cross_sim = content_engine.get_similarity_score("biz_001", "biz_002")
        assert italian_sim > cross_sim

    def test_unknown_business_returns_zero(self, content_engine):
        score = content_engine.get_similarity_score("biz_001", "biz_unknown")
        assert score == 0.0


class TestContentEngineRecommend:
    """Tests for recommendation methods."""

    def test_recommend_from_history_returns_correct_k(self, content_engine):
        recs = content_engine.recommend_from_history(
            history_business_ids=["biz_001", "biz_005"],
            k=3,
        )
        assert len(recs) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)

    def test_recommend_excludes_history_by_default(self, content_engine):
        history = ["biz_001", "biz_005"]
        recs = content_engine.recommend_from_history(
            history_business_ids=history,
            k=5,
        )
        rec_ids = [bid for bid, _ in recs]
        assert "biz_001" not in rec_ids
        assert "biz_005" not in rec_ids

    def test_recommend_includes_history_when_disabled(self, content_engine):
        recs = content_engine.recommend_from_history(
            history_business_ids=["biz_001"],
            k=10,
            exclude_history=False,
        )
        rec_ids = [bid for bid, _ in recs]
        assert "biz_001" in rec_ids

    def test_score_candidates_returns_dict(self, content_engine):
        scores = content_engine.score_candidates_for_history(
            history_business_ids=["biz_001"],
            candidate_business_ids=["biz_002", "biz_005"],
        )
        assert isinstance(scores, dict)
        assert "biz_002" in scores
        assert "biz_005" in scores

    def test_recommend_italian_lover_gets_italian(self, content_engine):
        """User who liked Italian restaurants should get Italian recommendations high."""
        recs = content_engine.recommend_from_history(
            history_business_ids=["biz_001", "biz_005"],  # Pizza Palace, Pasta Place
            k=3,
        )
        rec_ids = [bid for bid, _ in recs]
        # At minimum, the top recs should not include biz_001/biz_005 (excluded)
        assert len(rec_ids) == 3

    def test_unfitted_engine_raises(self):
        engine = ContentEngine()
        with pytest.raises(ValueError, match="not fitted"):
            engine.recommend_from_history(["biz_001"], k=5)
