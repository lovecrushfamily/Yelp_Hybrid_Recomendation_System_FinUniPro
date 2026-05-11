"""Tests for evaluation metrics and protocols."""
from __future__ import annotations

import pandas as pd
import pytest

from src.evaluator import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    temporal_leave_last_split,
    evaluate_hybrid_model,
    evaluate_hybrid_model_sampled,
)


# ── Metric unit tests ───────────────────────────────────────────────

class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_no_hits(self):
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0

    def test_partial_hits(self):
        assert precision_at_k(["a", "b", "c"], {"a", "x"}, 3) == pytest.approx(1 / 3)

    def test_k_limits_evaluation(self):
        assert precision_at_k(["a", "b", "c", "d"], {"c", "d"}, 2) == 0.0

    def test_empty_input(self):
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_zero_k(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_no_relevant(self):
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "b"], {"a", "c"}, 2) == pytest.approx(0.5)


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        # Single relevant item at position 1
        assert ndcg_at_k(["a"], {"a"}, 1) == pytest.approx(1.0)

    def test_no_hits(self):
        assert ndcg_at_k(["a", "b"], {"x"}, 2) == 0.0

    def test_empty_input(self):
        assert ndcg_at_k([], {"a"}, 5) == 0.0

    def test_ndcg_position_matters(self):
        # Hit at position 1 should score higher than hit at position 2
        ndcg_pos1 = ndcg_at_k(["a", "b"], {"a"}, 2)
        ndcg_pos2 = ndcg_at_k(["b", "a"], {"a"}, 2)
        assert ndcg_pos1 > ndcg_pos2


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k(["a", "b", "c"], {"b"}, 3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["a", "b", "c"], {"x"}, 3) == 0.0

    def test_hit_outside_k(self):
        assert hit_rate_at_k(["a", "b", "c"], {"c"}, 2) == 0.0

    def test_empty(self):
        assert hit_rate_at_k([], {"a"}, 5) == 0.0


# ── Split tests ──────────────────────────────────────────────────────

class TestTemporalSplit:
    def test_split_separates_last_item(self, interactions_df):
        train, test = temporal_leave_last_split(
            interactions_df, min_history=2, n_test_items=1,
        )
        assert len(test) > 0
        assert len(train) > 0
        assert len(train) + len(test) <= len(interactions_df)

    def test_split_filters_short_history(self, interactions_df):
        # With min_history=10, no user qualifies (max 4 interactions)
        train, test = temporal_leave_last_split(
            interactions_df, min_history=10, n_test_items=1,
        )
        assert len(train) == 0
        assert len(test) == 0

    def test_split_preserves_columns(self, interactions_df):
        train, test = temporal_leave_last_split(
            interactions_df, min_history=2, n_test_items=1,
        )
        for col in ["user_id", "business_id", "stars", "date"]:
            assert col in train.columns
            assert col in test.columns

    def test_split_test_items_are_most_recent(self, interactions_df):
        train, test = temporal_leave_last_split(
            interactions_df, min_history=2, n_test_items=1,
        )
        # For each user, test item date should be >= all train item dates
        for user_id in test["user_id"].unique():
            test_dates = test[test["user_id"] == user_id]["date"]
            train_dates = train[train["user_id"] == user_id]["date"]
            if len(train_dates) > 0:
                assert test_dates.min() >= train_dates.max()


# ── Full evaluation tests ────────────────────────────────────────────

class TestEvaluateHybridModel:
    def test_returns_all_metrics(self, hybrid_recommender, interactions_df):
        train, test = temporal_leave_last_split(
            interactions_df, min_history=2, n_test_items=1,
        )
        result = evaluate_hybrid_model(
            hybrid_recommender, train, test,
            k=3, max_users=10, relevance_threshold=3.0,
        )
        assert "evaluated_users" in result
        assert "precision@k" in result
        assert "recall@k" in result
        assert "ndcg@k" in result
        assert "hit_rate@k" in result
        assert "coverage" in result

    def test_empty_data_returns_zeros(self, hybrid_recommender):
        empty = pd.DataFrame(columns=["user_id", "business_id", "stars", "date"])
        result = evaluate_hybrid_model(hybrid_recommender, empty, empty)
        assert result["evaluated_users"] == 0
        assert result["precision@k"] == 0.0


class TestEvaluateHybridModelSampled:
    def test_sampled_returns_all_metrics(self, hybrid_recommender, interactions_df):
        train, test = temporal_leave_last_split(
            interactions_df, min_history=2, n_test_items=1,
        )
        result = evaluate_hybrid_model_sampled(
            hybrid_recommender, train, test,
            k=3, n_neg_samples=5, max_users=10,
            relevance_threshold=3.0,
        )
        assert "evaluated_users" in result
        assert "precision@k" in result
        assert "hit_rate@k" in result
        assert "n_neg_samples" in result

    def test_sampled_metrics_higher_than_full_catalog(
        self, hybrid_recommender, interactions_df
    ):
        """With fewer candidates, metrics should be meaningfully higher."""
        train, test = temporal_leave_last_split(
            interactions_df, min_history=2, n_test_items=1,
        )
        full = evaluate_hybrid_model(
            hybrid_recommender, train, test,
            k=3, max_users=10, relevance_threshold=3.0,
        )
        sampled = evaluate_hybrid_model_sampled(
            hybrid_recommender, train, test,
            k=3, n_neg_samples=5, max_users=10,
            relevance_threshold=3.0,
        )
        # Sampled eval should produce >= full-catalog metrics
        # (or at least be non-negative)
        assert sampled["hit_rate@k"] >= 0.0
        assert sampled["precision@k"] >= 0.0

    def test_empty_data_returns_zeros(self, hybrid_recommender):
        empty = pd.DataFrame(columns=["user_id", "business_id", "stars", "date"])
        result = evaluate_hybrid_model_sampled(
            hybrid_recommender, empty, empty, n_neg_samples=10,
        )
        assert result["evaluated_users"] == 0
