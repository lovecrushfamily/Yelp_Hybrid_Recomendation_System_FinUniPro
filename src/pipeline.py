from __future__ import annotations

import pandas as pd

from .collab_filter import CFEngine
from .content_base import ContentEngine
from .evaluator import evaluate_hybrid_model, evaluate_hybrid_model_sampled, temporal_leave_last_split
from .hybrid_engine import HybridRecommender
from .preprocess import DataConfig, DataLoader


class YelpHybridPipeline:
    """High-level orchestration wrapper for training, serving, and evaluation."""

    def __init__(
        self,
        content_engine: ContentEngine | None = None,
        cf_engine: CFEngine | None = None,
        min_alpha: float = 0.2,
        half_life: float = 12.0,
        prior_weight: float = 0.1,
    ):
        self.content_engine = content_engine or ContentEngine()
        self.cf_engine = cf_engine or CFEngine()
        self.hybrid = HybridRecommender(
            self.content_engine,
            self.cf_engine,
            min_alpha=min_alpha,
            half_life=half_life,
            prior_weight=prior_weight,
        )
        self.business_df: pd.DataFrame | None = None
        self.interactions_df: pd.DataFrame | None = None

    def fit(self, business_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Fit content and CF engines, then wire optional business priors."""
        self.business_df = business_df.copy()
        self.interactions_df = interactions_df.copy()
        self.content_engine.fit(self.business_df)
        self.cf_engine.fit(self.interactions_df)
        if "popularity_prior" in self.business_df.columns:
            priors = (
                self.business_df[["business_id", "popularity_prior"]]
                .dropna(subset=["business_id"])
                .assign(business_id=lambda df: df["business_id"].astype(str))
            )
            self.hybrid.set_business_priors(
                {
                    row.business_id: float(row.popularity_prior)
                    for row in priors.itertuples(index=False)
                }
            )
        return self

    def _history_for_user(self, user_id: str) -> tuple[list[str], list[float]]:
        if self.interactions_df is None:
            return [], []
        user_rows = self.interactions_df[self.interactions_df["user_id"].astype(str) == str(user_id)]
        if user_rows.empty:
            return [], []

        if "date" in user_rows.columns:
            user_rows = user_rows.sort_values("date")
        items = user_rows["business_id"].astype(str).tolist()
        ratings = user_rows["stars"].astype(float).tolist()
        return items, ratings

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        user_history_business_ids: list[str] | None = None,
        user_history_ratings: list[float] | None = None,
        candidate_business_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-K hybrid recommendations for a user."""
        history_ids = user_history_business_ids
        history_ratings = user_history_ratings

        if history_ids is None:
            auto_items, auto_ratings = self._history_for_user(str(user_id))
            history_ids = auto_items
            history_ratings = auto_ratings

        return self.hybrid.recommend(
            user_id=str(user_id),
            k=k,
            user_history_business_ids=history_ids,
            user_history_ratings=history_ratings,
            candidate_business_ids=candidate_business_ids,
            exclude_history=True,
        )

    def evaluate(
        self,
        interactions_df: pd.DataFrame | None = None,
        k: int = 10,
        min_history: int = 3,
        n_test_items: int = 1,
        max_users: int | None = None,
        progress_every: int = 200,
        verbose: bool = False,
        relevance_threshold: float = 4.0,
    ) -> dict[str, float]:
        """Evaluate ranking quality with temporal holdout and top-K metrics."""
        source_df = interactions_df if interactions_df is not None else self.interactions_df
        if source_df is None or source_df.empty:
            return {
                "evaluated_users": 0.0,
                "precision@k": 0.0,
                "recall@k": 0.0,
                "ndcg@k": 0.0,
                "hit_rate@k": 0.0,
                "coverage": 0.0,
            }

        train_df, test_df = temporal_leave_last_split(
            source_df,
            min_history=min_history,
            n_test_items=n_test_items,
        )
        if train_df.empty or test_df.empty:
            return {
                "evaluated_users": 0.0,
                "precision@k": 0.0,
                "recall@k": 0.0,
                "ndcg@k": 0.0,
                "hit_rate@k": 0.0,
                "coverage": 0.0,
            }

        self.cf_engine.fit(train_df)
        self.interactions_df = train_df.copy()
        return evaluate_hybrid_model(
            self.hybrid,
            train_df,
            test_df,
            k=k,
            max_users=max_users,
            progress_every=progress_every,
            verbose=verbose,
            relevance_threshold=relevance_threshold,
        )

    def evaluate_sampled(
        self,
        interactions_df: pd.DataFrame | None = None,
        k: int = 10,
        n_neg_samples: int = 100,
        min_history: int = 3,
        n_test_items: int = 1,
        max_users: int | None = None,
        progress_every: int = 200,
        verbose: bool = False,
        relevance_threshold: float = 4.0,
    ) -> dict[str, float]:
        """Evaluate with sampled negatives for interpretable metrics."""
        source_df = interactions_df if interactions_df is not None else self.interactions_df
        if source_df is None or source_df.empty:
            return {
                "evaluated_users": 0.0,
                "precision@k": 0.0,
                "recall@k": 0.0,
                "ndcg@k": 0.0,
                "hit_rate@k": 0.0,
                "n_neg_samples": float(n_neg_samples),
            }

        train_df, test_df = temporal_leave_last_split(
            source_df,
            min_history=min_history,
            n_test_items=n_test_items,
        )
        if train_df.empty or test_df.empty:
            return {
                "evaluated_users": 0.0,
                "precision@k": 0.0,
                "recall@k": 0.0,
                "ndcg@k": 0.0,
                "hit_rate@k": 0.0,
                "n_neg_samples": float(n_neg_samples),
            }

        self.cf_engine.fit(train_df)
        self.interactions_df = train_df.copy()
        return evaluate_hybrid_model_sampled(
            self.hybrid,
            train_df,
            test_df,
            k=k,
            n_neg_samples=n_neg_samples,
            max_users=max_users,
            progress_every=progress_every,
            verbose=verbose,
            relevance_threshold=relevance_threshold,
        )


def build_pipeline_from_paths(
    business_path: str,
    review_path: str,
    data_config: DataConfig | None = None,
) -> tuple[YelpHybridPipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = data_config or DataConfig()
    loader = DataLoader(business_path=business_path, review_path=review_path)
    biz, rev, interactions = loader.run(cfg)

    pipeline = YelpHybridPipeline()
    pipeline.fit(business_df=biz, interactions_df=interactions)
    return pipeline, biz, rev, interactions
