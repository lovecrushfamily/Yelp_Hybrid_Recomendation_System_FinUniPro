from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentEngine:
    """Content-based recommender using TF-IDF features from business metadata."""

    def __init__(
        self,
        max_features: int = 80_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ):
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        self.tfidf_matrix = None
        self.biz_ids: list[str] = []
        self.id_to_index: dict[str, int] = {}

    def fit(self, df_biz, text_col: str = "soup", id_col: str = "business_id"):
        """Fit TF-IDF representation for all businesses."""
        if text_col not in df_biz.columns or id_col not in df_biz.columns:
            raise ValueError(f"Missing required columns: {id_col}, {text_col}")

        text = df_biz[text_col].fillna("").astype(str)
        try:
            self.tfidf_matrix = self.tfidf.fit_transform(text)
        except ValueError as exc:
            # Small datasets can prune all terms when min_df is too strict.
            if "After pruning, no terms remain" not in str(exc):
                raise
            params = self.tfidf.get_params()
            params["min_df"] = 1
            self.tfidf = TfidfVectorizer(**params)
            self.tfidf_matrix = self.tfidf.fit_transform(text)

        self.biz_ids = df_biz[id_col].astype(str).tolist()
        self.id_to_index = {biz_id: idx for idx, biz_id in enumerate(self.biz_ids)}
        return self

    def _require_fit(self):
        if self.tfidf_matrix is None or not self.biz_ids:
            raise ValueError("ContentEngine is not fitted. Call fit() first.")

    def _resolve_indices(self, business_ids: list[str]) -> list[int]:
        return [self.id_to_index[b] for b in business_ids if b in self.id_to_index]

    def get_similarity_score(self, biz_id_a: str, biz_id_b: str) -> float:
        """Return cosine similarity between two businesses."""
        self._require_fit()
        if biz_id_a not in self.id_to_index or biz_id_b not in self.id_to_index:
            return 0.0

        idx_a = self.id_to_index[biz_id_a]
        idx_b = self.id_to_index[biz_id_b]
        score = cosine_similarity(
            self.tfidf_matrix[idx_a], self.tfidf_matrix[idx_b]
        )[0, 0]
        return float(score)

    def score_candidates_for_history(
        self,
        history_business_ids: list[str],
        candidate_business_ids: list[str] | None = None,
        history_weights: list[float] | None = None,
    ) -> dict[str, float]:
        """Score candidate businesses against aggregated user history profile."""
        self._require_fit()

        history_indices = self._resolve_indices(history_business_ids)
        if not history_indices:
            return {}

        history_matrix = self.tfidf_matrix[history_indices]
        if history_weights is not None and len(history_weights) == len(history_business_ids):
            weights = np.array(
                [history_weights[i] for i, bid in enumerate(history_business_ids) if bid in self.id_to_index],
                dtype=float,
            )
            if weights.sum() <= 0:
                profile = history_matrix.mean(axis=0)
            else:
                profile = (history_matrix.multiply(weights[:, None])).sum(axis=0) / weights.sum()
        else:
            profile = history_matrix.mean(axis=0)
        profile = np.asarray(profile).reshape(1, -1)

        if candidate_business_ids is None:
            candidate_indices = list(range(len(self.biz_ids)))
            candidate_ids = self.biz_ids
        else:
            candidate_indices = self._resolve_indices(candidate_business_ids)
            candidate_ids = [self.biz_ids[idx] for idx in candidate_indices]

        if not candidate_indices:
            return {}

        candidate_matrix = self.tfidf_matrix[candidate_indices]
        scores = cosine_similarity(candidate_matrix, profile).ravel()
        return {bid: float(scores[i]) for i, bid in enumerate(candidate_ids)}

    def recommend_from_history(
        self,
        history_business_ids: list[str],
        candidate_business_ids: list[str] | None = None,
        history_weights: list[float] | None = None,
        k: int = 10,
        exclude_history: bool = True,
    ) -> list[tuple[str, float]]:
        """Return top-K businesses based on content similarity to history."""
        scores = self.score_candidates_for_history(
            history_business_ids=history_business_ids,
            candidate_business_ids=candidate_business_ids,
            history_weights=history_weights,
        )
        if not scores:
            return []

        if exclude_history:
            history_set = set(history_business_ids)
            scores = {k_: v for k_, v in scores.items() if k_ not in history_set}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
