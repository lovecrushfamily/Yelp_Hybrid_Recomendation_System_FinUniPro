from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD


class CFEngine:
    """Bias-aware matrix factorization recommender (TruncatedSVD on residuals)."""

    def __init__(
        self,
        n_components: int = 64,
        random_state: int = 42,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.svd: TruncatedSVD | None = None
        self.residual_matrix: csr_matrix | None = None

        self.users: list[str] = []
        self.items: list[str] = []
        self.user_to_idx: dict[str, int] = {}
        self.item_to_idx: dict[str, int] = {}

        self.global_mean: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

        self.seen_items_by_user: dict[str, set[str]] = {}

    def fit(
        self,
        df_rev,
        user_col: str = "user_id",
        item_col: str = "business_id",
        rating_col: str = "stars",
    ):
        """Fit collaborative model from interaction dataframe."""
        required = {user_col, item_col, rating_col}
        if not required.issubset(df_rev.columns):
            raise ValueError(f"Missing required columns: {required}")

        interactions = (
            df_rev[[user_col, item_col, rating_col]]
            .dropna()
            .groupby([user_col, item_col], as_index=False)[rating_col]
            .mean()
        )
        interactions[user_col] = interactions[user_col].astype(str)
        interactions[item_col] = interactions[item_col].astype(str)
        interactions[rating_col] = interactions[rating_col].astype(float)

        self.users = interactions[user_col].drop_duplicates().tolist()
        self.items = interactions[item_col].drop_duplicates().tolist()
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.item_to_idx = {b: i for i, b in enumerate(self.items)}

        user_idx = interactions[user_col].map(self.user_to_idx).to_numpy()
        item_idx = interactions[item_col].map(self.item_to_idx).to_numpy()
        ratings = interactions[rating_col].to_numpy(dtype=float)

        self.global_mean = float(ratings.mean()) if len(ratings) else 0.0

        n_users = len(self.users)
        n_items = len(self.items)

        user_sum = np.bincount(user_idx, weights=ratings, minlength=n_users)
        user_count = np.bincount(user_idx, minlength=n_users)
        user_mean = np.divide(
            user_sum,
            np.maximum(user_count, 1),
            where=np.maximum(user_count, 1) > 0,
        )
        self.user_bias = user_mean - self.global_mean

        item_sum = np.bincount(item_idx, weights=ratings, minlength=n_items)
        item_count = np.bincount(item_idx, minlength=n_items)
        item_mean = np.divide(
            item_sum,
            np.maximum(item_count, 1),
            where=np.maximum(item_count, 1) > 0,
        )
        self.item_bias = item_mean - self.global_mean

        baseline = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
        residual = ratings - baseline

        self.residual_matrix = coo_matrix(
            (residual, (user_idx, item_idx)),
            shape=(n_users, n_items),
            dtype=np.float32,
        ).tocsr()

        max_components = min(self.residual_matrix.shape) - 1
        effective_k = max(1, min(self.n_components, max_components))
        self.svd = TruncatedSVD(n_components=effective_k, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(self.residual_matrix)
        self.item_factors = self.svd.components_.T

        self.seen_items_by_user = (
            interactions.groupby(user_col)[item_col].apply(lambda s: set(s.tolist())).to_dict()
        )
        return self

    def _baseline(self, user_id: str | None, item_id: str | None) -> float:
        score = self.global_mean
        if user_id is not None and user_id in self.user_to_idx and self.user_bias is not None:
            score += float(self.user_bias[self.user_to_idx[user_id]])
        if item_id is not None and item_id in self.item_to_idx and self.item_bias is not None:
            score += float(self.item_bias[self.item_to_idx[item_id]])
        return score

    def predict_rating(self, user_id: str, biz_id: str) -> float:
        """Predict user rating for a business within configured rating bounds."""
        user_id = str(user_id)
        biz_id = str(biz_id)

        score = self._baseline(user_id, biz_id)

        if (
            user_id in self.user_to_idx
            and biz_id in self.item_to_idx
            and self.user_factors is not None
            and self.item_factors is not None
        ):
            u = self.user_to_idx[user_id]
            i = self.item_to_idx[biz_id]
            score += float(np.dot(self.user_factors[u], self.item_factors[i]))

        score = float(np.clip(score, self.min_rating, self.max_rating))
        return score

    def recommend_for_user(
        self,
        user_id: str,
        k: int = 10,
        candidate_business_ids: list[str] | None = None,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        """Return top-K CF predictions for one user."""
        if not self.items:
            return []

        user_id = str(user_id)
        if candidate_business_ids is None:
            candidate_ids = self.items.copy()
        else:
            candidate_ids = [str(b) for b in candidate_business_ids if str(b) in self.item_to_idx]

        if exclude_seen:
            seen = self.seen_items_by_user.get(user_id, set())
            candidate_ids = [b for b in candidate_ids if b not in seen]

        if not candidate_ids:
            return []

        scores = [(b, self.predict_rating(user_id, b)) for b in candidate_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
