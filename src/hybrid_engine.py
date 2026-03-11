from __future__ import annotations

import numpy as np


class HybridRecommender:
    """Blend content-based and collaborative signals into one ranking score.

    The model adapts to user cold-start:
    - New users receive a higher content-based weight.
    - Users with deeper history receive a higher collaborative weight.

    Optionally, a business-level prior can be injected (e.g., review/check-in/tip popularity).
    """

    def __init__(
        self,
        content_engine,
        cf_engine,
        min_alpha: float = 0.2,
        half_life: float = 12.0,
        prior_weight: float = 0.1,
        business_prior_scores: dict[str, float] | None = None,
    ):
        self.content_engine = content_engine
        self.cf_engine = cf_engine
        self.min_alpha = min_alpha
        self.half_life = half_life
        self.prior_weight = float(np.clip(prior_weight, 0.0, 0.5))
        self.business_prior_scores = business_prior_scores or {}

    def set_business_priors(self, prior_scores: dict[str, float] | None):
        """Inject normalized prior scores in [0, 1] by business_id."""
        self.business_prior_scores = prior_scores or {}

    def calculate_alpha(self, user_interaction_count: int) -> float:
        """
        Adaptive weight for content-based branch.
        0 interactions -> alpha ~= 1.0
        many interactions -> alpha approaches min_alpha
        """
        n = max(0, int(user_interaction_count))
        alpha = self.min_alpha + (1.0 - self.min_alpha) * np.exp(-n / self.half_life)
        return float(np.clip(alpha, self.min_alpha, 1.0))

    def _content_score(
        self,
        candidate_biz_id: str,
        history_business_ids: list[str],
        history_ratings: list[float] | None = None,
    ) -> float:
        if not history_business_ids:
            return 0.0

        weights = history_ratings if history_ratings and len(history_ratings) == len(history_business_ids) else None
        score_map = self.content_engine.score_candidates_for_history(
            history_business_ids=history_business_ids,
            candidate_business_ids=[candidate_biz_id],
            history_weights=weights,
        )
        return float(score_map.get(candidate_biz_id, 0.0))

    def _normalize_cf(self, rating: float) -> float:
        rating_range = self.cf_engine.max_rating - self.cf_engine.min_rating
        if rating_range <= 0:
            return 0.0
        return float(np.clip((rating - self.cf_engine.min_rating) / rating_range, 0.0, 1.0))

    def _prior_score(self, biz_id: str) -> float:
        return float(np.clip(self.business_prior_scores.get(str(biz_id), 0.0), 0.0, 1.0))

    def get_hybrid_score(
        self,
        user_id: str,
        biz_id: str,
        user_history_count: int | None = None,
        user_history_business_ids: list[str] | None = None,
        user_history_ratings: list[float] | None = None,
    ) -> float:
        history_ids = user_history_business_ids or []
        history_count = len(history_ids) if user_history_count is None else user_history_count
        alpha = self.calculate_alpha(history_count)

        content_score = self._content_score(
            candidate_biz_id=str(biz_id),
            history_business_ids=[str(x) for x in history_ids],
            history_ratings=user_history_ratings,
        )
        cf_pred = self.cf_engine.predict_rating(str(user_id), str(biz_id))
        cf_score = self._normalize_cf(cf_pred)

        hybrid_core = alpha * content_score + (1.0 - alpha) * cf_score
        prior_score = self._prior_score(str(biz_id))
        final_score = (1.0 - self.prior_weight) * hybrid_core + self.prior_weight * prior_score
        return float(final_score)

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        user_history_business_ids: list[str] | None = None,
        user_history_ratings: list[float] | None = None,
        candidate_business_ids: list[str] | None = None,
        exclude_history: bool = True,
    ) -> list[tuple[str, float]]:
        user_id = str(user_id)
        history_ids = [str(x) for x in (user_history_business_ids or [])]

        if candidate_business_ids is None:
            # Fast candidate generation: CF preselection then hybrid rerank.
            if getattr(self.cf_engine, "items", None):
                pre_k = min(800, len(self.cf_engine.items))
                cf_candidates = self.cf_engine.recommend_for_user(
                    user_id=user_id,
                    k=pre_k,
                    exclude_seen=exclude_history,
                )
                candidate_ids = [bid for bid, _ in cf_candidates]
                if not candidate_ids:
                    candidate_ids = self.cf_engine.items.copy()
            elif getattr(self.content_engine, "biz_ids", None):
                candidate_ids = self.content_engine.biz_ids.copy()
            else:
                candidate_ids = []

            if getattr(self.content_engine, "biz_ids", None):
                valid = set(self.content_engine.biz_ids)
                candidate_ids = [c for c in candidate_ids if c in valid]
        else:
            candidate_ids = [str(x) for x in candidate_business_ids]

        if exclude_history and history_ids:
            history_set = set(history_ids)
            candidate_ids = [c for c in candidate_ids if c not in history_set]

        if not candidate_ids:
            return []

        alpha = self.calculate_alpha(len(history_ids))
        content_scores = self.content_engine.score_candidates_for_history(
            history_business_ids=history_ids,
            candidate_business_ids=candidate_ids,
            history_weights=user_history_ratings,
        ) if history_ids else {}

        scores = []
        for candidate_id in candidate_ids:
            content_score = float(content_scores.get(candidate_id, 0.0))
            cf_pred = self.cf_engine.predict_rating(user_id, candidate_id)
            cf_score = self._normalize_cf(cf_pred)
            hybrid_core = alpha * content_score + (1.0 - alpha) * cf_score
            prior_score = self._prior_score(candidate_id)
            final_score = (1.0 - self.prior_weight) * hybrid_core + self.prior_weight * prior_score
            scores.append((candidate_id, float(final_score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
