from __future__ import annotations

from collections import Counter
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class ModelBundle:
    """Deployment-ready container for recommendation inference.

    The bundle keeps model engines, business metadata, and precomputed
    user histories for fast recommendation serving in APIs/demo apps.
    """

    content_engine: object
    cf_engine: object
    hybrid_recommender: object
    business_index: pd.DataFrame
    user_histories: dict[str, tuple[list[str], list[float]]]
    user_events: dict[str, list[dict]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    _business_map: dict[str, dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        index_cols = [
            c
            for c in [
                "business_id",
                "name",
                "city",
                "state",
                "stars",
                "review_count",
                "categories",
                "popularity_prior",
            ]
            if c in self.business_index.columns
        ]
        indexed = self.business_index[index_cols].copy()
        indexed["business_id"] = indexed["business_id"].astype(str)
        self._business_map = {
            row.business_id: row._asdict() for row in indexed.itertuples(index=False)
        }

    def _fallback_popularity(self, k: int, exclude_ids: set[str] | None = None) -> list[tuple[str, float]]:
        exclude_ids = exclude_ids or set()
        rows = []
        for business_id, payload in self._business_map.items():
            if business_id in exclude_ids:
                continue
            prior = float(payload.get("popularity_prior", 0.0) or 0.0)
            stars = float(payload.get("stars", 0.0) or 0.0) / 5.0
            score = 0.7 * prior + 0.3 * stars
            rows.append((business_id, score))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows[:k]

    @staticmethod
    def _activity_level(interaction_count: int) -> str:
        if interaction_count < 5:
            return "low"
        if interaction_count < 25:
            return "medium"
        return "high"

    def has_business(self, business_id: str) -> bool:
        return str(business_id) in self._business_map

    @staticmethod
    def _split_categories(value: object) -> list[str]:
        if value is None:
            return []
        return [c.strip() for c in str(value).split(",") if c and c.strip()]

    def _user_category_counts(self, history_business_ids: list[str]) -> Counter:
        counts: Counter = Counter()
        for business_id in history_business_ids:
            info = self._business_map.get(str(business_id), {})
            counts.update(self._split_categories(info.get("categories")))
        return counts

    def _build_activity_summary(
        self,
        activity_level: str,
        interaction_count: int,
        avg_given_stars: float,
        top_categories: list[str],
    ) -> str:
        if interaction_count == 0:
            return "No interaction history yet. The system treats this user as cold-start."
        cat_text = ", ".join(top_categories[:3]) if top_categories else "mixed categories"
        return (
            f"User has {activity_level} activity with {interaction_count} interactions, "
            f"average given stars {avg_given_stars:.2f}, and most often engages with {cat_text}."
        )

    def get_user_profile(self, user_id: str) -> dict:
        user_id = str(user_id)
        legacy_profiles = getattr(self, "user_profiles", {})
        profile = dict(legacy_profiles.get(user_id, {})) if isinstance(legacy_profiles, dict) else {}
        hist_items, hist_ratings = self.user_histories.get(user_id, ([], []))
        interaction_count = len(hist_items)
        avg_given = float(sum(hist_ratings) / len(hist_ratings)) if hist_ratings else 0.0
        activity_level = self._activity_level(interaction_count)
        pos_ratio = (
            float(sum(1 for x in hist_ratings if float(x) >= 4.0) / len(hist_ratings))
            if hist_ratings
            else 0.0
        )
        category_counts = self._user_category_counts(hist_items)
        top_categories = [k for k, _ in category_counts.most_common(5)]
        recent_event = self.user_events.get(user_id, [])
        recent_activity_date = recent_event[0].get("date") if recent_event else None
        profile.update(
            {
                "user_id": user_id,
                "interaction_count": interaction_count,
                "activity_level": activity_level,
                "avg_given_stars": avg_given,
                "positive_rating_ratio": pos_ratio,
                "top_categories": top_categories,
                "recent_activity_date": recent_activity_date,
                "activity_summary": self._build_activity_summary(
                    activity_level=activity_level,
                    interaction_count=interaction_count,
                    avg_given_stars=avg_given,
                    top_categories=top_categories,
                ),
            }
        )
        return profile

    def get_user_activities(self, user_id: str, limit: int = 30) -> list[dict]:
        user_id = str(user_id)
        events = self.user_events.get(user_id, [])
        out: list[dict] = []
        for event in events[: max(1, limit)]:
            business_id = str(event.get("business_id"))
            info = self._business_map.get(business_id, {})
            out.append(
                {
                    "date": event.get("date"),
                    "business_id": business_id,
                    "business_name": info.get("name"),
                    "city": info.get("city"),
                    "stars": event.get("stars"),
                    "categories": info.get("categories"),
                }
            )
        return out

    def search_businesses(self, query: str, limit: int = 20) -> list[dict]:
        q = query.strip().lower() if query else ""
        if not q:
            return []
        rows: list[dict] = []
        for business_id, payload in self._business_map.items():
            hay = " ".join(
                [
                    str(payload.get("name", "")),
                    str(payload.get("city", "")),
                    str(payload.get("categories", "")),
                ]
            ).lower()
            if q in hay:
                row = dict(payload)
                row["business_id"] = business_id
                rows.append(row)
                if len(rows) >= limit:
                    break
        return rows

    def search_businesses_filtered(
        self,
        query: str = "",
        limit: int = 20,
        categories: list[str] | None = None,
        name_only: bool = False,
    ) -> list[dict]:
        """Search businesses by name/text and optional category filters."""
        q = (query or "").strip().lower()
        wanted = {c.strip().lower() for c in (categories or []) if c and c.strip()}
        rows: list[dict] = []
        for business_id, payload in self._business_map.items():
            name = str(payload.get("name", ""))
            categories_text = str(payload.get("categories", ""))
            hay = name.lower() if name_only else f"{name} {payload.get('city', '')} {categories_text}".lower()

            if q and q not in hay:
                continue
            if wanted:
                biz_categories = {c.strip().lower() for c in categories_text.split(",") if c and c.strip()}
                if not biz_categories.intersection(wanted):
                    continue

            row = dict(payload)
            row["business_id"] = business_id
            rows.append(row)
            if len(rows) >= max(1, limit):
                break
        return rows

    def list_categories(self, limit: int = 300) -> list[dict]:
        """Return top business categories with counts for UI filters."""
        counter: Counter = Counter()
        for payload in self._business_map.values():
            for category in self._split_categories(payload.get("categories")):
                counter[category] += 1
        return [{"category": name, "count": int(count)} for name, count in counter.most_common(max(1, limit))]

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        history_business_ids: list[str] | None = None,
        history_ratings: list[float] | None = None,
        algorithm: str = "hybrid",
    ) -> list[dict]:
        """Generate enriched recommendations for API/demo output."""
        user_id = str(user_id)
        algo = algorithm.lower().strip()

        hist_ids = history_business_ids
        hist_ratings = history_ratings
        if hist_ids is None:
            hist_ids, hist_ratings = self.user_histories.get(user_id, ([], []))
        history_set = set(hist_ids or [])
        history_count = len(hist_ids or [])
        activity_level = self._activity_level(history_count)
        category_counts = self._user_category_counts(hist_ids or [])
        user_top_categories = {k for k, _ in category_counts.most_common(5)}

        candidate_ids_for_explain: list[str] = []
        content_scores: dict[str, float] = {}

        if algo == "content":
            if hist_ids:
                recs = self.content_engine.recommend_from_history(
                    history_business_ids=hist_ids,
                    history_weights=hist_ratings,
                    k=k,
                    exclude_history=True,
                )
            else:
                recs = self._fallback_popularity(k=k, exclude_ids=history_set)
        elif algo in ("cf", "collaborative"):
            recs = self.cf_engine.recommend_for_user(
                user_id=user_id,
                k=k,
                exclude_seen=True,
            )
            if not recs:
                recs = self._fallback_popularity(k=k, exclude_ids=history_set)
        else:
            recs = self.hybrid_recommender.recommend(
                user_id=user_id,
                k=k,
                user_history_business_ids=hist_ids,
                user_history_ratings=hist_ratings,
                exclude_history=True,
            )
            if not recs:
                recs = self._fallback_popularity(k=k, exclude_ids=history_set)

        if hist_ids and recs:
            candidate_ids_for_explain = [str(business_id) for business_id, _ in recs]
            content_weights = None
            if hasattr(self.hybrid_recommender, "prepare_content_history_weights"):
                content_weights = self.hybrid_recommender.prepare_content_history_weights(
                    history_business_ids=hist_ids,
                    history_ratings=hist_ratings,
                )
            content_scores = self.content_engine.score_candidates_for_history(
                history_business_ids=hist_ids,
                candidate_business_ids=candidate_ids_for_explain,
                history_weights=content_weights,
            )

        enriched: list[dict] = []
        for business_id, score in recs:
            info = self._business_map.get(str(business_id), {"business_id": str(business_id)})
            content_score = float(content_scores.get(str(business_id), 0.0)) if content_scores else 0.0
            cf_pred = float(self.cf_engine.predict_rating(user_id, str(business_id)))
            prior_score = float(getattr(self.hybrid_recommender, "business_prior_scores", {}).get(str(business_id), 0.0))
            alpha = float(self.hybrid_recommender.calculate_alpha(history_count))
            biz_categories = set(self._split_categories(info.get("categories")))
            overlap = sorted(biz_categories.intersection(user_top_categories))
            reason_tags: list[str] = []
            if history_count == 0:
                reason_tags.append(
                    "No history available, so ranking leans on popularity priors and global collaborative signal."
                )
            else:
                if overlap:
                    reason_tags.append(
                        f"Category match with frequent interests: {', '.join(overlap[:3])}."
                    )
                if content_score >= 0.18:
                    reason_tags.append(
                        f"Content similarity is strong ({content_score:.3f}) against the user's past businesses."
                    )
                if cf_pred >= 4.0:
                    reason_tags.append(
                        f"Collaborative branch predicts a high rating ({cf_pred:.2f}/5)."
                    )
                if algo == "hybrid":
                    reason_tags.append(
                        f"Hybrid blend uses alpha={alpha:.3f}, adapting to {activity_level} activity."
                    )
            if not reason_tags:
                reason_tags.append(
                    "Business remains competitive after blending content, collaborative, and prior signals."
                )
            explanation = " ".join(reason_tags)
            enriched.append(
                {
                    "business_id": str(business_id),
                    "score": float(score),
                    "name": info.get("name"),
                    "city": info.get("city"),
                    "state": info.get("state"),
                    "address": info.get("address"),
                    "latitude": info.get("latitude"),
                    "longitude": info.get("longitude"),
                    "stars": info.get("stars"),
                    "review_count": info.get("review_count"),
                    "categories": info.get("categories"),
                    "popularity_prior": info.get("popularity_prior"),
                    "algorithm": algo,
                    "explanation": explanation,
                    "reason_tags": reason_tags,
                    "category_overlap": overlap[:5],
                    "activity_level": activity_level,
                    "history_count": history_count,
                    "content_score": content_score,
                    "cf_pred_rating": cf_pred,
                    "hybrid_alpha": alpha,
                    "prior_score": prior_score,
                }
            )
        return enriched


def build_user_histories(interactions_df: pd.DataFrame) -> dict[str, tuple[list[str], list[float]]]:
    """Build ordered user histories used by serving/inference."""
    if interactions_df.empty:
        return {}

    df = interactions_df.copy()
    if "date" in df.columns:
        df = df.sort_values(["user_id", "date"])
    else:
        df = df.sort_values(["user_id"])

    histories = (
        df.groupby("user_id")
        .agg(items=("business_id", lambda s: [str(x) for x in s.tolist()]), ratings=("stars", lambda s: [float(x) for x in s.tolist()]))
        .to_dict(orient="index")
    )
    return {str(user_id): (payload["items"], payload["ratings"]) for user_id, payload in histories.items()}


def build_user_events(interactions_df: pd.DataFrame, max_events_per_user: int = 200) -> dict[str, list[dict]]:
    """Build readable user activity logs for demo/explainability."""
    if interactions_df.empty:
        return {}
    df = interactions_df.copy()
    if "date" in df.columns:
        df = df.sort_values(["user_id", "date"], ascending=[True, False])
    else:
        df = df.sort_values(["user_id"])
    out: dict[str, list[dict]] = {}
    for user_id, grp in df.groupby("user_id", sort=False):
        rows = grp.head(max_events_per_user)
        out[str(user_id)] = [
            {
                "business_id": str(r.business_id),
                "stars": float(r.stars),
                "date": str(r.date) if hasattr(r, "date") else None,
            }
            for r in rows.itertuples(index=False)
        ]
    return out


def build_user_events_for_subset(
    interactions_df: pd.DataFrame,
    user_ids: set[str],
    max_events_per_user: int = 200,
) -> dict[str, list[dict]]:
    """Build user activity logs only for selected users to keep bundles slim."""
    if interactions_df.empty or not user_ids:
        return {}
    filtered = interactions_df[interactions_df["user_id"].astype(str).isin({str(x) for x in user_ids})].copy()
    if filtered.empty:
        return {}
    return build_user_events(filtered, max_events_per_user=max_events_per_user)


def save_model_bundle(bundle: ModelBundle, artifact_dir: str) -> Path:
    """Persist model bundle and summary metadata for deployment."""
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = out_dir / "model_bundle.pkl"
    with bundle_path.open("wb") as f:
        pickle.dump(bundle, f)

    summary = {
        "created_at_utc": bundle.created_at_utc,
        "n_businesses": int(len(bundle.business_index)),
        "n_users_with_history": int(len(bundle.user_histories)),
        "n_users_with_events": int(len(bundle.user_events)),
        "metrics": bundle.metrics,
        "bundle_path": str(bundle_path),
    }
    (out_dir / "bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return bundle_path


def load_model_bundle(bundle_path: str) -> ModelBundle:
    """Load persisted bundle from disk."""
    path = Path(bundle_path)
    with path.open("rb") as f:
        bundle = pickle.load(f)
    return bundle
