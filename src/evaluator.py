from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_leave_last_split(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "business_id",
    time_col: str = "date",
    min_history: int = 3,
    n_test_items: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions per user by holding out the most recent items."""
    required = {user_col, item_col, time_col}
    if not required.issubset(interactions.columns):
        raise ValueError(f"Missing required columns: {required}")

    df = interactions[[user_col, item_col, "stars", time_col]].copy()
    df = df.dropna(subset=[user_col, item_col])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([user_col, time_col])

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, grp in df.groupby(user_col, sort=False):
        if len(grp) < (min_history + n_test_items):
            continue
        split_at = len(grp) - n_test_items
        train_parts.append(grp.iloc[:split_at])
        test_parts.append(grp.iloc[split_at:])

    if not train_parts or not test_parts:
        empty = pd.DataFrame(columns=df.columns)
        return empty, empty

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def precision_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K for binary relevance."""
    if k <= 0 or not recommended_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K for binary relevance."""
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@K for binary relevance."""
    if k <= 0 or not recommended_ids or not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]
    dcg = 0.0
    for rank, item_id in enumerate(top_k, start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / np.log2(rank + 1.0)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / np.log2(i + 1.0) for i in range(2, ideal_hits + 2))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def evaluate_hybrid_model(
    recommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "business_id",
    rating_col: str = "stars",
    time_col: str = "date",
    k: int = 10,
    max_users: int | None = None,
) -> dict[str, float]:
    """Evaluate hybrid recommender over users with temporal holdout."""
    if train_df.empty or test_df.empty:
        return {
            "evaluated_users": 0,
            "precision@k": 0.0,
            "recall@k": 0.0,
            "ndcg@k": 0.0,
            "coverage": 0.0,
        }

    train_df = train_df.sort_values([user_col, time_col])
    test_df = test_df.sort_values([user_col, time_col])

    train_histories = train_df.groupby(user_col).agg(
        history_items=(item_col, list),
        history_ratings=(rating_col, list),
    )
    test_relevant = test_df.groupby(user_col)[item_col].apply(lambda s: set(s.astype(str)))
    if max_users is not None and max_users > 0:
        test_relevant = test_relevant.iloc[:max_users]

    prec_scores: list[float] = []
    rec_scores: list[float] = []
    ndcg_scores: list[float] = []
    unique_recommended: set[str] = set()

    for user_id, relevant in test_relevant.items():
        if user_id not in train_histories.index:
            continue
        history_items = [str(x) for x in train_histories.loc[user_id, "history_items"]]
        history_ratings = [float(x) for x in train_histories.loc[user_id, "history_ratings"]]
        if not history_items:
            continue

        recs = recommender.recommend(
            user_id=str(user_id),
            k=k,
            user_history_business_ids=history_items,
            user_history_ratings=history_ratings,
            exclude_history=True,
        )
        recommended_ids = [str(item_id) for item_id, _ in recs]
        if not recommended_ids:
            continue

        unique_recommended.update(recommended_ids[:k])
        rel = {str(x) for x in relevant}
        prec_scores.append(precision_at_k(recommended_ids, rel, k))
        rec_scores.append(recall_at_k(recommended_ids, rel, k))
        ndcg_scores.append(ndcg_at_k(recommended_ids, rel, k))

    evaluated_users = len(prec_scores)
    item_universe = set()
    if hasattr(recommender, "cf_engine") and getattr(recommender.cf_engine, "items", None):
        item_universe = {str(x) for x in recommender.cf_engine.items}
    elif hasattr(recommender, "content_engine") and getattr(
        recommender.content_engine, "biz_ids", None
    ):
        item_universe = {str(x) for x in recommender.content_engine.biz_ids}
    else:
        item_universe = set(train_df[item_col].astype(str).tolist())
    coverage = len(unique_recommended) / len(item_universe) if item_universe else 0.0

    return {
        "evaluated_users": float(evaluated_users),
        "precision@k": float(np.mean(prec_scores)) if prec_scores else 0.0,
        "recall@k": float(np.mean(rec_scores)) if rec_scores else 0.0,
        "ndcg@k": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "coverage": float(coverage),
    }
