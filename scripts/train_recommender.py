#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_user_store import LocalUserStore
from src.model_bundle import (
    ModelBundle,
    build_user_events,
    build_user_histories,
    save_model_bundle,
)
from src.pipeline import YelpHybridPipeline


def _checkpoint(name: str, ok: bool, details: str):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hybrid recommender from processed Yelp tables."
    )
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--local-data-dir", default="local_data")
    parser.add_argument("--user-path", default="raw_data/yelp_academic_dataset_user.json")
    parser.add_argument("--profile-chunksize", type=int, default=200_000)
    parser.add_argument("--max-user-chunks", type=int, default=0)
    parser.add_argument("--max-events-per-user", type=int, default=200)
    parser.add_argument("--skip-user-profiles", action="store_true")
    parser.add_argument(
        "--dataset-mode",
        choices=["yelp_only", "merged", "local_only"],
        default="yelp_only",
        help="Select interactions source for training/retraining.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--n-test-items", type=int, default=1)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=300,
        help="Cap evaluation users for faster terminal/deploy checkpoints.",
    )
    return parser.parse_args()


def extract_user_profiles(
    user_path: str,
    target_user_ids: set[str],
    chunksize: int = 200_000,
    max_chunks: int | None = None,
) -> dict[str, dict]:
    """Extract compact user profiles for explainability demo."""
    if not Path(user_path).exists() or not target_user_ids:
        return {}
    profiles: dict[str, dict] = {}
    cols = [
        "user_id",
        "name",
        "review_count",
        "yelping_since",
        "average_stars",
        "fans",
        "useful",
        "funny",
        "cool",
        "elite",
    ]
    for idx, chunk in enumerate(pd.read_json(user_path, lines=True, chunksize=chunksize), start=1):
        filtered = chunk[chunk["user_id"].astype(str).isin(target_user_ids)]
        if not filtered.empty:
            for row in filtered[cols].itertuples(index=False):
                profiles[str(row.user_id)] = {
                    "name": row.name,
                    "review_count_profile": int(row.review_count),
                    "yelping_since": str(row.yelping_since),
                    "average_stars_profile": float(row.average_stars),
                    "fans": int(row.fans),
                    "useful_votes": int(row.useful),
                    "funny_votes": int(row.funny),
                    "cool_votes": int(row.cool),
                    "elite": str(row.elite) if row.elite is not None else "",
                }
        if len(profiles) >= len(target_user_ids):
            break
        if max_chunks is not None and idx >= max_chunks:
            break
    return profiles


def main():
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    businesses_path = processed_dir / "businesses.pkl"
    interactions_path = processed_dir / "interactions.pkl"

    if not businesses_path.exists() or not interactions_path.exists():
        _checkpoint(
            "Load Processed",
            False,
            f"missing required files: {businesses_path}, {interactions_path}",
        )
        raise SystemExit(1)

    businesses = pd.read_pickle(businesses_path)
    yelp_interactions = pd.read_pickle(interactions_path)
    _checkpoint(
        "Load Processed",
        True,
        f"businesses={len(businesses)}, interactions={len(yelp_interactions)}",
    )

    local_store = LocalUserStore(args.local_data_dir)
    local_interactions = local_store.load_interactions_df()

    valid_businesses = set(businesses["business_id"].astype(str).tolist())
    if not local_interactions.empty:
        local_interactions = local_interactions[
            local_interactions["business_id"].astype(str).isin(valid_businesses)
        ].copy()

    if args.dataset_mode == "yelp_only":
        interactions = yelp_interactions.copy()
    elif args.dataset_mode == "local_only":
        interactions = local_interactions.copy()
    else:
        interactions = pd.concat([yelp_interactions, local_interactions], ignore_index=True)

    if interactions.empty:
        _checkpoint("Dataset Mode", False, f"no interactions for mode={args.dataset_mode}")
        raise SystemExit(1)

    _checkpoint(
        "Dataset Mode",
        True,
        f"mode={args.dataset_mode}, yelp={len(yelp_interactions)}, local={len(local_interactions)}, train={len(interactions)}",
    )

    pipeline = YelpHybridPipeline(prior_weight=args.prior_weight)
    pipeline.fit(businesses, interactions)
    _checkpoint("Train Models", True, "content + collaborative + hybrid fitted")

    metrics = pipeline.evaluate(
        interactions_df=interactions,
        k=args.k,
        min_history=args.min_history,
        n_test_items=args.n_test_items,
        max_users=args.max_eval_users,
    )
    _checkpoint("Evaluate", True, json.dumps(metrics))

    # Refit on full interactions for final serving bundle.
    pipeline.fit(businesses, interactions)

    business_index = businesses[
        [
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
            if c in businesses.columns
        ]
    ].copy()
    user_histories = build_user_histories(interactions)
    user_events = build_user_events(interactions, max_events_per_user=args.max_events_per_user)
    max_user_chunks = args.max_user_chunks if args.max_user_chunks > 0 else None
    user_profiles = {}
    if not args.skip_user_profiles:
        user_profiles = extract_user_profiles(
            user_path=args.user_path,
            target_user_ids=set(user_histories.keys()),
            chunksize=args.profile_chunksize,
            max_chunks=max_user_chunks,
        )
    _checkpoint("User Profiles", True, f"profiles={len(user_profiles)}, events={len(user_events)}")

    bundle = ModelBundle(
        content_engine=pipeline.content_engine,
        cf_engine=pipeline.cf_engine,
        hybrid_recommender=pipeline.hybrid,
        business_index=business_index,
        user_histories=user_histories,
        user_events=user_events,
        user_profiles=user_profiles,
        metrics=metrics,
    )
    bundle_path = save_model_bundle(bundle, args.artifact_dir)
    _checkpoint("Save Bundle", True, str(bundle_path))

    top_users = list(user_histories.keys())[:3]
    for user_id in top_users:
        preview = bundle.recommend(user_id=user_id, k=min(5, args.k))
        _checkpoint(
            f"Sample Recommend user={user_id}",
            len(preview) > 0,
            f"n_results={len(preview)}",
        )


if __name__ == "__main__":
    main()
