#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_user_store import LocalUserStore
from src.model_bundle import (
    ModelBundle,
    build_user_events_for_subset,
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
    parser.add_argument("--max-events-per-user", type=int, default=200)
    parser.add_argument(
        "--bundle-event-users",
        type=int,
        default=300,
        help="Persist readable activity logs only for a small demo subset of users.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["yelp_only", "merged", "local_only"],
        default="yelp_only",
        help="Select interactions source for training/retraining.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--n-test-items", type=int, default=1)
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Only held-out ratings >= threshold are treated as relevant in offline ranking metrics.",
    )
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=300,
        help="Cap evaluation users for faster terminal/deploy checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
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
    if verbose:
        print("[INFO] Fitting content + collaborative models...")
    t0 = time.perf_counter()
    pipeline.fit(businesses, interactions)
    if verbose:
        print(f"[INFO] Fit completed in {time.perf_counter() - t0:.1f}s")
    _checkpoint("Train Models", True, "content + collaborative + hybrid fitted")

    if verbose:
        print("[INFO] Evaluating hybrid model...")
    metrics = pipeline.evaluate(
        interactions_df=interactions,
        k=args.k,
        min_history=args.min_history,
        n_test_items=args.n_test_items,
        max_users=args.max_eval_users,
        progress_every=args.progress_every,
        verbose=verbose,
        relevance_threshold=args.relevance_threshold,
    )
    _checkpoint("Evaluate", True, json.dumps(metrics))

    # Refit on full interactions for final serving bundle.
    if verbose:
        print("[INFO] Refit on full interactions for serving bundle...")
    t1 = time.perf_counter()
    pipeline.fit(businesses, interactions)
    if verbose:
        print(f"[INFO] Refit completed in {time.perf_counter() - t1:.1f}s")

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
    local_event_users = []
    if not local_interactions.empty:
        local_event_users = local_interactions["user_id"].astype(str).drop_duplicates().tolist()
    demo_event_users = list(user_histories.keys())[: max(0, args.bundle_event_users)]
    event_user_ids = set(local_event_users + demo_event_users)
    user_events = build_user_events_for_subset(
        interactions,
        user_ids=event_user_ids,
        max_events_per_user=args.max_events_per_user,
    )
    _checkpoint("Bundle Events", True, f"event_users={len(user_events)}, histories={len(user_histories)}")

    bundle = ModelBundle(
        content_engine=pipeline.content_engine,
        cf_engine=pipeline.cf_engine,
        hybrid_recommender=pipeline.hybrid,
        business_index=business_index,
        user_histories=user_histories,
        user_events=user_events,
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
