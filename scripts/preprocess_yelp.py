#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import DataConfig, DataLoader


def _checkpoint(name: str, ok: bool, details: str):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {details}")


def _normalize_01(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    min_v = values.min()
    max_v = values.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v <= min_v:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - min_v) / (max_v - min_v)


def _count_tips(
    tip_path: str,
    business_ids: set[str],
    chunksize: int = 200_000,
    max_chunks: int | None = None,
    verbose: bool = False,
    progress_every: int = 5,
) -> pd.Series:
    """Count tips per business from yelp_academic_dataset_tip.json."""
    counts: dict[str, int] = {}
    total_rows = 0
    start_ts = time.perf_counter()
    for idx, chunk in enumerate(pd.read_json(tip_path, lines=True, chunksize=chunksize), start=1):
        total_rows += len(chunk)
        filtered = chunk[chunk["business_id"].isin(business_ids)]
        if not filtered.empty:
            vc = filtered["business_id"].astype(str).value_counts()
            for business_id, n in vc.items():
                counts[business_id] = counts.get(business_id, 0) + int(n)
        if verbose and progress_every > 0 and idx % progress_every == 0:
            elapsed = time.perf_counter() - start_ts
            print(f"[INFO] Tip chunks={idx} rows={total_rows} elapsed_sec={elapsed:.1f}")
        if max_chunks is not None and idx >= max_chunks:
            break
    if verbose:
        elapsed = time.perf_counter() - start_ts
        print(f"[INFO] Tip scan done chunks={idx} rows={total_rows} elapsed_sec={elapsed:.1f}")
    return pd.Series(counts, name="tip_count", dtype=float)


def _count_checkins(
    checkin_path: str,
    business_ids: set[str],
    chunksize: int = 200_000,
    max_chunks: int | None = None,
    verbose: bool = False,
    progress_every: int = 5,
) -> pd.Series:
    """Count check-in events per business from yelp_academic_dataset_checkin.json."""
    counts: dict[str, int] = {}
    total_rows = 0
    start_ts = time.perf_counter()
    for idx, chunk in enumerate(pd.read_json(checkin_path, lines=True, chunksize=chunksize), start=1):
        total_rows += len(chunk)
        filtered = chunk[chunk["business_id"].isin(business_ids)]
        if not filtered.empty:
            filtered = filtered[["business_id", "date"]].copy()
            filtered["date"] = filtered["date"].fillna("")
            filtered["checkin_count"] = filtered["date"].apply(
                lambda x: 0 if not x else len(str(x).split(","))
            )
            grouped = filtered.groupby("business_id")["checkin_count"].sum()
            for business_id, n in grouped.items():
                counts[str(business_id)] = counts.get(str(business_id), 0) + int(n)
        if verbose and progress_every > 0 and idx % progress_every == 0:
            elapsed = time.perf_counter() - start_ts
            print(f"[INFO] Checkin chunks={idx} rows={total_rows} elapsed_sec={elapsed:.1f}")
        if max_chunks is not None and idx >= max_chunks:
            break
    if verbose:
        elapsed = time.perf_counter() - start_ts
        print(f"[INFO] Checkin scan done chunks={idx} rows={total_rows} elapsed_sec={elapsed:.1f}")
    return pd.Series(counts, name="checkin_count", dtype=float)


def _add_business_priors(
    businesses: pd.DataFrame,
    tip_counts: pd.Series | None = None,
    checkin_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Create normalized popularity prior from Yelp auxiliary signals."""
    df = businesses.copy()
    if "tip_count" in df.columns:
        df["tip_count"] = pd.to_numeric(df["tip_count"], errors="coerce").fillna(0.0)
    else:
        df["tip_count"] = 0.0
    if "checkin_count" in df.columns:
        df["checkin_count"] = pd.to_numeric(df["checkin_count"], errors="coerce").fillna(0.0)
    else:
        df["checkin_count"] = 0.0

    if tip_counts is not None and not tip_counts.empty:
        df = df.merge(
            tip_counts.rename_axis("business_id").reset_index(),
            on="business_id",
            how="left",
            suffixes=("", "_tip"),
        )
        df["tip_count"] = df["tip_count_tip"].fillna(df["tip_count"])
        df = df.drop(columns=["tip_count_tip"])

    if checkin_counts is not None and not checkin_counts.empty:
        df = df.merge(
            checkin_counts.rename_axis("business_id").reset_index(),
            on="business_id",
            how="left",
            suffixes=("", "_checkin"),
        )
        df["checkin_count"] = df["checkin_count_checkin"].fillna(df["checkin_count"])
        df = df.drop(columns=["checkin_count_checkin"])

    stars_norm = _normalize_01(df["stars"].fillna(0))
    review_norm = _normalize_01(np.log1p(df["review_count"].fillna(0)))
    tip_norm = _normalize_01(np.log1p(df["tip_count"].fillna(0)))
    checkin_norm = _normalize_01(np.log1p(df["checkin_count"].fillna(0)))

    df["popularity_prior"] = (
        0.35 * review_norm
        + 0.25 * stars_norm
        + 0.20 * tip_norm
        + 0.20 * checkin_norm
    ).astype(float)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Yelp raw JSON into model-ready tables."
    )
    parser.add_argument(
        "--business-path",
        default="raw_data/yelp_academic_dataset_business.json",
    )
    parser.add_argument(
        "--review-path",
        default="raw_data/yelp_academic_dataset_review.json",
    )
    parser.add_argument(
        "--tip-path",
        default="raw_data/yelp_academic_dataset_tip.json",
    )
    parser.add_argument(
        "--checkin-path",
        default="raw_data/yelp_academic_dataset_checkin.json",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--input-format",
        choices=["auto", "json", "parquet"],
        default="auto",
        help="Prefer parquet exports in data/ when available, otherwise fallback to raw JSON.",
    )
    parser.add_argument("--out-dir", default="processed")
    parser.add_argument("--city", default="")
    parser.add_argument("--state", default="")
    parser.add_argument(
        "--usa-only",
        action="store_true",
        default=True,
        help="Limit businesses to U.S. state codes.",
    )
    parser.add_argument(
        "--no-usa-only",
        action="store_false",
        dest="usa_only",
        help="Disable U.S.-only filter.",
    )
    parser.add_argument("--category-keyword", default="Restaurants")
    parser.add_argument("--min-business-reviews", type=int, default=20)
    parser.add_argument("--min-user-reviews", type=int, default=3)
    parser.add_argument("--min-item-reviews", type=int, default=5)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-review-chunks", type=int, default=0)
    parser.add_argument("--max-aux-chunks", type=int, default=0)
    parser.add_argument("--save-reviews", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_review_chunks = args.max_review_chunks if args.max_review_chunks > 0 else None
    max_aux_chunks = args.max_aux_chunks if args.max_aux_chunks > 0 else None

    verbose = not args.quiet
    cfg = DataConfig(
        min_business_reviews=args.min_business_reviews,
        min_user_reviews=args.min_user_reviews,
        min_item_reviews=args.min_item_reviews,
        category_keyword=args.category_keyword,
        city=args.city or None,
        state=args.state or None,
        usa_only=args.usa_only,
        chunksize=args.chunksize,
        max_review_chunks=max_review_chunks,
        verbose=verbose,
        progress_every=args.progress_every,
    )

    loader = DataLoader(
        args.business_path,
        args.review_path,
        source_format=args.input_format,
        data_dir=args.data_dir,
    )
    businesses, reviews, interactions = loader.run(cfg)
    resolved_source = loader.resolve_source_format()
    _checkpoint("Input Source", True, f"source_format={resolved_source}")
    _checkpoint(
        "Preprocess Core",
        not businesses.empty and not interactions.empty,
        f"businesses={len(businesses)}, reviews={len(reviews)}, interactions={len(interactions)}",
    )

    business_id_set = set(businesses["business_id"].astype(str).tolist())

    tip_counts = pd.Series(dtype=float)
    checkin_counts = pd.Series(dtype=float)
    if "popularity_prior" in businesses.columns and businesses["popularity_prior"].notna().any():
        tip_count_present = int((pd.to_numeric(businesses.get("tip_count", 0), errors="coerce").fillna(0) > 0).sum()) if "tip_count" in businesses.columns else 0
        checkin_count_present = int((pd.to_numeric(businesses.get("checkin_count", 0), errors="coerce").fillna(0) > 0).sum()) if "checkin_count" in businesses.columns else 0
        _checkpoint("Tip Signals", True, f"loaded_from={resolved_source}, businesses_with_tips={tip_count_present}")
        _checkpoint("Check-in Signals", True, f"loaded_from={resolved_source}, businesses_with_checkins={checkin_count_present}")
    elif resolved_source == "parquet" and {"tip_count", "checkin_count"}.issubset(businesses.columns):
        businesses = _add_business_priors(businesses)
        tip_count_present = int((businesses["tip_count"] > 0).sum())
        checkin_count_present = int((businesses["checkin_count"] > 0).sum())
        _checkpoint("Tip Signals", True, f"computed_from=parquet, businesses_with_tips={tip_count_present}")
        _checkpoint("Check-in Signals", True, f"computed_from=parquet, businesses_with_checkins={checkin_count_present}")
    else:
        if Path(args.tip_path).exists():
            tip_counts = _count_tips(
                args.tip_path,
                business_id_set,
                chunksize=args.chunksize,
                max_chunks=max_aux_chunks,
                verbose=verbose,
                progress_every=args.progress_every,
            )
            _checkpoint("Tip Signals", True, f"businesses_with_tips={len(tip_counts)}")
        else:
            _checkpoint("Tip Signals", False, f"missing_file={args.tip_path}")

        if Path(args.checkin_path).exists():
            checkin_counts = _count_checkins(
                args.checkin_path,
                business_id_set,
                chunksize=args.chunksize,
                max_chunks=max_aux_chunks,
                verbose=verbose,
                progress_every=args.progress_every,
            )
            _checkpoint("Check-in Signals", True, f"businesses_with_checkins={len(checkin_counts)}")
        else:
            _checkpoint("Check-in Signals", False, f"missing_file={args.checkin_path}")

        businesses = _add_business_priors(businesses, tip_counts=tip_counts, checkin_counts=checkin_counts)

    businesses_path = out_dir / "businesses.pkl"
    interactions_path = out_dir / "interactions.pkl"
    businesses.to_pickle(businesses_path)
    interactions.to_pickle(interactions_path)
    _checkpoint("Persist Processed", True, f"{businesses_path}, {interactions_path}")

    review_path_out = None
    if args.save_reviews:
        review_path_out = out_dir / "reviews.pkl"
        reviews.to_pickle(review_path_out)
        _checkpoint("Persist Reviews", True, str(review_path_out))

    summary = {
        "config": vars(args),
        "n_businesses": int(len(businesses)),
        "n_reviews": int(len(reviews)),
        "n_interactions": int(len(interactions)),
        "paths": {
            "businesses": str(businesses_path),
            "interactions": str(interactions_path),
            "reviews": str(review_path_out) if review_path_out else None,
        },
        "signals": {
            "businesses_with_tip_count": int(len(tip_counts)),
            "businesses_with_checkin_count": int(len(checkin_counts)),
        },
    }
    summary_path = out_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _checkpoint("Summary", True, str(summary_path))


if __name__ == "__main__":
    main()
