#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_bundle import load_model_bundle


def _print_check(name: str, ok: bool, detail: str):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminal checkpoint to classify codebase/deployment health."
    )
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--bundle-path", default="artifacts/model_bundle.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    all_ok = True

    deps = ["numpy", "pandas", "scipy", "sklearn", "fastapi", "uvicorn"]
    for mod in deps:
        try:
            importlib.import_module(mod)
            _print_check(f"Dependency {mod}", True, "import ok")
        except Exception as exc:
            all_ok = False
            _print_check(f"Dependency {mod}", False, str(exc))

    raw_dir = Path(args.raw_data_dir)
    data_dir = Path(args.data_dir)
    expected_inputs = [
        ("Business source", raw_dir / "yelp_academic_dataset_business.json", data_dir / "business_main.parquet"),
        ("Review source", raw_dir / "yelp_academic_dataset_review.json", data_dir / "reviews.parquet"),
        ("Tip source", raw_dir / "yelp_academic_dataset_tip.json", data_dir / "tips.parquet"),
        ("Check-in source", raw_dir / "yelp_academic_dataset_checkin.json", data_dir / "checkins.parquet"),
    ]
    for name, raw_path, parquet_path in expected_inputs:
        ok = raw_path.exists() or parquet_path.exists()
        all_ok = all_ok and ok
        _print_check(
            name,
            ok,
            f"raw_exists={raw_path.exists()} ({raw_path}) | parquet_exists={parquet_path.exists()} ({parquet_path})",
        )

    processed_dir = Path(args.processed_dir)
    expected_processed = ["businesses.pkl", "interactions.pkl", "preprocess_summary.json"]
    for filename in expected_processed:
        path = processed_dir / filename
        ok = path.exists()
        all_ok = all_ok and ok
        _print_check(f"Processed file {filename}", ok, str(path))

    bundle_path = Path(args.bundle_path)
    if not bundle_path.exists():
        all_ok = False
        _print_check("Bundle file", False, str(bundle_path))
    else:
        try:
            bundle = load_model_bundle(str(bundle_path))
            _print_check(
                "Bundle load",
                True,
                f"businesses={len(bundle.business_index)}, users={len(bundle.user_histories)}",
            )
            sample_user = next(iter(bundle.user_histories.keys()), None)
            if sample_user is None:
                all_ok = False
                _print_check("Bundle sample user", False, "no user history")
            else:
                recs = bundle.recommend(sample_user, k=5)
                ok = len(recs) > 0
                all_ok = all_ok and ok
                _print_check("Inference test", ok, f"user_id={sample_user}, n_recs={len(recs)}")
        except Exception as exc:
            all_ok = False
            _print_check("Bundle load", False, str(exc))

    if all_ok:
        print("\nOVERALL: GOOD (deployment-ready checkpoint passed)")
        raise SystemExit(0)

    print("\nOVERALL: BAD (fix failed checks before deploy)")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
