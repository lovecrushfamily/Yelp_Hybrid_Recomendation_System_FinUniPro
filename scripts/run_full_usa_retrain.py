#!/usr/bin/env python3
from __future__ import annotations

"""Run full-data USA restaurant retraining and deployment checkpoint."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _checkpoint(name: str, ok: bool, details: str):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {details}")


def _run_step(name: str, cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - start
    ok = proc.returncode == 0
    _checkpoint(name, ok, f"exit_code={proc.returncode}, elapsed_sec={elapsed:.1f}")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full retrain for USA restaurants: preprocess -> train -> checkpoint."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--input-format",
        choices=["auto", "json", "parquet"],
        default="auto",
    )
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--local-data-dir", default="local_data")
    parser.add_argument(
        "--dataset-mode",
        choices=["yelp_only", "merged", "local_only"],
        default="yelp_only",
    )
    parser.add_argument("--max-eval-users", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--n-test-items", type=int, default=1)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    py = args.python

    preprocess_cmd = [
        py,
        str(root / "scripts" / "preprocess_yelp.py"),
        "--data-dir",
        args.data_dir,
        "--input-format",
        args.input_format,
        "--out-dir",
        args.processed_dir,
        "--city",
        "",
        "--state",
        "",
        "--usa-only",
        "--max-review-chunks",
        "0",
        "--max-aux-chunks",
        "0",
        "--progress-every",
        str(max(1, args.progress_every // 4)),
    ]
    if args.quiet:
        preprocess_cmd.append("--quiet")
    train_cmd = [
        py,
        str(root / "scripts" / "train_recommender.py"),
        "--processed-dir",
        args.processed_dir,
        "--artifact-dir",
        args.artifact_dir,
        "--local-data-dir",
        args.local_data_dir,
        "--dataset-mode",
        args.dataset_mode,
        "--max-eval-users",
        str(args.max_eval_users),
        "--k",
        str(args.k),
        "--min-history",
        str(args.min_history),
        "--n-test-items",
        str(args.n_test_items),
        "--relevance-threshold",
        str(args.relevance_threshold),
        "--prior-weight",
        str(args.prior_weight),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.quiet:
        train_cmd.append("--quiet")
    checkpoint_cmd = [
        py,
        str(root / "scripts" / "checkpoint.py"),
        "--raw-data-dir",
        "raw_data",
        "--data-dir",
        args.data_dir,
        "--processed-dir",
        args.processed_dir,
        "--bundle-path",
        str(Path(args.artifact_dir) / "model_bundle.pkl"),
    ]

    for name, cmd in [
        ("Preprocess Full USA", preprocess_cmd),
        ("Train Hybrid Bundle", train_cmd),
        ("Deployment Checkpoint", checkpoint_cmd),
    ]:
        code = _run_step(name, cmd)
        if code != 0:
            raise SystemExit(code)

    print("\nFull USA retrain completed successfully.")


if __name__ == "__main__":
    main()
