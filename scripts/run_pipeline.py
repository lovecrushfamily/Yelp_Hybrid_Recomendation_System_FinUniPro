#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full recommendation pipeline: preprocess -> train -> checkpoint."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--city", default="Philadelphia")
    parser.add_argument("--state", default="PA")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--local-data-dir", default="local_data")
    parser.add_argument(
        "--dataset-mode",
        choices=["yelp_only", "merged", "local_only"],
        default="yelp_only",
    )
    parser.add_argument("--full-data", action="store_true", help="Use full Yelp review/checkin/tip files.")
    parser.add_argument("--max-review-chunks", type=int, default=8)
    parser.add_argument("--max-aux-chunks", type=int, default=8)
    parser.add_argument("--max-eval-users", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = args.python

    max_review_chunks = "0" if args.full_data else str(args.max_review_chunks)
    max_aux_chunks = "0" if args.full_data else str(args.max_aux_chunks)

    preprocess_cmd = [
        python,
        str(project_root / "scripts" / "preprocess_yelp.py"),
        "--out-dir", args.processed_dir,
        "--city", args.city,
        "--state", args.state,
        "--max-review-chunks", max_review_chunks,
        "--max-aux-chunks", max_aux_chunks,
    ]
    train_cmd = [
        python,
        str(project_root / "scripts" / "train_recommender.py"),
        "--processed-dir", args.processed_dir,
        "--artifact-dir", args.artifact_dir,
        "--local-data-dir", args.local_data_dir,
        "--dataset-mode", args.dataset_mode,
        "--max-eval-users", str(args.max_eval_users),
    ]
    checkpoint_cmd = [
        python,
        str(project_root / "scripts" / "checkpoint.py"),
        "--raw-data-dir", "raw_data",
        "--processed-dir", args.processed_dir,
        "--bundle-path", str(Path(args.artifact_dir) / "model_bundle.pkl"),
    ]

    for cmd in [preprocess_cmd, train_cmd, checkpoint_cmd]:
        code = run_cmd(cmd)
        if code != 0:
            raise SystemExit(code)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
