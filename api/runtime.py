from __future__ import annotations

import json
import os
import random
import secrets
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from fastapi import Header, HTTPException

from src.local_user_store import LocalUserStore
from src.model_bundle import ModelBundle, load_model_bundle

from .schemas import CheckResult

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_BUNDLE_PATH = "artifacts/model_bundle.pkl"
DEFAULT_LOCAL_DATA_DIR = "local_data"
DEFAULT_RAW_DATA_DIR = "raw_data"
DEFAULT_USER_PROFILE_PARQUET = "data/user_profiles.parquet"
DEFAULT_BUSINESS_MAIN_PARQUET = "data/business_main.parquet"
DEFAULT_BUSINESS_HOURS_PARQUET = "data/business_hours.parquet"
ACTIVE_WINDOW_MINUTES = 20
MAX_EVENT_HISTORY = 5000
DEFAULT_PROFILE_SAMPLE_SIZE = 160

STATE: dict[str, object] = {
    "bundle": None,
    "load_error": None,
    "sessions": {},
    "ux_sessions": {},
    "user_store": None,
    "active_users": {},
    "recommendation_events": [],
    "runtime_metrics": {"recommend_count": 0, "latency_ms_sum": 0.0},
    "global_config": {"default_k": 15, "default_algorithm": "hybrid"},
    "ux_rankings": {},
    "retrain_job": {
        "running": False,
        "last_status": "idle",
        "last_started_utc": None,
        "last_finished_utc": None,
        "last_error": None,
        "last_command": None,
        "last_output_tail": "",
    },
    "app_started_ts": time.time(),
    "business_hours": {},
    "sample_yelp_user_ids": [],
    "profile_cache": {},
    "profile_cache_stats": {"path": None, "loaded": 0, "requested": 0, "error": None},
    "business_geo": {},
    "business_geo_stats": {"path": None, "loaded": 0, "error": None},
}
RETRAIN_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bundle_path() -> str:
    return os.environ.get("MODEL_BUNDLE_PATH", DEFAULT_BUNDLE_PATH)


def local_data_dir() -> str:
    return os.environ.get("LOCAL_DATA_DIR", DEFAULT_LOCAL_DATA_DIR)


def raw_data_dir() -> str:
    return os.environ.get("RAW_DATA_DIR", DEFAULT_RAW_DATA_DIR)


def user_profile_parquet_path() -> str:
    return os.environ.get("USER_PROFILE_PARQUET_PATH", DEFAULT_USER_PROFILE_PARQUET)


def business_main_parquet_path() -> str:
    return os.environ.get("BUSINESS_MAIN_PARQUET_PATH", DEFAULT_BUSINESS_MAIN_PARQUET)


def business_hours_parquet_path() -> str:
    return os.environ.get("BUSINESS_HOURS_PARQUET_PATH", DEFAULT_BUSINESS_HOURS_PARQUET)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def profile_sample_size() -> int:
    raw = os.environ.get("USER_PROFILE_SAMPLE_SIZE", str(DEFAULT_PROFILE_SAMPLE_SIZE))
    try:
        return max(1, min(int(raw), 1000))
    except Exception:
        return DEFAULT_PROFILE_SAMPLE_SIZE


def load_business_hours_map() -> dict[str, dict]:
    parquet_path = resolve_project_path(business_hours_parquet_path())
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            day_cols = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            out: dict[str, dict] = {}
            for row in df.itertuples(index=False):
                business_id = str(getattr(row, "business_id", "") or "")
                if not business_id:
                    continue
                hours = {}
                for day in day_cols:
                    value = getattr(row, day, None)
                    if value is None or pd.isna(value):
                        continue
                    text = str(value).strip()
                    if text:
                        hours[day] = text
                if hours:
                    out[business_id] = hours
            return out
        except Exception:
            pass

    path = resolve_project_path(raw_data_dir()) / "yelp_academic_dataset_business.json"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                business_id = str(row.get("business_id", ""))
                if not business_id:
                    continue
                hours = row.get("hours")
                if isinstance(hours, dict):
                    out[business_id] = hours
    except Exception:
        return {}
    return out


def parse_hour_token(token: str) -> tuple[int, int] | None:
    token = str(token).strip()
    parts = token.split(":")
    if len(parts) != 2:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
    except Exception:
        return None
    if h < 0 or h > 24 or m < 0 or m > 59:
        return None
    if h == 24 and m != 0:
        return None
    return h, m


def is_open_now(hours_payload: object, now_local: datetime | None = None) -> tuple[bool | None, str | None]:
    if not isinstance(hours_payload, dict):
        return None, None

    now = now_local or datetime.now()
    day_name = now.strftime("%A")
    hours_range = hours_payload.get(day_name)
    if not hours_range:
        return None, None

    text = str(hours_range)
    if "-" not in text:
        return None, text
    open_token, close_token = text.split("-", 1)
    open_hm = parse_hour_token(open_token)
    close_hm = parse_hour_token(close_token)
    if open_hm is None or close_hm is None:
        return None, text

    now_minutes = now.hour * 60 + now.minute
    open_minutes = open_hm[0] * 60 + open_hm[1]
    close_minutes = close_hm[0] * 60 + close_hm[1]

    if close_minutes == open_minutes:
        return True, text
    if close_minutes > open_minutes:
        return open_minutes <= now_minutes < close_minutes, text
    return now_minutes >= open_minutes or now_minutes < close_minutes, text


def load_bundle() -> tuple[ModelBundle | None, str | None]:
    path = resolve_project_path(bundle_path())
    if not path.exists():
        return None, f"bundle_not_found: {path}"
    try:
        bundle = load_model_bundle(str(path))
        return bundle, None
    except Exception as exc:  # pragma: no cover
        return None, f"bundle_load_failed: {exc}"


def get_bundle() -> ModelBundle:
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail=str(STATE.get("load_error")))
    return bundle  # type: ignore[return-value]


def get_sample_yelp_user_ids() -> list[str]:
    value = STATE.get("sample_yelp_user_ids")
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def demo_ready_user_ids(bundle: ModelBundle | None) -> list[str]:
    if bundle is None:
        return []
    if getattr(bundle, "user_events", None):
        return [str(x) for x in bundle.user_events.keys()]
    return [str(x) for x in bundle.user_histories.keys()]


def get_profile_cache() -> dict[str, dict]:
    value = STATE.get("profile_cache")
    if isinstance(value, dict):
        return value  # type: ignore[return-value]
    value = {}
    STATE["profile_cache"] = value
    return value  # type: ignore[return-value]


def normalize_profile_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def load_profile_cache_for_users(user_ids: list[str]) -> tuple[dict[str, dict], dict[str, object]]:
    path = resolve_project_path(user_profile_parquet_path())
    requested_ids = [str(x) for x in user_ids if str(x).strip()]
    stats: dict[str, object] = {
        "path": str(path),
        "requested": len(requested_ids),
        "loaded": 0,
        "error": None,
    }
    if not requested_ids:
        return {}, stats
    if not path.exists():
        stats["error"] = f"profile_parquet_not_found: {path}"
        return {}, stats

    try:
        dataset = ds.dataset(path, format="parquet")
        columns = [field.name for field in dataset.schema if field.name != "friends"]
        table = dataset.to_table(
            columns=columns,
            filter=ds.field("user_id").isin(requested_ids),
        )
        df = table.to_pandas()
    except Exception as exc:
        stats["error"] = f"profile_parquet_load_failed: {exc}"
        return {}, stats

    out: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        payload = {col: normalize_profile_value(getattr(row, col)) for col in df.columns}
        user_id = str(payload.get("user_id", "") or "")
        if not user_id:
            continue
        payload["review_count_profile"] = payload.get("review_count")
        payload["average_stars_profile"] = payload.get("average_stars")
        payload["useful_votes"] = payload.get("useful")
        payload["funny_votes"] = payload.get("funny")
        payload["cool_votes"] = payload.get("cool")
        out[user_id] = payload

    stats["loaded"] = len(out)
    return out, stats


def load_business_geo_map() -> tuple[dict[str, dict], dict[str, object]]:
    path = resolve_project_path(business_main_parquet_path())
    stats: dict[str, object] = {"path": str(path), "loaded": 0, "error": None}
    if not path.exists():
        stats["error"] = f"business_main_parquet_not_found: {path}"
        return {}, stats

    try:
        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table(columns=["business_id", "latitude", "longitude", "address"])
        df = table.to_pandas()
    except Exception as exc:
        stats["error"] = f"business_main_parquet_load_failed: {exc}"
        return {}, stats

    out: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        bid = str(row.business_id)
        lat = row.latitude
        lon = row.longitude
        if pd.isna(lat) or pd.isna(lon):
            continue
        out[bid] = {
            "latitude": float(lat),
            "longitude": float(lon),
            "address": None if pd.isna(row.address) else str(row.address),
        }

    stats["loaded"] = len(out)
    return out, stats


def get_business_geo() -> dict[str, dict]:
    value = STATE.get("business_geo")
    if isinstance(value, dict):
        return value  # type: ignore[return-value]
    value = {}
    STATE["business_geo"] = value
    return value  # type: ignore[return-value]


def apply_geo(rec: dict) -> None:
    if rec.get("latitude") is not None and rec.get("longitude") is not None:
        return
    payload = get_business_geo().get(str(rec.get("business_id", "")))
    if not payload:
        return
    rec["latitude"] = payload.get("latitude")
    rec["longitude"] = payload.get("longitude")
    if rec.get("address") is None:
        rec["address"] = payload.get("address")


def compose_user_profile(user_id: str) -> dict:
    bundle = get_bundle()
    profile = bundle.get_user_profile(user_id)
    cached = get_profile_cache().get(str(user_id))
    if cached:
        merged = dict(cached)
        merged.update(profile)
        for key in [
            "name",
            "yelping_since",
            "review_count_profile",
            "average_stars_profile",
            "fans",
            "useful_votes",
            "funny_votes",
            "cool_votes",
            "elite_year_count",
            "friends_count",
            "compliment_hot",
            "compliment_more",
            "compliment_profile",
            "compliment_cute",
            "compliment_list",
            "compliment_note",
            "compliment_plain",
            "compliment_cool",
            "compliment_funny",
            "compliment_writer",
            "compliment_photos",
        ]:
            if key in cached:
                merged[key] = cached[key]
        profile = merged

    elite_year_count = int(profile.get("elite_year_count") or 0) if profile.get("elite_year_count") is not None else 0
    elite_text = str(profile.get("elite") or "").strip()
    is_elite = elite_year_count > 0 or bool(elite_text)
    profile["is_elite_user"] = bool(is_elite)
    profile["user_tier"] = "elite" if is_elite else "regular"
    return profile


def get_user_store() -> LocalUserStore:
    store = STATE.get("user_store")
    if store is None:
        raise HTTPException(status_code=503, detail="local_user_store_not_ready")
    return store  # type: ignore[return-value]


def get_sessions() -> dict[str, str]:
    sessions = STATE.get("sessions")
    if sessions is None:
        sessions = {}
        STATE["sessions"] = sessions
    return sessions  # type: ignore[return-value]


def get_ux_sessions() -> dict[str, dict]:
    sessions = STATE.get("ux_sessions")
    if sessions is None:
        sessions = {}
        STATE["ux_sessions"] = sessions
    return sessions  # type: ignore[return-value]


def get_active_users() -> dict[str, dict]:
    active = STATE.get("active_users")
    if active is None:
        active = {}
        STATE["active_users"] = active
    return active  # type: ignore[return-value]


def get_global_config() -> dict[str, object]:
    cfg = STATE.get("global_config")
    if cfg is None:
        cfg = {"default_k": 15, "default_algorithm": "hybrid"}
        STATE["global_config"] = cfg
    return cfg  # type: ignore[return-value]


def slim_business_payload(row: dict) -> dict:
    return {
        "business_id": row.get("business_id"),
        "name": row.get("name"),
        "city": row.get("city"),
        "state": row.get("state"),
        "stars": row.get("stars"),
        "review_count": row.get("review_count"),
        "categories": row.get("categories"),
    }


def get_ux_rankings() -> dict[str, dict]:
    rankings = STATE.get("ux_rankings")
    if rankings is None:
        rankings = {}
        STATE["ux_rankings"] = rankings
    return rankings  # type: ignore[return-value]


def ensure_ux_ranking(
    ux_token: str,
    user_id: str,
    force_refresh: bool = False,
) -> dict:
    rankings = get_ux_rankings()
    cfg = get_global_config()
    target_algo = str(cfg.get("default_algorithm", "hybrid")).lower().strip()
    payload = rankings.get(ux_token)
    should_rebuild = (
        force_refresh
        or payload is None
        or str(payload.get("user_id")) != str(user_id)
        or str(payload.get("algorithm")) != target_algo
    )
    if not should_rebuild:
        return payload

    bundle = get_bundle()
    history_ids = None
    history_ratings = None
    if str(user_id).startswith("local_"):
        history_ids, history_ratings = get_user_store().get_user_history(str(user_id))

    all_business_ids = bundle.business_index["business_id"].astype(str).tolist()
    if history_ids is None:
        history_ids, history_ratings = bundle.user_histories.get(str(user_id), ([], []))
    history_set = set(history_ids or [])

    start = time.perf_counter()
    if target_algo == "content":
        if history_ids:
            rec_tuples = bundle.content_engine.recommend_from_history(
                history_business_ids=history_ids,
                candidate_business_ids=all_business_ids,
                k=len(all_business_ids),
                exclude_history=True,
            )
        else:
            rec_tuples = bundle._fallback_popularity(k=len(all_business_ids), exclude_ids=history_set)
    elif target_algo in {"cf", "collaborative"}:
        rec_tuples = bundle.cf_engine.recommend_for_user(
            user_id=str(user_id),
            k=len(all_business_ids),
            candidate_business_ids=all_business_ids,
            exclude_seen=True,
        )
        if len(rec_tuples) < len(all_business_ids):
            known = {x[0] for x in rec_tuples}
            missing = [x for x in all_business_ids if x not in known and x not in history_set]
            if missing:
                rec_tuples.extend(bundle._fallback_popularity(k=len(missing), exclude_ids=history_set.union(known)))
    else:
        rec_tuples = bundle.hybrid_recommender.recommend(
            user_id=str(user_id),
            k=len(all_business_ids),
            user_history_business_ids=history_ids,
            user_history_ratings=history_ratings,
            candidate_business_ids=all_business_ids,
            exclude_history=True,
        )
        if len(rec_tuples) < len(all_business_ids):
            known = {x[0] for x in rec_tuples}
            missing = [x for x in all_business_ids if x not in known and x not in history_set]
            if missing:
                rec_tuples.extend(bundle._fallback_popularity(k=len(missing), exclude_ids=history_set.union(known)))

    seen_ids: set[str] = set()
    ordered_ids: list[str] = []
    for business_id, _ in rec_tuples:
        bid = str(business_id)
        if bid in seen_ids:
            continue
        seen_ids.add(bid)
        ordered_ids.append(bid)

    cols = [
        c
        for c in [
            "business_id",
            "name",
            "city",
            "state",
            "stars",
            "review_count",
            "categories",
            "address",
            "is_open",
            "latitude",
            "longitude",
        ]
        if c in bundle.business_index.columns
    ]
    biz_map: dict[str, dict] = {}
    for row in bundle.business_index[cols].itertuples(index=False):
        biz_map[str(row.business_id)] = row._asdict()
    recs = []
    for business_id in ordered_ids:
        info = biz_map.get(str(business_id), {})
        row = {
            "business_id": str(business_id),
            "name": info.get("name"),
            "city": info.get("city"),
            "state": info.get("state"),
            "stars": info.get("stars"),
            "review_count": info.get("review_count"),
            "categories": info.get("categories"),
            "address": info.get("address"),
            "is_open": info.get("is_open"),
            "latitude": info.get("latitude"),
            "longitude": info.get("longitude"),
        }
        apply_geo(row)
        recs.append(row)

    latency_ms = (time.perf_counter() - start) * 1000.0
    record_recommend_event(
        user_id=str(user_id),
        algorithm=target_algo,
        k=len(recs),
        latency_ms=latency_ms,
        source="ux_ranking_build",
    )

    payload = {
        "user_id": str(user_id),
        "algorithm": target_algo,
        "created_at_utc": utc_now_iso(),
        "items": recs,
    }
    rankings[ux_token] = payload
    return payload


def touch_user(user_id: str, source: str):
    get_active_users()[str(user_id)] = {
        "last_seen_ts": time.time(),
        "source": source,
    }


def active_users_snapshot(window_minutes: int = ACTIVE_WINDOW_MINUTES) -> list[dict]:
    now_ts = time.time()
    cutoff = now_ts - (window_minutes * 60)
    out: list[dict] = []
    for user_id, payload in get_active_users().items():
        last_seen_ts = float(payload.get("last_seen_ts", 0.0))
        if last_seen_ts < cutoff:
            continue
        out.append(
            {
                "user_id": user_id,
                "source": payload.get("source", "unknown"),
                "last_seen_utc": datetime.fromtimestamp(last_seen_ts, tz=timezone.utc).isoformat(),
            }
        )
    out.sort(key=lambda x: x["last_seen_utc"], reverse=True)
    return out


def trim_events():
    events = STATE.get("recommendation_events")
    if isinstance(events, list) and len(events) > MAX_EVENT_HISTORY:
        del events[:-MAX_EVENT_HISTORY]


def record_recommend_event(
    user_id: str,
    algorithm: str,
    k: int,
    latency_ms: float,
    source: str,
):
    events = STATE.get("recommendation_events")
    if events is None:
        events = []
        STATE["recommendation_events"] = events
    if isinstance(events, list):
        events.append(
            {
                "ts": time.time(),
                "user_id": str(user_id),
                "algorithm": str(algorithm),
                "k": int(k),
                "latency_ms": float(latency_ms),
                "source": str(source),
            }
        )
    trim_events()
    metrics = STATE.get("runtime_metrics")
    if not isinstance(metrics, dict):
        metrics = {"recommend_count": 0, "latency_ms_sum": 0.0}
        STATE["runtime_metrics"] = metrics
    metrics["recommend_count"] = int(metrics.get("recommend_count", 0)) + 1
    metrics["latency_ms_sum"] = float(metrics.get("latency_ms_sum", 0.0)) + float(latency_ms)


def resolve_user_id(payload_user_id: str | None, token: str | None) -> str:
    if payload_user_id:
        return str(payload_user_id)
    if token:
        user_id = get_sessions().get(token)
        if user_id:
            return user_id
    raise HTTPException(status_code=400, detail="missing_user_id_or_auth_token")


def resolve_ux_user_id(token: str | None) -> str:
    if not token:
        raise HTTPException(status_code=401, detail="missing_ux_token")
    payload = get_ux_sessions().get(token)
    if not payload:
        raise HTTPException(status_code=401, detail="invalid_or_expired_ux_token")
    payload["last_seen_ts"] = time.time()
    return str(payload["user_id"])


def resolve_recommend_params(k: int | None, algorithm: str | None) -> tuple[int, str]:
    cfg = get_global_config()
    resolved_k = int(k if k is not None else cfg.get("default_k", 10))
    algo = str(algorithm if algorithm is not None else cfg.get("default_algorithm", "hybrid")).lower().strip()
    if algo not in {"content", "cf", "hybrid", "collaborative"}:
        raise HTTPException(status_code=400, detail="invalid_algorithm")
    if resolved_k < 1 or resolved_k > 50:
        raise HTTPException(status_code=400, detail="invalid_k")
    return resolved_k, algo


def run_checkpoints() -> list[CheckResult]:
    checks: list[CheckResult] = []
    path = resolve_project_path(bundle_path())
    checks.append(CheckResult(name="Bundle file exists", ok=path.exists(), detail=str(path)))

    bundle = STATE.get("bundle")
    if bundle is None:
        checks.append(CheckResult(name="Bundle loaded", ok=False, detail=str(STATE.get("load_error"))))
        return checks

    bundle = bundle  # type: ignore[assignment]
    checks.append(CheckResult(name="Bundle loaded", ok=True, detail="ok"))
    checks.append(CheckResult(name="Business index", ok=len(bundle.business_index) > 0, detail=f"n_businesses={len(bundle.business_index)}"))
    checks.append(CheckResult(name="User histories", ok=len(bundle.user_histories) > 0, detail=f"n_users={len(bundle.user_histories)}"))

    sample_user = next(iter(bundle.user_histories.keys()), None)
    if sample_user is None:
        checks.append(CheckResult(name="Sample inference", ok=False, detail="no user history"))
    else:
        recs = bundle.recommend(sample_user, k=5)
        checks.append(CheckResult(name="Sample inference", ok=len(recs) > 0, detail=f"user_id={sample_user}, n_recs={len(recs)}"))
    return checks


def local_interaction_summary() -> dict:
    df = get_user_store().load_interactions_df()
    if df.empty:
        return {"count": 0, "avg_stars": 0.0}
    return {
        "count": int(len(df)),
        "avg_stars": float(df["stars"].astype(float).mean()),
    }


def series_last_hours(hours: int = 24) -> list[dict]:
    now_bucket = int(time.time() // 3600)
    start_bucket = now_bucket - max(1, hours) + 1
    buckets = list(range(start_bucket, now_bucket + 1))

    local_counter: Counter = Counter()
    local_df = get_user_store().load_interactions_df()
    if not local_df.empty:
        for dt in local_df["date"].dropna().tolist():
            ts = pd.Timestamp(dt).timestamp()
            bucket = int(ts // 3600)
            if bucket >= start_bucket:
                local_counter[bucket] += 1

    rec_counter: Counter = Counter()
    events = STATE.get("recommendation_events")
    if isinstance(events, list):
        for event in events:
            bucket = int(float(event.get("ts", 0.0)) // 3600)
            if bucket >= start_bucket:
                rec_counter[bucket] += 1

    out: list[dict] = []
    for bucket in buckets:
        dt = datetime.fromtimestamp(bucket * 3600, tz=timezone.utc)
        out.append(
            {
                "label": dt.strftime("%m-%d %H:00"),
                "local_interactions": int(local_counter.get(bucket, 0)),
                "recommend_calls": int(rec_counter.get(bucket, 0)),
            }
        )
    return out


def run_retrain_job(dataset_mode: str, max_eval_users: int):
    job = STATE.get("retrain_job")
    if not isinstance(job, dict):
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_recommender.py"),
        "--processed-dir",
        "processed",
        "--artifact-dir",
        "artifacts",
        "--local-data-dir",
        local_data_dir(),
        "--dataset-mode",
        dataset_mode,
        "--max-eval-users",
        str(max_eval_users),
    ]

    job["last_started_utc"] = utc_now_iso()
    job["last_finished_utc"] = None
    job["last_status"] = "running"
    job["last_error"] = None
    job["last_command"] = " ".join(cmd)
    job["last_output_tail"] = ""

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        job["last_output_tail"] = output[-6000:]
        if proc.returncode != 0:
            job["last_status"] = "failed"
            job["last_error"] = f"train_exit_code_{proc.returncode}"
            return

        bundle, err = load_bundle()
        STATE["bundle"] = bundle
        STATE["load_error"] = err
        if bundle is None:
            job["last_status"] = "failed"
            job["last_error"] = str(err)
        else:
            geo_map, geo_stats = load_business_geo_map()
            STATE["business_geo"] = geo_map
            STATE["business_geo_stats"] = geo_stats
            sample_user_ids = demo_ready_user_ids(bundle)[: profile_sample_size()]
            STATE["sample_yelp_user_ids"] = sample_user_ids
            cache, cache_stats = load_profile_cache_for_users(sample_user_ids)
            STATE["profile_cache"] = cache
            STATE["profile_cache_stats"] = cache_stats
            job["last_status"] = "success"
            job["last_error"] = None
    except Exception as exc:
        job["last_status"] = "failed"
        job["last_error"] = str(exc)
    finally:
        job["running"] = False
        job["last_finished_utc"] = utc_now_iso()


def startup():
    bundle, err = load_bundle()
    STATE["bundle"] = bundle
    STATE["load_error"] = err
    STATE["user_store"] = LocalUserStore(local_data_dir())
    STATE["sessions"] = {}
    STATE["ux_sessions"] = {}
    STATE["active_users"] = {}
    STATE["recommendation_events"] = []
    STATE["runtime_metrics"] = {"recommend_count": 0, "latency_ms_sum": 0.0}
    STATE["global_config"] = {"default_k": 15, "default_algorithm": "hybrid"}
    STATE["ux_rankings"] = {}
    STATE["retrain_job"] = {
        "running": False,
        "last_status": "idle",
        "last_started_utc": None,
        "last_finished_utc": None,
        "last_error": None,
        "last_command": None,
        "last_output_tail": "",
    }
    STATE["app_started_ts"] = time.time()
    STATE["business_hours"] = load_business_hours_map()
    geo_map, geo_stats = load_business_geo_map()
    STATE["business_geo"] = geo_map
    STATE["business_geo_stats"] = geo_stats
    sample_user_ids: list[str] = []
    if bundle is not None:
        sample_user_ids = demo_ready_user_ids(bundle)[: profile_sample_size()]
    STATE["sample_yelp_user_ids"] = sample_user_ids
    cache, cache_stats = load_profile_cache_for_users(sample_user_ids)
    STATE["profile_cache"] = cache
    STATE["profile_cache_stats"] = cache_stats


def health_payload() -> dict:
    return {
        "status": "ok" if STATE.get("bundle") is not None else "degraded",
        "bundle_path": str(resolve_project_path(bundle_path())),
        "load_error": STATE.get("load_error"),
        "profile_cache": STATE.get("profile_cache_stats"),
        "business_geo": STATE.get("business_geo_stats"),
        "started_utc": datetime.fromtimestamp(float(STATE.get("app_started_ts", time.time())), tz=timezone.utc).isoformat(),
    }


def checkpoint_payload() -> dict:
    checks = run_checkpoints()
    overall_ok = all(c.ok for c in checks)
    return {"overall_ok": overall_ok, "checks": [c.model_dump() for c in checks]}


def users_payload(limit: int = 30, source: str = "all") -> dict:
    bundle = get_bundle()
    safe_limit = max(1, min(limit, 1000))
    user_ids: list[str] = []
    records: list[dict] = []

    if source in ("all", "yelp"):
        all_yelp_ids = demo_ready_user_ids(bundle)
        if all_yelp_ids:
            sample_size = min(safe_limit, len(all_yelp_ids))
            sampled_ids = random.sample(all_yelp_ids, sample_size)
            cache = get_profile_cache()
            missing_ids = [user_id for user_id in sampled_ids if user_id not in cache]
            if missing_ids:
                loaded, _ = load_profile_cache_for_users(missing_ids)
                cache.update(loaded)

            for user_id in sampled_ids:
                profile = compose_user_profile(user_id)
                name = str(profile.get("name") or "User").strip() or "User"
                short_id = str(user_id)[:6]
                is_elite = bool(profile.get("is_elite_user"))
                label = f"{name} - {short_id}"
                if is_elite:
                    label = f"Elite | {label}"
                records.append(
                    {
                        "user_id": str(user_id),
                        "name": name,
                        "short_id": short_id,
                        "label": label,
                        "is_elite_user": is_elite,
                        "user_tier": profile.get("user_tier", "regular"),
                    }
                )
                user_ids.append(str(user_id))

    if source in ("all", "local") and len(records) < safe_limit:
        local_users = get_user_store().list_users()
        remaining = max(0, safe_limit - len(records))
        for payload in local_users[:remaining]:
            user_id = str(payload["user_id"])
            name = str(payload.get("username") or "Local").strip() or "Local"
            records.append(
                {
                    "user_id": user_id,
                    "name": name,
                    "short_id": user_id[:6],
                    "label": f"{name} - {user_id[:6]}",
                    "is_elite_user": False,
                    "user_tier": "regular",
                }
            )
            user_ids.append(user_id)

    return {"users": user_ids[:safe_limit], "user_records": records[:safe_limit]}


def user_profile_payload(user_id: str) -> dict:
    return compose_user_profile(user_id)


def user_activities_payload(user_id: str, limit: int = 30) -> dict:
    activities = get_bundle().get_user_activities(user_id=user_id, limit=max(1, min(limit, 200)))
    for row in activities:
        apply_geo(row)
    return {"user_id": user_id, "activities": activities}


def search_business_payload(
    q: str = "",
    categories: str = "",
    limit: int = 20,
    name_only: bool = False,
) -> dict:
    bundle = get_bundle()
    category_list = [x.strip() for x in categories.split(",") if x.strip()]
    result = bundle.search_businesses_filtered(
        query=q,
        limit=max(1, min(limit, 100)),
        categories=category_list,
        name_only=name_only,
    )
    return {"query": q, "categories": category_list, "businesses": result}


def business_categories_payload(limit: int = 200) -> dict:
    bundle = get_bundle()
    return {"categories": bundle.list_categories(limit=max(1, min(limit, 1000)))}


def signup_payload(username: str, password: str) -> dict:
    try:
        rec = get_user_store().signup(username, password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    get_sessions()[rec.token] = rec.user_id
    touch_user(rec.user_id, "auth_signup")
    return {"username": rec.username, "user_id": rec.user_id, "auth_token": rec.token}


def signin_payload(username: str, password: str) -> dict:
    try:
        rec = get_user_store().signin(username, password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    get_sessions()[rec.token] = rec.user_id
    touch_user(rec.user_id, "auth_signin")
    return {"username": rec.username, "user_id": rec.user_id, "auth_token": rec.token}


def rate_payload(business_id: str, stars: float, x_auth_token: str | None = None) -> dict:
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="missing_auth_token")
    user_id = get_sessions().get(x_auth_token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid_or_expired_token")

    bundle = get_bundle()
    if not bundle.has_business(business_id):
        raise HTTPException(status_code=404, detail="business_not_found")

    store = get_user_store()
    store.add_interaction(user_id=user_id, business_id=business_id, stars=stars)
    history_items, _ = store.get_user_history(user_id)
    touch_user(user_id, "rate")
    return {
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        "history_size": len(history_items),
    }


def me_history_payload(x_auth_token: str | None = None) -> dict:
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="missing_auth_token")
    user_id = get_sessions().get(x_auth_token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid_or_expired_token")
    items, ratings = get_user_store().get_user_history(user_id)
    touch_user(user_id, "history")
    return {"user_id": user_id, "history": [{"business_id": bid, "stars": stars} for bid, stars in zip(items, ratings)]}


def me_profile_payload(x_auth_token: str | None = Header(default=None)) -> dict:
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="missing_auth_token")
    user_id = get_sessions().get(x_auth_token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid_or_expired_token")
    touch_user(user_id, "profile")
    return compose_user_profile(user_id)


def recommend_payload(
    user_id: str | None,
    k: int | None,
    algorithm: str | None,
    history_business_ids: list[str] | None,
    history_ratings: list[float] | None,
    source: str | None,
    x_auth_token: str | None = None,
) -> dict:
    bundle = get_bundle()
    resolved_user_id = resolve_user_id(user_id, x_auth_token)
    resolved_k, algo = resolve_recommend_params(k, algorithm)

    history_ids = history_business_ids
    rating_history = history_ratings
    if history_ids is None and resolved_user_id.startswith("local_"):
        history_ids, rating_history = get_user_store().get_user_history(resolved_user_id)

    start = time.perf_counter()
    recs = bundle.recommend(
        user_id=resolved_user_id,
        k=resolved_k,
        history_business_ids=history_ids,
        history_ratings=rating_history,
        algorithm=algo,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    event_source = str(source or ("auth_api" if x_auth_token else "public_api"))
    record_recommend_event(user_id=resolved_user_id, algorithm=algo, k=resolved_k, latency_ms=latency_ms, source=event_source)
    touch_user(resolved_user_id, event_source)

    hours_map = STATE.get("business_hours", {})
    now_local = datetime.now()
    if isinstance(hours_map, dict):
        for rec in recs:
            business_id = str(rec.get("business_id", ""))
            open_now, today_hours = is_open_now(hours_map.get(business_id), now_local=now_local)
            rec["open_now"] = open_now
            rec["today_hours"] = today_hours
            apply_geo(rec)
    else:
        for rec in recs:
            rec["open_now"] = None
            rec["today_hours"] = None
            apply_geo(rec)

    return {
        "user_id": resolved_user_id,
        "k": resolved_k,
        "algorithm": algo,
        "latency_ms": round(latency_ms, 3),
        "recommendations": recs,
    }


def ux_login_payload(user_id: str) -> dict:
    bundle = get_bundle()
    resolved_user_id = str(user_id).strip()
    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="user_id_required")

    local_ids = {x["user_id"] for x in get_user_store().list_users()}
    if resolved_user_id not in bundle.user_histories and resolved_user_id not in local_ids:
        raise HTTPException(status_code=404, detail="user_id_not_found")

    token = secrets.token_urlsafe(24)
    get_ux_sessions()[token] = {
        "user_id": resolved_user_id,
        "created_at_utc": utc_now_iso(),
        "last_seen_ts": time.time(),
    }
    get_ux_rankings().pop(token, None)
    touch_user(resolved_user_id, "ux_login")
    return {"user_id": resolved_user_id, "ux_token": token}


def ux_me_payload(x_ux_token: str | None = None) -> dict:
    user_id = resolve_ux_user_id(x_ux_token)
    touch_user(user_id, "ux_me")
    return compose_user_profile(user_id)


def ux_recommend_payload(
    offset: int = 0,
    limit: int = 15,
    refresh: bool = False,
    x_ux_token: str | None = None,
) -> dict:
    user_id = resolve_ux_user_id(x_ux_token)
    if x_ux_token is None:
        raise HTTPException(status_code=401, detail="missing_ux_token")

    ranking = ensure_ux_ranking(x_ux_token, user_id=user_id, force_refresh=bool(refresh))
    items = ranking.get("items", [])
    total = len(items)
    safe_offset = max(0, int(offset))
    safe_limit = max(1, min(int(limit), 40))
    end = min(total, safe_offset + safe_limit)
    chunk = items[safe_offset:end]
    record_recommend_event(
        user_id=user_id,
        algorithm=str(ranking.get("algorithm", "hybrid")),
        k=len(chunk),
        latency_ms=0.0,
        source="ux_page_chunk",
    )
    touch_user(user_id, "ux_recommend")
    return {
        "user_id": user_id,
        "algorithm": ranking.get("algorithm"),
        "offset": safe_offset,
        "limit": safe_limit,
        "next_offset": end,
        "has_more": end < total,
        "total": total,
        "recommendations": chunk,
    }


def ux_search_payload(
    q: str = "",
    categories: str = "",
    limit: int = 20,
    x_ux_token: str | None = None,
) -> dict:
    user_id = resolve_ux_user_id(x_ux_token)
    touch_user(user_id, "ux_search")
    category_list = [x.strip() for x in categories.split(",") if x.strip()]
    result = search_business_payload(
        q=q,
        categories=",".join(category_list),
        limit=max(1, min(limit, 20)),
        name_only=True,
    )
    slim = [slim_business_payload(x) for x in result["businesses"]]
    return {"query": q, "categories": category_list, "businesses": slim}


def ux_categories_payload(limit: int = 250) -> dict:
    return business_categories_payload(limit=limit)


def admin_get_config_payload() -> dict[str, object]:
    return get_global_config()


def admin_set_config_payload(default_k: int | None, default_algorithm: str | None) -> dict[str, object]:
    cfg = get_global_config()
    if default_k is not None:
        cfg["default_k"] = int(default_k)
    if default_algorithm is not None:
        algo = str(default_algorithm).lower().strip()
        if algo not in {"content", "cf", "hybrid", "collaborative"}:
            raise HTTPException(status_code=400, detail="invalid_default_algorithm")
        cfg["default_algorithm"] = algo
    return cfg


def admin_stats_payload() -> dict:
    bundle = get_bundle()
    active = active_users_snapshot()
    local_summary = local_interaction_summary()
    metrics = STATE.get("runtime_metrics", {})
    rec_count = int(metrics.get("recommend_count", 0)) if isinstance(metrics, dict) else 0
    latency_sum = float(metrics.get("latency_ms_sum", 0.0)) if isinstance(metrics, dict) else 0.0
    avg_latency = latency_sum / rec_count if rec_count > 0 else 0.0

    event_counter: Counter = Counter()
    events = STATE.get("recommendation_events")
    if isinstance(events, list):
        cutoff = time.time() - 86400
        for event in events:
            if float(event.get("ts", 0.0)) >= cutoff:
                event_counter[str(event.get("algorithm", "unknown"))] += 1

    return {
        "active_users_window_minutes": ACTIVE_WINDOW_MINUTES,
        "active_users_count": len(active),
        "active_users_preview": active[:20],
        "model_metrics": bundle.metrics,
        "runtime_metrics": {
            "recommend_requests": rec_count,
            "average_recommend_latency_ms": round(avg_latency, 3),
        },
        "algorithm_usage_24h": dict(event_counter),
        "local_user_count": len(get_user_store().list_users()),
        "local_interactions": local_summary,
        "global_config": get_global_config(),
        "total_yelp_users_in_bundle": len(bundle.user_histories),
        "total_businesses_in_bundle": len(bundle.business_index),
        "profile_cache": STATE.get("profile_cache_stats"),
        "business_geo": STATE.get("business_geo_stats"),
    }


def admin_activity_payload(hours: int = 24) -> dict:
    safe_hours = max(6, min(hours, 168))
    return {"hours": safe_hours, "points": series_last_hours(hours=safe_hours)}


def admin_reload_bundle_payload() -> dict:
    bundle, err = load_bundle()
    STATE["bundle"] = bundle
    STATE["load_error"] = err
    if bundle is None:
        raise HTTPException(status_code=500, detail=str(err))
    geo_map, geo_stats = load_business_geo_map()
    STATE["business_geo"] = geo_map
    STATE["business_geo_stats"] = geo_stats
    sample_user_ids = demo_ready_user_ids(bundle)[: profile_sample_size()]
    STATE["sample_yelp_user_ids"] = sample_user_ids
    cache, cache_stats = load_profile_cache_for_users(sample_user_ids)
    STATE["profile_cache"] = cache
    STATE["profile_cache_stats"] = cache_stats
    return {"ok": True, "businesses": len(bundle.business_index), "users": len(bundle.user_histories)}


def admin_retrain_payload(dataset_mode: str, max_eval_users: int) -> dict:
    mode = str(dataset_mode).strip().lower()
    if mode not in {"yelp_only", "merged", "local_only"}:
        raise HTTPException(status_code=400, detail="invalid_dataset_mode")

    job = STATE.get("retrain_job")
    if not isinstance(job, dict):
        raise HTTPException(status_code=500, detail="retrain_state_not_initialized")

    with RETRAIN_LOCK:
        if bool(job.get("running", False)):
            raise HTTPException(status_code=409, detail="retrain_already_running")
        job["running"] = True
        t = threading.Thread(
            target=run_retrain_job,
            args=(mode, int(max_eval_users)),
            daemon=True,
        )
        t.start()
    return {"ok": True, "status": "started", "dataset_mode": mode, "max_eval_users": int(max_eval_users)}


def admin_retrain_status_payload() -> dict:
    return STATE.get("retrain_job", {})
