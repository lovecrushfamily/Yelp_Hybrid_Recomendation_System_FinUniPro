from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.local_user_store import LocalUserStore
from src.model_bundle import ModelBundle, load_model_bundle

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))
DEFAULT_BUNDLE_PATH = "artifacts/model_bundle.pkl"
DEFAULT_LOCAL_DATA_DIR = "local_data"
ACTIVE_WINDOW_MINUTES = 20
MAX_EVENT_HISTORY = 5000

app = FastAPI(title="Yelp Hybrid Recommender API", version="2.0.0")
_retrain_lock = threading.Lock()
_state: dict[str, object] = {
    "bundle": None,
    "load_error": None,
    "sessions": {},  # auth token -> user_id
    "ux_sessions": {},  # ux token -> dict(user_id, created_at_utc, last_seen_ts)
    "user_store": None,
    "active_users": {},  # user_id -> dict(last_seen_ts, source)
    "recommendation_events": [],  # in-memory recent recommend calls
    "runtime_metrics": {"recommend_count": 0, "latency_ms_sum": 0.0},
    "global_config": {"default_k": 15, "default_algorithm": "hybrid"},
    "ux_rankings": {},  # ux token -> cached ranking payload
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
}


class RecommendRequest(BaseModel):
    """Request payload for top-K recommendation."""

    user_id: str | None = Field(None, description="Yelp/local user id")
    k: int | None = Field(None, ge=1, le=50)
    algorithm: str | None = Field(None, description="content | cf | hybrid")
    history_business_ids: list[str] | None = None
    history_ratings: list[float] | None = None
    source: str | None = Field(None, description="basic | experience | management | api")


class CheckResult(BaseModel):
    name: str
    ok: bool
    detail: str


class SignUpRequest(BaseModel):
    username: str
    password: str


class SignInRequest(BaseModel):
    username: str
    password: str


class RateRequest(BaseModel):
    business_id: str
    stars: float = Field(..., ge=1.0, le=5.0)


class UXLoginRequest(BaseModel):
    user_id: str


class AdminConfigRequest(BaseModel):
    default_k: int | None = Field(None, ge=1, le=50)
    default_algorithm: str | None = None


class AdminRetrainRequest(BaseModel):
    dataset_mode: str = "yelp_only"
    max_eval_users: int = Field(200, ge=50, le=5000)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bundle_path() -> str:
    return os.environ.get("MODEL_BUNDLE_PATH", DEFAULT_BUNDLE_PATH)


def _local_data_dir() -> str:
    return os.environ.get("LOCAL_DATA_DIR", DEFAULT_LOCAL_DATA_DIR)


def _load_bundle() -> tuple[ModelBundle | None, str | None]:
    path = _bundle_path()
    if not Path(path).exists():
        return None, f"bundle_not_found: {path}"
    try:
        bundle = load_model_bundle(path)
        return bundle, None
    except Exception as exc:  # pragma: no cover
        return None, f"bundle_load_failed: {exc}"


def _get_bundle() -> ModelBundle:
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail=str(_state.get("load_error")))
    return bundle  # type: ignore[return-value]


def _get_user_store() -> LocalUserStore:
    store = _state.get("user_store")
    if store is None:
        raise HTTPException(status_code=503, detail="local_user_store_not_ready")
    return store  # type: ignore[return-value]


def _get_sessions() -> dict[str, str]:
    sessions = _state.get("sessions")
    if sessions is None:
        sessions = {}
        _state["sessions"] = sessions
    return sessions  # type: ignore[return-value]


def _get_ux_sessions() -> dict[str, dict]:
    sessions = _state.get("ux_sessions")
    if sessions is None:
        sessions = {}
        _state["ux_sessions"] = sessions
    return sessions  # type: ignore[return-value]


def _get_active_users() -> dict[str, dict]:
    active = _state.get("active_users")
    if active is None:
        active = {}
        _state["active_users"] = active
    return active  # type: ignore[return-value]


def _get_global_config() -> dict[str, object]:
    cfg = _state.get("global_config")
    if cfg is None:
        cfg = {"default_k": 15, "default_algorithm": "hybrid"}
        _state["global_config"] = cfg
    return cfg  # type: ignore[return-value]


def _slim_business_payload(row: dict) -> dict:
    return {
        "business_id": row.get("business_id"),
        "name": row.get("name"),
        "city": row.get("city"),
        "state": row.get("state"),
        "stars": row.get("stars"),
        "review_count": row.get("review_count"),
        "categories": row.get("categories"),
    }


def _get_ux_rankings() -> dict[str, dict]:
    rankings = _state.get("ux_rankings")
    if rankings is None:
        rankings = {}
        _state["ux_rankings"] = rankings
    return rankings  # type: ignore[return-value]


def _ensure_ux_ranking(
    ux_token: str,
    user_id: str,
    force_refresh: bool = False,
) -> dict:
    rankings = _get_ux_rankings()
    cfg = _get_global_config()
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

    bundle = _get_bundle()
    history_ids = None
    history_ratings = None
    if str(user_id).startswith("local_"):
        history_ids, history_ratings = _get_user_store().get_user_history(str(user_id))

    all_business_ids = bundle.business_index["business_id"].astype(str).tolist()
    history_ids = history_ids if history_ids is not None else bundle.user_histories.get(str(user_id), ([], []))[0]
    history_ratings = history_ratings if history_ratings is not None else bundle.user_histories.get(str(user_id), ([], []))[1]
    history_set = set(history_ids or [])

    start = time.perf_counter()
    rec_tuples: list[tuple[str, float]]
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

    # Preserve ranking order while de-duplicating.
    seen_ids: set[str] = set()
    ordered_ids: list[str] = []
    for business_id, _ in rec_tuples:
        bid = str(business_id)
        if bid in seen_ids:
            continue
        seen_ids.add(bid)
        ordered_ids.append(bid)

    cols = [c for c in ["business_id", "name", "city", "state", "stars", "review_count", "categories"] if c in bundle.business_index.columns]
    biz_map: dict[str, dict] = {}
    for row in bundle.business_index[cols].itertuples(index=False):
        biz_map[str(row.business_id)] = row._asdict()
    recs = []
    for business_id in ordered_ids:
        info = biz_map.get(str(business_id), {})
        recs.append(
            {
                "business_id": str(business_id),
                "name": info.get("name"),
                "city": info.get("city"),
                "state": info.get("state"),
                "stars": info.get("stars"),
                "review_count": info.get("review_count"),
                "categories": info.get("categories"),
            }
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    _record_recommend_event(
        user_id=str(user_id),
        algorithm=target_algo,
        k=len(recs),
        latency_ms=latency_ms,
        source="ux_ranking_build",
    )

    payload = {
        "user_id": str(user_id),
        "algorithm": target_algo,
        "created_at_utc": _utc_now_iso(),
        "items": recs,
    }
    rankings[ux_token] = payload
    return payload


def _touch_user(user_id: str, source: str):
    _get_active_users()[str(user_id)] = {
        "last_seen_ts": time.time(),
        "source": source,
    }


def _active_users_snapshot(window_minutes: int = ACTIVE_WINDOW_MINUTES) -> list[dict]:
    now_ts = time.time()
    cutoff = now_ts - (window_minutes * 60)
    out: list[dict] = []
    for user_id, payload in _get_active_users().items():
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


def _trim_events():
    events = _state.get("recommendation_events")
    if isinstance(events, list) and len(events) > MAX_EVENT_HISTORY:
        del events[:-MAX_EVENT_HISTORY]


def _record_recommend_event(
    user_id: str,
    algorithm: str,
    k: int,
    latency_ms: float,
    source: str,
):
    events = _state.get("recommendation_events")
    if events is None:
        events = []
        _state["recommendation_events"] = events
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
    _trim_events()
    metrics = _state.get("runtime_metrics")
    if not isinstance(metrics, dict):
        metrics = {"recommend_count": 0, "latency_ms_sum": 0.0}
        _state["runtime_metrics"] = metrics
    metrics["recommend_count"] = int(metrics.get("recommend_count", 0)) + 1
    metrics["latency_ms_sum"] = float(metrics.get("latency_ms_sum", 0.0)) + float(latency_ms)


def _resolve_user_id(payload_user_id: str | None, token: str | None) -> str:
    if payload_user_id:
        return str(payload_user_id)
    if token:
        user_id = _get_sessions().get(token)
        if user_id:
            return user_id
    raise HTTPException(status_code=400, detail="missing_user_id_or_auth_token")


def _resolve_ux_user_id(token: str | None) -> str:
    if not token:
        raise HTTPException(status_code=401, detail="missing_ux_token")
    payload = _get_ux_sessions().get(token)
    if not payload:
        raise HTTPException(status_code=401, detail="invalid_or_expired_ux_token")
    payload["last_seen_ts"] = time.time()
    return str(payload["user_id"])


def _resolve_recommend_params(payload: RecommendRequest) -> tuple[int, str]:
    cfg = _get_global_config()
    k = int(payload.k if payload.k is not None else cfg.get("default_k", 10))
    algo = str(payload.algorithm if payload.algorithm is not None else cfg.get("default_algorithm", "hybrid")).lower().strip()
    if algo not in {"content", "cf", "hybrid", "collaborative"}:
        raise HTTPException(status_code=400, detail="invalid_algorithm")
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="invalid_k")
    return k, algo


def run_checkpoints() -> list[CheckResult]:
    """Runtime checks to quickly classify deployment as healthy/unhealthy."""
    checks: list[CheckResult] = []
    path = _bundle_path()
    checks.append(CheckResult(name="Bundle file exists", ok=Path(path).exists(), detail=path))

    bundle = _state.get("bundle")
    if bundle is None:
        checks.append(CheckResult(name="Bundle loaded", ok=False, detail=str(_state.get("load_error"))))
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


def _local_interaction_summary() -> dict:
    store = _get_user_store()
    df = store.load_interactions_df()
    if df.empty:
        return {"count": 0, "avg_stars": 0.0}
    return {
        "count": int(len(df)),
        "avg_stars": float(df["stars"].astype(float).mean()),
    }


def _series_last_hours(hours: int = 24) -> list[dict]:
    now_bucket = int(time.time() // 3600)
    start_bucket = now_bucket - max(1, hours) + 1
    buckets = list(range(start_bucket, now_bucket + 1))

    local_counter: Counter = Counter()
    local_df = _get_user_store().load_interactions_df()
    if not local_df.empty:
        for dt in local_df["date"].dropna().tolist():
            ts = pd.Timestamp(dt).timestamp()
            bucket = int(ts // 3600)
            if bucket >= start_bucket:
                local_counter[bucket] += 1

    rec_counter: Counter = Counter()
    events = _state.get("recommendation_events")
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


def _run_retrain_job(dataset_mode: str, max_eval_users: int):
    job = _state.get("retrain_job")
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
        _local_data_dir(),
        "--dataset-mode",
        dataset_mode,
        "--max-eval-users",
        str(max_eval_users),
    ]

    job["last_started_utc"] = _utc_now_iso()
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

        bundle, err = _load_bundle()
        _state["bundle"] = bundle
        _state["load_error"] = err
        if bundle is None:
            job["last_status"] = "failed"
            job["last_error"] = str(err)
        else:
            job["last_status"] = "success"
            job["last_error"] = None
    except Exception as exc:
        job["last_status"] = "failed"
        job["last_error"] = str(exc)
    finally:
        job["running"] = False
        job["last_finished_utc"] = _utc_now_iso()


@app.on_event("startup")
def on_startup():
    bundle, err = _load_bundle()
    _state["bundle"] = bundle
    _state["load_error"] = err
    _state["user_store"] = LocalUserStore(_local_data_dir())
    _state["sessions"] = {}
    _state["ux_sessions"] = {}
    _state["active_users"] = {}
    _state["recommendation_events"] = []
    _state["runtime_metrics"] = {"recommend_count": 0, "latency_ms_sum": 0.0}
    _state["global_config"] = {"default_k": 15, "default_algorithm": "hybrid"}
    _state["ux_rankings"] = {}
    _state["retrain_job"] = {
        "running": False,
        "last_status": "idle",
        "last_started_utc": None,
        "last_finished_utc": None,
        "last_error": None,
        "last_command": None,
        "last_output_tail": "",
    }
    _state["app_started_ts"] = time.time()


@app.get("/health")
def health():
    return {
        "status": "ok" if _state.get("bundle") is not None else "degraded",
        "bundle_path": _bundle_path(),
        "load_error": _state.get("load_error"),
        "started_utc": datetime.fromtimestamp(float(_state.get("app_started_ts", time.time())), tz=timezone.utc).isoformat(),
    }


@app.get("/checkpoint")
def checkpoint():
    checks = run_checkpoints()
    overall_ok = all(c.ok for c in checks)
    return {"overall_ok": overall_ok, "checks": [c.model_dump() for c in checks]}


@app.get("/users")
def users(limit: int = 30, source: str = "all"):
    bundle = _get_bundle()
    user_ids = []
    if source in ("all", "yelp"):
        user_ids.extend(list(bundle.user_histories.keys()))
    if source in ("all", "local"):
        local_users = _get_user_store().list_users()
        user_ids.extend([u["user_id"] for u in local_users])
    user_ids = user_ids[: max(1, min(limit, 1000))]
    return {"users": user_ids}


@app.get("/users/{user_id}/profile")
def user_profile(user_id: str):
    bundle = _get_bundle()
    return bundle.get_user_profile(user_id)


@app.get("/users/{user_id}/activities")
def user_activities(user_id: str, limit: int = 30):
    bundle = _get_bundle()
    activities = bundle.get_user_activities(user_id=user_id, limit=max(1, min(limit, 200)))
    return {"user_id": user_id, "activities": activities}


@app.get("/business/search")
def search_business(
    q: str = "",
    categories: str = "",
    limit: int = 20,
    name_only: bool = False,
):
    bundle = _get_bundle()
    category_list = [x.strip() for x in categories.split(",") if x.strip()]
    result = bundle.search_businesses_filtered(
        query=q,
        limit=max(1, min(limit, 100)),
        categories=category_list,
        name_only=name_only,
    )
    return {"query": q, "categories": category_list, "businesses": result}


@app.get("/business/categories")
def business_categories(limit: int = 200):
    bundle = _get_bundle()
    return {"categories": bundle.list_categories(limit=max(1, min(limit, 1000)))}


@app.post("/auth/signup")
def auth_signup(payload: SignUpRequest):
    store = _get_user_store()
    try:
        rec = store.signup(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _get_sessions()[rec.token] = rec.user_id
    _touch_user(rec.user_id, "auth_signup")
    return {"username": rec.username, "user_id": rec.user_id, "auth_token": rec.token}


@app.post("/auth/signin")
def auth_signin(payload: SignInRequest):
    store = _get_user_store()
    try:
        rec = store.signin(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    _get_sessions()[rec.token] = rec.user_id
    _touch_user(rec.user_id, "auth_signin")
    return {"username": rec.username, "user_id": rec.user_id, "auth_token": rec.token}


@app.post("/me/rate")
def rate_business(payload: RateRequest, x_auth_token: str | None = Header(default=None)):
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="missing_auth_token")
    user_id = _get_sessions().get(x_auth_token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid_or_expired_token")

    bundle = _get_bundle()
    if not bundle.has_business(payload.business_id):
        raise HTTPException(status_code=404, detail="business_not_found")

    store = _get_user_store()
    store.add_interaction(user_id=user_id, business_id=payload.business_id, stars=payload.stars)
    history_items, _ = store.get_user_history(user_id)
    _touch_user(user_id, "rate")
    return {
        "user_id": user_id,
        "business_id": payload.business_id,
        "stars": payload.stars,
        "history_size": len(history_items),
    }


@app.get("/me/history")
def me_history(x_auth_token: str | None = Header(default=None)):
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="missing_auth_token")
    user_id = _get_sessions().get(x_auth_token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid_or_expired_token")
    store = _get_user_store()
    items, ratings = store.get_user_history(user_id)
    _touch_user(user_id, "history")
    return {"user_id": user_id, "history": [{"business_id": bid, "stars": stars} for bid, stars in zip(items, ratings)]}


@app.get("/me/profile")
def me_profile(x_auth_token: str | None = Header(default=None)):
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="missing_auth_token")
    user_id = _get_sessions().get(x_auth_token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid_or_expired_token")
    bundle = _get_bundle()
    _touch_user(user_id, "profile")
    return bundle.get_user_profile(user_id)


@app.post("/recommend")
def recommend(payload: RecommendRequest, x_auth_token: str | None = Header(default=None)):
    bundle = _get_bundle()
    user_id = _resolve_user_id(payload.user_id, x_auth_token)
    k, algo = _resolve_recommend_params(payload)

    history_ids = payload.history_business_ids
    history_ratings = payload.history_ratings
    if history_ids is None and user_id.startswith("local_"):
        history_ids, history_ratings = _get_user_store().get_user_history(user_id)

    start = time.perf_counter()
    recs = bundle.recommend(
        user_id=user_id,
        k=k,
        history_business_ids=history_ids,
        history_ratings=history_ratings,
        algorithm=algo,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    source = str(payload.source or ("auth_api" if x_auth_token else "public_api"))
    _record_recommend_event(user_id=user_id, algorithm=algo, k=k, latency_ms=latency_ms, source=source)
    _touch_user(user_id, source)
    return {
        "user_id": user_id,
        "k": k,
        "algorithm": algo,
        "latency_ms": round(latency_ms, 3),
        "recommendations": recs,
    }


@app.post("/ux/login")
def ux_login(payload: UXLoginRequest):
    bundle = _get_bundle()
    user_id = str(payload.user_id).strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id_required")

    local_ids = {x["user_id"] for x in _get_user_store().list_users()}
    if user_id not in bundle.user_histories and user_id not in local_ids:
        raise HTTPException(status_code=404, detail="user_id_not_found")

    token = secrets.token_urlsafe(24)
    _get_ux_sessions()[token] = {
        "user_id": user_id,
        "created_at_utc": _utc_now_iso(),
        "last_seen_ts": time.time(),
    }
    _get_ux_rankings().pop(token, None)
    _touch_user(user_id, "ux_login")
    return {"user_id": user_id, "ux_token": token}


@app.get("/ux/me")
def ux_me(x_ux_token: str | None = Header(default=None)):
    user_id = _resolve_ux_user_id(x_ux_token)
    bundle = _get_bundle()
    _touch_user(user_id, "ux_me")
    return bundle.get_user_profile(user_id)


@app.get("/ux/recommend")
def ux_recommend(
    offset: int = 0,
    limit: int = 15,
    refresh: bool = False,
    x_ux_token: str | None = Header(default=None),
):
    token = x_ux_token
    user_id = _resolve_ux_user_id(token)
    if token is None:
        raise HTTPException(status_code=401, detail="missing_ux_token")

    ranking = _ensure_ux_ranking(token, user_id=user_id, force_refresh=bool(refresh))
    items = ranking.get("items", [])
    total = len(items)
    safe_offset = max(0, int(offset))
    safe_limit = max(1, min(int(limit), 40))
    end = min(total, safe_offset + safe_limit)
    chunk = items[safe_offset:end]
    _record_recommend_event(
        user_id=user_id,
        algorithm=str(ranking.get("algorithm", "hybrid")),
        k=len(chunk),
        latency_ms=0.0,
        source="ux_page_chunk",
    )
    _touch_user(user_id, "ux_recommend")
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


@app.get("/ux/search")
def ux_search(
    q: str = "",
    categories: str = "",
    limit: int = 20,
    x_ux_token: str | None = Header(default=None),
):
    user_id = _resolve_ux_user_id(x_ux_token)
    _touch_user(user_id, "ux_search")
    category_list = [x.strip() for x in categories.split(",") if x.strip()]
    result = search_business(
        q=q,
        categories=",".join(category_list),
        limit=max(1, min(limit, 20)),
        name_only=True,
    )
    slim = [_slim_business_payload(x) for x in result["businesses"]]
    return {"query": q, "categories": category_list, "businesses": slim}


@app.get("/ux/categories")
def ux_categories(limit: int = 250):
    data = business_categories(limit=limit)
    return data


@app.get("/admin/config")
def admin_get_config():
    return _get_global_config()


@app.post("/admin/config")
def admin_set_config(payload: AdminConfigRequest):
    cfg = _get_global_config()
    if payload.default_k is not None:
        cfg["default_k"] = int(payload.default_k)
    if payload.default_algorithm is not None:
        algo = str(payload.default_algorithm).lower().strip()
        if algo not in {"content", "cf", "hybrid", "collaborative"}:
            raise HTTPException(status_code=400, detail="invalid_default_algorithm")
        cfg["default_algorithm"] = algo
    return cfg


@app.get("/admin/stats")
def admin_stats():
    bundle = _get_bundle()
    active = _active_users_snapshot()
    local_summary = _local_interaction_summary()
    metrics = _state.get("runtime_metrics", {})
    rec_count = int(metrics.get("recommend_count", 0)) if isinstance(metrics, dict) else 0
    latency_sum = float(metrics.get("latency_ms_sum", 0.0)) if isinstance(metrics, dict) else 0.0
    avg_latency = latency_sum / rec_count if rec_count > 0 else 0.0

    event_counter: Counter = Counter()
    events = _state.get("recommendation_events")
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
        "local_user_count": len(_get_user_store().list_users()),
        "local_interactions": local_summary,
        "global_config": _get_global_config(),
        "total_yelp_users_in_bundle": len(bundle.user_histories),
        "total_businesses_in_bundle": len(bundle.business_index),
    }


@app.get("/admin/activity")
def admin_activity(hours: int = 24):
    points = _series_last_hours(hours=max(6, min(hours, 168)))
    return {"hours": max(6, min(hours, 168)), "points": points}


@app.post("/admin/reload")
def admin_reload_bundle():
    bundle, err = _load_bundle()
    _state["bundle"] = bundle
    _state["load_error"] = err
    if bundle is None:
        raise HTTPException(status_code=500, detail=str(err))
    return {"ok": True, "businesses": len(bundle.business_index), "users": len(bundle.user_histories)}


@app.post("/admin/retrain")
def admin_retrain(payload: AdminRetrainRequest):
    mode = str(payload.dataset_mode).strip().lower()
    if mode not in {"yelp_only", "merged", "local_only"}:
        raise HTTPException(status_code=400, detail="invalid_dataset_mode")

    job = _state.get("retrain_job")
    if not isinstance(job, dict):
        raise HTTPException(status_code=500, detail="retrain_state_not_initialized")

    with _retrain_lock:
        if bool(job.get("running", False)):
            raise HTTPException(status_code=409, detail="retrain_already_running")
        job["running"] = True
        t = threading.Thread(
            target=_run_retrain_job,
            args=(mode, int(payload.max_eval_users)),
            daemon=True,
        )
        t.start()

    return {"accepted": True, "dataset_mode": mode, "max_eval_users": int(payload.max_eval_users)}


@app.get("/admin/retrain/status")
def admin_retrain_status():
    return _state.get("retrain_job", {})


@app.get("/basic", response_class=HTMLResponse)
def demo_page_basic(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/experience", response_class=HTMLResponse)
def experience_login(request: Request):
    return TEMPLATES.TemplateResponse("login.html", {"request": request})


@app.get("/experience/app", response_class=HTMLResponse)
def experience_app(request: Request):
    return TEMPLATES.TemplateResponse("app.html", {"request": request})


@app.get("/management", response_class=HTMLResponse)
def management_page(request: Request):
    return TEMPLATES.TemplateResponse("management.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
def demo_page_login(request: Request):
    return TEMPLATES.TemplateResponse("login.html", {"request": request})


@app.get("/app", response_class=HTMLResponse)
def demo_page_app_main(request: Request):
    return TEMPLATES.TemplateResponse("app.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
def root_page(request: Request):
    return TEMPLATES.TemplateResponse("login.html", {"request": request})
