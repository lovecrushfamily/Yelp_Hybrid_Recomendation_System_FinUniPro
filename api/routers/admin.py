from __future__ import annotations

from fastapi import APIRouter

from .. import runtime
from ..schemas import AdminConfigRequest, AdminRetrainRequest

router = APIRouter()


@router.get("/admin/config")
def admin_get_config():
    return runtime.admin_get_config_payload()


@router.post("/admin/config")
def admin_set_config(payload: AdminConfigRequest):
    return runtime.admin_set_config_payload(
        default_k=payload.default_k,
        default_algorithm=payload.default_algorithm,
    )


@router.get("/admin/stats")
def admin_stats():
    return runtime.admin_stats_payload()


@router.get("/admin/activity")
def admin_activity(hours: int = 24):
    return runtime.admin_activity_payload(hours=hours)


@router.post("/admin/reload")
def admin_reload_bundle():
    return runtime.admin_reload_bundle_payload()


@router.post("/admin/retrain")
def admin_retrain(payload: AdminRetrainRequest):
    return runtime.admin_retrain_payload(
        dataset_mode=payload.dataset_mode,
        max_eval_users=payload.max_eval_users,
    )


@router.get("/admin/retrain/status")
def admin_retrain_status():
    return runtime.admin_retrain_status_payload()
