from __future__ import annotations

from fastapi import APIRouter, Header

from .. import runtime
from ..schemas import RecommendRequest

router = APIRouter()


@router.get("/health")
def health():
    return runtime.health_payload()


@router.get("/checkpoint")
def checkpoint():
    return runtime.checkpoint_payload()


@router.get("/users")
def users(limit: int = 30, source: str = "all"):
    return runtime.users_payload(limit=limit, source=source)


@router.get("/users/{user_id}/profile")
def user_profile(user_id: str):
    return runtime.user_profile_payload(user_id)


@router.get("/users/{user_id}/activities")
def user_activities(user_id: str, limit: int = 30):
    return runtime.user_activities_payload(user_id=user_id, limit=limit)


@router.get("/business/search")
def search_business(
    q: str = "",
    categories: str = "",
    limit: int = 20,
    name_only: bool = False,
):
    return runtime.search_business_payload(
        q=q,
        categories=categories,
        limit=limit,
        name_only=name_only,
    )


@router.get("/business/categories")
def business_categories(limit: int = 200):
    return runtime.business_categories_payload(limit=limit)


@router.post("/recommend")
def recommend(payload: RecommendRequest, x_auth_token: str | None = Header(default=None)):
    return runtime.recommend_payload(
        user_id=payload.user_id,
        k=payload.k,
        algorithm=payload.algorithm,
        history_business_ids=payload.history_business_ids,
        history_ratings=payload.history_ratings,
        source=payload.source,
        x_auth_token=x_auth_token,
    )
