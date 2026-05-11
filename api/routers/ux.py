from __future__ import annotations

from fastapi import APIRouter, Header

from .. import runtime
from ..schemas import UXLoginRequest

router = APIRouter()


@router.post("/ux/login")
def ux_login(payload: UXLoginRequest):
    return runtime.ux_login_payload(payload.user_id)


@router.get("/ux/me")
def ux_me(x_ux_token: str | None = Header(default=None)):
    return runtime.ux_me_payload(x_ux_token=x_ux_token)


@router.get("/ux/recommend")
def ux_recommend(
    offset: int = 0,
    limit: int = 15,
    refresh: bool = False,
    x_ux_token: str | None = Header(default=None),
):
    return runtime.ux_recommend_payload(
        offset=offset,
        limit=limit,
        refresh=refresh,
        x_ux_token=x_ux_token,
    )


@router.get("/ux/search")
def ux_search(
    q: str = "",
    categories: str = "",
    limit: int = 20,
    x_ux_token: str | None = Header(default=None),
):
    return runtime.ux_search_payload(
        q=q,
        categories=categories,
        limit=limit,
        x_ux_token=x_ux_token,
    )


@router.get("/ux/categories")
def ux_categories(limit: int = 250):
    return runtime.ux_categories_payload(limit=limit)
