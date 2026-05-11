from __future__ import annotations

from fastapi import APIRouter, Header

from .. import runtime
from ..schemas import RateRequest, SignInRequest, SignUpRequest

router = APIRouter()


@router.post("/auth/signup")
def auth_signup(payload: SignUpRequest):
    return runtime.signup_payload(payload.username, payload.password)


@router.post("/auth/signin")
def auth_signin(payload: SignInRequest):
    return runtime.signin_payload(payload.username, payload.password)


@router.post("/me/rate")
def rate_business(payload: RateRequest, x_auth_token: str | None = Header(default=None)):
    return runtime.rate_payload(
        business_id=payload.business_id,
        stars=payload.stars,
        x_auth_token=x_auth_token,
    )


@router.get("/me/history")
def me_history(x_auth_token: str | None = Header(default=None)):
    return runtime.me_history_payload(x_auth_token=x_auth_token)


@router.get("/me/profile")
def me_profile(x_auth_token: str | None = Header(default=None)):
    return runtime.me_profile_payload(x_auth_token=x_auth_token)
