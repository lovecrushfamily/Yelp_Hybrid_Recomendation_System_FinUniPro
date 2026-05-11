from __future__ import annotations

from pydantic import BaseModel, Field


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
