from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


def _safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _attributes_to_text(value: object) -> str:
    if isinstance(value, dict):
        pairs = [f"{k} {_safe_text(v)}" for k, v in value.items()]
        return " ".join(pairs)
    return _safe_text(value)


@dataclass
class DataConfig:
    """Configuration for raw Yelp preprocessing."""

    min_business_reviews: int = 10
    min_user_reviews: int = 3
    min_item_reviews: int = 5
    category_keyword: str = "Restaurants"
    city: str | None = None
    state: str | None = None
    chunksize: int = 200_000
    max_review_chunks: int | None = None


class DataLoader:
    """Load and transform raw Yelp JSON files into model-ready tables."""

    def __init__(self, business_path: str, review_path: str):
        self.business_path = business_path
        self.review_path = review_path
        self.biz: pd.DataFrame | None = None
        self.rev: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None

    def load_businesses(
        self,
        min_reviews: int = 10,
        category_keyword: str = "Restaurants",
        city: str | None = None,
        state: str | None = None,
    ) -> pd.DataFrame:
        """Load business metadata and build text features for content modeling."""
        biz = pd.read_json(self.business_path, lines=True)
        biz = biz[biz["review_count"] >= min_reviews].copy()
        biz = biz[biz["categories"].fillna("").str.contains(category_keyword, case=False)]

        if city is not None:
            biz = biz[biz["city"].str.lower() == city.lower()]
        if state is not None:
            biz = biz[biz["state"].str.lower() == state.lower()]

        biz["categories"] = biz["categories"].fillna("")
        biz["attributes"] = biz["attributes"].apply(_attributes_to_text)
        biz["name"] = biz["name"].fillna("")
        biz["soup"] = (
            biz["name"].astype(str)
            + " "
            + biz["categories"].astype(str)
            + " "
            + biz["attributes"].astype(str)
        ).str.lower()

        keep_cols = [
            "business_id",
            "name",
            "city",
            "state",
            "stars",
            "review_count",
            "categories",
            "attributes",
            "soup",
        ]
        self.biz = biz[keep_cols].drop_duplicates("business_id").reset_index(drop=True)
        return self.biz

    def load_reviews(
        self,
        business_ids: Iterable[str],
        chunksize: int = 200_000,
        max_chunks: int | None = None,
        include_text: bool = False,
    ) -> pd.DataFrame:
        """Chunk-load reviews filtered by selected business IDs."""
        business_id_set = set(business_ids)
        selected_chunks: list[pd.DataFrame] = []

        for idx, chunk in enumerate(
            pd.read_json(self.review_path, lines=True, chunksize=chunksize), start=1
        ):
            filtered = chunk[chunk["business_id"].isin(business_id_set)]
            if filtered.empty:
                pass
            else:
                cols = ["user_id", "business_id", "stars", "date"]
                if include_text and "text" in filtered.columns:
                    cols.append("text")
                selected_chunks.append(
                    filtered[cols].copy()
                )
            if max_chunks is not None and idx >= max_chunks:
                break

        if not selected_chunks:
            self.rev = pd.DataFrame(
                columns=["user_id", "business_id", "stars", "date"] + (["text"] if include_text else [])
            )
            return self.rev

        rev = pd.concat(selected_chunks, ignore_index=True)
        rev["date"] = pd.to_datetime(rev["date"], errors="coerce")
        rev = rev.dropna(subset=["user_id", "business_id", "stars"])
        self.rev = rev.reset_index(drop=True)
        return self.rev

    def build_interactions(
        self,
        min_user_reviews: int = 3,
        min_item_reviews: int = 5,
    ) -> pd.DataFrame:
        """Aggregate review events into one (user, business) interaction row."""
        if self.rev is None:
            raise ValueError("Reviews are not loaded. Call load_reviews first.")

        interactions = self.rev[["user_id", "business_id", "stars", "date"]].copy()
        interactions = interactions.sort_values("date")
        interactions = (
            interactions.groupby(["user_id", "business_id"], as_index=False)
            .agg(stars=("stars", "mean"), date=("date", "max"))
            .reset_index(drop=True)
        )

        user_counts = interactions["user_id"].value_counts()
        item_counts = interactions["business_id"].value_counts()

        interactions = interactions[
            interactions["user_id"].isin(user_counts[user_counts >= min_user_reviews].index)
        ]
        interactions = interactions[
            interactions["business_id"].isin(
                item_counts[item_counts >= min_item_reviews].index
            )
        ]

        self.interactions = interactions.reset_index(drop=True)
        return self.interactions

    def clean_data(
        self,
        min_reviews: int = 10,
        city: str | None = None,
        state: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Backward-compatible convenience call for business/review loading."""
        biz = self.load_businesses(min_reviews=min_reviews, city=city, state=state)
        rev = self.load_reviews(business_ids=biz["business_id"], chunksize=200_000)
        return biz, rev

    def run(self, config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """End-to-end preprocessing run controlled by DataConfig."""
        biz = self.load_businesses(
            min_reviews=config.min_business_reviews,
            category_keyword=config.category_keyword,
            city=config.city,
            state=config.state,
        )
        rev = self.load_reviews(
            business_ids=biz["business_id"],
            chunksize=config.chunksize,
            max_chunks=config.max_review_chunks,
            include_text=False,
        )
        interactions = self.build_interactions(
            min_user_reviews=config.min_user_reviews,
            min_item_reviews=config.min_item_reviews,
        )
        return biz, rev, interactions
