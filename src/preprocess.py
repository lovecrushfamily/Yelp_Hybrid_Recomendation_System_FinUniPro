from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import pandas as pd
import pyarrow.dataset as ds

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}


def _safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _attributes_to_text(value: object) -> str:
    if isinstance(value, dict):
        pairs = [f"{k} {_safe_text(v)}" for k, v in value.items()]
        return " ".join(pairs)
    return _safe_text(value)


def _attributes_frame_to_text(attr_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten wide attribute parquet export into one text column."""
    if attr_df.empty or "business_id" not in attr_df.columns:
        return pd.DataFrame(columns=["business_id", "attributes"])

    attr_cols = [c for c in attr_df.columns if c != "business_id"]
    if not attr_cols:
        return pd.DataFrame(
            {
                "business_id": attr_df["business_id"].astype(str),
                "attributes": "",
            }
        )

    filled = attr_df[attr_cols].fillna("")

    def row_to_text(row: pd.Series) -> str:
        parts: list[str] = []
        for col, value in row.items():
            text = _safe_text(value).strip()
            if not text or text.lower() in {"nan", "none", "null"}:
                continue
            parts.append(f"{col} {text}")
        return " ".join(parts)

    return pd.DataFrame(
        {
            "business_id": attr_df["business_id"].astype(str),
            "attributes": filled.apply(row_to_text, axis=1).astype(str),
        }
    )


@dataclass
class DataConfig:
    """Configuration for raw Yelp preprocessing."""

    min_business_reviews: int = 10
    min_user_reviews: int = 3
    min_item_reviews: int = 5
    category_keyword: str = "Restaurants"
    city: str | None = None
    state: str | None = None
    usa_only: bool = False
    chunksize: int = 200_000
    max_review_chunks: int | None = None
    verbose: bool = False
    progress_every: int = 5


class DataLoader:
    """Load and transform raw Yelp JSON files into model-ready tables."""

    def __init__(
        self,
        business_path: str,
        review_path: str,
        source_format: str = "json",
        data_dir: str = "data",
    ):
        self.business_path = business_path
        self.review_path = review_path
        self.source_format = source_format
        self.data_dir = data_dir
        self.biz: pd.DataFrame | None = None
        self.rev: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None
        self.active_source_format: str | None = None

    def resolve_source_format(self) -> str:
        """Choose between raw JSON and parquet exports."""
        if self.source_format in {"json", "parquet"}:
            self.active_source_format = self.source_format
            return self.source_format

        data_root = Path(self.data_dir)
        if (data_root / "business_main.parquet").exists() and (data_root / "reviews.parquet").exists():
            self.active_source_format = "parquet"
            return "parquet"

        self.active_source_format = "json"
        return "json"

    def _data_path(self, filename: str) -> Path:
        return Path(self.data_dir) / filename

    def _load_businesses_from_json(
        self,
        min_reviews: int,
        category_keyword: str,
        city: str | None,
        state: str | None,
        usa_only: bool,
    ) -> pd.DataFrame:
        biz = pd.read_json(self.business_path, lines=True)
        biz = biz[biz["review_count"] >= min_reviews].copy()
        biz = biz[biz["categories"].fillna("").str.contains(category_keyword, case=False)]
        if usa_only:
            biz = biz[biz["state"].astype(str).str.upper().isin(US_STATE_CODES)]

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
        return biz

    def _load_businesses_from_parquet(
        self,
        min_reviews: int,
        category_keyword: str,
        city: str | None,
        state: str | None,
        usa_only: bool,
    ) -> pd.DataFrame:
        main_path = self._data_path("business_main.parquet")
        if not main_path.exists():
            raise FileNotFoundError(f"Missing parquet source: {main_path}")

        biz = pd.read_parquet(main_path)
        biz["business_id"] = biz["business_id"].astype(str)
        biz = biz[biz["review_count"] >= min_reviews].copy()
        biz = biz[biz["categories"].fillna("").str.contains(category_keyword, case=False)]
        if usa_only:
            biz = biz[biz["state"].astype(str).str.upper().isin(US_STATE_CODES)]

        if city is not None:
            biz = biz[biz["city"].astype(str).str.lower() == city.lower()]
        if state is not None:
            biz = biz[biz["state"].astype(str).str.lower() == state.lower()]

        attr_path = self._data_path("business_attributes.parquet")
        if attr_path.exists():
            attr_df = pd.read_parquet(attr_path)
            attr_df["business_id"] = attr_df["business_id"].astype(str)
            attr_text = _attributes_frame_to_text(attr_df)
            biz = biz.merge(attr_text, on="business_id", how="left")
        else:
            biz["attributes"] = ""

        soup_path = self._data_path("business_soup.parquet")
        if soup_path.exists():
            soup_df = pd.read_parquet(soup_path)
            soup_df["business_id"] = soup_df["business_id"].astype(str)
            biz = biz.merge(soup_df[["business_id", "soup"]], on="business_id", how="left")
        else:
            biz["soup"] = ""

        pop_path = self._data_path("business_popularity.parquet")
        if pop_path.exists():
            pop_df = pd.read_parquet(pop_path)
            pop_df["business_id"] = pop_df["business_id"].astype(str)
            biz = biz.merge(pop_df[["business_id", "popularity_prior"]], on="business_id", how="left")

        biz["categories"] = biz["categories"].fillna("")
        biz["name"] = biz["name"].fillna("")
        biz["attributes"] = biz["attributes"].fillna("")
        if "soup" not in biz.columns:
            biz["soup"] = ""
        biz["soup"] = biz["soup"].fillna("")
        missing_soup = biz["soup"].astype(str).str.strip() == ""
        if missing_soup.any():
            biz.loc[missing_soup, "soup"] = (
                biz.loc[missing_soup, "name"].astype(str)
                + " "
                + biz.loc[missing_soup, "categories"].astype(str)
                + " "
                + biz.loc[missing_soup, "attributes"].astype(str)
            ).str.lower()
        return biz

    def load_businesses(
        self,
        min_reviews: int = 10,
        category_keyword: str = "Restaurants",
        city: str | None = None,
        state: str | None = None,
        usa_only: bool = False,
    ) -> pd.DataFrame:
        """Load business metadata and build text features for content modeling."""
        source_format = self.resolve_source_format()
        if source_format == "parquet":
            biz = self._load_businesses_from_parquet(
                min_reviews=min_reviews,
                category_keyword=category_keyword,
                city=city,
                state=state,
                usa_only=usa_only,
            )
        else:
            biz = self._load_businesses_from_json(
                min_reviews=min_reviews,
                category_keyword=category_keyword,
                city=city,
                state=state,
                usa_only=usa_only,
            )

        keep_cols = [
            "business_id",
            "name",
            "city",
            "state",
            "address",
            "latitude",
            "longitude",
            "stars",
            "review_count",
            "is_open",
            "categories",
            "attributes",
            "soup",
            "tip_count",
            "checkin_count",
            "popularity_prior",
        ]
        keep_cols = [col for col in keep_cols if col in biz.columns]
        self.biz = biz[keep_cols].drop_duplicates("business_id").reset_index(drop=True)
        return self.biz

    def _load_reviews_from_json(
        self,
        business_ids: Iterable[str],
        chunksize: int,
        max_chunks: int | None,
        include_text: bool,
        verbose: bool,
        progress_every: int,
    ) -> pd.DataFrame:
        business_id_set = set(business_ids)
        selected_chunks: list[pd.DataFrame] = []
        total_rows = 0
        total_selected = 0
        start_ts = time.perf_counter()

        for idx, chunk in enumerate(
            pd.read_json(self.review_path, lines=True, chunksize=chunksize), start=1
        ):
            total_rows += len(chunk)
            filtered = chunk[chunk["business_id"].isin(business_id_set)]
            if not filtered.empty:
                cols = ["user_id", "business_id", "stars", "date"]
                if include_text and "text" in filtered.columns:
                    cols.append("text")
                selected_chunks.append(filtered[cols].copy())
                total_selected += len(filtered)
            if verbose and progress_every > 0 and idx % progress_every == 0:
                elapsed = time.perf_counter() - start_ts
                print(
                    f"[INFO] Review chunks={idx} rows={total_rows} selected={total_selected} "
                    f"elapsed_sec={elapsed:.1f}"
                )
            if max_chunks is not None and idx >= max_chunks:
                break

        if verbose:
            elapsed = time.perf_counter() - start_ts
            print(
                f"[INFO] Review scan done chunks={idx} rows={total_rows} selected={total_selected} "
                f"elapsed_sec={elapsed:.1f}"
            )

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

    def _load_reviews_from_parquet(
        self,
        business_ids: Iterable[str],
        chunksize: int,
        max_chunks: int | None,
        include_text: bool,
        verbose: bool,
        progress_every: int,
    ) -> pd.DataFrame:
        review_path = self._data_path("reviews.parquet")
        if not review_path.exists():
            raise FileNotFoundError(f"Missing parquet source: {review_path}")

        business_id_list = [str(x) for x in business_ids]
        cols = ["user_id", "business_id", "stars", "date"]
        if include_text:
            cols.append("text")

        if verbose:
            print(f"[INFO] Loading parquet reviews from {review_path}")
        dataset = ds.dataset(str(review_path), format="parquet")
        table = dataset.to_table(
            columns=cols,
            filter=ds.field("business_id").isin(business_id_list),
        )
        if max_chunks is not None:
            max_rows = max(1, int(max_chunks) * max(1, chunksize))
            table = table.slice(0, max_rows)
        rev = table.to_pandas()
        rev["date"] = pd.to_datetime(rev["date"], errors="coerce")
        rev = rev.dropna(subset=["user_id", "business_id", "stars"]).reset_index(drop=True)
        if verbose:
            print(
                f"[INFO] Review parquet load done rows={len(rev)} "
                f"selected_businesses={len(set(rev['business_id'].astype(str)))}"
            )
        self.rev = rev
        return self.rev

    def load_reviews(
        self,
        business_ids: Iterable[str],
        chunksize: int = 200_000,
        max_chunks: int | None = None,
        include_text: bool = False,
        verbose: bool = False,
        progress_every: int = 5,
    ) -> pd.DataFrame:
        """Chunk-load reviews filtered by selected business IDs."""
        source_format = self.resolve_source_format()
        if source_format == "parquet":
            return self._load_reviews_from_parquet(
                business_ids=business_ids,
                chunksize=chunksize,
                max_chunks=max_chunks,
                include_text=include_text,
                verbose=verbose,
                progress_every=progress_every,
            )
        return self._load_reviews_from_json(
            business_ids=business_ids,
            chunksize=chunksize,
            max_chunks=max_chunks,
            include_text=include_text,
            verbose=verbose,
            progress_every=progress_every,
        )

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
            usa_only=config.usa_only,
        )
        rev = self.load_reviews(
            business_ids=biz["business_id"],
            chunksize=config.chunksize,
            max_chunks=config.max_review_chunks,
            include_text=False,
            verbose=config.verbose,
            progress_every=config.progress_every,
        )
        interactions = self.build_interactions(
            min_user_reviews=config.min_user_reviews,
            min_item_reviews=config.min_item_reviews,
        )
        return biz, rev, interactions
