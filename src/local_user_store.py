from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class AuthRecord:
    username: str
    user_id: str
    token: str


class LocalUserStore:
    """Simple local auth + interaction storage for demo/testing on localhost."""

    def __init__(self, root_dir: str = "local_data"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.users_path = self.root / "users.json"
        self.interactions_path = self.root / "interactions.jsonl"

    def _load_users(self) -> dict[str, dict]:
        if not self.users_path.exists():
            return {}
        try:
            return json.loads(self.users_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_users(self, users: dict[str, dict]):
        self.users_path.write_text(json.dumps(users, indent=2), encoding="utf-8")

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()

    def signup(self, username: str, password: str) -> AuthRecord:
        username = username.strip().lower()
        if not username or len(password) < 4:
            raise ValueError("username/password_invalid")

        users = self._load_users()
        if username in users:
            raise ValueError("username_exists")

        user_id = f"local_{secrets.token_hex(8)}"
        salt = secrets.token_hex(8)
        users[username] = {
            "username": username,
            "user_id": user_id,
            "salt": salt,
            "password_hash": self._hash_password(password, salt),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save_users(users)
        token = secrets.token_urlsafe(24)
        return AuthRecord(username=username, user_id=user_id, token=token)

    def signin(self, username: str, password: str) -> AuthRecord:
        username = username.strip().lower()
        users = self._load_users()
        payload = users.get(username)
        if payload is None:
            raise ValueError("user_not_found")
        candidate_hash = self._hash_password(password, payload["salt"])
        if candidate_hash != payload["password_hash"]:
            raise ValueError("bad_credentials")
        token = secrets.token_urlsafe(24)
        return AuthRecord(username=username, user_id=payload["user_id"], token=token)

    def add_interaction(
        self,
        user_id: str,
        business_id: str,
        stars: float,
        source: str = "local",
    ):
        stars = float(max(1.0, min(5.0, stars)))
        payload = {
            "user_id": str(user_id),
            "business_id": str(business_id),
            "stars": stars,
            "date": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        with self.interactions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def get_user_history(self, user_id: str) -> tuple[list[str], list[float]]:
        df = self.load_interactions_df()
        if df.empty:
            return [], []
        user_df = df[df["user_id"].astype(str) == str(user_id)].sort_values("date")
        if user_df.empty:
            return [], []
        return (
            user_df["business_id"].astype(str).tolist(),
            user_df["stars"].astype(float).tolist(),
        )

    def load_interactions_df(self) -> pd.DataFrame:
        if not self.interactions_path.exists():
            return pd.DataFrame(columns=["user_id", "business_id", "stars", "date", "source"])
        rows: list[dict] = []
        with self.interactions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if not rows:
            return pd.DataFrame(columns=["user_id", "business_id", "stars", "date", "source"])
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def list_users(self) -> list[dict]:
        users = self._load_users()
        return [
            {"username": payload["username"], "user_id": payload["user_id"]}
            for payload in users.values()
        ]
