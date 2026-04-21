import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import MANIFEST_DIR


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    target = ensure_parent_dir(path)
    df.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    target = ensure_parent_dir(path)
    df.to_parquet(target, index=False)
    return target


def write_manifest(stage: str, payload: dict) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    target = MANIFEST_DIR / f"{stage}.json"
    body = {"stage": stage, "ts": datetime.now(timezone.utc).isoformat(), **payload}
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None
