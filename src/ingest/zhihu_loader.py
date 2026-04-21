from pathlib import Path

import pandas as pd

from config.settings import DATA_RAW_DIR
from src.preprocess.normalize import load_zhihu_comment_csv, load_zhihu_content_csv


def load_latest_zhihu_exports(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = raw_dir or DATA_RAW_DIR
    content_path = base_dir / "search_contents_2026-04-12.csv"
    comment_path = base_dir / "search_comments_2026-04-12.csv"

    content_df = load_zhihu_content_csv(content_path) if content_path.exists() else pd.DataFrame()
    comment_df = load_zhihu_comment_csv(comment_path) if comment_path.exists() else pd.DataFrame()
    return content_df, comment_df
