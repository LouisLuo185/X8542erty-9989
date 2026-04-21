from pathlib import Path

import pandas as pd

from config.settings import DATA_RAW_DIR, RAW_SCHEMA


def load_external_csv(file_path: str | Path, platform: str) -> pd.DataFrame:
    """
    用于导入你手工整理或其他方式导出的原始 CSV。
    只要最终能映射到统一字段即可继续跑流程。
    """
    df = pd.read_csv(file_path)
    for column in RAW_SCHEMA:
        if column not in df.columns:
            df[column] = None
    df["platform"] = platform
    return df[RAW_SCHEMA]


def save_imported_csv(file_path: str | Path, platform: str) -> Path:
    df = load_external_csv(file_path, platform)
    output_path = DATA_RAW_DIR / f"{platform}_raw.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
