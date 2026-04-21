from pathlib import Path

import pandas as pd

from config.settings import DATA_RAW_DIR, RAW_SCHEMA


def fetch_zhihu_data(keyword: str = "原神", limit: int = 100) -> pd.DataFrame:
    """
    知乎数据采集占位函数。
    后续可替换为 requests / Playwright / 半自动导入逻辑。
    """
    records = [
        {
            "platform": "zhihu",
            "title": f"{keyword} 示例问题",
            "text": "原神的美术和角色设计确实很强，但后期玩法有些重复。",
            "time": "2026-04-01",
            "url": "https://example.com/zhihu-demo",
            "likes": 120,
        }
    ]
    return pd.DataFrame(records, columns=RAW_SCHEMA).head(limit)


def save_demo_csv(path: Path | None = None) -> Path:
    output_path = path or DATA_RAW_DIR / "zhihu_raw.csv"
    df = fetch_zhihu_data()
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


if __name__ == "__main__":
    saved = save_demo_csv()
    print(f"saved: {saved}")
