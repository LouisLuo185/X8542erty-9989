from pathlib import Path

import pandas as pd

from config.settings import DATA_RAW_DIR, RAW_SCHEMA


def fetch_bilibili_data(keyword: str = "原神", limit: int = 100) -> pd.DataFrame:
    """
    B站数据采集占位函数。
    后续可替换为实际抓取逻辑或导入导出的原始 CSV。
    """
    records = [
        {
            "platform": "bilibili",
            "title": f"{keyword} 示例视频",
            "text": "新版本剧情挺有意思，音乐也很顶，就是抽卡压力还是大。",
            "time": "2026-04-02",
            "url": "https://example.com/bilibili-demo",
            "likes": 88,
        }
    ]
    return pd.DataFrame(records, columns=RAW_SCHEMA).head(limit)


def save_demo_csv(path: Path | None = None) -> Path:
    output_path = path or DATA_RAW_DIR / "bilibili_raw.csv"
    df = fetch_bilibili_data()
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


if __name__ == "__main__":
    saved = save_demo_csv()
    print(f"saved: {saved}")
