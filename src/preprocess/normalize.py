from pathlib import Path

import pandas as pd

from config.settings import RAW_COMMENT_SCHEMA, RAW_CONTENT_SCHEMA


def load_zhihu_content_csv(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    normalized = pd.DataFrame(
        {
            "platform": "zhihu",
            "source_type": "answer",
            "content_id": df.get("content_id"),
            "question_id": df.get("question_id"),
            "title": df.get("title"),
            "text": df.get("content_text"),
            "time": df.get("created_time"),
            "url": df.get("content_url"),
            "likes": df.get("voteup_count", 0),
            "comment_count": df.get("comment_count", 0),
            "source_keyword": df.get("source_keyword"),
        }
    )
    return normalized[RAW_CONTENT_SCHEMA]


def load_zhihu_comment_csv(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    normalized = pd.DataFrame(
        {
            "platform": "zhihu",
            "source_type": "comment",
            "content_id": df.get("content_id"),
            "comment_id": df.get("comment_id"),
            "parent_comment_id": df.get("parent_comment_id", 0),
            "text": df.get("content"),
            "time": df.get("publish_time"),
            "likes": df.get("like_count", 0),
            "ip_location": df.get("ip_location"),
            "sub_comment_count": df.get("sub_comment_count", 0),
        }
    )
    return normalized[RAW_COMMENT_SCHEMA]
