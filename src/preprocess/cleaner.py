import re

import pandas as pd

from config.settings import MIN_TEXT_LENGTH
from config.taxonomies import LOW_VALUE_TEXTS, SPAM_PATTERNS
from src.preprocess.dedup import enrich_dedup_columns


URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\S+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = re.sub(r"[^\w\u4e00-\u9fff，。！？；：“”‘’、,.!?;:()（）\-\[\] ]", "", text)
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


def detect_low_value_reason(text: str) -> str:
    if not text:
        return "empty"
    if len(text) < MIN_TEXT_LENGTH:
        return "too_short"
    if text in LOW_VALUE_TEXTS:
        return "slang_only"
    if any(pattern in text for pattern in SPAM_PATTERNS):
        return "spam_like"
    return ""


def clean_thread_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["clean_text"] = cleaned["comment_text"].fillna("").map(clean_text)
    cleaned["text_length"] = cleaned["clean_text"].map(len)
    cleaned["low_value_reason"] = cleaned["clean_text"].map(detect_low_value_reason)
    cleaned["is_low_value"] = cleaned["low_value_reason"].ne("")
    cleaned = cleaned[cleaned["low_value_reason"] != "empty"]
    cleaned = cleaned.drop_duplicates(subset=["thread_id", "clean_text"])
    cleaned = enrich_dedup_columns(cleaned.reset_index(drop=True))
    return cleaned


def clean_content_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["clean_text"] = cleaned["answer_text"].fillna("").map(clean_text)
    cleaned["text_length"] = cleaned["clean_text"].map(len)
    cleaned["low_value_reason"] = cleaned["clean_text"].map(detect_low_value_reason)
    cleaned["is_low_value"] = cleaned["low_value_reason"].ne("")
    cleaned = cleaned[cleaned["low_value_reason"] != "empty"]
    cleaned = cleaned.drop_duplicates(subset=["question_id", "clean_text"])
    cleaned = enrich_dedup_columns(cleaned.reset_index(drop=True))
    return cleaned
