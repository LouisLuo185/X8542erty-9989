from collections import Counter

import jieba
import pandas as pd


def extract_top_keywords(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    words: list[str] = []
    for text in df["text"].fillna(""):
        words.extend([word.strip() for word in jieba.lcut(text) if len(word.strip()) >= 2])

    counter = Counter(words)
    return pd.DataFrame(counter.most_common(top_n), columns=["keyword", "count"])
