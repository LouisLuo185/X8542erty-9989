import pandas as pd

from config.settings import FEATURE_SCHEMA, HIGH_VALUE_WEIGHT, LOW_VALUE_WEIGHT, MEDIUM_VALUE_WEIGHT
from src.preprocess.slang_rules import build_term_features


def get_base_weight(actionable: bool, entropy_score: int) -> float:
    if actionable and entropy_score >= 6:
        return HIGH_VALUE_WEIGHT
    if actionable:
        return MEDIUM_VALUE_WEIGHT
    return LOW_VALUE_WEIGHT


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    features = [
        build_term_features(text, title)
        for text, title in zip(featured["clean_text"], featured["question_title"])
    ]
    feature_df = pd.DataFrame(features)
    featured = pd.concat([featured.reset_index(drop=True), feature_df], axis=1)

    for column in FEATURE_SCHEMA:
        if column not in featured.columns:
            featured[column] = None
    return featured[FEATURE_SCHEMA]
