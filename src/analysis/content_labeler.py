import pandas as pd

from config.settings import CONTENT_RULE_SCHEMA
from src.analysis.llm_labeler import rule_based_label


def label_content_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        merged = row.to_dict()
        merged.update(rule_based_label(row))
        rows.append(merged)
    labeled_df = pd.DataFrame(rows)
    for column in CONTENT_RULE_SCHEMA:
        if column not in labeled_df.columns:
            labeled_df[column] = None
    return labeled_df[CONTENT_RULE_SCHEMA]
