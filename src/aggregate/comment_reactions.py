import pandas as pd


def build_comment_reaction_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "answer_id",
                "question_title",
                "comment_count",
                "supportive_ratio",
                "opposing_ratio",
                "controversy_score",
                "sarcasm_ratio",
                "high_value_comment_count",
                "avg_comment_business_value_score",
                "avg_comment_entropy_score",
            ]
        )

    comments = df.copy()
    comments["is_supportive"] = comments["sentiment"].eq("positive")
    comments["is_opposing"] = comments["sentiment"].eq("negative")
    comments["is_sarcasm"] = comments["sarcasm_hit_count"].fillna(0).astype(float).gt(0)
    comments["is_high_value_comment"] = comments["is_high_value_feedback"].fillna(False).astype(bool)
    comments["is_controversial"] = comments["sentiment"].isin(["negative", "mixed"])

    grouped = (
        comments.groupby(["answer_id", "question_title"], dropna=False)
        .agg(
            comment_count=("comment_id", "count"),
            supportive_ratio=("is_supportive", "mean"),
            opposing_ratio=("is_opposing", "mean"),
            controversy_score=("is_controversial", "mean"),
            sarcasm_ratio=("is_sarcasm", "mean"),
            high_value_comment_count=("is_high_value_comment", "sum"),
            avg_comment_business_value_score=("business_value_score", "mean"),
            avg_comment_entropy_score=("feedback_entropy_score", "mean"),
        )
        .reset_index()
    )

    ratio_cols = ["supportive_ratio", "opposing_ratio", "controversy_score", "sarcasm_ratio"]
    for column in ratio_cols:
        grouped[column] = (grouped[column] * 100).round(2)
    grouped["avg_comment_business_value_score"] = grouped["avg_comment_business_value_score"].round(2)
    grouped["avg_comment_entropy_score"] = grouped["avg_comment_entropy_score"].round(2)
    return grouped.sort_values(["high_value_comment_count", "comment_count"], ascending=[False, False]).reset_index(drop=True)
