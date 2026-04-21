import pandas as pd


def build_content_feedback_report(content_df: pd.DataFrame, comment_summary_df: pd.DataFrame) -> pd.DataFrame:
    if content_df.empty:
        return pd.DataFrame()

    report = content_df.copy()
    if not comment_summary_df.empty:
        report = report.merge(comment_summary_df, on=["answer_id", "question_title"], how="left")
        rename_map = {}
        if "comment_count_x" in report.columns:
            rename_map["comment_count_x"] = "source_comment_count"
        if "comment_count_y" in report.columns:
            rename_map["comment_count_y"] = "reaction_comment_count"
        if rename_map:
            report = report.rename(columns=rename_map)
    else:
        for column in [
            "reaction_comment_count",
            "supportive_ratio",
            "opposing_ratio",
            "controversy_score",
            "sarcasm_ratio",
            "high_value_comment_count",
            "avg_comment_business_value_score",
            "avg_comment_entropy_score",
        ]:
            report[column] = None

    return report.sort_values(
        ["is_high_value_feedback_pre_llm", "business_value_score", "feedback_entropy_score", "likes"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
