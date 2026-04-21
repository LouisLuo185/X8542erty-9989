import pandas as pd

from config.settings import FINAL_LABELED_SCHEMA


def _build_custom_id(row: pd.Series) -> str:
    comment_id = row.get("comment_id")
    if pd.notna(comment_id):
        return f"comment_{int(comment_id)}"
    return f"row_{row.name}"


def merge_llm_results(rule_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    final_df = rule_df.copy()
    final_df["custom_id"] = final_df.apply(_build_custom_id, axis=1)

    if not llm_df.empty:
        final_df = final_df.merge(llm_df, on="custom_id", how="left")
    else:
        final_df["llm_sentiment"] = None
        final_df["llm_dimension"] = None
        final_df["llm_is_comparative"] = None
        final_df["llm_target_game"] = None
        final_df["llm_stance_summary"] = None
        final_df["llm_confidence"] = None
        final_df["llm_needs_manual_review"] = None

    has_llm = final_df["llm_sentiment"].notna()

    final_df["annotation_source"] = has_llm.map(lambda value: "llm" if value else "rule")
    final_df["sentiment"] = final_df["llm_sentiment"].fillna(final_df["rule_sentiment"])
    final_df["dimension"] = final_df["llm_dimension"].fillna(final_df["rule_dimension"])
    final_df["reason"] = final_df["llm_stance_summary"].fillna(final_df["rule_reason"])
    final_df["confidence"] = final_df["llm_confidence"].fillna(final_df["rule_confidence"])
    final_df["is_comparative"] = final_df["llm_is_comparative"].fillna(final_df["rule_is_comparative"])
    final_df["target_game"] = final_df["llm_target_game"].fillna(final_df["rule_target_game"])
    final_df["needs_manual_review"] = final_df["llm_needs_manual_review"].fillna(False)

    for column in FINAL_LABELED_SCHEMA:
        if column not in final_df.columns:
            final_df[column] = None

    return final_df[FINAL_LABELED_SCHEMA]
