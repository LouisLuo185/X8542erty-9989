import pandas as pd

from config.settings import CONTENT_FINAL_SCHEMA, FINAL_SCHEMA
from src.llm.batch_prepare import build_context_ready_text
from src.utils.business_modes import normalize_business_modes


def _build_custom_id(row: pd.Series) -> str:
    comment_id = row.get("comment_id")
    if pd.notna(comment_id):
        return f"comment_{int(comment_id)}"
    answer_id = row.get("answer_id")
    if pd.notna(answer_id):
        return f"answer_{int(answer_id)}"
    return f"row_{row.name}"


def _merge_llm_results_base(rule_df: pd.DataFrame, llm_df: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    final_df = rule_df.copy()
    final_df["custom_id"] = final_df.apply(_build_custom_id, axis=1)
    final_df["context_ready_text"] = final_df.apply(build_context_ready_text, axis=1)

    if not llm_df.empty:
        final_df = final_df.merge(llm_df, on="custom_id", how="left")
    else:
        for column in [
            "llm_sentiment",
            "llm_dimension",
            "llm_is_comparative",
            "llm_target_game",
            "llm_comment_target",
            "llm_business_mode",
            "llm_stance_summary",
            "llm_confidence",
            "llm_is_high_value_feedback",
            "llm_needs_manual_review",
        ]:
            final_df[column] = None

    has_llm = final_df["llm_sentiment"].notna()
    final_df["annotation_source"] = has_llm.map(lambda item: "llm" if item else "rule")
    final_df["sentiment"] = final_df["llm_sentiment"].where(has_llm, final_df["rule_sentiment"])
    final_df["dimension"] = final_df["llm_dimension"].where(has_llm, final_df["rule_dimension"])
    final_df["reason"] = final_df["llm_stance_summary"].where(has_llm, final_df["rule_reason"])
    final_df["confidence"] = pd.to_numeric(final_df["llm_confidence"], errors="coerce").fillna(final_df["rule_confidence"])
    final_df["is_comparative"] = final_df["llm_is_comparative"].where(has_llm, final_df["rule_is_comparative"])
    final_df["target_game"] = final_df["llm_target_game"].where(has_llm, final_df["rule_target_game"])
    final_df["comment_target"] = final_df["llm_comment_target"].where(has_llm, final_df["rule_comment_target"])
    llm_mode = final_df["llm_business_mode"].where(has_llm, "")
    final_df["business_mode"] = [
        normalize_business_modes(raw_mode, fallback_mode)
        for raw_mode, fallback_mode in zip(llm_mode, final_df["feedback_modes"])
    ]
    pre_llm = final_df["is_high_value_feedback_pre_llm"] if "is_high_value_feedback_pre_llm" in final_df.columns else final_df["is_actionable_feedback"]
    final_df["is_high_value_feedback"] = final_df["llm_is_high_value_feedback"].where(has_llm, pre_llm)
    final_df["needs_manual_review"] = final_df["llm_needs_manual_review"].fillna(value=False).astype(bool)

    if "high_value_reason" not in final_df.columns:
        final_df["high_value_reason"] = ""
    final_df["high_value_reason"] = final_df["high_value_reason"].fillna("").astype(str)
    if bool(has_llm.any()):
        mask = has_llm.fillna(False)
        thv = final_df.loc[mask, "llm_is_high_value_feedback"].fillna(value=False).astype(bool).astype(str)
        tman = final_df.loc[mask, "llm_needs_manual_review"].fillna(value=False).astype(bool).astype(str)
        final_df.loc[mask, "high_value_reason"] = (
            final_df.loc[mask, "high_value_reason"].astype(str) + "|teacher_high_value=" + thv + "|teacher_manual=" + tman
        )

    for column in schema:
        if column not in final_df.columns:
            final_df[column] = None
    return final_df[schema]


def merge_llm_results(rule_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    return _merge_llm_results_base(rule_df, llm_df, FINAL_SCHEMA)


def merge_content_llm_results(rule_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    return _merge_llm_results_base(rule_df, llm_df, CONTENT_FINAL_SCHEMA)
