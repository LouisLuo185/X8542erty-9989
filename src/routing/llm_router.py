import pandas as pd

from config.settings import LLM_ROUTE_MIN_LENGTH, LLM_ROUTE_MIN_PRIORITY


def route_for_llm(df: pd.DataFrame) -> pd.DataFrame:
    routed = df.copy()
    priorities = []
    reasons = []
    needs_flags = []

    for _, row in routed.iterrows():
        score = 0
        route_reasons = []

        if row.get("is_high_value_feedback_pre_llm", False):
            score += 3
            route_reasons.append("actionable_feedback")
        if row.get("text_length", 0) >= LLM_ROUTE_MIN_LENGTH:
            score += 2
            route_reasons.append("long_comment")
        if row.get("context_dependency_score", 0) >= 3:
            score += 2
            route_reasons.append("context_heavy")
        if row.get("feedback_entropy_score", 0) >= 5:
            score += 2
            route_reasons.append("high_entropy")
        if row.get("actionability_score", 0) >= 2:
            score += 2
            route_reasons.append("actionable_pattern")
        if row.get("target_specificity_score", 0) >= 2:
            score += 1
            route_reasons.append("clear_target")
        if "competition" in str(row.get("feedback_modes", "")):
            score += 2
            route_reasons.append("competition_mode")
        if row.get("sarcasm_hit_count", 0) > 0 and row.get("business_value_score", 0) > 0:
            score += 1
            route_reasons.append("sarcasm_with_signal")

        needs_llm = bool(row.get("is_high_value_feedback_pre_llm", False)) and (
            score >= LLM_ROUTE_MIN_PRIORITY or row.get("text_length", 0) >= LLM_ROUTE_MIN_LENGTH
        )

        priorities.append(score)
        reasons.append("|".join(route_reasons) if needs_llm else "")
        needs_flags.append(needs_llm)

    routed["llm_priority_score"] = priorities
    routed["llm_route_reason"] = reasons
    routed["needs_llm_analysis"] = needs_flags
    return routed


def select_llm_candidates(df: pd.DataFrame) -> pd.DataFrame:
    selected = df[df["needs_llm_analysis"]].copy()
    if selected.empty:
        return selected
    return selected.sort_values(
        ["llm_priority_score", "business_value_score", "feedback_entropy_score", "likes"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
