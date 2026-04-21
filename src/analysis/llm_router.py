import pandas as pd

from config.settings import (
    LLM_ROUTE_MAX_RULE_CONFIDENCE,
    LLM_ROUTE_MIN_LENGTH,
    LLM_ROUTE_MIN_LIKES,
    LLM_ROUTE_MIN_RATIONAL_SCORE,
)


def route_for_llm(df: pd.DataFrame) -> pd.DataFrame:
    routed = df.copy()
    route_reasons: list[str] = []
    scores: list[int] = []
    needs: list[bool] = []

    for _, row in routed.iterrows():
        reasons: list[str] = []
        score = 0

        strong_trigger = False
        soft_trigger_count = 0

        if row.get("text_length", 0) >= LLM_ROUTE_MIN_LENGTH:
            reasons.append("long_comment")
            score += 2
            strong_trigger = True

        if row.get("rational_score", 0) >= LLM_ROUTE_MIN_RATIONAL_SCORE:
            reasons.append("high_rationality")
            score += 2
            strong_trigger = True

        if row.get("value_level") == "high":
            reasons.append("high_value")
            score += 2
            strong_trigger = True

        if bool(row.get("rule_is_comparative", False)):
            reasons.append("competitive_comparison")
            score += 2
            strong_trigger = True

        if float(row.get("likes", 0) or 0) >= LLM_ROUTE_MIN_LIKES:
            reasons.append("high_engagement")
            score += 1
            soft_trigger_count += 1

        if float(row.get("rule_confidence", 1) or 1) <= LLM_ROUTE_MAX_RULE_CONFIDENCE:
            reasons.append("low_rule_confidence")
            score += 1
            soft_trigger_count += 1

        if row.get("positive_hits", 0) > 0 and row.get("negative_hits", 0) > 0:
            reasons.append("mixed_sentiment_signal")
            score += 1
            soft_trigger_count += 1

        needs_llm = strong_trigger or soft_trigger_count >= 2 or score >= 4
        needs.append(needs_llm)
        scores.append(score)
        route_reasons.append(";".join(reasons) if needs_llm else "")

    routed["needs_llm_analysis"] = needs
    routed["llm_priority_score"] = scores
    routed["llm_route_reason"] = route_reasons
    return routed


def select_llm_candidates(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df[df["needs_llm_analysis"]].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["llm_priority_score", "final_weight", "likes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
