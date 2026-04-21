import pandas as pd


def compute_quality_score(row: pd.Series) -> int:
    score = 0
    if row.get("text_length", 0) >= 20:
        score += 1
    if row.get("text_length", 0) >= 80:
        score += 1
    if row.get("text_length", 0) >= 180:
        score += 1
    if row.get("causal_hit_count", 0) > 0:
        score += 1
    if row.get("comparative_hit_count", 0) > 0:
        score += 1
    if row.get("causal_hit_count", 0) > 0 and row.get("comparative_hit_count", 0) > 0:
        score += 1
    if row.get("target_specificity_group_count", 0) >= 1:
        score += 1
    if row.get("target_specificity_group_count", 0) >= 2:
        score += 1
    if row.get("conclusion_hit_count", 0) > 0:
        score += 1
    if row.get("is_low_value", False):
        score -= 2
    if row.get("sarcasm_hit_count", 0) > 0 and row.get("target_specificity_group_count", 0) == 0:
        score -= 1
    if row.get("community_term_hit_count", 0) > 0 and len([mode for mode in str(row.get("feedback_modes", "")).split("|") if mode]) == 1:
        score -= 1
    return max(score, 0)
