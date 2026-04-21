import pandas as pd


def compute_context_dependency_score(row: pd.Series) -> int:
    score = 0
    if row.get("text_length", 0) <= 12:
        score += 2
    elif row.get("text_length", 0) <= 25:
        score += 1
    if row.get("sarcasm_hit_count", 0) > 0:
        score += 1
    if row.get("sarcasm_marker_hit_count", 0) > 0:
        score += 1
    if row.get("abstract_slang_hit_count", 0) > 0:
        score += 1
    if row.get("reference_dependency_hit_count", 0) > 0:
        score += 1
    if row.get("reference_dependency_hit_count", 0) > 0 and row.get("target_specificity_group_count", 0) == 0:
        score += 1
    if row.get("answer_coupling_hit_count", 0) > 0:
        score += 2
    if row.get("answer_coupling_hit_count", 0) >= 2:
        score += 1
    if row.get("community_term_hit_count", 0) > 0 and (
        row.get("sarcasm_hit_count", 0) > 0 or row.get("abstract_slang_hit_count", 0) > 0
    ):
        score += 1
    if row.get("target_specificity_group_count", 0) >= 2:
        score -= 1
    if row.get("causal_hit_count", 0) > 0 and row.get("text_length", 0) >= 40:
        score -= 1
    return max(score, 0)
