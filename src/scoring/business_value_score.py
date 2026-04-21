import math

import pandas as pd


def compute_target_specificity_score(row: pd.Series) -> int:
    score = 0
    if row.get("target_specificity_group_count", 0) >= 1:
        score += 1
    if row.get("target_specificity_group_count", 0) >= 2:
        score += 1
    if row.get("canonical_entity_hit_count", 0) >= 1 or row.get("target_specificity_term_hit_count", 0) >= 3:
        score += 1
    return min(score, 3)


def compute_actionability_score(row: pd.Series) -> int:
    score = 0
    if row.get("impact_hit_count", 0) > 0:
        score += 1
    if row.get("actionability_term_hit_count", 0) > 0:
        score += 1
    if row.get("causal_hit_count", 0) > 0 and row.get("target_specificity_score", 0) >= 1:
        score += 1
    modes = set(filter(None, str(row.get("feedback_modes", "")).split("|")))
    if row.get("impact_hit_count", 0) > 0 and modes & {
        "product_experience",
        "monetization",
        "brand_communication",
        "competition",
    }:
        score += 1
    return min(score, 4)


def compute_business_value_score(row: pd.Series) -> int:
    score = 0
    if row.get("product_term_hit_count", 0) > 0:
        score += 2
    if row.get("monetization_term_hit_count", 0) > 0:
        score += 2
    if row.get("brand_term_hit_count", 0) > 0:
        score += 2
    if row.get("competitor_term_hit_count", 0) > 0:
        score += 2
    if row.get("plot_term_hit_count", 0) > 0:
        score += 1
    if row.get("community_term_hit_count", 0) > 0:
        score += 1
    if row.get("target_specificity_group_count", 0) >= 1:
        score += 1
    if row.get("target_specificity_group_count", 0) >= 2:
        score += 1
    if row.get("impact_hit_count", 0) > 0:
        score += 2
    if row.get("actionability_term_hit_count", 0) > 0:
        score += 1
    likes = float(row.get("likes", 0) or 0)
    if likes >= 20:
        score += 1
    if likes >= 100:
        score += 1

    modes = set(filter(None, str(row.get("feedback_modes", "")).split("|")))
    if modes == {"community_conflict"} and row.get("target_specificity_group_count", 0) == 0:
        score -= 2
    if row.get("abstract_slang_hit_count", 0) > 0 and score <= 2 and row.get("target_specificity_group_count", 0) == 0:
        score -= 2
    if row.get("answer_coupling_hit_count", 0) > 0 and not (
        modes & {"product_experience", "monetization", "brand_communication", "competition", "plot_discussion"}
    ):
        score -= 1
    return max(score, 0)


def compute_feedback_entropy_score(row: pd.Series) -> float:
    score = (
        row.get("quality_score", 0) * 0.45
        + row.get("target_specificity_score", 0) * 0.8
        + row.get("actionability_score", 0) * 0.9
        + min(row.get("business_value_score", 0), 10) * 0.3
    )
    if row.get("context_dependency_score", 0) >= 5:
        score -= 1.0
    elif row.get("context_dependency_score", 0) >= 3:
        score -= 0.5
    if row.get("sarcasm_hit_count", 0) > 0 and row.get("business_value_score", 0) == 0:
        score -= 1.5
    return round(max(score, 0), 2)


def rule_high_value_feedback(row: pd.Series) -> tuple[bool, str]:
    modes = set(filter(None, str(row.get("feedback_modes", "")).split("|")))
    reasons: list[str] = []

    qs = int(row.get("quality_score", 0) or 0)
    bv = int(row.get("business_value_score", 0) or 0)
    action = int(row.get("actionability_score", 0) or 0)
    target = int(row.get("target_specificity_score", 0) or 0)
    context = int(row.get("context_dependency_score", 0) or 0)
    abstract = int(row.get("abstract_slang_hit_count", 0) or 0)
    community = int(row.get("community_term_hit_count", 0) or 0)

    if row.get("text_length", 0) < 10:
        return False, "excluded_short_vent"
    if abstract >= 2 and community > 0 and target == 0 and action == 0:
        return False, "excluded_abstract_conflict_only"
    if modes == {"community_conflict"} and target == 0 and action == 0:
        return False, "excluded_community_flame_only"

    hit = False
    if bv >= 6 and qs >= 4:
        hit = True
        reasons.append("business_value_and_quality")
    if action >= 2 and modes & {"product_experience", "monetization", "brand_communication", "competition"}:
        hit = True
        reasons.append("actionable_business_feedback")
    if "plot_discussion" in modes and qs >= 6 and target >= 1:
        hit = True
        reasons.append("high_quality_plot_feedback")
    if "competition" in modes and target >= 1 and bv >= 5:
        hit = True
        reasons.append("competition_signal")
    if context >= 4 and action >= 2 and bv >= 5:
        hit = True
        reasons.append("context_heavy_but_actionable")

    if not hit:
        return False, ""

    deduped: list[str] = []
    seen: set[str] = set()
    for item in reasons:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return True, "|".join(deduped)


def is_actionable_feedback(row: pd.Series) -> tuple[bool, str]:
    return rule_high_value_feedback(row)


def compute_engagement_weight(likes: float | int | str | None) -> float:
    try:
        numeric_likes = float(likes or 0)
    except (TypeError, ValueError):
        numeric_likes = 0.0
    return round(1 + math.log1p(max(numeric_likes, 0)) / 5, 4)
