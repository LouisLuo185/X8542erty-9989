import pandas as pd

from src.scoring.business_value_score import (
    compute_actionability_score,
    compute_business_value_score,
    compute_engagement_weight,
    compute_feedback_entropy_score,
    compute_target_specificity_score,
    rule_high_value_feedback,
)
from src.scoring.context_score import compute_context_dependency_score
from src.scoring.quality_score import compute_quality_score


def score_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["target_specificity_score"] = scored.apply(compute_target_specificity_score, axis=1)
    scored["actionability_score"] = scored.apply(compute_actionability_score, axis=1)
    scored["quality_score"] = scored.apply(compute_quality_score, axis=1)
    scored["context_dependency_score"] = scored.apply(compute_context_dependency_score, axis=1)
    scored["business_value_score"] = scored.apply(compute_business_value_score, axis=1)
    scored["feedback_entropy_score"] = scored.apply(compute_feedback_entropy_score, axis=1)
    hv = scored.apply(rule_high_value_feedback, axis=1)
    scored["is_high_value_feedback_pre_llm"] = hv.map(lambda item: item[0])
    scored["high_value_reason"] = hv.map(lambda item: item[1])
    scored["is_actionable_feedback"] = scored["is_high_value_feedback_pre_llm"]
    scored["actionable_reason"] = scored["high_value_reason"].where(scored["is_high_value_feedback_pre_llm"], "")
    scored["engagement_weight"] = scored["likes"].map(compute_engagement_weight)
    return scored
