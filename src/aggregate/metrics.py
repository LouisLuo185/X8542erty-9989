import pandas as pd

from src.utils.business_modes import normalize_business_modes


def build_feedback_mode_multilabel_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        modes = [mode for mode in str(row.get("feedback_modes", "")).split("|") if mode]
        for mode in modes:
            rows.append(
                {
                    "platform": row["platform"],
                    "feedback_mode": mode,
                    "final_weight": float(pd.to_numeric(row.get("final_weight", 0), errors="coerce") or 0),
                    "is_high_value_feedback": bool(row.get("is_high_value_feedback", False)),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["platform", "feedback_mode", "count", "weighted_count", "high_value_count"]
        )
    mode_df = pd.DataFrame(rows)
    return (
        mode_df.groupby(["platform", "feedback_mode"])
        .agg(
            count=("feedback_mode", "size"),
            weighted_count=("final_weight", "sum"),
            high_value_count=("is_high_value_feedback", "sum"),
        )
        .reset_index()
        .sort_values(["platform", "weighted_count"], ascending=[True, False])
    )


def build_mode_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        normalized = normalize_business_modes(row.get("business_mode", ""), row.get("feedback_modes", ""))
        modes = [mode for mode in str(normalized).split("|") if mode]
        for mode in modes:
            rows.append(
                {
                    "platform": row["platform"],
                    "business_mode": mode,
                    "final_weight": row["final_weight"],
                    "is_high_value_feedback": bool(row["is_high_value_feedback"]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["platform", "business_mode", "count", "weighted_count", "high_value_count"])
    mode_df = pd.DataFrame(rows)
    return (
        mode_df.groupby(["platform", "business_mode"])
        .agg(
            count=("business_mode", "size"),
            weighted_count=("final_weight", "sum"),
            high_value_count=("is_high_value_feedback", "sum"),
        )
        .reset_index()
        .sort_values(["platform", "weighted_count"], ascending=[True, False])
    )


def build_platform_summary(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for platform, group in df.groupby("platform"):
        top_sentiment = group.groupby("sentiment")["final_weight"].sum().sort_values(ascending=False).index[0]
        actionable_share = round(group["is_high_value_feedback"].mean() * 100, 2)
        context_heavy_share = round((group["context_dependency_score"] >= 3).mean() * 100, 2)
        llm_share = round((group["annotation_source"] == "llm").mean() * 100, 2)
        summaries.append(
            {
                "platform": platform,
                "summary": (
                    f"{platform} 当前加权主情绪为 {top_sentiment}，"
                    f"高价值反馈占比约 {actionable_share}%，"
                    f"强上下文依赖评论占比约 {context_heavy_share}%，"
                    f"LLM 深度分析覆盖约 {llm_share}%。"
                ),
            }
        )
    return pd.DataFrame(summaries)
