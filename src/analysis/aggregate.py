import pandas as pd


def build_platform_summary(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []

    for platform, group in df.groupby("platform"):
        weighted_sentiment = group.groupby("sentiment")["final_weight"].sum().sort_values(ascending=False)
        weighted_dimension = group.groupby("dimension")["final_weight"].sum().sort_values(ascending=False)
        top_sentiment = weighted_sentiment.index[0] if not weighted_sentiment.empty else "neutral"
        top_dimension = weighted_dimension.index[0] if not weighted_dimension.empty else "未知"
        rational_share = round((group["value_level"] == "high").mean() * 100, 2) if not group.empty else 0
        llm_share = round((group["annotation_source"] == "llm").mean() * 100, 2) if "annotation_source" in group.columns else 0

        summaries.append(
            {
                "platform": platform,
                "summary": (
                    f"{platform} 平台在加权后以 {top_sentiment} 情绪为主，"
                    f"最常讨论的维度是 {top_dimension}，"
                    f"高理性评论占比约 {rational_share}%，LLM 深度分析覆盖约 {llm_share}%。"
                ),
            }
        )

    return pd.DataFrame(summaries)


def build_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["platform", "sentiment", "dimension"], dropna=False)
        .agg(
            count=("text", "size"),
            weighted_count=("final_weight", "sum"),
        )
        .reset_index()
        .sort_values(["platform", "weighted_count"], ascending=[True, False])
    )


def build_value_level_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["platform", "value_level"], dropna=False)
        .agg(
            count=("text", "size"),
            weighted_count=("final_weight", "sum"),
        )
        .reset_index()
        .sort_values(["platform", "weighted_count"], ascending=[True, False])
    )
