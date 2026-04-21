import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import DATA_DEMO_DIR, DATA_PROCESSED_DIR, MANIFEST_DIR


LABELED_COMMENT_PATH = DATA_PROCESSED_DIR / "labeled_final.csv"
CONTENT_REPORT_PATH = DATA_PROCESSED_DIR / "content_feedback_report.csv"
CONTENT_FINAL_PATH = DATA_PROCESSED_DIR / "labeled_content_final.csv"
CONTENT_RULE_PATH = DATA_PROCESSED_DIR / "labeled_content_rule.csv"
COMMENT_REACTION_PATH = DATA_PROCESSED_DIR / "comment_reaction_summary.csv"
SUMMARY_PATH = DATA_PROCESSED_DIR / "platform_summary.csv"
MODES_PATH = DATA_PROCESSED_DIR / "business_modes.csv"
MULTILABEL_PATH = DATA_PROCESSED_DIR / "feedback_mode_multilabel.csv"
DEMO_COMMENT_PATH = DATA_DEMO_DIR / "demo_comment_report.csv"
DEMO_CONTENT_REPORT_PATH = DATA_DEMO_DIR / "demo_content_report.csv"
DEMO_CONTENT_FINAL_PATH = DATA_DEMO_DIR / "demo_content_report.csv"
DEMO_COMMENT_REACTION_PATH = DATA_DEMO_DIR / "demo_comment_reaction_summary.csv"
DEMO_SUMMARY_PATH = DATA_DEMO_DIR / "demo_platform_summary.csv"
DEMO_MODES_PATH = DATA_DEMO_DIR / "demo_business_modes.csv"
DEMO_MULTILABEL_PATH = DATA_DEMO_DIR / "demo_feedback_multilabel.csv"
DEMO_MANIFEST_PATH = DATA_DEMO_DIR / "demo_pipeline_manifest.csv"

STAGE_LABELS = {
    "ingest": "原始数据接入",
    "threads": "评论线程构建",
    "content_base": "回答主表构建",
    "clean_features": "评论清洗",
    "content_clean_features": "回答清洗",
    "score_label": "评论规则评分",
    "content_score_label": "回答规则评分",
    "route": "评论 LLM 路由",
    "content_route": "回答 LLM 路由",
    "llm_merge": "评论 LLM 结果回填",
    "content_llm_merge": "回答 LLM 结果回填",
    "aggregate": "聚合统计",
    "funnel": "整体漏斗汇总",
    "labeled_final": "评论最终样本表",
    "needs_llm_analysis": "进入 LLM 候选池",
    "annotation_llm": "已完成 LLM 标注",
}

BUSINESS_MODE_LABELS = {
    "product_experience": "产品体验",
    "monetization": "商业化",
    "brand_communication": "品牌传播",
    "community_conflict": "社区冲突",
    "competition": "竞品比较",
    "plot_discussion": "剧情内容",
    "other_high_value": "其他高价值",
}

COMMENT_TARGET_LABELS = {
    "game": "游戏本体",
    "answer": "回答观点",
    "community": "玩家/社区",
    "mixed": "混合对象",
    "unclear": "不明确",
}

SENTIMENT_LABELS = {
    "positive": "正向",
    "neutral": "中性",
    "negative": "负向",
    "mixed": "复杂/混合",
}

PLATFORM_LABELS = {"zhihu": "知乎"}

ANNOTATION_SOURCE_LABELS = {
    "llm": "LLM 标注",
    "rule": "规则标注",
}

ROUTE_REASON_LABELS = {
    "actionable_feedback": "高价值反馈",
    "long_comment": "长评论",
    "context_heavy": "强上下文依赖",
    "high_entropy": "高信息熵",
    "actionable_pattern": "具备可行动性",
    "clear_target": "对象明确",
    "competition_mode": "竞品比较",
    "sarcasm_with_signal": "反讽但有业务信号",
}

COLUMN_LABELS = {
    "stage": "阶段",
    "rows": "行数",
    "seconds": "耗时（秒）",
    "platform": "平台",
    "business_mode": "业务模式",
    "feedback_mode": "反馈模式",
    "weighted_count": "加权数量",
    "count": "评论数",
    "high_value_count": "高价值评论数",
    "question_title": "问题标题",
    "answer_title": "回答标题",
    "answer_summary": "回答摘要",
    "clean_text": "文本",
    "feedback_entropy_score": "信息密度分",
    "business_value_score": "业务价值分",
    "quality_score": "质量分",
    "context_dependency_score": "上下文依赖分",
    "high_value_reason": "高价值原因",
    "llm_priority_score": "LLM 优先级",
    "llm_route_reason": "LLM 路由原因",
    "needs_llm_analysis": "是否进入 LLM",
    "sentiment": "情绪倾向",
    "confidence": "置信度",
    "summary": "摘要",
    "comment_count": "评论数",
    "source_comment_count": "原始评论数",
    "reaction_comment_count": "反应评论数",
    "supportive_ratio": "支持占比（%）",
    "opposing_ratio": "反对占比（%）",
    "controversy_score": "争议度（%）",
    "sarcasm_ratio": "反讽占比（%）",
    "high_value_comment_count": "高价值评论数",
    "avg_comment_business_value_score": "评论平均业务价值分",
    "avg_comment_entropy_score": "评论平均信息密度分",
}


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def resolve_path(primary: Path, demo: Path) -> tuple[Path, bool]:
    if primary.exists():
        return primary, False
    return demo, demo.exists()


def load_manifests() -> pd.DataFrame:
    if not MANIFEST_DIR.exists():
        return load_dataframe(DEMO_MANIFEST_PATH)
    rows = []
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                rows.append(json.load(handle))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if rows:
        return pd.DataFrame(rows)
    return load_dataframe(DEMO_MANIFEST_PATH)


def map_series(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.fillna("").astype(str).map(lambda value: mapping.get(value, value))


def map_pipe_text(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    def _convert(text: object) -> str:
        parts = [part for part in str(text or "").split("|") if part]
        return "、".join(mapping.get(part, part) for part in parts)

    return series.map(_convert)


def localize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    if "platform" in localized.columns:
        localized["platform"] = map_series(localized["platform"], PLATFORM_LABELS)
    if "stage" in localized.columns:
        localized["stage"] = map_series(localized["stage"], STAGE_LABELS)
    if "business_mode" in localized.columns:
        localized["business_mode"] = map_pipe_text(localized["business_mode"], BUSINESS_MODE_LABELS)
    if "feedback_mode" in localized.columns:
        localized["feedback_mode"] = map_series(localized["feedback_mode"], BUSINESS_MODE_LABELS)
    if "comment_target" in localized.columns:
        localized["comment_target"] = map_series(localized["comment_target"], COMMENT_TARGET_LABELS)
    if "sentiment" in localized.columns:
        localized["sentiment"] = map_series(localized["sentiment"], SENTIMENT_LABELS)
    if "annotation_source" in localized.columns:
        localized["annotation_source"] = map_series(localized["annotation_source"], ANNOTATION_SOURCE_LABELS)
    if "llm_route_reason" in localized.columns:
        localized["llm_route_reason"] = map_pipe_text(localized["llm_route_reason"], ROUTE_REASON_LABELS)
    return localized


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: COLUMN_LABELS.get(column, column) for column in df.columns})


def render_overview(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader("整体概览")
    cols = st.columns(5)
    cols[0].metric("回答总数", len(content_df))
    cols[1].metric(
        "高价值回答",
        int(content_df["is_high_value_feedback_pre_llm"].sum()) if "is_high_value_feedback_pre_llm" in content_df.columns else 0,
    )
    cols[2].metric(
        "回答 LLM 候选",
        int(content_df["needs_llm_analysis"].sum()) if "needs_llm_analysis" in content_df.columns else 0,
    )
    cols[3].metric("评论总数", len(comment_df))
    cols[4].metric(
        "高价值评论",
        int(comment_df["is_high_value_feedback"].sum()) if "is_high_value_feedback" in comment_df.columns else 0,
    )


def render_pipeline_funnel(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader("管线漏斗")
    manifest_df = load_manifests()
    if not manifest_df.empty and "stage" in manifest_df.columns:
        display_df = localize_common_fields(manifest_df[["stage", "rows", "seconds"]])
        st.dataframe(rename_columns(display_df), use_container_width=True, hide_index=True)

    funnel = pd.DataFrame(
        {
            "阶段": ["回答总样本", "高价值回答", "回答 LLM 候选", "评论总样本", "评论 LLM 候选"],
            "数量": [
                len(content_df),
                int(content_df["is_high_value_feedback_pre_llm"].sum()) if "is_high_value_feedback_pre_llm" in content_df.columns else 0,
                int(content_df["needs_llm_analysis"].sum()) if "needs_llm_analysis" in content_df.columns else 0,
                len(comment_df),
                int(comment_df["needs_llm_analysis"].sum()) if "needs_llm_analysis" in comment_df.columns else 0,
            ],
        }
    )
    fig = px.bar(funnel, x="阶段", y="数量", title="回答与评论双视角漏斗")
    st.plotly_chart(fig, use_container_width=True)


def render_summary(summary_df: pd.DataFrame) -> None:
    st.subheader("平台摘要")
    if summary_df.empty:
        return
    display_df = localize_common_fields(summary_df)
    for _, row in display_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['platform']}")
            st.write(row["summary"])


def render_content_modes(content_df: pd.DataFrame) -> None:
    st.subheader("回答视角：业务模式分布")
    if content_df.empty or "feedback_modes" not in content_df.columns:
        st.info("暂无回答业务模式统计。")
        return
    chart_df = content_df.copy()
    chart_df["business_mode"] = map_pipe_text(chart_df["feedback_modes"], BUSINESS_MODE_LABELS)
    exploded = chart_df.assign(business_mode=chart_df["business_mode"].str.split("、")).explode("business_mode")
    exploded = exploded[exploded["business_mode"].astype(str).str.len() > 0]
    summary = (
        exploded.groupby("business_mode")
        .agg(回答数=("answer_id", "count"), 加权数量=("final_weight", "sum"))
        .reset_index()
        .sort_values("加权数量", ascending=False)
    )
    fig = px.bar(summary, x="business_mode", y="加权数量", title="回答侧业务模式（加权）")
    fig.update_xaxes(title="业务模式")
    st.plotly_chart(fig, use_container_width=True)


def render_content_scatter(content_df: pd.DataFrame) -> None:
    st.subheader("回答视角：信息密度与业务价值")
    if content_df.empty:
        return
    sample = content_df.sort_values("feedback_entropy_score", ascending=False).head(300)
    fig = px.scatter(
        sample,
        x="feedback_entropy_score",
        y="business_value_score",
        color="is_high_value_feedback_pre_llm",
        hover_data=["question_title", "answer_summary"],
        title="回答的信息密度与业务价值分布",
    )
    fig.update_xaxes(title="信息密度分")
    fig.update_yaxes(title="业务价值分")
    st.plotly_chart(fig, use_container_width=True)


def render_top_contents(content_report_df: pd.DataFrame, content_df: pd.DataFrame) -> None:
    st.subheader("回答视角：高价值回答 Top 15")
    source_df = content_report_df if not content_report_df.empty else content_df
    if source_df.empty:
        st.info("暂无回答主视角结果。")
        return
    ranked = source_df.sort_values(
        ["is_high_value_feedback_pre_llm", "business_value_score", "feedback_entropy_score", "likes"],
        ascending=[False, False, False, False],
    ).head(15)
    cols = [
        "question_title",
        "answer_summary",
        "feedback_modes",
        "high_value_reason",
        "business_value_score",
        "feedback_entropy_score",
        "likes",
        "reaction_comment_count",
        "supportive_ratio",
        "opposing_ratio",
        "controversy_score",
    ]
    display_df = ranked[[c for c in cols if c in ranked.columns]].copy()
    if "feedback_modes" in display_df.columns:
        display_df["feedback_modes"] = map_pipe_text(display_df["feedback_modes"], BUSINESS_MODE_LABELS)
    st.dataframe(rename_columns(localize_common_fields(display_df)), use_container_width=True, hide_index=True)


def render_comment_reactions(comment_summary_df: pd.DataFrame) -> None:
    st.subheader("回答视角：评论辅助验证")
    if comment_summary_df.empty:
        st.info("暂无评论反应汇总。")
        return
    st.dataframe(rename_columns(localize_common_fields(comment_summary_df.head(20))), use_container_width=True, hide_index=True)


def render_comment_modes(mode_df: pd.DataFrame, multilabel_df: pd.DataFrame) -> None:
    st.subheader("评论视角：业务模式分布")
    if not mode_df.empty:
        chart_df = localize_common_fields(mode_df.copy())
        order = chart_df.groupby("business_mode")["weighted_count"].sum().sort_values(ascending=False).index.tolist()
        fig = px.bar(
            chart_df,
            x="business_mode",
            y="weighted_count",
            color="platform",
            barmode="group",
            category_orders={"business_mode": order},
            title="评论侧业务模式（加权）",
        )
        fig.update_xaxes(title="业务模式")
        fig.update_yaxes(title="加权数量")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无评论业务模式统计。")

    st.subheader("评论视角：多标签反馈模式")
    if not multilabel_df.empty:
        chart_df = localize_common_fields(multilabel_df.copy())
        fig = px.bar(
            chart_df,
            x="feedback_mode",
            y="weighted_count",
            color="platform",
            barmode="group",
            title="评论侧多标签反馈模式（加权）",
        )
        fig.update_xaxes(title="反馈模式")
        fig.update_yaxes(title="加权数量")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无评论多标签统计。")


def render_comment_entropy(comment_df: pd.DataFrame) -> None:
    st.subheader("评论视角：信息密度与业务价值")
    if comment_df.empty:
        return
    sample = comment_df.sort_values("feedback_entropy_score", ascending=False).head(300)
    fig = px.scatter(
        sample,
        x="feedback_entropy_score",
        y="business_value_score",
        color="is_high_value_feedback",
        hover_data=["question_title", "clean_text"],
        title="评论的信息密度与业务价值分布",
    )
    fig.update_xaxes(title="信息密度分")
    fig.update_yaxes(title="业务价值分")
    st.plotly_chart(fig, use_container_width=True)

    melted = comment_df.melt(
        id_vars=["comment_id"],
        value_vars=["quality_score", "context_dependency_score", "business_value_score"],
        var_name="score_name",
        value_name="value",
    )
    score_labels = {
        "quality_score": "质量分",
        "context_dependency_score": "上下文依赖分",
        "business_value_score": "业务价值分",
    }
    melted["score_name"] = map_series(melted["score_name"], score_labels)
    fig2 = px.box(melted, x="score_name", y="value", color="score_name", title="评论三分分布")
    fig2.update_xaxes(title="评分类型")
    fig2.update_yaxes(title="分值")
    st.plotly_chart(fig2, use_container_width=True)


def render_comment_route(comment_df: pd.DataFrame) -> None:
    st.subheader("评论视角：LLM 路由")
    if comment_df.empty:
        return
    reason_keys = sorted({r for cell in comment_df["llm_route_reason"].fillna("") for r in str(cell).split("|") if r})
    selected = st.multiselect(
        "按评论路由原因过滤",
        options=reason_keys,
        format_func=lambda item: ROUTE_REASON_LABELS.get(item, item),
        default=[],
    )
    filtered = comment_df.copy()
    if selected:
        mask = filtered["llm_route_reason"].fillna("").apply(lambda text: all(item in str(text) for item in selected))
        filtered = filtered[mask]
    st.metric("命中评论数", len(filtered))
    cols = [
        "question_title",
        "clean_text",
        "high_value_reason",
        "llm_priority_score",
        "llm_route_reason",
        "needs_llm_analysis",
        "quality_score",
        "context_dependency_score",
        "business_value_score",
    ]
    display_df = filtered[[c for c in cols if c in filtered.columns]].head(200)
    st.dataframe(rename_columns(localize_common_fields(display_df)), use_container_width=True, hide_index=True)


def render_comment_examples(comment_df: pd.DataFrame) -> None:
    st.subheader("评论视角：样本")
    if comment_df.empty:
        return

    st.markdown("**强上下文依赖评论**")
    context_sample = comment_df[comment_df["context_dependency_score"] >= 3].head(10)
    cols = ["question_title", "answer_summary", "clean_text", "comment_target", "business_mode", "feedback_entropy_score"]
    display_df = context_sample[[c for c in cols if c in context_sample.columns]]
    st.dataframe(rename_columns(localize_common_fields(display_df)), use_container_width=True, hide_index=True)

    st.markdown("**高价值评论**")
    sort_cols = [c for c in ["business_value_score", "feedback_entropy_score", "likes"] if c in comment_df.columns]
    high_value_sample = comment_df[comment_df["is_high_value_feedback"]].sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(15)
    cols2 = ["question_title", "clean_text", "business_mode", "high_value_reason", "comment_target", "sentiment", "confidence"]
    display_df2 = high_value_sample[[c for c in cols2 if c in high_value_sample.columns]]
    st.dataframe(rename_columns(localize_common_fields(display_df2)), use_container_width=True, hide_index=True)


def render_export(content_report_df: pd.DataFrame, content_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader("导出")
    export_view = st.radio("导出视角", ["回答视角", "评论视角"], horizontal=True)
    if export_view == "回答视角":
        subset = content_report_df.copy() if not content_report_df.empty else content_df.copy()
        only_high = st.checkbox("仅导出高价值回答", value=True)
        if only_high and "is_high_value_feedback_pre_llm" in subset.columns:
            subset = subset[subset["is_high_value_feedback_pre_llm"]]
        st.download_button(
            "下载回答视角 CSV",
            data=subset.to_csv(index=False).encode("utf-8-sig"),
            file_name="content_feedback_report.csv",
            mime="text/csv",
        )
        st.caption(f"当前导出行数：{len(subset)}")
    else:
        subset = comment_df.copy()
        only_high = st.checkbox("仅导出高价值评论", value=True)
        only_llm = st.checkbox("仅导出 LLM 候选评论", value=False)
        if only_high and "is_high_value_feedback" in subset.columns:
            subset = subset[subset["is_high_value_feedback"]]
        if only_llm and "needs_llm_analysis" in subset.columns:
            subset = subset[subset["needs_llm_analysis"]]
        st.download_button(
            "下载评论视角 CSV",
            data=subset.to_csv(index=False).encode("utf-8-sig"),
            file_name="comment_feedback_report.csv",
            mime="text/csv",
        )
        st.caption(f"当前导出行数：{len(subset)}")


def main() -> None:
    st.set_page_config(page_title="回答与评论双视角控制台", layout="wide")
    st.title("知乎《原神》回答与评论双视角反馈控制台")
    st.caption("回答视角负责识别主观点与高价值内容；评论视角负责验证社区反应、争议与噪声。")

    comment_path, comment_demo_mode = resolve_path(LABELED_COMMENT_PATH, DEMO_COMMENT_PATH)
    content_report_path, content_report_demo_mode = resolve_path(CONTENT_REPORT_PATH, DEMO_CONTENT_REPORT_PATH)
    content_final_path, content_final_demo_mode = resolve_path(CONTENT_FINAL_PATH, DEMO_CONTENT_FINAL_PATH)
    content_rule_path, content_rule_demo_mode = resolve_path(CONTENT_RULE_PATH, DEMO_CONTENT_FINAL_PATH)
    comment_reaction_path, comment_reaction_demo_mode = resolve_path(COMMENT_REACTION_PATH, DEMO_COMMENT_REACTION_PATH)
    summary_path, summary_demo_mode = resolve_path(SUMMARY_PATH, DEMO_SUMMARY_PATH)
    mode_path, mode_demo_mode = resolve_path(MODES_PATH, DEMO_MODES_PATH)
    multilabel_path, multilabel_demo_mode = resolve_path(MULTILABEL_PATH, DEMO_MULTILABEL_PATH)

    demo_mode = any(
        [
            comment_demo_mode,
            content_report_demo_mode,
            content_final_demo_mode,
            content_rule_demo_mode,
            comment_reaction_demo_mode,
            summary_demo_mode,
            mode_demo_mode,
            multilabel_demo_mode,
        ]
    )

    if demo_mode:
        st.info("当前为公开演示数据模式：页面使用脱敏与改写后的 demo 数据，不包含平台原始文本。")

    comment_df = load_dataframe(comment_path)
    content_report_df = load_dataframe(content_report_path)
    content_df = load_dataframe(content_final_path)
    if content_df.empty:
        content_df = load_dataframe(content_rule_path)
    comment_summary_df = load_dataframe(comment_reaction_path)
    summary_df = load_dataframe(summary_path)
    mode_df = load_dataframe(mode_path)
    multilabel_df = load_dataframe(multilabel_path)

    if comment_df.empty and content_df.empty:
        st.warning("尚未发现处理结果，请先运行 python -m src.pipeline.run_pipeline --stage pre_llm 或 full。")
        return

    tabs = st.tabs(["概览", "管线漏斗", "回答视角", "评论视角", "导出"])

    with tabs[0]:
        render_overview(content_df, comment_df)
        render_summary(summary_df)

    with tabs[1]:
        render_pipeline_funnel(content_df, comment_df)

    with tabs[2]:
        render_content_modes(content_df)
        render_content_scatter(content_df)
        render_top_contents(content_report_df, content_df)
        render_comment_reactions(comment_summary_df)

    with tabs[3]:
        render_comment_modes(mode_df, multilabel_df)
        render_comment_entropy(comment_df)
        render_comment_route(comment_df)
        render_comment_examples(comment_df)

    with tabs[4]:
        render_export(content_report_df, content_df, comment_df)


if __name__ == "__main__":
    main()
