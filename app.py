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

UI_LANG = "zh"

TEXTS = {
    "title": {
        "zh": "中文长文本语料筛选与 LLM 弱监督标注控制台",
        "en": "Chinese Long-Text Filtering and LLM Weak-Supervision Dashboard",
    },
    "caption": {
        "zh": "展示从原始社区文本到训练候选样本、Hard Sample 路由和结构化弱监督标签的完整 pipeline。",
        "en": "Shows the full pipeline from raw community text to training candidates, hard-sample routing, and structured weak-supervision labels.",
    },
    "demo_notice": {
        "zh": "线上演示因版权与公开展示限制，不展示完整原始社区文本、用户标识或可回溯链接；当前页面使用脱敏与改写后的 demo 数据。",
        "en": "Due to copyright and public-display restrictions, the online demo does not expose full original community text, user identifiers, or traceable links; the current page uses sanitized and rewritten demo data.",
    },
    "missing_warning": {
        "zh": "尚未发现处理结果，请先运行 python -m src.pipeline.run_pipeline --stage pre_llm 或 full。",
        "en": "No processed results were found. Please run python -m src.pipeline.run_pipeline --stage pre_llm or full first.",
    },
    "tab_overview": {"zh": "概览", "en": "Overview"},
    "tab_training": {"zh": "训练数据视角", "en": "Training View"},
    "tab_funnel": {"zh": "管线漏斗", "en": "Pipeline Funnel"},
    "tab_content": {"zh": "回答视角", "en": "Content View"},
    "tab_comment": {"zh": "评论视角", "en": "Comment View"},
    "tab_export": {"zh": "导出", "en": "Export"},
    "lang_label": {"zh": "语言 / Language", "en": "Language / 语言"},
    "sub_overview": {"zh": "整体概览", "en": "Overview Metrics"},
    "metric_content_total": {"zh": "回答总数", "en": "Content Rows"},
    "metric_content_high": {"zh": "高价值回答", "en": "High-Value Content"},
    "metric_content_llm": {"zh": "回答 LLM 候选", "en": "Content Routed to LLM"},
    "metric_comment_total": {"zh": "评论总数", "en": "Comment Rows"},
    "metric_comment_high": {"zh": "高价值评论", "en": "High-Value Comments"},
    "sub_training_position": {"zh": "训练数据项目定位", "en": "Training-Data Positioning"},
    "sub_training_metrics": {"zh": "训练候选指标", "en": "Training-Candidate Metrics"},
    "sub_training_scatter": {"zh": "训练样本质量分布", "en": "Training Sample Distribution"},
    "sub_training_route": {"zh": "Hard Sample 路由视角", "en": "Hard-Sample Routing"},
    "sub_training_examples": {"zh": "训练候选样本示例", "en": "Training Candidate Examples"},
    "sub_funnel": {"zh": "管线漏斗", "en": "Pipeline Funnel"},
    "sub_platform_summary": {"zh": "平台摘要", "en": "Platform Summary"},
    "sub_content_modes": {"zh": "回答视角：业务模式分布", "en": "Content View: Mode Distribution"},
    "sub_content_scatter": {"zh": "回答视角：信息密度与业务价值", "en": "Content View: Density vs. Value"},
    "sub_content_top": {"zh": "回答视角：高价值回答 Top 15", "en": "Content View: Top High-Value Samples"},
    "sub_comment_reaction": {"zh": "回答视角：评论辅助验证", "en": "Content View: Comment Validation"},
    "sub_comment_modes": {"zh": "评论视角：业务模式分布", "en": "Comment View: Mode Distribution"},
    "sub_comment_multilabel": {"zh": "评论视角：多标签反馈模式", "en": "Comment View: Multilabel Modes"},
    "sub_comment_entropy": {"zh": "评论视角：信息密度与业务价值", "en": "Comment View: Density vs. Value"},
    "sub_comment_route": {"zh": "评论视角：LLM 路由", "en": "Comment View: LLM Routing"},
    "sub_comment_examples": {"zh": "评论视角：样本", "en": "Comment View: Sample Cases"},
    "sub_export": {"zh": "导出", "en": "Export"},
}


def tr(key: str) -> str:
    return TEXTS.get(key, {}).get(UI_LANG, key)

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
STAGE_LABELS_EN = {
    "ingest": "Raw Ingestion",
    "threads": "Thread Construction",
    "content_base": "Content Base Table",
    "clean_features": "Comment Cleaning",
    "content_clean_features": "Content Cleaning",
    "score_label": "Comment Rule Scoring",
    "content_score_label": "Content Rule Scoring",
    "route": "Comment LLM Routing",
    "content_route": "Content LLM Routing",
    "llm_merge": "Comment LLM Merge",
    "content_llm_merge": "Content LLM Merge",
    "aggregate": "Aggregation",
    "funnel": "Overall Funnel",
    "labeled_final": "Final Comment Table",
    "needs_llm_analysis": "LLM Candidate Pool",
    "annotation_llm": "LLM Annotation Done",
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
BUSINESS_MODE_LABELS_EN = {
    "product_experience": "Product Experience",
    "monetization": "Monetization",
    "brand_communication": "Brand Communication",
    "community_conflict": "Community Conflict",
    "competition": "Competition",
    "plot_discussion": "Narrative / Plot",
    "other_high_value": "Other High Value",
}

COMMENT_TARGET_LABELS = {
    "game": "游戏本体",
    "answer": "回答观点",
    "community": "玩家/社区",
    "mixed": "混合对象",
    "unclear": "不明确",
}
COMMENT_TARGET_LABELS_EN = {
    "game": "Game",
    "answer": "Answer",
    "community": "Community",
    "mixed": "Mixed Target",
    "unclear": "Unclear",
}

SENTIMENT_LABELS = {
    "positive": "正向",
    "neutral": "中性",
    "negative": "负向",
    "mixed": "复杂/混合",
}
SENTIMENT_LABELS_EN = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "mixed": "Mixed",
}

PLATFORM_LABELS = {"zhihu": "知乎"}
PLATFORM_LABELS_EN = {"zhihu": "Zhihu"}

ANNOTATION_SOURCE_LABELS = {
    "llm": "LLM 标注",
    "rule": "规则标注",
}
ANNOTATION_SOURCE_LABELS_EN = {
    "llm": "LLM",
    "rule": "Rule",
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
ROUTE_REASON_LABELS_EN = {
    "actionable_feedback": "Actionable Feedback",
    "long_comment": "Long Text",
    "context_heavy": "Context Heavy",
    "high_entropy": "High Information Density",
    "actionable_pattern": "Actionable Pattern",
    "clear_target": "Clear Target",
    "competition_mode": "Competition",
    "sarcasm_with_signal": "Sarcasm with Signal",
}

TRAINING_LABELS = {
    "keep_for_training": "保留为训练候选",
    "drop_as_noise": "噪声样本丢弃",
    "needs_context": "需要上下文补全",
    "hard_sample_for_llm": "LLM 难样本",
    "review_required": "需要人工复核",
}
TRAINING_LABELS_EN = {
    "keep_for_training": "Keep for Training",
    "drop_as_noise": "Drop as Noise",
    "needs_context": "Needs Context",
    "hard_sample_for_llm": "Hard Sample for LLM",
    "review_required": "Review Required",
}

MODEL_TARGET_LABELS = {
    "quality_filter_model": "质量过滤模型",
    "sample_router_model": "样本路由模型",
    "context_dependency_model": "上下文依赖检测模型",
    "weak_label_classifier": "弱监督标签分类器",
}
MODEL_TARGET_LABELS_EN = {
    "quality_filter_model": "Quality Filter Model",
    "sample_router_model": "Sample Router Model",
    "context_dependency_model": "Context Dependency Model",
    "weak_label_classifier": "Weak Label Classifier",
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
    "training_data_label": "训练数据标签",
    "training_data_reason": "训练标签原因",
    "training_sample_score": "训练样本分",
    "model_target": "适合训练的模型",
}
COLUMN_LABELS_EN = {
    "stage": "Stage",
    "rows": "Rows",
    "seconds": "Seconds",
    "platform": "Platform",
    "business_mode": "Business Mode",
    "feedback_mode": "Feedback Mode",
    "weighted_count": "Weighted Count",
    "count": "Count",
    "high_value_count": "High-Value Count",
    "question_title": "Case Title",
    "answer_title": "Answer Title",
    "answer_summary": "Summary",
    "clean_text": "Text",
    "feedback_entropy_score": "Information Density Score",
    "business_value_score": "Value Score",
    "quality_score": "Quality Score",
    "context_dependency_score": "Context Dependency Score",
    "high_value_reason": "Reason",
    "llm_priority_score": "LLM Priority Score",
    "llm_route_reason": "LLM Routing Reason",
    "needs_llm_analysis": "Needs LLM",
    "sentiment": "Sentiment",
    "confidence": "Confidence",
    "summary": "Summary",
    "comment_count": "Comment Count",
    "source_comment_count": "Source Comment Count",
    "reaction_comment_count": "Reaction Comment Count",
    "supportive_ratio": "Supportive Ratio (%)",
    "opposing_ratio": "Opposing Ratio (%)",
    "controversy_score": "Controversy Score (%)",
    "sarcasm_ratio": "Sarcasm Ratio (%)",
    "high_value_comment_count": "High-Value Comment Count",
    "avg_comment_business_value_score": "Avg Comment Value Score",
    "avg_comment_entropy_score": "Avg Comment Density Score",
    "training_data_label": "Training Label",
    "training_data_reason": "Training Label Reason",
    "training_sample_score": "Training Sample Score",
    "model_target": "Model Target",
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
    stage_map = STAGE_LABELS_EN if UI_LANG == "en" else STAGE_LABELS
    business_mode_map = BUSINESS_MODE_LABELS_EN if UI_LANG == "en" else BUSINESS_MODE_LABELS
    feedback_mode_map = business_mode_map
    comment_target_map = COMMENT_TARGET_LABELS_EN if UI_LANG == "en" else COMMENT_TARGET_LABELS
    sentiment_map = SENTIMENT_LABELS_EN if UI_LANG == "en" else SENTIMENT_LABELS
    platform_map = PLATFORM_LABELS_EN if UI_LANG == "en" else PLATFORM_LABELS
    annotation_map = ANNOTATION_SOURCE_LABELS_EN if UI_LANG == "en" else ANNOTATION_SOURCE_LABELS
    route_map = ROUTE_REASON_LABELS_EN if UI_LANG == "en" else ROUTE_REASON_LABELS
    training_map = TRAINING_LABELS_EN if UI_LANG == "en" else TRAINING_LABELS
    model_target_map = MODEL_TARGET_LABELS_EN if UI_LANG == "en" else MODEL_TARGET_LABELS
    if "platform" in localized.columns:
        localized["platform"] = map_series(localized["platform"], platform_map)
    if "stage" in localized.columns:
        localized["stage"] = map_series(localized["stage"], stage_map)
    if "business_mode" in localized.columns:
        localized["business_mode"] = map_pipe_text(localized["business_mode"], business_mode_map)
    if "feedback_mode" in localized.columns:
        localized["feedback_mode"] = map_series(localized["feedback_mode"], feedback_mode_map)
    if "comment_target" in localized.columns:
        localized["comment_target"] = map_series(localized["comment_target"], comment_target_map)
    if "sentiment" in localized.columns:
        localized["sentiment"] = map_series(localized["sentiment"], sentiment_map)
    if "annotation_source" in localized.columns:
        localized["annotation_source"] = map_series(localized["annotation_source"], annotation_map)
    if "llm_route_reason" in localized.columns:
        localized["llm_route_reason"] = map_pipe_text(localized["llm_route_reason"], route_map)
    if "training_data_label" in localized.columns:
        localized["training_data_label"] = map_series(localized["training_data_label"], training_map)
    if "model_target" in localized.columns:
        localized["model_target"] = map_pipe_text(localized["model_target"], model_target_map)
    return localized


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = COLUMN_LABELS_EN if UI_LANG == "en" else COLUMN_LABELS
    return df.rename(columns={column: column_map.get(column, column) for column in df.columns})


def render_overview(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader(tr("sub_overview"))
    cols = st.columns(5)
    cols[0].metric(tr("metric_content_total"), len(content_df))
    cols[1].metric(
        tr("metric_content_high"),
        int(content_df["is_high_value_feedback_pre_llm"].sum()) if "is_high_value_feedback_pre_llm" in content_df.columns else 0,
    )
    cols[2].metric(
        tr("metric_content_llm"),
        int(content_df["needs_llm_analysis"].sum()) if "needs_llm_analysis" in content_df.columns else 0,
    )
    cols[3].metric(tr("metric_comment_total"), len(comment_df))
    cols[4].metric(
        tr("metric_comment_high"),
        int(comment_df["is_high_value_feedback"].sum()) if "is_high_value_feedback" in comment_df.columns else 0,
    )


def render_training_positioning() -> None:
    st.subheader(tr("sub_training_position"))
    if UI_LANG == "en":
        st.markdown(
            """
            This demo is framed not only as community analytics, but also as a
            **Chinese long-text corpus cleaning, training-candidate filtering, hard-sample routing, and weak-supervision pipeline**.

            In this view:

            - `quality_score` measures structural quality
            - `context_dependency_score` measures context-loss risk
            - `business_value_score` can be interpreted as training-signal strength
            - `feedback_entropy_score` measures information density
            - `needs_llm_analysis` marks hard samples that should be sent to a stronger model
            """
        )
    else:
        st.markdown(
            """
            这个 demo 在展示层上不再只强调“社区反馈分析”，而是把整条链路解释为一条
            **中文长文本语料清洗、训练候选样本筛选、难样本路由与弱监督标注 pipeline**。

            在这个视角下：

            - `quality_score` 代表文本表达质量与结构完整度
            - `context_dependency_score` 代表样本脱离上下文后的失真风险
            - `business_value_score` 在训练视角中可理解为训练信号强度或样本效用
            - `feedback_entropy_score` 代表样本的信息密度与优先级
            - `needs_llm_analysis` 代表需要送入更强模型处理的 hard samples
            """
        )


def render_training_metrics(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader(tr("sub_training_metrics"))
    content_candidates = (
        int(content_df["is_high_value_feedback_pre_llm"].sum())
        if "is_high_value_feedback_pre_llm" in content_df.columns
        else 0
    )
    comment_candidates = (
        int(comment_df["is_high_value_feedback"].sum())
        if "is_high_value_feedback" in comment_df.columns
        else 0
    )
    hard_samples = (
        int(content_df["needs_llm_analysis"].sum()) if "needs_llm_analysis" in content_df.columns else 0
    ) + (int(comment_df["needs_llm_analysis"].sum()) if "needs_llm_analysis" in comment_df.columns else 0)
    context_risky = (
        int((content_df["context_dependency_score"] >= 3).sum()) if "context_dependency_score" in content_df.columns else 0
    ) + (int((comment_df["context_dependency_score"] >= 3).sum()) if "context_dependency_score" in comment_df.columns else 0)
    llm_labeled = (
        int((content_df.get("annotation_source", pd.Series(dtype=str)) == "llm").sum())
        + int((comment_df.get("annotation_source", pd.Series(dtype=str)) == "llm").sum())
    )

    cols = st.columns(5)
    cols[0].metric("回答训练候选", content_candidates)
    cols[1].metric("评论训练候选", comment_candidates)
    cols[2].metric("Hard Samples", hard_samples)
    cols[3].metric("上下文风险样本", context_risky)
    cols[4].metric("LLM 弱监督标注样本", llm_labeled)

    summary_rows = [
        {"样本池": "回答主文本", "总量": len(content_df), "训练候选": content_candidates},
        {"样本池": "评论与回复", "总量": len(comment_df), "训练候选": comment_candidates},
    ]
    summary_df = pd.DataFrame(summary_rows)
    fig = px.bar(summary_df, x="样本池", y=["总量", "训练候选"], barmode="group", title="样本池与训练候选规模")
    st.plotly_chart(fig, use_container_width=True)

    label_frames = []
    for pool_name, df in [("回答主文本", content_df), ("评论与回复", comment_df)]:
        if not df.empty and "training_data_label" in df.columns:
            label_df = df.groupby("training_data_label").size().reset_index(name="数量")
            label_df["样本池"] = pool_name
            label_frames.append(label_df)
    if label_frames:
        label_summary = pd.concat(label_frames, ignore_index=True)
        label_summary["训练数据标签"] = map_series(label_summary["training_data_label"], TRAINING_LABELS)
        fig2 = px.bar(
            label_summary,
            x="训练数据标签",
            y="数量",
            color="样本池",
            barmode="group",
            title="训练侧标签分布：保留、丢弃、补上下文、难样本与复核",
        )
        st.plotly_chart(fig2, use_container_width=True)

    target_frames = []
    for pool_name, df in [("回答主文本", content_df), ("评论与回复", comment_df)]:
        if not df.empty and "model_target" in df.columns:
            rows = []
            for cell in df["model_target"].fillna(""):
                for target in [part for part in str(cell).split("|") if part]:
                    rows.append({"model_target": target, "样本池": pool_name})
            if rows:
                target_frames.append(pd.DataFrame(rows))
    if target_frames:
        target_df = pd.concat(target_frames, ignore_index=True)
        target_summary = target_df.groupby(["样本池", "model_target"]).size().reset_index(name="数量")
        target_summary["适合训练的模型"] = map_series(target_summary["model_target"], MODEL_TARGET_LABELS)
        fig3 = px.bar(
            target_summary,
            x="适合训练的模型",
            y="数量",
            color="样本池",
            barmode="group",
            title="样本可支持的模型目标",
        )
        st.plotly_chart(fig3, use_container_width=True)


def render_training_scatter(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader(tr("sub_training_scatter"))
    frames = []
    if not content_df.empty:
        temp = content_df.copy()
        temp["sample_pool"] = "回答主文本"
        temp["candidate_flag"] = temp.get("is_high_value_feedback_pre_llm", False)
        frames.append(temp)
    if not comment_df.empty:
        temp = comment_df.copy()
        temp["sample_pool"] = "评论与回复"
        temp["candidate_flag"] = temp.get("is_high_value_feedback", False)
        frames.append(temp)
    if not frames:
        st.info("暂无可展示的训练样本分布。")
        return
    sample_df = pd.concat(frames, ignore_index=True)
    sort_col = "training_sample_score" if "training_sample_score" in sample_df.columns else "feedback_entropy_score"
    sample_df = sample_df.sort_values(sort_col, ascending=False).head(500)
    hover_cols = [col for col in ["question_title", "answer_summary", "clean_text", "high_value_reason"] if col in sample_df.columns]
    fig = px.scatter(
        sample_df,
        x="quality_score",
        y=sort_col,
        color="candidate_flag",
        symbol="sample_pool",
        hover_data=hover_cols,
        title="质量分与训练样本分布：候选样本更集中在右上区域",
    )
    fig.update_xaxes(title="质量分")
    fig.update_yaxes(title="训练样本分" if sort_col == "training_sample_score" else "信息密度分")
    st.plotly_chart(fig, use_container_width=True)


def render_training_routing(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader(tr("sub_training_route"))
    rows = []
    for pool_name, df, candidate_col in [
        ("回答主文本", content_df, "is_high_value_feedback_pre_llm"),
        ("评论与回复", comment_df, "is_high_value_feedback"),
    ]:
        if df.empty:
            continue
        rows.append(
            {
                "样本池": pool_name,
                "训练候选": int(df[candidate_col].sum()) if candidate_col in df.columns else 0,
                "进入 LLM": int(df["needs_llm_analysis"].sum()) if "needs_llm_analysis" in df.columns else 0,
                "高上下文风险": int((df["context_dependency_score"] >= 3).sum())
                if "context_dependency_score" in df.columns
                else 0,
            }
        )
    routing_df = pd.DataFrame(rows)
    if routing_df.empty:
        st.info("暂无可展示的路由结果。")
        return
    fig = px.bar(
        routing_df.melt(id_vars="样本池", var_name="阶段", value_name="数量"),
        x="样本池",
        y="数量",
        color="阶段",
        barmode="group",
        title="训练候选、Hard Sample 与上下文风险样本分布",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_training_examples(content_report_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader(tr("sub_training_examples"))
    content_source = content_report_df.copy() if not content_report_df.empty else pd.DataFrame()
    if not content_source.empty and "is_high_value_feedback_pre_llm" in content_source.columns:
        top_content = content_source[content_source["is_high_value_feedback_pre_llm"]].sort_values(
            ["feedback_entropy_score", "llm_priority_score", "business_value_score"],
            ascending=[False, False, False],
        ).head(8)
        cols = [
            "question_title",
            "answer_summary",
            "high_value_reason",
            "training_data_label",
            "training_data_reason",
            "training_sample_score",
            "model_target",
            "quality_score",
            "context_dependency_score",
            "feedback_entropy_score",
            "needs_llm_analysis",
        ]
        st.markdown("**回答主文本候选**")
        st.dataframe(
            rename_columns(localize_common_fields(top_content[[c for c in cols if c in top_content.columns]])),
            use_container_width=True,
            hide_index=True,
        )

    if not comment_df.empty and "is_high_value_feedback" in comment_df.columns:
        top_comment = comment_df[comment_df["is_high_value_feedback"]].sort_values(
            ["feedback_entropy_score", "llm_priority_score", "business_value_score"],
            ascending=[False, False, False],
        ).head(8)
        cols = [
            "question_title",
            "clean_text",
            "high_value_reason",
            "training_data_label",
            "training_data_reason",
            "training_sample_score",
            "model_target",
            "comment_target",
            "quality_score",
            "context_dependency_score",
            "feedback_entropy_score",
            "needs_llm_analysis",
        ]
        st.markdown("**评论与回复候选**")
        st.dataframe(
            rename_columns(localize_common_fields(top_comment[[c for c in cols if c in top_comment.columns]])),
            use_container_width=True,
            hide_index=True,
        )


def render_pipeline_funnel(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> None:
    st.subheader(tr("sub_funnel"))
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
    st.subheader(tr("sub_platform_summary"))
    if summary_df.empty:
        return
    display_df = localize_common_fields(summary_df)
    for _, row in display_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['platform']}")
            st.write(row["summary"])


def render_content_modes(content_df: pd.DataFrame) -> None:
    st.subheader(tr("sub_content_modes"))
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
    st.subheader(tr("sub_content_scatter"))
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
    st.subheader(tr("sub_content_top"))
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
    st.subheader(tr("sub_comment_reaction"))
    if comment_summary_df.empty:
        st.info("暂无评论反应汇总。")
        return
    st.dataframe(rename_columns(localize_common_fields(comment_summary_df.head(20))), use_container_width=True, hide_index=True)


def render_comment_modes(mode_df: pd.DataFrame, multilabel_df: pd.DataFrame) -> None:
    st.subheader(tr("sub_comment_modes"))
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

    st.subheader(tr("sub_comment_multilabel"))
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
    st.subheader(tr("sub_comment_entropy"))
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
    st.subheader(tr("sub_comment_route"))
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
    st.subheader(tr("sub_comment_examples"))
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
    st.subheader(tr("sub_export"))
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
    global UI_LANG
    st.set_page_config(page_title="Bilingual Demo", layout="wide")
    UI_LANG = "en" if st.sidebar.radio(tr("lang_label"), ["中文", "English"], index=0) == "English" else "zh"
    st.title(tr("title"))
    st.caption(tr("caption"))

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
        st.info(tr("demo_notice"))

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
        st.warning(tr("missing_warning"))
        return

    tabs = st.tabs(
        [
            tr("tab_overview"),
            tr("tab_training"),
            tr("tab_funnel"),
            tr("tab_content"),
            tr("tab_comment"),
            tr("tab_export"),
        ]
    )

    with tabs[0]:
        render_overview(content_df, comment_df)
        render_summary(summary_df)

    with tabs[1]:
        render_training_positioning()
        render_training_metrics(content_df, comment_df)
        render_training_scatter(content_df, comment_df)
        render_training_routing(content_df, comment_df)
        render_training_examples(content_report_df, comment_df)

    with tabs[2]:
        render_pipeline_funnel(content_df, comment_df)

    with tabs[3]:
        render_content_modes(content_df)
        render_content_scatter(content_df)
        render_top_contents(content_report_df, content_df)
        render_comment_reactions(comment_summary_df)

    with tabs[4]:
        render_comment_modes(mode_df, multilabel_df)
        render_comment_entropy(comment_df)
        render_comment_route(comment_df)
        render_comment_examples(comment_df)

    with tabs[5]:
        render_export(content_report_df, content_df, comment_df)


if __name__ == "__main__":
    main()
