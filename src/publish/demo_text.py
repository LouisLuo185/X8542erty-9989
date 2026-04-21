from __future__ import annotations

import math

import pandas as pd


BUSINESS_MODE_LABELS = {
    "product_experience": "产品体验",
    "monetization": "商业化",
    "brand_communication": "品牌传播",
    "community_conflict": "社区冲突",
    "competition": "竞品比较",
    "plot_discussion": "剧情内容",
    "other_high_value": "其他高价值",
}

SENTIMENT_LABELS = {
    "positive": "正向",
    "neutral": "中性",
    "negative": "负向",
    "mixed": "混合",
}

COMMENT_TARGET_LABELS = {
    "game": "游戏本体",
    "answer": "回答观点",
    "community": "社区群体",
    "mixed": "混合对象",
    "unclear": "对象不明确",
}


def _primary_mode(raw_value: object) -> str:
    parts = [part for part in str(raw_value or "").split("|") if part]
    if not parts:
        return "other_high_value"
    return parts[0]


def _mode_label(raw_value: object) -> str:
    return BUSINESS_MODE_LABELS.get(_primary_mode(raw_value), "其他高价值")


def _sentiment_label(raw_value: object) -> str:
    return SENTIMENT_LABELS.get(str(raw_value or ""), "中性")


def _target_label(raw_value: object) -> str:
    return COMMENT_TARGET_LABELS.get(str(raw_value or ""), "对象不明确")


def _score_level(value: float | int | object) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        number = 0.0
    if number >= 8:
        return "很高"
    if number >= 5:
        return "较高"
    if number >= 3:
        return "中等"
    return "较低"


def _percent_text(value: object) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        number = 0.0
    if number <= 1:
        number *= 100
    return f"{round(number, 1)}%"


def build_content_question_title(row: pd.Series, idx: int) -> str:
    mode_label = _mode_label(row.get("business_mode") or row.get("feedback_modes"))
    return f"演示案例 {idx:03d}：{mode_label}主题"


def build_content_answer_summary(row: pd.Series) -> str:
    mode_label = _mode_label(row.get("business_mode") or row.get("feedback_modes"))
    sentiment_label = _sentiment_label(row.get("sentiment"))
    entropy_level = _score_level(row.get("feedback_entropy_score", 0))
    return (
        f"该回答样本在公开演示版中被归纳为“{mode_label}”主题，"
        f"整体情绪倾向为{sentiment_label}，信息密度处于{entropy_level}水平。"
    )


def build_content_issue_abstract(row: pd.Series) -> str:
    mode_label = _mode_label(row.get("business_mode") or row.get("feedback_modes"))
    business_level = _score_level(row.get("business_value_score", 0))
    quality_level = _score_level(row.get("quality_score", 0))
    llm_text = "需要进一步进入 LLM 深分析" if bool(row.get("needs_llm_analysis", False)) else "规则层已可完成初步判断"
    return (
        f"该样本主要体现为{mode_label}反馈，业务价值{business_level}、表达质量{quality_level}，"
        f"因此在演示版中被记录为“{llm_text}”的案例。"
    )


def build_comment_question_title(row: pd.Series, idx: int) -> str:
    mode_label = _mode_label(row.get("business_mode") or row.get("feedback_modes"))
    return f"评论案例 {idx:03d}：{mode_label}回应"


def build_comment_answer_summary(row: pd.Series) -> str:
    target_label = _target_label(row.get("comment_target"))
    context_level = _score_level(row.get("context_dependency_score", 0))
    return (
        f"该评论样本在公开演示版中被整理为针对“{target_label}”的回应，"
        f"上下文依赖程度为{context_level}。"
    )


def build_comment_clean_text(row: pd.Series) -> str:
    mode_label = _mode_label(row.get("business_mode") or row.get("feedback_modes"))
    sentiment_label = _sentiment_label(row.get("sentiment"))
    target_label = _target_label(row.get("comment_target"))
    entropy_level = _score_level(row.get("feedback_entropy_score", 0))
    llm_text = "已进入 LLM 候选池" if bool(row.get("needs_llm_analysis", False)) else "未进入 LLM 候选池"
    return (
        f"该评论被改写为公开展示摘要：围绕{mode_label}主题，对{target_label}表达了{sentiment_label}态度，"
        f"整体信息密度为{entropy_level}，{llm_text}。"
    )


def build_comment_issue_abstract(row: pd.Series) -> str:
    business_level = _score_level(row.get("business_value_score", 0))
    context_level = _score_level(row.get("context_dependency_score", 0))
    target_label = _target_label(row.get("comment_target"))
    return (
        f"该评论主要作为{target_label}相关的社区回应样本被保留，"
        f"业务价值{business_level}，上下文依赖{context_level}。"
    )


def build_safe_summary(platform: str, content_rows: int, high_value_rows: int, llm_rows: int, context_ratio: float) -> str:
    share = 0.0 if content_rows == 0 else (high_value_rows / content_rows) * 100
    llm_share = 0.0 if content_rows == 0 else (llm_rows / content_rows) * 100
    return (
        f"{platform} 当前演示数据中，高价值样本占比约 {round(share, 2)}%，"
        f"进入 LLM 的样本占比约 {round(llm_share, 2)}%，"
        f"高上下文依赖样本占比约 {round(context_ratio, 2)}%。"
    )


def stable_demo_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:04d}"


def bucket_likes(value: object) -> int:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        number = 0.0
    return int(math.floor(number))
