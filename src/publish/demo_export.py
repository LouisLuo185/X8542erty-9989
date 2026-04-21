from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.settings import DATA_DEMO_DIR, DATA_PROCESSED_DIR, MANIFEST_DIR
from src.publish.demo_text import (
    build_comment_answer_summary,
    build_comment_clean_text,
    build_comment_issue_abstract,
    build_comment_question_title,
    build_content_answer_summary,
    build_content_issue_abstract,
    build_content_question_title,
    build_safe_summary,
    stable_demo_id,
)
from src.utils.io_utils import save_csv


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _safe_bool(series: pd.Series | None, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.fillna(default).astype(bool)


def build_demo_content_report(content_report_df: pd.DataFrame) -> pd.DataFrame:
    if content_report_df.empty:
        return pd.DataFrame(
            columns=[
                "platform",
                "question_title",
                "answer_summary",
                "feedback_modes",
                "business_mode",
                "sentiment",
                "quality_score",
                "context_dependency_score",
                "business_value_score",
                "feedback_entropy_score",
                "likes",
                "high_value_reason",
                "is_high_value_feedback_pre_llm",
                "needs_llm_analysis",
                "llm_priority_score",
                "llm_route_reason",
                "reaction_comment_count",
                "supportive_ratio",
                "opposing_ratio",
                "controversy_score",
                "sarcasm_ratio",
                "high_value_comment_count",
                "avg_comment_business_value_score",
                "avg_comment_entropy_score",
                "demo_case_id",
            ]
        )

    ranked = content_report_df.copy().reset_index(drop=True)
    ranked["demo_case_id"] = [stable_demo_id("content", idx + 1) for idx in range(len(ranked))]
    ranked["question_title"] = [
        build_content_question_title(row, idx + 1) for idx, (_, row) in enumerate(ranked.iterrows())
    ]
    ranked["answer_summary"] = ranked.apply(build_content_answer_summary, axis=1)
    ranked["high_value_reason"] = ranked.apply(build_content_issue_abstract, axis=1)
    ranked["likes"] = pd.to_numeric(ranked.get("likes", 0), errors="coerce").fillna(0).astype(int)

    keep_cols = [
        "platform",
        "question_title",
        "answer_summary",
        "feedback_modes",
        "business_mode",
        "sentiment",
        "quality_score",
        "context_dependency_score",
        "business_value_score",
        "feedback_entropy_score",
        "likes",
        "high_value_reason",
        "is_high_value_feedback_pre_llm",
        "needs_llm_analysis",
        "llm_priority_score",
        "llm_route_reason",
        "reaction_comment_count",
        "supportive_ratio",
        "opposing_ratio",
        "controversy_score",
        "sarcasm_ratio",
        "high_value_comment_count",
        "avg_comment_business_value_score",
        "avg_comment_entropy_score",
        "demo_case_id",
    ]
    for column in keep_cols:
        if column not in ranked.columns:
            ranked[column] = ""
    return ranked[keep_cols]


def build_demo_comment_report(comment_df: pd.DataFrame) -> pd.DataFrame:
    if comment_df.empty:
        return pd.DataFrame(
            columns=[
                "platform",
                "comment_id",
                "question_title",
                "answer_summary",
                "clean_text",
                "business_mode",
                "feedback_modes",
                "comment_target",
                "sentiment",
                "confidence",
                "quality_score",
                "context_dependency_score",
                "business_value_score",
                "feedback_entropy_score",
                "likes",
                "high_value_reason",
                "is_high_value_feedback",
                "needs_llm_analysis",
                "llm_priority_score",
                "llm_route_reason",
                "annotation_source",
                "demo_case_id",
            ]
        )

    demo_df = comment_df.copy().reset_index(drop=True)
    demo_df["demo_case_id"] = [stable_demo_id("comment", idx + 1) for idx in range(len(demo_df))]
    demo_df["comment_id"] = demo_df["demo_case_id"]
    demo_df["question_title"] = [
        build_comment_question_title(row, idx + 1) for idx, (_, row) in enumerate(demo_df.iterrows())
    ]
    demo_df["answer_summary"] = demo_df.apply(build_comment_answer_summary, axis=1)
    demo_df["clean_text"] = demo_df.apply(build_comment_clean_text, axis=1)
    demo_df["high_value_reason"] = demo_df.apply(build_comment_issue_abstract, axis=1)
    demo_df["likes"] = pd.to_numeric(demo_df.get("likes", 0), errors="coerce").fillna(0).astype(int)
    if "confidence" not in demo_df.columns:
        demo_df["confidence"] = 0.0

    keep_cols = [
        "platform",
        "comment_id",
        "question_title",
        "answer_summary",
        "clean_text",
        "business_mode",
        "feedback_modes",
        "comment_target",
        "sentiment",
        "confidence",
        "quality_score",
        "context_dependency_score",
        "business_value_score",
        "feedback_entropy_score",
        "likes",
        "high_value_reason",
        "is_high_value_feedback",
        "needs_llm_analysis",
        "llm_priority_score",
        "llm_route_reason",
        "annotation_source",
        "demo_case_id",
    ]
    for column in keep_cols:
        if column not in demo_df.columns:
            demo_df[column] = ""
    return demo_df[keep_cols]


def build_demo_platform_summary(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> pd.DataFrame:
    platforms = sorted(set(content_df.get("platform", pd.Series(dtype=str)).dropna().astype(str)) | set(comment_df.get("platform", pd.Series(dtype=str)).dropna().astype(str)))
    rows: list[dict[str, object]] = []
    for platform in platforms:
        content_group = content_df[content_df.get("platform", "") == platform] if not content_df.empty else pd.DataFrame()
        comment_group = comment_df[comment_df.get("platform", "") == platform] if not comment_df.empty else pd.DataFrame()
        total_contents = len(content_group)
        high_value_contents = int(_safe_bool(content_group.get("is_high_value_feedback_pre_llm")).sum()) if not content_group.empty else 0
        llm_routed_contents = int(_safe_bool(content_group.get("needs_llm_analysis")).sum()) if not content_group.empty else 0
        total_comments = len(comment_group)
        high_value_comments = int(_safe_bool(comment_group.get("is_high_value_feedback")).sum()) if not comment_group.empty else 0
        llm_routed_comments = int(_safe_bool(comment_group.get("needs_llm_analysis")).sum()) if not comment_group.empty else 0
        context_ratio = 0.0
        if not comment_group.empty and "context_dependency_score" in comment_group.columns:
            context_ratio = round((pd.to_numeric(comment_group["context_dependency_score"], errors="coerce").fillna(0) >= 3).mean() * 100, 2)
        rows.append(
            {
                "platform": platform,
                "total_contents": total_contents,
                "high_value_contents": high_value_contents,
                "llm_routed_contents": llm_routed_contents,
                "total_comments": total_comments,
                "high_value_comments": high_value_comments,
                "llm_routed_comments": llm_routed_comments,
                "summary": build_safe_summary(
                    platform=platform,
                    content_rows=max(total_contents, 1),
                    high_value_rows=high_value_contents,
                    llm_rows=llm_routed_contents,
                    context_ratio=context_ratio,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_demo_comment_reaction_summary(content_report_df: pd.DataFrame) -> pd.DataFrame:
    if content_report_df.empty:
        return pd.DataFrame(
            columns=[
                "question_title",
                "reaction_comment_count",
                "supportive_ratio",
                "opposing_ratio",
                "controversy_score",
                "sarcasm_ratio",
                "high_value_comment_count",
                "avg_comment_business_value_score",
                "avg_comment_entropy_score",
            ]
        )
    cols = [
        "question_title",
        "reaction_comment_count",
        "supportive_ratio",
        "opposing_ratio",
        "controversy_score",
        "sarcasm_ratio",
        "high_value_comment_count",
        "avg_comment_business_value_score",
        "avg_comment_entropy_score",
    ]
    return content_report_df[[column for column in cols if column in content_report_df.columns]].copy()


def build_demo_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if MANIFEST_DIR.exists():
        for path in sorted(MANIFEST_DIR.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "stage": record.get("stage", path.stem),
                    "rows": record.get("rows"),
                    "seconds": record.get("seconds"),
                }
            )
    return pd.DataFrame(rows)


def publish_demo_assets() -> dict[str, Path]:
    DATA_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    content_report_df = _safe_read_csv(DATA_PROCESSED_DIR / "content_feedback_report.csv")
    comment_df = _safe_read_csv(DATA_PROCESSED_DIR / "labeled_final.csv")
    modes_df = _safe_read_csv(DATA_PROCESSED_DIR / "business_modes.csv")
    multilabel_df = _safe_read_csv(DATA_PROCESSED_DIR / "feedback_mode_multilabel.csv")

    demo_content_df = build_demo_content_report(content_report_df)
    demo_comment_df = build_demo_comment_report(comment_df)
    demo_summary_df = build_demo_platform_summary(demo_content_df, demo_comment_df)
    demo_comment_reaction_df = build_demo_comment_reaction_summary(demo_content_df)
    demo_manifest_df = build_demo_manifest()

    paths = {
        "demo_content_report": save_csv(demo_content_df, DATA_DEMO_DIR / "demo_content_report.csv"),
        "demo_comment_report": save_csv(demo_comment_df, DATA_DEMO_DIR / "demo_comment_report.csv"),
        "demo_comment_reaction_summary": save_csv(
            demo_comment_reaction_df,
            DATA_DEMO_DIR / "demo_comment_reaction_summary.csv",
        ),
        "demo_platform_summary": save_csv(demo_summary_df, DATA_DEMO_DIR / "demo_platform_summary.csv"),
        "demo_business_modes": save_csv(modes_df, DATA_DEMO_DIR / "demo_business_modes.csv"),
        "demo_feedback_multilabel": save_csv(multilabel_df, DATA_DEMO_DIR / "demo_feedback_multilabel.csv"),
        "demo_pipeline_manifest": save_csv(demo_manifest_df, DATA_DEMO_DIR / "demo_pipeline_manifest.csv"),
    }
    return paths
