import argparse
import time
from pathlib import Path

import pandas as pd

from config.settings import DATA_INTERIM_DIR, DATA_PROCESSED_DIR
from src.aggregate.metrics import (
    build_feedback_mode_multilabel_table,
    build_mode_table,
    build_platform_summary,
)
from src.aggregate.comment_reactions import build_comment_reaction_summary
from src.aggregate.content_reports import build_content_feedback_report
from src.analysis.content_labeler import label_content_dataframe
from src.analysis.keywords import extract_top_keywords
from src.analysis.llm_labeler import label_dataframe
from src.context.thread_builder import build_threads
from src.context.content_builder import build_content_base_table
from src.ingest.zhihu_loader import load_latest_zhihu_exports
from src.llm.batch_parse import parse_batch_results
from src.llm.batch_prepare import export_batch_requests
from src.llm.merge_results import merge_content_llm_results, merge_llm_results
from src.preprocess.cleaner import clean_content_dataframe, clean_thread_dataframe
from src.preprocess.content_feature_builder import build_content_feature_dataframe, get_content_base_weight
from src.preprocess.feature_builder import build_feature_dataframe, get_base_weight
from src.publish.demo_export import publish_demo_assets
from src.routing.llm_router import route_for_llm, select_llm_candidates
from src.scoring.pipeline_scores import score_feature_dataframe
from src.utils.io_utils import read_parquet_if_exists, save_csv, save_parquet, write_manifest
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _dual_save(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> tuple[Path, Path]:
    return save_csv(df, csv_path), save_parquet(df, parquet_path)


def _manifest(stage: str, rows: int | None = None, seconds: float | None = None, extra: dict | None = None) -> None:
    payload: dict = {}
    if rows is not None:
        payload["rows"] = rows
    if seconds is not None:
        payload["seconds"] = round(seconds, 4)
    if extra:
        payload.update(extra)
    path = write_manifest(stage, payload)
    logger.info("manifest %s -> %s", stage, path)


def stage_ingest() -> dict[str, Path]:
    t0 = time.perf_counter()
    content_df, comment_df = load_latest_zhihu_exports()
    if content_df.empty or comment_df.empty:
        raise ValueError("Zhihu raw exports are missing.")
    c_csv, c_pq = _dual_save(content_df, DATA_INTERIM_DIR / "contents.csv", DATA_INTERIM_DIR / "contents.parquet")
    m_csv, m_pq = _dual_save(comment_df, DATA_INTERIM_DIR / "comments.csv", DATA_INTERIM_DIR / "comments.parquet")
    _manifest(
        "ingest",
        rows=len(comment_df),
        seconds=time.perf_counter() - t0,
        extra={"rows_content": len(content_df), "rows_comments": len(comment_df)},
    )
    return {"contents": c_csv, "contents_parquet": c_pq, "comments": m_csv, "comments_parquet": m_pq}


def _load_ingest_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    content_pq = DATA_INTERIM_DIR / "contents.parquet"
    comment_pq = DATA_INTERIM_DIR / "comments.parquet"
    content_df = read_parquet_if_exists(content_pq)
    comment_df = read_parquet_if_exists(comment_pq)
    if content_df is None:
        content_df = pd.read_csv(DATA_INTERIM_DIR / "contents.csv")
    if comment_df is None:
        comment_df = pd.read_csv(DATA_INTERIM_DIR / "comments.csv")
    return content_df, comment_df


def stage_threads() -> dict[str, Path]:
    t0 = time.perf_counter()
    content_df, comment_df = _load_ingest_frames()
    thread_df = build_threads(content_df, comment_df)
    csv_path, pq_path = _dual_save(thread_df, DATA_INTERIM_DIR / "threads.csv", DATA_INTERIM_DIR / "threads.parquet")
    _manifest("threads", rows=len(thread_df), seconds=time.perf_counter() - t0)
    return {"threads": csv_path, "threads_parquet": pq_path}


def stage_content_base() -> dict[str, Path]:
    t0 = time.perf_counter()
    content_df, _ = _load_ingest_frames()
    base_df = build_content_base_table(content_df)
    csv_path, pq_path = _dual_save(base_df, DATA_INTERIM_DIR / "content_base.csv", DATA_INTERIM_DIR / "content_base.parquet")
    _manifest("content_base", rows=len(base_df), seconds=time.perf_counter() - t0)
    return {"content_base": csv_path, "content_base_parquet": pq_path}


def _load_threads() -> pd.DataFrame:
    pq = read_parquet_if_exists(DATA_INTERIM_DIR / "threads.parquet")
    if pq is not None:
        return pq
    return pd.read_csv(DATA_INTERIM_DIR / "threads.csv")


def _load_content_base() -> pd.DataFrame:
    pq = read_parquet_if_exists(DATA_INTERIM_DIR / "content_base.parquet")
    if pq is not None:
        return pq
    return pd.read_csv(DATA_INTERIM_DIR / "content_base.csv")


def stage_clean_features() -> dict[str, Path]:
    t0 = time.perf_counter()
    thread_df = _load_threads()
    cleaned_df = clean_thread_dataframe(thread_df)
    csv_path, pq_path = _dual_save(
        cleaned_df,
        DATA_PROCESSED_DIR / "cleaned_comments.csv",
        DATA_PROCESSED_DIR / "cleaned_comments.parquet",
    )
    _manifest("clean_features", rows=len(cleaned_df), seconds=time.perf_counter() - t0)
    return {"cleaned_comments": csv_path, "cleaned_comments_parquet": pq_path}


def stage_content_clean_features() -> dict[str, Path]:
    t0 = time.perf_counter()
    content_df = _load_content_base()
    cleaned_df = clean_content_dataframe(content_df)
    csv_path, pq_path = _dual_save(
        cleaned_df,
        DATA_PROCESSED_DIR / "cleaned_contents.csv",
        DATA_PROCESSED_DIR / "cleaned_contents.parquet",
    )
    _manifest("content_clean_features", rows=len(cleaned_df), seconds=time.perf_counter() - t0)
    return {"cleaned_contents": csv_path, "cleaned_contents_parquet": pq_path}


def _load_cleaned() -> pd.DataFrame:
    pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "cleaned_comments.parquet")
    if pq is not None:
        return pq
    return pd.read_csv(DATA_PROCESSED_DIR / "cleaned_comments.csv")


def _load_cleaned_contents() -> pd.DataFrame:
    pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "cleaned_contents.parquet")
    if pq is not None:
        return pq
    return pd.read_csv(DATA_PROCESSED_DIR / "cleaned_contents.csv")


def stage_score_label() -> dict[str, Path]:
    t0 = time.perf_counter()
    cleaned_df = _load_cleaned()
    feature_df = build_feature_dataframe(cleaned_df)
    scored_df = score_feature_dataframe(feature_df)
    scored_df["base_weight"] = [
        get_base_weight(actionable, entropy)
        for actionable, entropy in zip(scored_df["is_actionable_feedback"], scored_df["feedback_entropy_score"])
    ]
    scored_df["final_weight"] = (scored_df["base_weight"] * scored_df["engagement_weight"]).round(4)
    csv_path, pq_path = _dual_save(
        scored_df,
        DATA_PROCESSED_DIR / "scored_comments.csv",
        DATA_PROCESSED_DIR / "scored_comments.parquet",
    )
    _manifest("score_label", rows=len(scored_df), seconds=time.perf_counter() - t0)
    return {"scored_comments": csv_path, "scored_comments_parquet": pq_path}


def stage_content_score_label() -> dict[str, Path]:
    t0 = time.perf_counter()
    cleaned_df = _load_cleaned_contents()
    feature_df = build_content_feature_dataframe(cleaned_df)
    scored_df = score_feature_dataframe(feature_df)
    scored_df["base_weight"] = [
        get_content_base_weight(actionable, density)
        for actionable, density in zip(scored_df["is_actionable_feedback"], scored_df["feedback_entropy_score"])
    ]
    scored_df["final_weight"] = (scored_df["base_weight"] * scored_df["engagement_weight"]).round(4)
    csv_path, pq_path = _dual_save(
        scored_df,
        DATA_PROCESSED_DIR / "scored_contents.csv",
        DATA_PROCESSED_DIR / "scored_contents.parquet",
    )
    _manifest("content_score_label", rows=len(scored_df), seconds=time.perf_counter() - t0)
    return {"scored_contents": csv_path, "scored_contents_parquet": pq_path}


def _load_scored() -> pd.DataFrame:
    pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "scored_comments.parquet")
    if pq is not None:
        return pq
    return pd.read_csv(DATA_PROCESSED_DIR / "scored_comments.csv")


def _load_scored_contents() -> pd.DataFrame:
    pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "scored_contents.parquet")
    if pq is not None:
        return pq
    return pd.read_csv(DATA_PROCESSED_DIR / "scored_contents.csv")


def stage_route() -> dict[str, Path | pd.DataFrame]:
    t0 = time.perf_counter()
    scored_df = _load_scored()
    rule_df = label_dataframe(scored_df)
    routed_df = route_for_llm(rule_df)
    csv_path, pq_path = _dual_save(
        routed_df,
        DATA_PROCESSED_DIR / "labeled_rule.csv",
        DATA_PROCESSED_DIR / "labeled_rule.parquet",
    )
    llm_candidates = select_llm_candidates(routed_df)
    llm_candidates_path = save_csv(llm_candidates, DATA_PROCESSED_DIR / "llm_candidates.csv")
    llm_requests_path = export_batch_requests(llm_candidates, DATA_PROCESSED_DIR / "llm_requests.jsonl")
    _manifest(
        "route",
        rows=len(routed_df),
        seconds=time.perf_counter() - t0,
        extra={"llm_candidates": len(llm_candidates), "needs_llm": int(routed_df["needs_llm_analysis"].sum())},
    )
    return {
        "labeled_rule": csv_path,
        "labeled_rule_parquet": pq_path,
        "llm_candidates": llm_candidates_path,
        "llm_requests": llm_requests_path,
        "routed_df": routed_df,
    }


def stage_content_route() -> dict[str, Path | pd.DataFrame]:
    t0 = time.perf_counter()
    scored_df = _load_scored_contents()
    rule_df = label_content_dataframe(scored_df)
    routed_df = route_for_llm(rule_df)
    csv_path, pq_path = _dual_save(
        routed_df,
        DATA_PROCESSED_DIR / "labeled_content_rule.csv",
        DATA_PROCESSED_DIR / "labeled_content_rule.parquet",
    )
    llm_candidates = select_llm_candidates(routed_df)
    llm_candidates_path = save_csv(llm_candidates, DATA_PROCESSED_DIR / "content_llm_candidates.csv")
    llm_requests_path = export_batch_requests(llm_candidates, DATA_PROCESSED_DIR / "content_llm_requests.jsonl")
    _manifest(
        "content_route",
        rows=len(routed_df),
        seconds=time.perf_counter() - t0,
        extra={"llm_candidates": len(llm_candidates), "needs_llm": int(routed_df["needs_llm_analysis"].sum())},
    )
    return {
        "labeled_content_rule": csv_path,
        "labeled_content_rule_parquet": pq_path,
        "content_llm_candidates": llm_candidates_path,
        "content_llm_requests": llm_requests_path,
        "content_routed_df": routed_df,
    }


def merge_llm_only(routed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Path]]:
    llm_responses_path = DATA_PROCESSED_DIR / "llm_responses.jsonl"
    llm_results_path = DATA_PROCESSED_DIR / "llm_results.csv"
    llm_results_df = parse_batch_results(llm_responses_path)
    if llm_results_df.empty:
        save_csv(
            pd.DataFrame(
                columns=[
                    "custom_id",
                    "llm_sentiment",
                    "llm_dimension",
                    "llm_is_comparative",
                    "llm_target_game",
                    "llm_comment_target",
                    "llm_business_mode",
                    "llm_stance_summary",
                    "llm_confidence",
                    "llm_is_high_value_feedback",
                    "llm_needs_manual_review",
                ]
            ),
            llm_results_path,
        )
    else:
        save_csv(llm_results_df, llm_results_path)

    final_df = merge_llm_results(routed_df, llm_results_df)
    final_csv, final_pq = _dual_save(
        final_df,
        DATA_PROCESSED_DIR / "labeled_final.csv",
        DATA_PROCESSED_DIR / "labeled_final.parquet",
    )
    labeled_path = save_csv(final_df, DATA_PROCESSED_DIR / "labeled.csv")
    paths = {
        "llm_results": llm_results_path,
        "labeled_final": final_csv,
        "labeled_final_parquet": final_pq,
        "labeled": labeled_path,
    }
    return final_df, paths


def merge_content_llm_only(routed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Path]]:
    llm_responses_path = DATA_PROCESSED_DIR / "content_llm_responses.jsonl"
    llm_results_path = DATA_PROCESSED_DIR / "content_llm_results.csv"
    llm_results_df = parse_batch_results(llm_responses_path)
    if llm_results_df.empty:
        save_csv(
            pd.DataFrame(
                columns=[
                    "custom_id",
                    "llm_sentiment",
                    "llm_dimension",
                    "llm_is_comparative",
                    "llm_target_game",
                    "llm_comment_target",
                    "llm_business_mode",
                    "llm_stance_summary",
                    "llm_confidence",
                    "llm_is_high_value_feedback",
                    "llm_needs_manual_review",
                ]
            ),
            llm_results_path,
        )
    else:
        save_csv(llm_results_df, llm_results_path)

    final_df = merge_content_llm_results(routed_df, llm_results_df)
    final_csv, final_pq = _dual_save(
        final_df,
        DATA_PROCESSED_DIR / "labeled_content_final.csv",
        DATA_PROCESSED_DIR / "labeled_content_final.parquet",
    )
    return final_df, {
        "content_llm_results": llm_results_path,
        "labeled_content_final": final_csv,
        "labeled_content_final_parquet": final_pq,
    }


def aggregate_from_final(final_df: pd.DataFrame) -> dict[str, Path]:
    mode_df = build_mode_table(final_df)
    mode_path = save_csv(mode_df, DATA_PROCESSED_DIR / "business_modes.csv")

    summary_df = build_platform_summary(final_df)
    summary_path = save_csv(summary_df, DATA_PROCESSED_DIR / "platform_summary.csv")

    keyword_df = extract_top_keywords(final_df.assign(text=final_df["clean_text"]))
    keyword_path = save_csv(keyword_df, DATA_PROCESSED_DIR / "keywords.csv")

    multilabel_df = build_feedback_mode_multilabel_table(final_df)
    multilabel_path = save_csv(multilabel_df, DATA_PROCESSED_DIR / "feedback_mode_multilabel.csv")

    content_final_pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "labeled_content_final.parquet")
    if content_final_pq is not None:
        content_df = content_final_pq
    elif (DATA_PROCESSED_DIR / "labeled_content_final.csv").exists():
        content_df = pd.read_csv(DATA_PROCESSED_DIR / "labeled_content_final.csv")
    else:
        content_rule_pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "labeled_content_rule.parquet")
        if content_rule_pq is not None:
            content_df = content_rule_pq
        elif (DATA_PROCESSED_DIR / "labeled_content_rule.csv").exists():
            content_df = pd.read_csv(DATA_PROCESSED_DIR / "labeled_content_rule.csv")
        else:
            content_df = pd.DataFrame()

    comment_summary_df = build_comment_reaction_summary(final_df)
    comment_summary_path = save_csv(comment_summary_df, DATA_PROCESSED_DIR / "comment_reaction_summary.csv")

    content_report_df = build_content_feedback_report(content_df, comment_summary_df)
    content_report_path = save_csv(content_report_df, DATA_PROCESSED_DIR / "content_feedback_report.csv")

    return {
        "business_modes": mode_path,
        "summary": summary_path,
        "keywords": keyword_path,
        "feedback_mode_multilabel": multilabel_path,
        "comment_reaction_summary": comment_summary_path,
        "content_feedback_report": content_report_path,
    }


def stage_llm_merge() -> dict[str, Path]:
    t0 = time.perf_counter()
    routed_pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "labeled_rule.parquet")
    if routed_pq is not None:
        routed_df = routed_pq
    else:
        routed_df = pd.read_csv(DATA_PROCESSED_DIR / "labeled_rule.csv")
    final_df, paths = merge_llm_only(routed_df)
    _manifest("llm_merge", rows=len(final_df), seconds=time.perf_counter() - t0)
    return paths


def stage_content_llm_merge() -> dict[str, Path]:
    t0 = time.perf_counter()
    routed_pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "labeled_content_rule.parquet")
    if routed_pq is not None:
        routed_df = routed_pq
    else:
        routed_df = pd.read_csv(DATA_PROCESSED_DIR / "labeled_content_rule.csv")
    final_df, paths = merge_content_llm_only(routed_df)
    _manifest("content_llm_merge", rows=len(final_df), seconds=time.perf_counter() - t0)
    return paths


def stage_aggregate() -> dict[str, Path]:
    t0 = time.perf_counter()
    final_pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "labeled_final.parquet")
    if final_pq is not None:
        final_df = final_pq
    else:
        final_df = pd.read_csv(DATA_PROCESSED_DIR / "labeled_final.csv")
    paths = aggregate_from_final(final_df)
    _manifest("aggregate", rows=len(final_df), seconds=time.perf_counter() - t0)
    return paths


def stage_publish_demo() -> dict[str, Path]:
    t0 = time.perf_counter()
    paths = publish_demo_assets()
    demo_content_df = pd.read_csv(next(path for name, path in paths.items() if name == "demo_content_report"))
    _manifest("publish_demo", rows=len(demo_content_df), seconds=time.perf_counter() - t0)
    return paths


def build_pre_llm_outputs() -> dict[str, Path | pd.DataFrame]:
    outputs: dict[str, Path | pd.DataFrame] = {}
    outputs.update(stage_ingest())
    outputs.update(stage_threads())
    outputs.update(stage_content_base())
    outputs.update(stage_clean_features())
    outputs.update(stage_content_clean_features())
    outputs.update(stage_score_label())
    outputs.update(stage_content_score_label())
    outputs.update(stage_route())
    outputs.update(stage_content_route())
    return outputs


def finalize_outputs(routed_df: pd.DataFrame) -> dict[str, Path]:
    final_df, merge_paths = merge_llm_only(routed_df)
    content_routed_pq = read_parquet_if_exists(DATA_PROCESSED_DIR / "labeled_content_rule.parquet")
    if content_routed_pq is not None:
        content_routed_df = content_routed_pq
    else:
        content_routed_df = pd.read_csv(DATA_PROCESSED_DIR / "labeled_content_rule.csv")
    _, content_merge_paths = merge_content_llm_only(content_routed_df)
    agg_paths = aggregate_from_final(final_df)
    demo_paths = publish_demo_assets()
    merge_paths.update(content_merge_paths)
    merge_paths.update(agg_paths)
    merge_paths.update(demo_paths)
    _manifest("funnel", extra={"final_rows": len(final_df)})
    return merge_paths


def run_pipeline(stage: str = "full") -> dict[str, Path]:
    if stage == "ingest":
        return stage_ingest()
    if stage == "threads":
        return stage_threads()
    if stage == "content_base":
        return stage_content_base()
    if stage == "clean_features":
        return stage_clean_features()
    if stage == "content_clean_features":
        return stage_content_clean_features()
    if stage == "score_label":
        return stage_score_label()
    if stage == "content_score_label":
        return stage_content_score_label()
    if stage == "route":
        return stage_route()  # type: ignore[return-value]
    if stage == "content_route":
        return stage_content_route()  # type: ignore[return-value]
    if stage == "llm_merge":
        return stage_llm_merge()
    if stage == "content_llm_merge":
        return stage_content_llm_merge()
    if stage == "aggregate":
        return stage_aggregate()
    if stage == "publish_demo":
        return stage_publish_demo()

    outputs = build_pre_llm_outputs()
    routed_df = outputs.pop("routed_df")  # type: ignore[assignment]
    if stage == "pre_llm":
        outputs["routed_df"] = routed_df
        return outputs  # type: ignore[return-value]

    final_outputs = finalize_outputs(routed_df)
    outputs.update(final_outputs)
    return outputs  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the thread-aware zhihu pipeline.")
    parser.add_argument(
        "--stage",
        choices=[
            "full",
            "pre_llm",
            "ingest",
            "threads",
            "content_base",
            "clean_features",
            "content_clean_features",
            "score_label",
            "content_score_label",
            "route",
            "content_route",
            "llm_merge",
            "content_llm_merge",
            "aggregate",
            "publish_demo",
        ],
        default="full",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(stage=args.stage)
    for name, path in result.items():
        if name == "routed_df":
            continue
        if isinstance(path, Path):
            logger.info("%s -> %s", name, path)
