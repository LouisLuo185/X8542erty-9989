"""Join manual gold labels with pipeline outputs and print precision@K style metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.settings import DATA_PROCESSED_DIR


GOLD_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "manual_eval.csv"
LABELED_PATH = DATA_PROCESSED_DIR / "labeled_final.csv"


def _load_gold(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_labeled(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def join_gold(labeled: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    if labeled.empty or gold.empty:
        return pd.DataFrame()
    gold = gold.copy()
    if "clean_text" not in gold.columns and "text" in gold.columns:
        gold = gold.rename(columns={"text": "clean_text"})
    if "comment_id" in gold.columns and "comment_id" in labeled.columns:
        merged = gold.merge(labeled, on="comment_id", how="inner", suffixes=("_gold", ""))
        if not merged.empty:
            return merged
    if "clean_text" in gold.columns and "clean_text" in labeled.columns:
        return gold.merge(labeled, on="clean_text", how="inner")
    return pd.DataFrame()


def precision_at_k(pred: pd.Series, gold: pd.Series, k: int) -> float | None:
    if gold.sum() == 0:
        return None
    order = pred.sort_values(ascending=False).head(k).index
    hits = int(gold.loc[order].sum())
    return round(hits / min(k, len(pred)), 4)


def build_report(labeled_path: Path = LABELED_PATH, gold_path: Path = GOLD_PATH) -> str:
    labeled = _load_labeled(labeled_path)
    gold = _load_gold(gold_path)
    if labeled.empty:
        return "labeled_final.csv 不存在，请先运行 pipeline。"
    if gold.empty:
        return "未找到人工金标文件 data/samples/manual_eval.csv。"

    merged = join_gold(labeled, gold)
    lines: list[str] = []
    lines.append(f"金标行数: {len(gold)}，成功对齐: {len(merged)}")
    if merged.empty:
        lines.append("无法对齐：请为金标提供 comment_id 或与 labeled_final 一致的 clean_text。")
        return "\n".join(lines)

    if "is_high_value_gold" in merged.columns:
        g = merged["is_high_value_gold"].astype(int)
        if "is_high_value_feedback" in merged.columns:
            pred = merged["is_high_value_feedback"].astype(int)
            lines.append(f"高价值准确率(对齐集): {(pred == g).mean():.4f}")
        if "business_value_score" in merged.columns:
            for k in (5, 10, 20, 50):
                p = precision_at_k(merged["business_value_score"], g, k)
                if p is not None:
                    lines.append(f"precision@{k} (按 business_value_score 排序): {p}")

    if "needs_llm_gold" in merged.columns and "needs_llm_analysis" in merged.columns:
        g2 = merged["needs_llm_gold"].astype(int)
        pred2 = merged["needs_llm_analysis"].astype(int)
        lines.append(f"LLM 路由一致率: {(pred2 == g2).mean():.4f}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual gold evaluation report.")
    parser.add_argument("--labeled", type=Path, default=LABELED_PATH)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(build_report(labeled_path=args.labeled, gold_path=args.gold))


if __name__ == "__main__":
    main()
