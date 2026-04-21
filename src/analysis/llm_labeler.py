import pandas as pd

from config.settings import ANALYSIS_DIMENSIONS, RULE_SCHEMA
from config.taxonomies import NEGATIVE_KEYWORDS, POSITIVE_KEYWORDS


DIMENSION_RULES = {
    "剧情内容": ["剧情", "故事", "设定", "世界观", "结局", "主线", "支线"],
    "角色设计": ["角色", "主角", "人设", "塑造"],
    "美术音乐": ["美术", "音乐", "画面", "配乐"],
    "玩法体验": ["玩法", "优化", "地图", "体验", "委托", "活动"],
    "商业化": ["抽卡", "氪金", "卡池", "保底", "流水"],
    "品牌传播": ["风评", "出圈", "宣传", "热度"],
}


def detect_dimension(text: str) -> str:
    for dimension, keywords in DIMENSION_RULES.items():
        if any(keyword in text for keyword in keywords):
            return dimension
    return "剧情内容"


def detect_target_game(text: str) -> str:
    has_genshin = any(keyword in text for keyword in ["原神", "提瓦特"])
    has_wuwa = "鸣潮" in text
    if has_genshin and has_wuwa:
        return "双方"
    if has_genshin:
        return "原神"
    if has_wuwa:
        return "鸣潮"
    return "不明确"


def detect_comment_target(row: pd.Series) -> str:
    text = row.get("clean_text", "")
    if any(term in text for term in ["答主", "你这", "偷换概念", "你是不是"]):
        return "answer"
    if row.get("community_term_hit_count", 0) > 0 and row.get("product_term_hit_count", 0) == 0 and row.get("monetization_term_hit_count", 0) == 0:
        return "community"
    if row.get("community_term_hit_count", 0) > 0 and (row.get("product_term_hit_count", 0) > 0 or row.get("plot_term_hit_count", 0) > 0):
        return "mixed"
    return "game"


def rule_based_label(row: pd.Series) -> dict:
    text = row.get("clean_text", "")
    positive_hits = sum(keyword in text for keyword in POSITIVE_KEYWORDS)
    negative_hits = sum(keyword in text for keyword in NEGATIVE_KEYWORDS)

    sentiment = "neutral"
    reason = "评论偏讨论性表达，暂按中性处理。"
    confidence = 0.58
    if positive_hits > negative_hits:
        sentiment = "positive"
        reason = "文本中正向评价线索更多。"
        confidence = 0.75
    elif negative_hits > positive_hits:
        sentiment = "negative"
        reason = "文本中负向评价线索更多。"
        confidence = 0.76
    elif positive_hits > 0 and negative_hits > 0:
        sentiment = "mixed"
        reason = "文本同时包含明显正负面线索。"
        confidence = 0.63

    return {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "rule_sentiment": sentiment,
        "rule_dimension": detect_dimension(text),
        "rule_reason": reason,
        "rule_confidence": confidence,
        "rule_is_comparative": row.get("competitor_term_hit_count", 0) > 0 or row.get("comparative_hit_count", 0) > 0,
        "rule_target_game": detect_target_game(text),
        "rule_comment_target": detect_comment_target(row),
    }


def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        merged = row.to_dict()
        merged.update(rule_based_label(row))
        rows.append(merged)
    labeled_df = pd.DataFrame(rows)
    for column in RULE_SCHEMA:
        if column not in labeled_df.columns:
            labeled_df[column] = None
    return labeled_df[RULE_SCHEMA]
