import re

from config.settings import BUSINESS_MODES


_NULL_LIKE = {
    "",
    "n/a",
    "na",
    "none",
    "unknown",
    "null",
    "无",
    "未提及",
    "未涉及",
    "不涉及",
    "无商业相关",
    "无商业相关信息",
}


def _split_modes(raw_value: object) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[|,，/；;]+", text) if part.strip()]


def _map_mode_token(token: str) -> str | None:
    lowered = token.strip().lower()
    if not lowered or lowered in _NULL_LIKE:
        return None

    if lowered in BUSINESS_MODES:
        return lowered

    competition_keys = ("competition", "竞品", "对比", "比较", "鸣潮", "崩铁", "崩坏", "塞尔达", "fgo", "dnf")
    monetization_keys = ("monetization", "商业化", "付费", "氪", "抽卡", "卡池", "流水", "命座", "专武", "gacha")
    brand_keys = ("brand", "品牌", "传播", "风评", "公关", "口碑", "出圈", "宣发", "营销", "热度")
    community_keys = ("community", "社区", "玩家群体", "米黑", "米孝子", "原友", "对线", "冲突", "节奏", "水军")
    product_keys = ("product", "体验", "优化", "ui", "操作", "玩法", "功能", "任务", "地图", "引擎", "画质", "性能")
    plot_keys = ("plot", "剧情", "叙事", "角色", "世界观", "设定", "主线", "支线", "npc")

    if any(key in lowered for key in competition_keys):
        return "competition"
    if any(key in lowered for key in monetization_keys):
        return "monetization"
    if any(key in lowered for key in brand_keys):
        return "brand_communication"
    if any(key in lowered for key in community_keys):
        return "community_conflict"
    if any(key in lowered for key in product_keys):
        return "product_experience"
    if any(key in lowered for key in plot_keys):
        return "plot_discussion"
    return None


def normalize_business_modes(raw_value: object, fallback_value: object = "") -> str:
    seen: list[str] = []
    for token in _split_modes(raw_value):
        mapped = _map_mode_token(token)
        if mapped and mapped not in seen:
            seen.append(mapped)

    if not seen:
        for token in _split_modes(fallback_value):
            mapped = _map_mode_token(token)
            if mapped and mapped not in seen:
                seen.append(mapped)

    if not seen:
        return "other_high_value" if str(raw_value or "").strip() else ""
    return "|".join(seen)
