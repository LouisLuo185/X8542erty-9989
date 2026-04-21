def summarize_answer(answer_text: str, limit: int = 140) -> str:
    text = (answer_text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def infer_answer_stance(answer_text: str) -> str:
    text = answer_text or ""
    positive_markers = ["好玩", "优秀", "惊艳", "很棒", "喜欢", "感动", "强"]
    negative_markers = ["无聊", "失望", "差", "拖沓", "骗氪", "不行"]
    pos = sum(marker in text for marker in positive_markers)
    neg = sum(marker in text for marker in negative_markers)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"
