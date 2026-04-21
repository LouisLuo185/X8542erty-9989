from config.settings import ANALYSIS_DIMENSIONS, COMMENT_TARGETS, SENTIMENT_LABELS, TARGET_GAMES


SYSTEM_PROMPT = """
你是一名中文游戏社区分析助手。请结合问题标题、回答摘要和评论文本，判断评论真正评价的对象与业务价值。
不要只根据表面情绪词下结论；如果评论是在嘲讽答主或玩家群体，不应误判为直接评价游戏本身。
请严格输出符合 JSON Schema 的结果。
""".strip()


def build_user_prompt(question_title: str, answer_summary: str, comment_text: str) -> str:
    return (
        f"问题标题：{question_title}\n"
        f"回答摘要：{answer_summary}\n"
        f"评论文本：{comment_text}\n"
        "请给出结构化分析。"
    )


def get_comment_analysis_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": SENTIMENT_LABELS},
            "dimension": {"type": "string", "enum": ANALYSIS_DIMENSIONS},
            "is_comparative": {"type": "boolean"},
            "target_game": {"type": "string", "enum": TARGET_GAMES},
            "comment_target": {"type": "string", "enum": COMMENT_TARGETS},
            "business_mode": {"type": "string"},
            "stance_summary": {"type": "string"},
            "confidence": {"type": "number"},
            "is_high_value_feedback": {"type": "boolean"},
            "needs_manual_review": {"type": "boolean"},
        },
        "required": [
            "sentiment",
            "dimension",
            "is_comparative",
            "target_game",
            "comment_target",
            "business_mode",
            "stance_summary",
            "confidence",
            "is_high_value_feedback",
            "needs_manual_review",
        ],
        "additionalProperties": False,
    }
