import pandas as pd

from config.settings import THREAD_SCHEMA
from src.context.answer_summarizer import infer_answer_stance, summarize_answer
from src.context.question_parser import infer_question_topic


def build_threads(content_df: pd.DataFrame, comment_df: pd.DataFrame) -> pd.DataFrame:
    if content_df.empty or comment_df.empty:
        return pd.DataFrame(columns=THREAD_SCHEMA)

    answers = content_df.copy()
    answers["answer_id"] = answers["content_id"]
    answers["question_title"] = answers["title"].fillna("")
    answers["question_topic"] = answers["question_title"].map(infer_question_topic)
    answers["answer_title"] = answers["title"].fillna("")
    answers["answer_text"] = answers["text"].fillna("")
    answers["answer_summary"] = answers["answer_text"].map(summarize_answer)
    answers["answer_stance"] = answers["answer_text"].map(infer_answer_stance)

    comments = comment_df.copy()
    comments["comment_text"] = comments["text"].fillna("")

    merged = comments.merge(
        answers[
            [
                "platform",
                "answer_id",
                "content_id",
                "question_id",
                "question_title",
                "question_topic",
                "answer_title",
                "answer_text",
                "answer_summary",
                "answer_stance",
            ]
        ],
        on=["platform", "content_id"],
        how="left",
    )

    merged["thread_id"] = merged["question_id"].fillna(merged["content_id"]).astype(str)
    merged["sub_comment_count"] = merged["sub_comment_count"].fillna(0)

    for column in THREAD_SCHEMA:
        if column not in merged.columns:
            merged[column] = None

    return merged[THREAD_SCHEMA].reset_index(drop=True)
