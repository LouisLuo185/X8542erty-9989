import pandas as pd

from config.settings import CONTENT_BASE_SCHEMA
from src.context.answer_summarizer import infer_answer_stance, summarize_answer
from src.context.question_parser import infer_question_topic


def build_content_base_table(content_df: pd.DataFrame) -> pd.DataFrame:
    if content_df.empty:
        return pd.DataFrame(columns=CONTENT_BASE_SCHEMA)

    answers = content_df.copy()
    answers["answer_id"] = answers["content_id"]
    answers["question_title"] = answers["title"].fillna("")
    answers["question_topic"] = answers["question_title"].map(infer_question_topic)
    answers["answer_title"] = answers["title"].fillna("")
    answers["answer_text"] = answers["text"].fillna("")
    answers["answer_summary"] = answers["answer_text"].map(summarize_answer)
    answers["answer_stance"] = answers["answer_text"].map(infer_answer_stance)
    answers["comment_count"] = pd.to_numeric(answers["comment_count"], errors="coerce").fillna(0).astype(int)
    answers["likes"] = pd.to_numeric(answers["likes"], errors="coerce").fillna(0).astype(int)

    for column in CONTENT_BASE_SCHEMA:
        if column not in answers.columns:
            answers[column] = None
    return answers[CONTENT_BASE_SCHEMA].reset_index(drop=True)
