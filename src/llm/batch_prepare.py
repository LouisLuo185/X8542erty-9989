import json
from pathlib import Path

import pandas as pd

from config.prompts import SYSTEM_PROMPT, build_user_prompt, get_comment_analysis_schema
from config.settings import LLM_MODEL


def build_context_ready_text(row: pd.Series) -> str:
    return build_user_prompt(
        str(row.get("question_title", "")),
        str(row.get("answer_summary", "")),
        str(row.get("clean_text", "")),
    )


def build_batch_request_row(row: pd.Series) -> dict:
    comment_id = row.get("comment_id")
    answer_id = row.get("answer_id")
    if pd.notna(comment_id):
        custom_id = f"comment_{int(comment_id)}"
    elif pd.notna(answer_id):
        custom_id = f"answer_{int(answer_id)}"
    else:
        custom_id = f"row_{row.name}"
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": LLM_MODEL,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_context_ready_text(row)},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "comment_analysis",
                    "schema": get_comment_analysis_schema(),
                }
            },
        },
    }


def export_batch_requests(df: pd.DataFrame, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for _, row in df.iterrows():
            file.write(json.dumps(build_batch_request_row(row), ensure_ascii=False) + "\n")
    return target
