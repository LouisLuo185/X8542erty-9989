import json
from pathlib import Path

import pandas as pd


def _extract_text_payload(record: dict) -> str:
    response_body = record.get("response", {}).get("body", {})

    if isinstance(response_body.get("output_text"), str):
        return response_body["output_text"]

    for item in response_body.get("output", []):
        for content in item.get("content", []):
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value

    return ""


def parse_batch_results(results_path: str | Path) -> pd.DataFrame:
    path = Path(results_path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            payload_text = _extract_text_payload(record)
            if not payload_text:
                continue

            payload = json.loads(payload_text)
            rows.append(
                {
                    "custom_id": record.get("custom_id"),
                    "llm_sentiment": payload.get("sentiment"),
                    "llm_dimension": payload.get("dimension"),
                    "llm_is_comparative": payload.get("is_comparative"),
                    "llm_target_game": payload.get("target_game"),
                    "llm_stance_summary": payload.get("stance_summary"),
                    "llm_confidence": payload.get("confidence"),
                    "llm_needs_manual_review": payload.get("needs_manual_review"),
                }
            )

    return pd.DataFrame(rows)
