# vLLM / SGLang offline teacher (POC notes)

This project’s teacher step is defined by:

- System + user text: `config/prompts.py` (`SYSTEM_PROMPT`, `build_user_prompt`)
- Structured JSON output: `get_comment_analysis_schema()`

To run the same task locally with **vLLM**:

1. Serve a chat/completions-compatible model with JSON constrained decoding (or post-parse JSON).
2. For each line in `data/processed/llm_requests.jsonl`, map `body.input` to your server’s chat template; keep field names aligned with `src/llm/batch_parse.py` expectations (`sentiment`, `dimension`, `is_high_value_feedback`, …).
3. Write results in the same **batch result JSONL** shape expected by `parse_batch_results` (see OpenAI batch response wrapper: `custom_id`, `response.body.output_text` as JSON string).

**SGLang**: same contract—batch prompts from `llm_candidates.csv` / prepared JSONL, aggregate into `llm_responses.jsonl`, then `python -m src.pipeline.run_pipeline --stage llm_merge`.

Throughput and **$/1K rows** comparisons belong in README or experiment notes once you have measured numbers from your hardware and API pricing.
