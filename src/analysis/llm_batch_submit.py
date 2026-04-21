from pathlib import Path


def get_batch_submit_instructions(requests_path: str | Path) -> str:
    path = Path(requests_path)
    return (
        "当前项目已生成可上传的 Batch API 请求文件：\n"
        f"- {path}\n\n"
        "下一步通常是：\n"
        "1. 上传 jsonl 文件到你的 LLM 服务\n"
        "2. 创建 batch job\n"
        "3. 下载结果文件到 data/processed/llm_responses.jsonl\n"
        "4. 重新运行 pipeline 完成结果回填\n"
    )
