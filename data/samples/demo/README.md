# Demo Data

This folder stores the public demo assets used by the Streamlit app when private processed data is not available.

These files are intended for portfolio display only:

- no raw Zhihu exports
- no original answer/comment body text
- no user identifiers or source links
- only sanitized, aggregated, or rewritten demo content

To regenerate these files locally after running the private pipeline, use:

```bash
python -m src.pipeline.run_pipeline --stage publish_demo
```
