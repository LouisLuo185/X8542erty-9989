[中文](./README.md) | English

## Online Demo Notice

Because of platform copyright and public-display restrictions, the online Streamlit demo **does not expose full original community text, user identifiers, or traceable links**.  
The public version only keeps sanitized aggregates, rewritten case summaries, and training-oriented sample labels for demonstrating the data cleaning, sample filtering, LLM routing, and weak-supervision workflow.

## Project Overview

This project is a **data cleaning, training-candidate selection, and LLM weak-supervision pipeline** for noisy Chinese long-form community text. Starting from public discussion data, it builds an end-to-end workflow covering raw text ingestion, context reconstruction, rule-based scoring, hard-sample routing, GPT Batch annotation, and sanitized demo export.

The main goal is not simple sentiment analysis. Instead, the project focuses on identifying which text samples are informative, structurally useful, context-complete enough, or valuable enough to be kept for downstream model-related tasks.

The current architecture has two layers:

- A rule layer for scalable, low-cost, explainable cleaning, feature extraction, and filtering
- An LLM layer for long text, context-heavy samples, mixed-stance text, and other hard samples

From a training-data perspective, this project should be understood as a **Chinese long-text corpus filtering and sample-routing prototype**, rather than a standard community analytics dashboard.

## What Problems It Solves

This project mainly addresses the following questions:

1. How to normalize raw `content/comment` text into a unified schema
2. How to detect noisy, low-quality, highly context-dependent, and high-information samples
3. How to use a rule-first workflow to reduce cost, then send only difficult samples to an LLM
4. How to assign training-side labels such as:
   - `keep_for_training`
   - `drop_as_noise`
   - `needs_context`
   - `hard_sample_for_llm`
   - `review_required`
5. How to map filtered samples to downstream model targets such as quality filters, sample routers, context-dependency detectors, and weak-label classifiers

## Pipeline Summary

```text
Raw Data -> Context Reconstruction -> Cleaning -> Feature Extraction
-> Rule Scoring -> Training Labels -> LLM Routing -> Batch Annotation
-> Merge -> Aggregation -> Public Demo Export
```

The project keeps two parallel views:

- `content` as the primary view for long-form, higher-information samples
- `comment` as the auxiliary view for community reaction, context risk, hard samples, and weak labels

## Outputs

The project currently produces three layers of outputs:

1. Private analysis outputs  
   Stored in `data/processed/`, containing full internal result tables that are not suitable for public release.

2. Public demo outputs  
   Stored in `data/samples/demo/`, containing only sanitized aggregates and rewritten display fields.

3. Streamlit dashboard  
   Supports both Chinese and English UI, and automatically falls back to the public demo data when private processed data is unavailable.

## Training-Side Labels and Model Targets

After rule scoring and again after LLM merge, the pipeline adds:

- `training_data_label`
- `training_data_reason`
- `training_sample_score`
- `model_target`

These fields answer two core questions:

- Should this sample be kept as a training candidate?
- If yes, what type of data-processing model is it most useful for?

Current model targets include:

- `quality_filter_model`
- `sample_router_model`
- `context_dependency_model`
- `weak_label_classifier`

## How To Run

### 1. Run the full private pipeline

```bash
python -m src.pipeline.run_pipeline --stage full
```

### 2. Generate only the public demo assets

```bash
python -m src.pipeline.run_pipeline --stage publish_demo
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

If `data/processed/` exists locally, the app loads the private processed outputs first.  
If those files are missing but `data/samples/demo/` exists, the app automatically switches to public demo mode.

## Open-Source Notice

The public repository does not include raw platform text, original answers/comments, user information, or traceable links.  
The public demo is only intended for learning, research, and portfolio presentation. If you want to reproduce the full pipeline, please collect compliant data yourself and run the private stages locally.
