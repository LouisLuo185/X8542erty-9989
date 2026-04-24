# 中文 | [English](./README_EN.md)

## 线上演示说明

由于平台内容版权与公开展示限制，线上 Streamlit demo **不会展示完整原始社区文本、用户标识或可回溯链接**。  
公开版本仅保留脱敏后的统计结果、改写后的案例摘要和训练候选样本标签，用于展示数据清洗、样本筛选、LLM 路由与弱监督标注流程。

## 项目简介

这是一个面向中文长文本社区语料的 **数据清洗、训练候选样本筛选与 LLM 弱监督标注** 项目。项目以公开讨论数据为原型，构建了一条从原始文本接入、上下文重建、规则打分、复杂样本路由到 GPT Batch 回填的完整流水线。它的重点不是单纯做情感分析，而是从高噪声文本中识别出更适合下游模型学习和数据治理的样本。

项目当前采用双层架构：

- 规则层负责大规模、低成本、可解释的清洗、特征提取和样本筛选
- LLM 层负责处理长文本、强上下文依赖文本、复杂立场文本和 hard samples

从训练数据视角看，这个项目更适合被理解为一条 **中文长文本语料过滤与样本路由原型**，而不是一个普通的社区舆情看板。

## 项目目标

这个项目主要解决以下问题：

1. 如何把原始 `content/comment` 文本整理成统一 schema。
2. 如何识别噪声、低质量、强上下文依赖和高信息密度样本。
3. 如何用规则层先做低成本过滤，再把复杂样本送入 LLM。
4. 如何给样本打上训练侧标签，例如：
   - `keep_for_training`
   - `drop_as_noise`
   - `needs_context`
   - `hard_sample_for_llm`
   - `review_required`
5. 如何将这些样本进一步映射到适合训练的模型目标，例如质量过滤模型、样本路由模型、上下文依赖检测模型和弱监督标签分类器。

## 流水线概览

```text
Raw Data -> Context Reconstruction -> Cleaning -> Feature Extraction
-> Rule Scoring -> Training Labels -> LLM Routing -> Batch Annotation
-> Merge -> Aggregation -> Public Demo Export
```

当前项目同时维护两条视角：

- `content` 主视角：用于观察长文本主内容是否具有高信息密度和训练价值
- `comment` 辅视角：用于观察社区回应、上下文风险、hard samples 和弱监督标签

## 当前输出

项目当前会输出三层结果：

1. 私有分析结果  
   位于 `data/processed/`，包含完整内部分析表，不适合公开。

2. 公开 demo 结果  
   位于 `data/samples/demo/`，仅包含脱敏后的聚合结果和改写后的展示字段。

3. Streamlit 展示页  
   支持中英文切换，并优先读取私有结果；如果私有结果不存在，则自动回退到公开 demo 数据。

## 训练侧标签与模型目标

当前项目在规则层和 LLM 回填后都会为样本补充以下字段：

- `training_data_label`
- `training_data_reason`
- `training_sample_score`
- `model_target`

这些字段用于回答两个核心问题：

- 这条文本是否适合作为训练候选样本？
- 如果适合，它更适合支持哪一类数据处理模型？

当前支持的模型目标包括：

- `quality_filter_model`
- `sample_router_model`
- `context_dependency_model`
- `weak_label_classifier`

## 运行方式

### 1. 运行完整分析链路

```bash
python -m src.pipeline.run_pipeline --stage full
```

### 2. 仅生成公开 demo 数据

```bash
python -m src.pipeline.run_pipeline --stage publish_demo
```

### 3. 启动展示页

```bash
streamlit run app.py
```

如果本地存在 `data/processed/` 的正式结果，应用会优先读取正式结果。  
如果正式结果不存在，但 `data/samples/demo/` 存在，应用会自动进入公开演示模式。

## 开源说明

本仓库公开版本不包含原始平台文本数据，不包含原始评论、回答、用户信息或可回溯链接。  
公开展示仅用于学习、研究和作品集演示。如需复现完整流程，请使用者自行获取符合平台规则的数据，并在本地环境运行。
