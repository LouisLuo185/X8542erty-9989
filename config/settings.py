from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SAMPLES_DIR = DATA_DIR / "samples"
DATA_DEMO_DIR = DATA_SAMPLES_DIR / "demo"
OUTPUTS_DIR = BASE_DIR / "outputs"
MANIFEST_DIR = DATA_DIR / "outputs" / "manifests"

TOPIC_KEYWORD = "原神"

ANALYSIS_DIMENSIONS = [
    "剧情内容",
    "角色设计",
    "美术音乐",
    "玩法体验",
    "商业化",
    "品牌传播",
]

BUSINESS_MODES = [
    "product_experience",
    "monetization",
    "brand_communication",
    "community_conflict",
    "competition",
    "plot_discussion",
    "other_high_value",
]

COMMENT_TARGETS = ["game", "answer", "community", "mixed", "unclear"]
SENTIMENT_LABELS = ["positive", "neutral", "negative", "mixed"]
TARGET_GAMES = ["原神", "鸣潮", "双方", "不明确"]

MIN_TEXT_LENGTH = 8
HIGH_VALUE_WEIGHT = 1.8
MEDIUM_VALUE_WEIGHT = 1.0
LOW_VALUE_WEIGHT = 0.2

LLM_MODEL = "gpt-5-mini"
LLM_ROUTE_MIN_LENGTH = 220
LLM_ROUTE_MIN_PRIORITY = 5

RAW_CONTENT_SCHEMA = [
    "platform",
    "source_type",
    "content_id",
    "question_id",
    "title",
    "text",
    "time",
    "url",
    "likes",
    "comment_count",
    "source_keyword",
]

RAW_COMMENT_SCHEMA = [
    "platform",
    "source_type",
    "content_id",
    "comment_id",
    "parent_comment_id",
    "text",
    "time",
    "likes",
    "ip_location",
    "sub_comment_count",
]

THREAD_SCHEMA = [
    "platform",
    "thread_id",
    "question_id",
    "question_title",
    "question_topic",
    "answer_id",
    "answer_title",
    "answer_text",
    "answer_summary",
    "answer_stance",
    "comment_id",
    "parent_comment_id",
    "comment_text",
    "time",
    "likes",
    "ip_location",
    "sub_comment_count",
]

CONTENT_BASE_SCHEMA = [
    "platform",
    "question_id",
    "question_title",
    "question_topic",
    "answer_id",
    "answer_title",
    "answer_text",
    "answer_summary",
    "answer_stance",
    "time",
    "url",
    "likes",
    "comment_count",
    "source_keyword",
]

CONTENT_FEATURE_SCHEMA = CONTENT_BASE_SCHEMA + [
    "clean_text",
    "text_length",
    "low_value_reason",
    "is_low_value",
    "simhash_u64",
    "dedup_cluster_id",
    "is_dedup_representative",
    "causal_hit_count",
    "comparative_hit_count",
    "conclusion_hit_count",
    "impact_hit_count",
    "actionability_term_hit_count",
    "answer_coupling_hit_count",
    "reference_dependency_hit_count",
    "question_term_hit_count",
    "plot_term_hit_count",
    "product_term_hit_count",
    "monetization_term_hit_count",
    "brand_term_hit_count",
    "community_term_hit_count",
    "competitor_term_hit_count",
    "target_specificity_term_hit_count",
    "target_specificity_group_count",
    "canonical_entity_hit_count",
    "matched_entities",
    "sarcasm_hit_count",
    "abstract_slang_hit_count",
    "sarcasm_marker_hit_count",
    "feedback_modes",
    "target_specificity_score",
    "actionability_score",
    "quality_score",
    "context_dependency_score",
    "business_value_score",
    "feedback_entropy_score",
    "is_actionable_feedback",
    "actionable_reason",
    "high_value_reason",
    "is_high_value_feedback_pre_llm",
    "base_weight",
    "engagement_weight",
    "final_weight",
]

CONTENT_RULE_SCHEMA = CONTENT_FEATURE_SCHEMA + [
    "positive_hits",
    "negative_hits",
    "rule_sentiment",
    "rule_dimension",
    "rule_reason",
    "rule_confidence",
    "rule_is_comparative",
    "rule_target_game",
    "rule_comment_target",
    "needs_llm_analysis",
    "llm_priority_score",
    "llm_route_reason",
]

CONTENT_FINAL_SCHEMA = CONTENT_RULE_SCHEMA + [
    "llm_sentiment",
    "llm_dimension",
    "llm_is_comparative",
    "llm_target_game",
    "llm_comment_target",
    "llm_stance_summary",
    "llm_confidence",
    "llm_is_high_value_feedback",
    "llm_business_mode",
    "llm_needs_manual_review",
    "annotation_source",
    "sentiment",
    "dimension",
    "reason",
    "confidence",
    "is_comparative",
    "target_game",
    "comment_target",
    "business_mode",
    "is_high_value_feedback",
    "needs_manual_review",
    "context_ready_text",
]

FEATURE_SCHEMA = THREAD_SCHEMA + [
    "clean_text",
    "text_length",
    "low_value_reason",
    "is_low_value",
    "simhash_u64",
    "dedup_cluster_id",
    "is_dedup_representative",
    "causal_hit_count",
    "comparative_hit_count",
    "conclusion_hit_count",
    "impact_hit_count",
    "actionability_term_hit_count",
    "answer_coupling_hit_count",
    "reference_dependency_hit_count",
    "question_term_hit_count",
    "plot_term_hit_count",
    "product_term_hit_count",
    "monetization_term_hit_count",
    "brand_term_hit_count",
    "community_term_hit_count",
    "competitor_term_hit_count",
    "target_specificity_term_hit_count",
    "target_specificity_group_count",
    "canonical_entity_hit_count",
    "matched_entities",
    "sarcasm_hit_count",
    "abstract_slang_hit_count",
    "sarcasm_marker_hit_count",
    "feedback_modes",
    "target_specificity_score",
    "actionability_score",
    "quality_score",
    "context_dependency_score",
    "business_value_score",
    "feedback_entropy_score",
    "is_actionable_feedback",
    "actionable_reason",
    "high_value_reason",
    "is_high_value_feedback_pre_llm",
    "base_weight",
    "engagement_weight",
    "final_weight",
]

RULE_SCHEMA = FEATURE_SCHEMA + [
    "positive_hits",
    "negative_hits",
    "rule_sentiment",
    "rule_dimension",
    "rule_reason",
    "rule_confidence",
    "rule_is_comparative",
    "rule_target_game",
    "rule_comment_target",
    "needs_llm_analysis",
    "llm_priority_score",
    "llm_route_reason",
]

FINAL_SCHEMA = RULE_SCHEMA + [
    "llm_sentiment",
    "llm_dimension",
    "llm_is_comparative",
    "llm_target_game",
    "llm_comment_target",
    "llm_stance_summary",
    "llm_confidence",
    "llm_is_high_value_feedback",
    "llm_business_mode",
    "llm_needs_manual_review",
    "annotation_source",
    "sentiment",
    "dimension",
    "reason",
    "confidence",
    "is_comparative",
    "target_game",
    "comment_target",
    "business_mode",
    "is_high_value_feedback",
    "needs_manual_review",
    "context_ready_text",
]
