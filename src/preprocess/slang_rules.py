from config.taxonomies import (
    ABSTRACT_SLANG_TERMS,
    ACTIONABILITY_TERMS,
    ANSWER_COUPLING_TERMS,
    BRAND_TERMS,
    CAUSAL_WORDS,
    CONCLUSION_WORDS,
    COMMUNITY_TERMS,
    COMPARATIVE_WORDS,
    COMPETITOR_TERMS,
    ENTITY_ALIAS_TO_CANONICAL_ZH,
    IMPACT_TERMS,
    MONETIZATION_TERMS,
    PLOT_TERMS,
    PRODUCT_TERMS,
    QUESTION_TOPIC_RULES,
    REFERENCE_DEPENDENCY_TERMS,
    SARCASM_MARKERS,
    SARCASM_TERMS,
    TARGET_SPECIFICITY_TERMS,
)


def count_hits(text: str, keywords: list[str]) -> int:
    return sum(keyword in text for keyword in keywords)


def count_group_hits(text: str, term_groups: dict[str, list[str]]) -> tuple[int, int]:
    hit_groups = 0
    hit_terms = 0
    for terms in term_groups.values():
        group_terms = [term for term in terms if term in text]
        if group_terms:
            hit_groups += 1
            hit_terms += len(group_terms)
    return hit_groups, hit_terms


def extract_canonical_entities(text: str) -> list[str]:
    seen: list[str] = []
    for alias, canonical in ENTITY_ALIAS_TO_CANONICAL_ZH.items():
        if alias in text and canonical not in seen:
            seen.append(canonical)
    return seen


def infer_feedback_modes(text: str, question_title: str) -> list[str]:
    modes: list[str] = []
    if count_hits(text, PRODUCT_TERMS) or any(keyword in (question_title or "") for keyword in QUESTION_TOPIC_RULES["product_experience"]):
        modes.append("product_experience")
    if count_hits(text, MONETIZATION_TERMS):
        modes.append("monetization")
    if count_hits(text, BRAND_TERMS):
        modes.append("brand_communication")
    if count_hits(text, COMMUNITY_TERMS):
        modes.append("community_conflict")
    if count_hits(text, COMPETITOR_TERMS) or count_hits(text, COMPARATIVE_WORDS):
        modes.append("competition")
    if count_hits(text, PLOT_TERMS):
        modes.append("plot_discussion")
    return modes


def build_term_features(text: str, question_title: str) -> dict[str, int | str]:
    feedback_modes = infer_feedback_modes(text, question_title)
    target_group_hits, target_term_hits = count_group_hits(text, TARGET_SPECIFICITY_TERMS)
    matched_entities = extract_canonical_entities(text)
    return {
        "causal_hit_count": count_hits(text, CAUSAL_WORDS),
        "comparative_hit_count": count_hits(text, COMPARATIVE_WORDS),
        "conclusion_hit_count": count_hits(text, CONCLUSION_WORDS),
        "impact_hit_count": count_hits(text, IMPACT_TERMS),
        "actionability_term_hit_count": count_hits(text, ACTIONABILITY_TERMS),
        "answer_coupling_hit_count": count_hits(text, ANSWER_COUPLING_TERMS),
        "reference_dependency_hit_count": count_hits(text, REFERENCE_DEPENDENCY_TERMS),
        "question_term_hit_count": count_hits(question_title or "", sum(QUESTION_TOPIC_RULES.values(), [])),
        "plot_term_hit_count": count_hits(text, PLOT_TERMS),
        "product_term_hit_count": count_hits(text, PRODUCT_TERMS),
        "monetization_term_hit_count": count_hits(text, MONETIZATION_TERMS),
        "brand_term_hit_count": count_hits(text, BRAND_TERMS),
        "community_term_hit_count": count_hits(text, COMMUNITY_TERMS),
        "competitor_term_hit_count": count_hits(text, COMPETITOR_TERMS),
        "target_specificity_term_hit_count": target_term_hits,
        "target_specificity_group_count": target_group_hits,
        "canonical_entity_hit_count": len(matched_entities),
        "matched_entities": "|".join(matched_entities),
        "sarcasm_hit_count": count_hits(text, SARCASM_TERMS),
        "abstract_slang_hit_count": count_hits(text, ABSTRACT_SLANG_TERMS),
        "sarcasm_marker_hit_count": count_hits(text, SARCASM_MARKERS),
        "feedback_modes": "|".join(feedback_modes),
    }
