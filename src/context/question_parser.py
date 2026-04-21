from config.taxonomies import QUESTION_TOPIC_RULES


def infer_question_topic(question_title: str) -> str:
    title = question_title or ""
    best_topic = "plot_discussion"
    best_score = 0

    for topic, keywords in QUESTION_TOPIC_RULES.items():
        score = sum(keyword in title for keyword in keywords)
        if score > best_score:
            best_topic = topic
            best_score = score

    return best_topic
