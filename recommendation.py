from llm import ask_llm


# =========================
# BUILD TOPIC PAYLOAD
# =========================
def build_topic_payload(df, topic_map, max_reviews=5):

    payload = {}

    # clean safely
    df = df.copy()

    df["cleaned_reviews"] = (
        df["cleaned_reviews"]
        .fillna("")
        .astype(str)
    )

    df["sentiment"] = (
        df["sentiment"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    for topic_id in df["topic"].unique():

        topic_name = topic_map.get(
            topic_id,
            f"Topic {topic_id}"
        )

        topic_df = df[df["topic"] == topic_id]

        # positive
        positive_reviews = (
            topic_df[
                topic_df["sentiment"] == "positive"
            ]["cleaned_reviews"]
            .head(max_reviews)
            .tolist()
        )

        # negative
        negative_reviews = (
            topic_df[
                topic_df["sentiment"] == "negative"
            ]["cleaned_reviews"]
            .head(max_reviews)
            .tolist()
        )

        # skip empty
        if len(positive_reviews) == 0 and len(negative_reviews) == 0:
            continue

        payload[topic_name] = {
            "positive_reviews": positive_reviews,
            "negative_reviews": negative_reviews
        }

    return payload


# =========================
# GENERATE SINGLE TOPIC
# =========================
def generate_topic_insights(topic_name, topic_data):

    positive_text = "\n".join(
        [f"- {r}" for r in topic_data["positive_reviews"]]
    )

    negative_text = "\n".join(
        [f"- {r}" for r in topic_data["negative_reviews"]]
    )

    prompt = f"""
أنت محلل احترافي لتقييمات المستخدمين.

حلل فقط البيانات التالية بدون أي افتراضات خارجية.

الموضوع:
{topic_name}

========================
التقييمات الإيجابية:
{positive_text if positive_text else "لا يوجد"}

========================
التقييمات السلبية:
{negative_text if negative_text else "لا يوجد"}

========================

المطلوب:

1- أكثر الأشياء التي أعجبت المستخدمين

2- أبرز المشاكل المتكررة

3- تحسينات عملية واضحة بناءً على الشكاوى

4- 3 توصيات واضحة للإدارة

========================

قواعد مهمة:
- لا تخترع معلومات
- لا تذكر أسباب جذرية
- لا تستخدم كلام عام
- اعتمد فقط على النصوص
- اكتب بالعربية فقط
- استخدم نقاط قصيرة وواضحة
"""

    try:

        response = ask_llm(
            prompt,
            max_tokens=400
        )

        # fallback if empty
        if response is None or response.strip() == "":
            return "لم يتم توليد تحليل من النموذج."

        return response

    except Exception as e:

        return f"خطأ في تحليل الموضوع: {str(e)}"


# =========================
# GENERATE ALL TOPICS
# =========================
def generate_all_topics_insights(df, topic_map):

    payload = build_topic_payload(df, topic_map)

    results = {}

    for topic_name, topic_data in payload.items():

        results[topic_name] = generate_topic_insights(
            topic_name,
            topic_data
        )

    return results