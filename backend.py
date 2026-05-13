import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from cleaning import DataCleaner


# =========================
# LOAD MODELS
# =========================

model_path = r"C:\Users\user\OneDrive\Desktop\rai.ai project\final_model (1)"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=5,
    nr_topics="auto",
    verbose=True
)


# =========================
# PIPELINE
# =========================

def run_pipeline(df):

    df = df.copy()

    # CLEANING
    #df = DataCleaner(df).process()
    df = df.head(200)

    # SENTIMENT
    reviews = (
    df["cleaned_reviews"]
    .fillna("")
    .astype(str)
    .str.strip()
    .tolist()
)

    results = classifier(
        reviews,
        batch_size=32,
        truncation=True,
        max_length=256
    )

    df["sentiment"] = [r["label"] for r in results]
    df["confidence"] = [r["score"] for r in results]

    # TOPICS
    topics, probs = topic_model.fit_transform(reviews)

    df["topic"] = topics

    topic_info = topic_model.get_topic_info()


    # CATEGORY PER TOPIC
    def get_category_mode(df, topic_id):

        sub = df[df["topic"] == topic_id]

        if sub.empty:
            return "Unknown"

        mode_series = sub["category"].mode()

        if mode_series.empty:
            return "Unknown"

        return mode_series.iloc[0]


    topic_info["category_name"] = topic_info["Topic"].apply(
        lambda t: get_category_mode(df, t)
    )


    # =========================
    # TOPIC WORDS + FINAL NAME
    # =========================

    def get_topic_words(topic_id):
        if topic_id == -1:
            return "Noise"
        
        words = topic_model.get_topic(topic_id)
        
        if words is None:
            return "Unknown"
        
        return ", ".join([w[0] for w in words[:3]])


    topic_info["topic_words"] = topic_info["Topic"].apply(get_topic_words)

    topic_info["final_name"] = topic_info.apply(
        lambda row: f"{row['category_name']}",
        axis=1
    )

    return df, topic_info