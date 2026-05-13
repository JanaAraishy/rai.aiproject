import streamlit as st
import plotly.express as px
import pandas as pd

from backend import run_pipeline
from recommendation import generate_all_topics_insights

from charts import (
  
    plot_topic_distribution,
    plot_sentiment_heatmap,
    
    
)


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Arabic AI Analytics Dashboard",
    page_icon=r"C:\Users\user\OneDrive\Desktop\rai.ai project\logo.jpeg",
    layout="wide"
)


# =========================
# LOGO + TITLE
# =========================
col1, col2 = st.columns([1, 6])

with col1:
    st.image(
        r"C:\Users\user\OneDrive\Desktop\rai.ai project\logo.jpeg",
        width=100
    )

with col2:
    st.markdown(
        """
        <h1 style='margin-top:20px;'>
        📊 Arabic Sentiment + Topic Dashboard
        </h1>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("aura_data.csv")


df_raw = load_data()


# =========================
# RUN PIPELINE
# =========================
df, topic_info = run_pipeline(df_raw)


# =========================
# CLEAN SENTIMENT
# =========================
df["sentiment"] = (
    df["sentiment"]
    .fillna("")
    .astype(str)
    .str.lower()
)


# =========================
# HANDLE OUTLIER TOPIC (-1)
# =========================
topic_info = topic_info.copy()

if -1 not in topic_info["Topic"].values:
    topic_info = pd.concat([
        topic_info,
        pd.DataFrame([{
            "Topic": -1,
            "final_name": "Noise / Uncategorized",
            "Representation": "Noise",
            "Count": 0
        }])
    ], ignore_index=True)


# =========================
# TOPIC MAP
# =========================
topic_map = dict(zip(topic_info["Topic"], topic_info["final_name"]))


# =========================
# ADD TOPIC NAME
# =========================
df["topic_name"] = df["topic"].map(topic_map).fillna("Unknown")
df["topic_name"] = df["topic_name"].fillna("Noise / Uncategorized")


# =========================
# KPI METRICS
# =========================
total_reviews = len(df)
positive_reviews = (df["sentiment"] == "positive").sum()
negative_reviews = (df["sentiment"] == "negative").sum()

positive_rate = round((positive_reviews / total_reviews) * 100, 2) if total_reviews else 0


col1, col2, col3, col4 = st.columns(4)

col1.metric("📝 Total Reviews", total_reviews)
col2.metric("✅ Positive", positive_reviews)
col3.metric("❌ Negative", negative_reviews)
col4.metric("📈 Positive Rate", f"{positive_rate}%")


st.markdown("---")


# =========================
# TOPIC DISTRIBUTION
# =========================
st.subheader("🧠 Topic Distribution")

topic_distribution = (
    df.groupby("topic_name")
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=True)
)

fig = plot_topic_distribution(df)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(topic_distribution, use_container_width=True)




# =========================
# HEATMAP
# =========================
st.subheader("🔥 Sentiment Heatmap")

fig = plot_sentiment_heatmap(df)
st.plotly_chart(fig, use_container_width=True)


# =========================
# NEGATIVE TOPICS
# =========================
st.subheader("🚨 Most Negative Topics")

negative_df = df[df["sentiment"] == "negative"]

neg_distribution = (
    negative_df.groupby("topic_name")
    .size()
    .reset_index(name="Negative Reviews")
    .sort_values("Negative Reviews", ascending=False)
)

if not neg_distribution.empty:
    fig = px.bar(
        neg_distribution,
        x="topic_name",
        y="Negative Reviews",
        color="Negative Reviews",
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(neg_distribution, use_container_width=True)

else:
    st.info("No negative reviews found.")


# =========================
# TOPIC EXPLORER
# =========================
st.subheader("📝 Topic Explorer")

unique_topics = sorted(df["topic_name"].unique())

selected_topic = st.selectbox("Choose Topic", unique_topics)

topic_reviews = df[df["topic_name"] == selected_topic][
    ["cleaned_reviews", "sentiment", "confidence"]
]

st.dataframe(topic_reviews.head(20), use_container_width=True)


# =========================
# AI BUSINESS ANALYST
# =========================
st.markdown("---")
st.subheader("🤖 AI Business Analyst (Topic-Based)")


@st.cache_data
def cached_ai(df_small, topic_map):
    return generate_all_topics_insights(df_small, topic_map)


if st.button("🚀 Generate AI Insights"):

    with st.spinner("Analyzing reviews with Groq AI..."):

        ai_results = cached_ai(
            df[["topic", "cleaned_reviews", "sentiment"]],
            topic_map
        )

    for topic, insight in ai_results.items():
        with st.expander(f"📌 {topic}"):
            st.markdown(insight)


