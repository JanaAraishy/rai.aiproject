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
# إعداد الصفحة
# =========================
st.set_page_config(
    page_title="لوحة تحليل رأي عربي",
    page_icon=r"C:\Users\user\OneDrive\Desktop\rai.ai project\logo.jpeg",
    layout="wide"
)

# =========================
# الشعار والعنوان
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
        لوحة تحليل المشاعر والمواضيع (العربية)
        </h1>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")


# =========================
# تحميل البيانات
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("aura_data.csv")

df_raw = load_data()


# =========================
# تشغيل المعالجة
# =========================
df, topic_info = run_pipeline(df_raw)

df["sentiment"] = (
    df["sentiment"]
    .fillna("")
    .astype(str)
    .str.lower()
)

# =========================
# معالجة المواضيع غير المصنفة
# =========================
topic_info = topic_info.copy()

if -1 not in topic_info["Topic"].values:
    topic_info = pd.concat([
        topic_info,
        pd.DataFrame([{
            "Topic": -1,
            "final_name": "ضوضاء / غير مصنف",
            "Representation": "Noise",
            "Count": 0
        }])
    ], ignore_index=True)


# =========================
# خريطة المواضيع
# =========================
topic_map = dict(zip(topic_info["Topic"], topic_info["final_name"]))

df["topic_name"] = df["topic"].map(topic_map).fillna("غير معروف")


# =========================
# الإحصائيات الرئيسية
# =========================
total_reviews = len(df)
positive_reviews = (df["sentiment"] == "positive").sum()
negative_reviews = (df["sentiment"] == "negative").sum()

positive_rate = round((positive_reviews / total_reviews) * 100, 2) if total_reviews else 0


col1, col2, col3, col4 = st.columns(4)

col1.metric("إجمالي المراجعات", total_reviews)
col2.metric("إيجابي", positive_reviews)
col3.metric("سلبي", negative_reviews)
col4.metric("نسبة الإيجابي", f"{positive_rate}%")


st.markdown("---")


# =========================
# توزيع المواضيع
# =========================
st.subheader("توزيع المواضيع")

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
# خريطة الحرارة للمشاعر
# =========================
st.subheader("خريطة حرارة المشاعر")

fig = plot_sentiment_heatmap(df)
st.plotly_chart(fig, use_container_width=True)


# =========================
# أكثر المواضيع سلبية
# =========================
st.subheader("أكثر المواضيع سلبية")

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
    st.info("لا توجد مراجعات سلبية")


# =========================
# مستكشف المواضيع
# =========================
st.subheader("مستكشف المواضيع")

unique_topics = sorted(df["topic_name"].unique())
selected_topic = st.selectbox("اختر الموضوع", unique_topics)

topic_reviews = df[df["topic_name"] == selected_topic][
    ["cleaned_reviews", "sentiment", "confidence"]
]

st.dataframe(topic_reviews.head(20), use_container_width=True)


# =========================
# محلل الذكاء الاصطناعي
# =========================
st.markdown("---")
st.subheader("تحليل الأعمال باستخدام الذكاء الاصطناعي")


@st.cache_data
def cached_ai(df_small, topic_map):
    return generate_all_topics_insights(df_small, topic_map)


if st.button("توليد الرؤى باستخدام الذكاء الاصطناعي"):

    with st.spinner("جاري تحليل المراجعات..."):

        ai_results = cached_ai(
            df[["topic", "cleaned_reviews", "sentiment"]],
            topic_map
        )

    for topic, insight in ai_results.items():
        with st.expander(topic):
            st.markdown(insight)
