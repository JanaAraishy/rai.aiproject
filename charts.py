import plotly.express as px
import pandas as pd
import plotly.express as px
import numpy as np

def plot_topic_distribution(df):
    topic_counts = (
        df.groupby("topic_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )

    fig = px.bar(
        topic_counts,
        x="count",
        y="topic_name",
        orientation="h",
        color="count",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        title="Topic Distribution",
        xaxis_title="Number of Reviews",
        yaxis_title="Topics",
        height=500
    )

    return fig



def plot_sentiment_heatmap(df):

    heatmap_data = (
        df.groupby(["topic_name", "sentiment"])
        .size()
        .reset_index(name="count")
        .pivot(index="topic_name", columns="sentiment", values="count")
        .fillna(0)
    )

    fig = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        title="Sentiment Heatmap (Topic vs Emotion)",
        height=500,
        xaxis_title="Sentiment",
        yaxis_title="Topic"
    )

    return fig


def plot_sentiment_trend(df, topic_name=None, freq="W"):

    df = df.copy()

    # OPTIONAL FILTER BY TOPIC
    if topic_name and topic_name != "All":
        df = df[df["topic_name"] == topic_name]

    # REAL DATE HANDLING (if no date → create stable index)
    if "date" not in df.columns:
        df["date"] = pd.date_range(
            start="2024-01-01",
            periods=len(df),
            freq="D"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # GROUP BY TIME + SENTIMENT
    trend = (
        df.groupby(
            [pd.Grouper(key="date", freq=freq), "sentiment"]
        )
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        trend,
        x="date",
        y="count",
        color="sentiment",
        markers=True,
        color_discrete_map={
            "positive": "#4CAF50",
            "negative": "#F44336",
            "neutral": "#2196F3"
        }
    )

    fig.update_layout(
        title="Sentiment Trend Over Time",
        xaxis_title="Time",
        yaxis_title="Number of Reviews",
        height=450
    )

    return fig