"""
Phase 3 & 4 — EDA & Visualization. Uses ai_toolkit.eda (visualizer).
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from utils.config import CONFIG
from utils.eda_pipeline import run_eda

st.set_page_config(page_title="EDA — SentimentScope", layout="wide")
st.title("Exploratory Data Analysis")
st.caption("Phase 3 & 4: analyses and visualizations via ai_toolkit.eda")

# Resolve data: session state first, then saved cleaned file
def _get_df():
    if "df_clean" in st.session_state and st.session_state["df_clean"] is not None:
        return st.session_state["df_clean"]
    cleaned_path = Path(CONFIG["processed_path"]) / "02_cleaned" / "cleaned.csv"
    if cleaned_path.exists():
        return pd.read_csv(cleaned_path, encoding="utf-8", nrows=None)
    return None

df = _get_df()

with st.sidebar:
    st.header("Config")
    text_col = st.text_input("Text column", value=CONFIG["text_column"], key="eda_text_col")
    sentiment_col = st.text_input("Sentiment column", value="sentiment", key="eda_sentiment_col")
    rating_col = st.text_input("Rating column", value=CONFIG["rating_column"], key="eda_rating_col")
    # Use clean_text for EDA when available
    if df is not None and "clean_text" in df.columns:
        text_col = "clean_text"
    max_sample = st.number_input(
        "Max samples for EDA",
        min_value=1000,
        value=min(50_000, CONFIG.get("sample_size", 100_000)),
        step=5000,
        help="Cap rows for faster runs (length dist, n-grams, wordclouds).",
    )
    top_ngram = st.number_input("Top N n-grams", min_value=5, value=20, step=5)
    run_clicked = st.button("Run EDA")

if df is None:
    st.info("Load and clean data on the **Data** page first. EDA uses `df_clean` from session or `data/processed/02_cleaned/cleaned.csv`.")
    st.stop()

# Ensure we have expected columns for EDA
if text_col not in df.columns:
    st.error(f"Text column `{text_col}` not found. Columns: {list(df.columns)}")
    st.stop()
if sentiment_col not in df.columns:
    st.warning(f"Sentiment column `{sentiment_col}` not found. Some EDA sections will be skipped.")
    sentiment_col = None
if rating_col not in df.columns:
    rating_col = None

if sentiment_col is None:
    # Create a dummy sentiment for display if missing
    df = df.copy()
    df["sentiment"] = "unknown"
    sentiment_col = "sentiment"

st.sidebar.metric("Rows available", f"{len(df):,}")

if run_clicked:
    with st.spinner("Running EDA…"):
        results = run_eda(
            df,
            text_col=text_col,
            sentiment_col=sentiment_col,
            rating_col=rating_col or "rating",
            max_sample=max_sample,
            top_ngram=top_ngram,
            wordcloud_sample_per_class=5_000,
            seed=CONFIG.get("random_seed", 42),
        )
    st.session_state["eda_results"] = results
    st.success(f"Done. {len(results)} findings.")

if "eda_results" not in st.session_state:
    st.info("Click **Run EDA** in the sidebar to generate analyses and plots.")
    st.stop()

results = st.session_state["eda_results"]

# Show error-only result
if len(results) == 1 and results[0].get("name") == "Error":
    st.error(results[0].get("summary", "Unknown error"))
    st.stop()

st.markdown("---")
st.subheader("Findings")

for i, r in enumerate(results):
    name = r.get("name", f"Finding {i+1}")
    summary = r.get("summary", {})
    fig = r.get("figure")

    with st.expander(f"**{name}**", expanded=(i < 3)):
        if isinstance(summary, dict):
            if "error" in summary:
                st.error(summary["error"])
            else:
                st.json(summary)
        else:
            st.write(summary)
        if fig is not None:
            st.pyplot(fig)
