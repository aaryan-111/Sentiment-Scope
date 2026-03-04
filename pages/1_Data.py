"""
Phase 1 & 2 — Data: load, inspect, clean. Uses ai_toolkit where indicated.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from utils.config import CONFIG
from utils.data_loader import download_dataset, load_and_sample, load_from_upload, initial_inspection
from utils.cleaning import clean_dataframe

st.set_page_config(page_title="Data — SentimentScope", layout="wide")
st.title("Data — Load, Inspect & Clean")

# Sidebar: config
with st.sidebar:
    st.header("Config")
    data_source = st.radio(
        "Data source",
        ["Download from URL", "Upload file"],
        help="Paste a download URL or upload a CSV/JSON/JSONL (.gz ok) file.",
    )
    sample_size = st.number_input("Sample size", min_value=1000, value=min(50000, CONFIG["sample_size"]), step=5000)
    seed = CONFIG["random_seed"]
    text_col = CONFIG["text_column"]
    rating_col = CONFIG["rating_column"]

    if data_source == "Download from URL":
        data_url = st.text_input("Data URL", value=CONFIG["data_url"], help="JSONL.gz URL for Amazon Reviews 2023")
        data_path = st.text_input("Data path", value=CONFIG["data_path"], help="Folder for raw data")
    else:
        data_url = data_path = None

raw_file = Path(data_path or CONFIG["data_path"]) / "reviews.jsonl.gz" if data_source == "Download from URL" else None
if data_source == "Download from URL" and data_path and not Path(data_path).exists():
    Path(data_path).mkdir(parents=True, exist_ok=True)

# Load: from URL path or from upload
if data_source == "Download from URL":
    if st.button("Download dataset"):
        with st.spinner("Downloading…"):
            try:
                download_dataset(data_url, str(raw_file))
                st.success(f"Saved to {raw_file}")
            except Exception as e:
                st.error(str(e))
    load_clicked = st.button("Load & sample from path")
    path = raw_file if raw_file and raw_file.exists() else (data_path if data_path else None)
    if load_clicked and path and Path(path).exists():
        with st.spinner("Loading…"):
            try:
                df = load_and_sample(path, n=sample_size, seed=seed, text_column=text_col, rating_column=rating_col)
                st.session_state["df_raw"] = df
                st.session_state["df_clean"] = None
                out_dir = Path(CONFIG["processed_path"]) / "01_loaded"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / "raw_sample.csv"
                df.to_csv(out_file, index=False, encoding="utf-8")
                st.success(f"Loaded {len(df):,} rows. Saved to `{out_file}`.")
            except Exception as e:
                st.error(str(e))
    elif load_clicked and (not path or not Path(path).exists()):
        st.warning("Download the dataset first or set Data path to an existing file.")
else:
    uploaded = st.file_uploader("Upload CSV, JSON, or JSONL (.gz allowed)", type=["csv", "json", "jsonl", "gz"])
    if uploaded is not None and st.button("Load from uploaded file"):
        with st.spinner("Loading…"):
            try:
                df = load_from_upload(
                    uploaded, filename=uploaded.name, n=sample_size, seed=seed,
                    text_column=text_col, rating_column=rating_col,
                )
                st.session_state["df_raw"] = df
                st.session_state["df_clean"] = None
                out_dir = Path(CONFIG["processed_path"]) / "01_loaded"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / "raw_sample.csv"
                df.to_csv(out_file, index=False, encoding="utf-8")
                st.success(f"Loaded {len(df):,} rows from {uploaded.name}. Saved to `{out_file}`.")
            except Exception as e:
                st.error(str(e))

if "df_raw" in st.session_state:
    df_raw = st.session_state["df_raw"]
    insp = initial_inspection(df_raw)
    st.subheader("Raw data — Inspection")
    st.write("**Shape:**", insp["shape"])
    st.write("**Null counts**")
    st.dataframe(
        pd.DataFrame({"count": insp["null_counts"], "pct": insp["null_pct"]}).T,
        width="stretch",
    )
    st.write("**Sample rows**")
    st.dataframe(insp["sample"], width="stretch")

    # Clean
    st.subheader("Cleaning")
    if st.button("Run cleaning pipeline"):
        with st.spinner("Cleaning…"):
            df_clean = clean_dataframe(df_raw, text_column=text_col, rating_column=rating_col)
            st.session_state["df_clean"] = df_clean
            out_dir = Path(CONFIG["processed_path"]) / "02_cleaned"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "cleaned.csv"
            df_clean.to_csv(out_file, index=False, encoding="utf-8")
            st.success(f"Cleaned {len(df_clean):,} rows. Saved to `{out_file}`. Columns: {list(df_clean.columns)}")
    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"]
        st.write("**Clean text & sentiment sample**")
        if df_clean is not None and isinstance(df_clean, pd.DataFrame) and len(df_clean.columns):
            display_cols = [c for c in ["clean_text", "sentiment", rating_col] if c in df_clean.columns]
            if display_cols:
                st.dataframe(df_clean[display_cols].head(20), width="stretch")
            else:
                st.info("Cleaned data has no clean_text/sentiment/rating columns to show. Check the cleaning step.")
        else:
            st.info("Run **Run cleaning pipeline** above to see the clean text & sentiment sample.")
