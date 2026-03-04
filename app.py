"""
SentimentScope — Streamlit entry point.
Run from project root: streamlit run app.py
"""
import os
import warnings
import logging

# Suppress BeautifulSoup spurious URL-resemblance warning (fires on review text that looks like a URL)
try:
    from bs4 import MarkupResemblesLocatorWarning
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
except ImportError:
    pass

# Suppress HuggingFace Hub unauthenticated-request nag
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress noisy model-load logs from transformers / sentence-transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st

st.set_page_config(
    page_title="SentimentScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("SentimentScope")
st.caption("Text analysis from data to models — SentimentScope (sentiment analysis showcase)")
st.markdown("---")
st.markdown(
    "Use the sidebar to open **Data** (load & clean), **EDA**, **Preprocessing**, **ML Models**, **DL Models**, or **Evaluation**."
)
st.info("Start with **1_Data** to load the dataset and run cleaning.")
