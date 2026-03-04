"""
Single source of truth for paths, splits, and hyperparameters.
Used by all phases and Streamlit pages.
"""
from pathlib import Path

# Base paths (relative to project root)
_ROOT = Path(__file__).resolve().parent.parent

# Amazon Reviews 2023 — McAuley Lab (UCSD). JSONL.gz per category.
# Swap category in URL: Digital_Music, Software, Electronics, etc.
DATA_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Digital_Music.jsonl.gz"

CONFIG = {
    # Data
    "data_url": DATA_URL,
    "data_path": str(_ROOT / "data" / "raw"),
    "processed_path": str(_ROOT / "data" / "processed"),
    "sample_size": 150_000,
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,
    # Column names (Amazon 2023 JSONL: rating, review_text/text, etc.)
    "text_column": "review_text",
    "rating_column": "rating",
    # Text
    "max_features": 50_000,
    "ngram_range": (1, 2),
    "max_seq_len": 256,
    "min_word_freq": 2,
    # Classical ML
    "cv_folds": 5,
    "n_jobs": -1,
    # Deep Learning
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-3,
    "patience": 3,
    # BERT (use reduced sample when running DistilBERT — CPU/low RAM friendly)
    "bert_model": "distilbert-base-uncased",
    "bert_sample_size": 20_000,
    "bert_epochs": 3,
    "bert_lr": 2e-5,
    "bert_batch_size": 32,
    # Sentence Transformers
    "sbert_model": "all-MiniLM-L6-v2",
    # Paths
    "outputs_dir": str(_ROOT / "outputs"),
    "saved_models_dir": str(_ROOT / "saved_models"),
}

# Sentiment mapping: 1–2 → negative, 3 → neutral, 4–5 → positive
SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
RATING_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
