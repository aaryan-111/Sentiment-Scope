"""
Phase 6 — Feature engineering: BoW, TF-IDF, Word2Vec, SBERT, text stats.
Uses ai_toolkit.nlp for BoW, TF-IDF, SBERT, text stats. Word2Vec in-repo (gensim).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.config import CONFIG

try:
    import ai_toolkit.nlp as nlp_toolkit
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False
    nlp_toolkit = None


def build_bow(
    train_texts: list[str],
    test_texts: Optional[list[str]] = None,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 1),
) -> tuple[Any, np.ndarray, Optional[np.ndarray]]:
    """CountVectorizer; fit on train, transform train and optionally test. Toolkit."""
    if not _NLP_AVAILABLE or nlp_toolkit is None:
        raise RuntimeError("ai_toolkit.nlp required for build_bow")
    return nlp_toolkit.build_bow(
        train_texts, test_texts=test_texts,
        max_features=max_features, ngram_range=ngram_range,
    )


def build_tfidf(
    train_texts: list[str],
    test_texts: Optional[list[str]] = None,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[Any, np.ndarray, Optional[np.ndarray]]:
    """TfidfVectorizer; fit on train, transform train and optionally test. Toolkit."""
    if not _NLP_AVAILABLE or nlp_toolkit is None:
        raise RuntimeError("ai_toolkit.nlp required for build_tfidf")
    return nlp_toolkit.build_tfidf(
        train_texts, test_texts=test_texts,
        max_features=max_features, ngram_range=ngram_range,
    )


def extract_text_stats(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Char count, word count, avg word length, punct count, capital ratio. Toolkit."""
    if not _NLP_AVAILABLE or nlp_toolkit is None:
        raise RuntimeError("ai_toolkit.nlp required for extract_text_stats")
    return nlp_toolkit.extract_text_stats(df, text_col)


def train_word2vec(
    tokenized_texts: list[list[str]],
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 5,
    seed: int = 42,
) -> Any:
    """Train Word2Vec on tokenized corpus. Gensim."""
    from gensim.models import Word2Vec
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        seed=seed,
    )
    return model


def get_tfidf_weighted_w2v(
    texts: list[str],
    w2v_model: Any,
    tfidf_vectorizer: Any,
) -> np.ndarray:
    """Document vector = weighted average of word vectors by TF-IDF weight. In-repo."""
    tokenized = [t.split() for t in texts]
    vocab_w2v = set(w2v_model.wv.index_to_key)
    X_tfidf = tfidf_vectorizer.transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    vec_size = w2v_model.wv.vector_size
    out = np.zeros((len(texts), vec_size), dtype=np.float32)
    for i in range(len(texts)):
        row = X_tfidf[i]
        inds = row.indices
        vals = row.data
        denom = 0.0
        for j in range(len(inds)):
            w = feature_names[inds[j]]
            if w in vocab_w2v and vals[j] > 0:
                out[i] += vals[j] * w2v_model.wv[w]
                denom += vals[j]
        if denom > 0:
            out[i] /= denom
    return out


def get_sbert_embeddings(
    texts: list[str],
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Sentence-BERT embeddings. Toolkit."""
    if not _NLP_AVAILABLE or nlp_toolkit is None:
        raise RuntimeError("ai_toolkit.nlp required for get_sbert_embeddings")
    model_name = model_name or CONFIG.get("sbert_model", "all-MiniLM-L6-v2")
    return nlp_toolkit.get_sbert_embeddings(texts, model_name=model_name, batch_size=batch_size)


def get_sbert_embeddings_cached(
    texts: list[str],
    cache_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """SBERT embeddings with disk cache. Load from cache if exists and same length else compute and save."""
    cache_path = cache_path or Path(CONFIG["processed_path"]) / "sbert_embeddings.npy"
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.txt")
    model_name = model_name or CONFIG.get("sbert_model", "all-MiniLM-L6-v2")
    n = len(texts)
    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                line = f.read().strip()
                parts = line.split(",")
                if len(parts) >= 2 and int(parts[0]) == n and parts[1] == model_name:
                    arr = np.load(cache_path)
                    if arr.shape[0] == n:
                        return arr
        except Exception:
            pass
    arr = get_sbert_embeddings(texts, model_name=model_name, batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    with open(meta_path, "w") as f:
        f.write(f"{n},{model_name}\n")
    return arr


def run_feature_engineering(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    text_col: str = "processed_text",
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    w2v_size: int = 128,
    use_sbert_cache: bool = True,
) -> dict[str, Any]:
    """
    Build all five representations. Returns dict with:
    - bow: (vectorizer, X_train, X_test)
    - tfidf: (vectorizer, X_train, X_test)
    - w2v: (model, X_train_w2v, X_test_w2v)  # TF-IDF weighted
    - sbert: (X_train_sbert, X_test_sbert)
    - text_stats: (df_train_stats, df_test_stats)
    - train_texts, test_texts (lists)
    """
    train_texts = df_train[text_col].astype(str).tolist()
    test_texts = df_test[text_col].astype(str).tolist()

    out: dict[str, Any] = {
        "train_texts": train_texts,
        "test_texts": test_texts,
    }

    # BoW
    vec_bow, X_bow_train, X_bow_test = build_bow(
        train_texts, test_texts, max_features=max_features, ngram_range=(1, 1),
    )
    out["bow"] = (vec_bow, X_bow_train, X_bow_test)

    # TF-IDF
    vec_tfidf, X_tfidf_train, X_tfidf_test = build_tfidf(
        train_texts, test_texts, max_features=max_features, ngram_range=ngram_range,
    )
    out["tfidf"] = (vec_tfidf, X_tfidf_train, X_tfidf_test)

    # Word2Vec (tokenized = split)
    tokenized_train = [t.split() for t in train_texts]
    tokenized_test = [t.split() for t in test_texts]
    w2v = train_word2vec(
        tokenized_train,
        vector_size=w2v_size,
        min_count=CONFIG.get("min_word_freq", 2),
        seed=CONFIG.get("random_seed", 42),
    )
    X_w2v_train = get_tfidf_weighted_w2v(train_texts, w2v, vec_tfidf)
    X_w2v_test = get_tfidf_weighted_w2v(test_texts, w2v, vec_tfidf)
    out["w2v"] = (w2v, X_w2v_train, X_w2v_test)

    # SBERT (with optional cache for full train+test)
    all_texts = train_texts + test_texts
    if use_sbert_cache:
        cache_dir = Path(CONFIG["processed_path"]) / "03_features"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "sbert_embeddings.npy"
        all_emb = get_sbert_embeddings_cached(all_texts, cache_path=cache_file)
    else:
        all_emb = get_sbert_embeddings(all_texts)
    n_train = len(train_texts)
    out["sbert"] = (all_emb[:n_train], all_emb[n_train:])

    # Text stats
    df_train_stats = extract_text_stats(df_train, text_col)
    df_test_stats = extract_text_stats(df_test, text_col)
    out["text_stats"] = (df_train_stats, df_test_stats)

    return out
