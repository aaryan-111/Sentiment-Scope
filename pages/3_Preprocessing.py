"""
Phase 5 & 6 — Preprocessing (NLP pipeline) and feature engineering.
Uses ai_toolkit.nlp for BoW, TF-IDF, SBERT, text stats; in-repo for lemma/stem, Word2Vec.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from utils.config import CONFIG, SENTIMENT_MAP
from utils.split_data import get_train_val_test_split
from utils.preprocessing import (
    run_preprocessing,
    compare_stem_vs_lemma,
    preprocess_for_ml,
)
from utils.features import run_feature_engineering

st.set_page_config(page_title="Preprocessing — SentimentScope", layout="wide")
st.title("Preprocessing & Features")
st.caption("Phase 5: NLP pipeline (lemma/stem, processed_text). Phase 6: BoW, TF-IDF, Word2Vec, SBERT, text stats.")

# Data from session or saved cleaned file
def _get_df():
    if "df_clean" in st.session_state and st.session_state["df_clean"] is not None:
        return st.session_state["df_clean"]
    p = Path(CONFIG["processed_path"]) / "02_cleaned" / "cleaned.csv"
    if p.exists():
        return pd.read_csv(p, encoding="utf-8", nrows=None)
    return None

df = _get_df()

with st.sidebar:
    st.header("Config")
    text_col = st.text_input("Text column", value="clean_text", key="prep_text_col")
    use_lemma = st.checkbox("Use lemmatization (else stemming)", value=True, key="prep_lemma")
    max_sample = st.number_input(
        "Max samples (0 = all)",
        min_value=0,
        value=0,
        step=5000,
        help="Cap rows for faster run. 0 = use all.",
    )
    st.markdown("---")
    st.markdown("**Split (from config)**")
    st.write("Seed:", CONFIG["random_seed"], "| Test:", CONFIG["test_size"], "| Val:", CONFIG["val_size"])

if df is None:
    st.info("Load and clean data on the **Data** page first (or ensure `data/processed/02_cleaned/cleaned.csv` exists).")
    st.stop()

if text_col not in df.columns:
    st.error(f"Column `{text_col}` not found. Available: {list(df.columns)}")
    st.stop()

if max_sample and len(df) > max_sample:
    df = df.sample(n=max_sample, random_state=CONFIG["random_seed"]).reset_index(drop=True)

st.sidebar.metric("Rows", f"{len(df):,}")

# --- One-go test ---
st.subheader("One-go test")
if st.button("Run full pipeline (Preprocess → Split → Build features)"):
    if "sentiment" not in df.columns:
        st.error("Data must have a `sentiment` column (from Phase 2 cleaning).")
    else:
        with st.spinner("Running full pipeline…"):
            try:
                # 1. Preprocess
                df_proc, vocab, token_stats = run_preprocessing(df, text_col=text_col, use_lemma=use_lemma)
                st.session_state["df_processed"] = df_proc
                st.session_state["prep_vocab"] = vocab
                st.session_state["prep_token_stats"] = token_stats
                st.session_state["preprocessing_done"] = True
                # 2. Split
                df_train, df_val, df_test = get_train_val_test_split(df_proc)
                st.session_state["df_train"] = df_train
                st.session_state["df_val"] = df_val
                st.session_state["df_test"] = df_test
                # 3. Build features
                feats = run_feature_engineering(
                    df_train, df_test,
                    text_col="processed_text",
                    max_features=CONFIG.get("max_features", 50_000),
                    ngram_range=CONFIG.get("ngram_range", (1, 2)),
                    w2v_size=CONFIG.get("embedding_dim", 128),
                    use_sbert_cache=True,
                )
                st.session_state["features"] = feats
                st.session_state["y_train"] = df_train["sentiment"].map(SENTIMENT_MAP).fillna(1).astype(int).values
                st.session_state["y_test"] = df_test["sentiment"].map(SENTIMENT_MAP).fillna(1).astype(int).values
                st.session_state["features_done"] = True
                st.success(
                    f"Done. Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,} | "
                    f"Vocab: {token_stats['vocab_size']:,} | BoW/TF-IDF/W2V/SBERT + text stats built."
                )
            except Exception as e:
                st.error(str(e))
                import traceback
                st.code(traceback.format_exc())
st.caption("Runs preprocessing, split, and all feature matrices in one go. Use **Max samples** in sidebar for a quick test (e.g. 5000).")
st.markdown("---")

# --- Phase 5: Preprocessing ---
st.subheader("Phase 5 — NLP pipeline")
col1, col2 = st.columns(2)
with col1:
    run_prep = st.button("Run preprocessing (lemma/stem → processed_text)")
with col2:
    st.caption("Adds `processed_text`, builds vocab for DL.")

if run_prep:
    with st.spinner("Preprocessing…"):
        try:
            df_proc, vocab, token_stats = run_preprocessing(df, text_col=text_col, use_lemma=use_lemma)
            st.session_state["df_processed"] = df_proc
            st.session_state["prep_vocab"] = vocab
            st.session_state["prep_token_stats"] = token_stats
            st.session_state["preprocessing_done"] = True
            st.success(f"Done. Vocab size: {token_stats['vocab_size']:,} | Avg tokens/doc: {token_stats['avg_tokens_per_doc']:.1f}")
        except Exception as e:
            st.error(str(e))
            raise

if "df_processed" in st.session_state:
    df_processed = st.session_state["df_processed"]
    st.write("**Token stats:**", st.session_state.get("prep_token_stats", {}))
    sample_texts = df_processed[text_col].astype(str).head(5).tolist()
    compare = compare_stem_vs_lemma(sample_texts, n_samples=5)
    with st.expander("Stem vs Lemma sample"):
        st.dataframe(pd.DataFrame(compare), width="stretch")
    st.write("**processed_text sample**")
    st.dataframe(df_processed[["processed_text"]].head(10) if "processed_text" in df_processed.columns else df_processed.head(10), width="stretch")

# --- Split ---
st.markdown("---")
st.subheader("Train / Val / Test split")
if st.button("Split data (train / val / test)"):
    source = st.session_state.get("df_processed")
    if source is None:
        source = df
    if "processed_text" not in source.columns:
        st.warning("Run preprocessing first so split includes processed_text.")
    else:
        with st.spinner("Splitting…"):
            df_train, df_val, df_test = get_train_val_test_split(source)
            st.session_state["df_train"] = df_train
            st.session_state["df_val"] = df_val
            st.session_state["df_test"] = df_test
            st.success(f"Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

if "df_train" in st.session_state and "df_test" in st.session_state:
    st.write("Train:", st.session_state["df_train"].shape, "| Val:", st.session_state["df_val"].shape, "| Test:", st.session_state["df_test"].shape)

# --- Phase 6: Features ---
st.markdown("---")
st.subheader("Phase 6 — Feature engineering")
st.caption("BoW, TF-IDF, Word2Vec (TF-IDF weighted), SBERT, text stats.")

if st.button("Build all features"):
    if "df_train" not in st.session_state or "df_test" not in st.session_state:
        st.error("Split data first (click **Split data** above).")
    else:
        df_train = st.session_state["df_train"]
        df_test = st.session_state["df_test"]
        if "processed_text" not in df_train.columns:
            st.error("Preprocessing required: run **Run preprocessing** then **Split data** again.")
        else:
            with st.spinner("Building BoW, TF-IDF, Word2Vec, SBERT, text stats…"):
                try:
                    feats = run_feature_engineering(
                        df_train,
                        df_test,
                        text_col="processed_text",
                        max_features=CONFIG.get("max_features", 50_000),
                        ngram_range=CONFIG.get("ngram_range", (1, 2)),
                        w2v_size=CONFIG.get("embedding_dim", 128),
                        use_sbert_cache=True,
                    )
                    st.session_state["features"] = feats
                    st.session_state["y_train"] = df_train["sentiment"].map(SENTIMENT_MAP).fillna(1).astype(int).values
                    st.session_state["y_test"] = df_test["sentiment"].map(SENTIMENT_MAP).fillna(1).astype(int).values
                    st.session_state["features_done"] = True
                    st.success("All feature matrices built.")
                except Exception as e:
                    st.error(str(e))
                    import traceback
                    st.code(traceback.format_exc())

if "features" in st.session_state:
    feats = st.session_state["features"]
    st.write("**Feature matrix shapes**")
    vec_bow, X_bow_train, X_bow_test = feats["bow"]
    vec_tfidf, X_tfidf_train, X_tfidf_test = feats["tfidf"]
    w2v, X_w2v_train, X_w2v_test = feats["w2v"]
    X_sbert_train, X_sbert_test = feats["sbert"]
    df_stats_train, df_stats_test = feats["text_stats"]
    rows = [
        ("BoW (train)", str(X_bow_train.shape)),
        ("BoW (test)", str(X_bow_test.shape)),
        ("TF-IDF (train)", str(X_tfidf_train.shape)),
        ("TF-IDF (test)", str(X_tfidf_test.shape)),
        ("Word2Vec (train)", str(X_w2v_train.shape)),
        ("Word2Vec (test)", str(X_w2v_test.shape)),
        ("SBERT (train)", str(X_sbert_train.shape)),
        ("SBERT (test)", str(X_sbert_test.shape)),
        ("Text stats (train)", str(df_stats_train.shape)),
        ("Text stats (test)", str(df_stats_test.shape)),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Feature", "Shape"]), width="stretch", hide_index=True)
    st.write("**Text stats columns:**", list(df_stats_train.columns))
    if "y_train" in st.session_state:
        st.write("**Labels (encoded):** train", st.session_state["y_train"].shape, "| test", st.session_state["y_test"].shape)

st.markdown("---")
if st.session_state.get("preprocessing_done") and st.session_state.get("features_done"):
    st.success("Preprocessing and features are ready. You can go to **4_ML_Models** for training.")
