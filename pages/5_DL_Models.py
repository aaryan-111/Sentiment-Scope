"""
Phase 13a–d — ANN, LSTM, Attention, BERT. Uses ai_toolkit.dl.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.config import CONFIG
from utils.dl_trainer import (
    ANNDataset,
    BERTDataset,
    ReviewDataset,
    SelfAttention,
    SentimentANN,
    SentimentBiLSTM,
    SentimentLSTM,
    SentimentLSTMWithAttention,
    Vocabulary,
    compute_metrics,
    fine_tune_bert,
    get_attention_weights_for_samples,
    get_predictions_ann,
    get_predictions_rnn,
    predict_bert,
    train_with_early_stopping,
)

st.set_page_config(page_title="DL Models — SentimentScope", layout="wide")
st.title("Deep Learning Models")
st.caption("Phase 13a–d: ANN → LSTM → BiLSTM → Self-Attention → DistilBERT")

# --- Session state init ---
for key in ("dl_results", "dl_fitted_models"):
    if key not in st.session_state:
        if "results" in key:
            st.session_state[key] = []
        else:
            st.session_state[key] = {}

# --- Check prerequisites ---
def _features_ready():
    return (
        st.session_state.get("features_done") is True
        and "features" in st.session_state
        and "y_train" in st.session_state
        and "y_test" in st.session_state
        and "df_train" in st.session_state
        and "df_test" in st.session_state
    )

# --- Sidebar configuration ---
with st.sidebar:
    st.header("DL Config")
    if _features_ready():
        st.success("Features ready")
        
        epochs = st.number_input("Epochs", 1, 50, CONFIG.get("epochs", 10), key="dl_epochs")
        batch_size = st.number_input("Batch Size", 8, 256, CONFIG.get("batch_size", 64), key="dl_batch")
        learning_rate = st.number_input(
            "Learning Rate", 0.0001, 0.1, CONFIG.get("learning_rate", 0.001),
            format="%.4f", key="dl_lr"
        )
        patience = st.number_input("Patience", 1, 10, CONFIG.get("patience", 3), key="dl_patience")
        
        st.divider()
        st.caption(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    else:
        st.warning("Run **Preprocessing** and build features first.")

# --- Gate check ---
if not _features_ready():
    st.info("⚠️ Load data and build features on the **Preprocessing** page first.")
    st.stop()

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get data from session state
features = st.session_state["features"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]
df_train = st.session_state["df_train"]
df_test = st.session_state["df_test"]
df_val = st.session_state.get("df_val")

# Prepare validation split from training data
if df_val is not None and len(df_val) > 0:
    y_val = df_val["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).values
    val_texts = df_val["processed_text"].tolist()
else:
    # Create validation split from training data
    train_texts = features.get("train_texts", df_train["processed_text"].tolist())
    train_indices = list(range(len(y_train)))
    train_idx, val_idx = train_test_split(
        train_indices, test_size=0.1, random_state=CONFIG["random_seed"], stratify=y_train
    )
    y_train_split = y_train[train_idx]
    y_val = y_train[val_idx]
    val_texts = [train_texts[i] for i in val_idx]
    train_texts = [train_texts[i] for i in train_idx]
    y_train = y_train_split

# Test data
test_texts = features.get("test_texts", df_test["processed_text"].tolist())

# Create tabs
tab_ann, tab_lstm, tab_bilstm, tab_attention, tab_bert = st.tabs([
    "ANN", "LSTM", "BiLSTM", "Attention-LSTM", "DistilBERT"
])

# =====================================================================
# TAB 1: ANN (Feedforward)
# =====================================================================

with tab_ann:
    st.subheader("Artificial Neural Network (Feedforward)")
    st.caption("Dense feedforward network on pre-computed embeddings.")
    
    # Representation selector
    representation = st.selectbox(
        "Feature Representation",
        ["tfidf", "sbert"],
        key="ann_repr"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        train_button = st.button("Train ANN", key="train_ann", type="primary")
    
    with col2:
        save_button = st.button("Save Model", key="save_ann")
    
    with col3:
        load_button = st.button("Load Model", key="load_ann")
    
    if train_button:
        with st.spinner("Training ANN..."):
            start_time = time.time()
            
            # Get features
            if representation == "tfidf":
                vectorizer, X_train_feat, X_test_feat = features["tfidf"]
                # Convert sparse to dense
                if issparse(X_train_feat):
                    X_train_feat = X_train_feat.toarray()
                if issparse(X_test_feat):
                    X_test_feat = X_test_feat.toarray()
            else:  # sbert
                X_train_feat, X_test_feat = features["sbert"]
            
            # Create validation split
            X_train_split, X_val, y_train_split, y_val_new = train_test_split(
                X_train_feat, y_train, test_size=0.1,
                random_state=CONFIG["random_seed"], stratify=y_train
            )
            
            # Create datasets
            train_dataset = ANNDataset(X_train_split, y_train_split)
            val_dataset = ANNDataset(X_val, y_val_new)
            test_dataset = ANNDataset(X_test_feat, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Create model
            input_dim = X_train_feat.shape[1]
            model = SentimentANN(
                input_dim=input_dim,
                hidden_dim=CONFIG.get("hidden_dim", 256),
                dropout=CONFIG.get("dropout", 0.3)
            ).to(device)
            
            # Train
            history = train_with_early_stopping(
                model, train_loader, val_loader,
                epochs=epochs, learning_rate=learning_rate,
                patience=patience, device=device, model_type="ANN"
            )
            
            # Evaluate on test set
            y_pred, y_true = get_predictions_ann(model, test_loader, device)
            metrics = compute_metrics(y_true, y_pred)
            
            train_time = time.time() - start_time
            
            # Store results
            result = {
                "model": "ANN",
                "representation": representation,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": metrics["per_class_f1"],
                "train_time": train_time,
            }
            
            st.session_state["dl_results"].append(result)
            st.session_state["dl_fitted_models"]["ANN"] = {
                "model": model,
                "representation": representation,
                "history": history,
            }
            
            st.success(f"✓ ANN trained in {train_time:.1f}s")
    
    # Display results if available
    if "ANN" in st.session_state["dl_fitted_models"]:
        model_data = st.session_state["dl_fitted_models"]["ANN"]
        history = model_data["history"]
        
        st.divider()
        st.subheader("Training History")
        
        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(1, len(history["train_loss"]) + 1)
        
        # Loss
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs_range, history["train_acc"], label="Train Acc", marker="o")
        ax2.plot(epochs_range, history["val_acc"], label="Val Acc", marker="s")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show metrics
        st.subheader("Test Set Metrics")
        
        # Find this model's results
        ann_results = [r for r in st.session_state["dl_results"] if r["model"] == "ANN"]
        if ann_results:
            latest = ann_results[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{latest['accuracy']:.4f}")
            col2.metric("Macro F1", f"{latest['macro_f1']:.4f}")
            col3.metric("Train Time", f"{latest['train_time']:.1f}s")
            
            # Per-class F1
            st.caption("Per-Class F1 Scores")
            per_class_df = pd.DataFrame({
                "Class": ["Negative (0)", "Neutral (1)", "Positive (2)"],
                "F1 Score": [
                    latest["per_class_f1"][0],
                    latest["per_class_f1"][1],
                    latest["per_class_f1"][2],
                ]
            })
            st.dataframe(per_class_df, hide_index=True, width='stretch')
    
    # Save/Load functionality
    if save_button and "ANN" in st.session_state["dl_fitted_models"]:
        save_dir = Path(CONFIG["saved_models_dir"]) / "dl"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "ann.pt"
        
        torch.save(st.session_state["dl_fitted_models"]["ANN"]["model"].state_dict(), save_path)
        st.success(f"✓ Model saved to {save_path}")
    
    if load_button:
        save_path = Path(CONFIG["saved_models_dir"]) / "dl" / "ann.pt"
        if save_path.exists():
            # Need to recreate model with same architecture
            st.info("Model loading requires feature representation to be set. Train first or specify architecture.")
        else:
            st.error("No saved model found.")

# =====================================================================
# TAB 2: LSTM
# =====================================================================

with tab_lstm:
    st.subheader("LSTM (Long Short-Term Memory)")
    st.caption("Recurrent neural network for sequential text processing.")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        train_button_lstm = st.button("Train LSTM", key="train_lstm", type="primary")
    
    with col2:
        save_button_lstm = st.button("Save Model", key="save_lstm")
    
    with col3:
        load_button_lstm = st.button("Load Model", key="load_lstm")
    
    if train_button_lstm:
        with st.spinner("Building vocabulary and training LSTM..."):
            start_time = time.time()
            
            # Build vocabulary
            train_texts_full = df_train["processed_text"].tolist()
            vocab = Vocabulary(min_freq=CONFIG.get("min_word_freq", 2))
            vocab.build(train_texts_full)
            
            st.info(f"Vocabulary size: {vocab.vocab_size:,}")
            
            # Create datasets
            max_len = CONFIG.get("max_seq_len", 256)
            
            # Re-split for LSTM
            train_idx, val_idx = train_test_split(
                list(range(len(train_texts_full))), test_size=0.1,
                random_state=CONFIG["random_seed"]
            )
            
            train_texts_split = [train_texts_full[i] for i in train_idx]
            val_texts_split = [train_texts_full[i] for i in val_idx]
            y_train_full = df_train["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).values
            y_train_split = y_train_full[train_idx]
            y_val_split = y_train_full[val_idx]
            
            train_dataset = ReviewDataset(train_texts_split, y_train_split, vocab, max_len)
            val_dataset = ReviewDataset(val_texts_split, y_val_split, vocab, max_len)
            test_dataset = ReviewDataset(test_texts, y_test, vocab, max_len)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Create model
            model = SentimentLSTM(
                vocab_size=vocab.vocab_size,
                embedding_dim=CONFIG.get("embedding_dim", 128),
                hidden_dim=CONFIG.get("hidden_dim", 256),
                num_layers=CONFIG.get("num_layers", 2),
                dropout=CONFIG.get("dropout", 0.3)
            ).to(device)
            
            # Train
            history = train_with_early_stopping(
                model, train_loader, val_loader,
                epochs=epochs, learning_rate=learning_rate,
                patience=patience, device=device, model_type="LSTM"
            )
            
            # Evaluate on test set
            y_pred, y_true = get_predictions_rnn(model, test_loader, device)
            metrics = compute_metrics(y_true, y_pred)
            
            train_time = time.time() - start_time
            
            # Store results
            result = {
                "model": "LSTM",
                "representation": "vocab",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": metrics["per_class_f1"],
                "train_time": train_time,
            }
            
            st.session_state["dl_results"].append(result)
            st.session_state["dl_fitted_models"]["LSTM"] = {
                "model": model,
                "vocab": vocab,
                "history": history,
            }
            
            st.success(f"✓ LSTM trained in {train_time:.1f}s")
    
    # Display results
    if "LSTM" in st.session_state["dl_fitted_models"]:
        model_data = st.session_state["dl_fitted_models"]["LSTM"]
        history = model_data["history"]
        
        st.divider()
        st.subheader("Training History")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(1, len(history["train_loss"]) + 1)
        
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs_range, history["train_acc"], label="Train Acc", marker="o")
        ax2.plot(epochs_range, history["val_acc"], label="Val Acc", marker="s")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Metrics
        st.subheader("Test Set Metrics")
        lstm_results = [r for r in st.session_state["dl_results"] if r["model"] == "LSTM"]
        if lstm_results:
            latest = lstm_results[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{latest['accuracy']:.4f}")
            col2.metric("Macro F1", f"{latest['macro_f1']:.4f}")
            col3.metric("Train Time", f"{latest['train_time']:.1f}s")
            
            per_class_df = pd.DataFrame({
                "Class": ["Negative (0)", "Neutral (1)", "Positive (2)"],
                "F1 Score": [
                    latest["per_class_f1"][0],
                    latest["per_class_f1"][1],
                    latest["per_class_f1"][2],
                ]
            })
            st.dataframe(per_class_df, hide_index=True, width='stretch')
    
    if save_button_lstm and "LSTM" in st.session_state["dl_fitted_models"]:
        save_dir = Path(CONFIG["saved_models_dir"]) / "dl"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(st.session_state["dl_fitted_models"]["LSTM"]["model"].state_dict(),
                   save_dir / "lstm.pt")
        st.success(f"✓ Model saved to {save_dir / 'lstm.pt'}")

# =====================================================================
# TAB 3: BiLSTM
# =====================================================================

with tab_bilstm:
    st.subheader("BiLSTM (Bidirectional LSTM)")
    st.caption("Processes text in both forward and backward directions.")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        train_button_bilstm = st.button("Train BiLSTM", key="train_bilstm", type="primary")
    
    with col2:
        save_button_bilstm = st.button("Save Model", key="save_bilstm")
    
    with col3:
        load_button_bilstm = st.button("Load Model", key="load_bilstm")
    
    if train_button_bilstm:
        with st.spinner("Building vocabulary and training BiLSTM..."):
            start_time = time.time()
            
            # Build vocabulary
            train_texts_full = df_train["processed_text"].tolist()
            vocab = Vocabulary(min_freq=CONFIG.get("min_word_freq", 2))
            vocab.build(train_texts_full)
            
            st.info(f"Vocabulary size: {vocab.vocab_size:,}")
            
            # Create datasets
            max_len = CONFIG.get("max_seq_len", 256)
            
            train_idx, val_idx = train_test_split(
                list(range(len(train_texts_full))), test_size=0.1,
                random_state=CONFIG["random_seed"]
            )
            
            train_texts_split = [train_texts_full[i] for i in train_idx]
            val_texts_split = [train_texts_full[i] for i in val_idx]
            y_train_full = df_train["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).values
            y_train_split = y_train_full[train_idx]
            y_val_split = y_train_full[val_idx]
            
            train_dataset = ReviewDataset(train_texts_split, y_train_split, vocab, max_len)
            val_dataset = ReviewDataset(val_texts_split, y_val_split, vocab, max_len)
            test_dataset = ReviewDataset(test_texts, y_test, vocab, max_len)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Create model
            model = SentimentBiLSTM(
                vocab_size=vocab.vocab_size,
                embedding_dim=CONFIG.get("embedding_dim", 128),
                hidden_dim=CONFIG.get("hidden_dim", 256),
                num_layers=CONFIG.get("num_layers", 2),
                dropout=CONFIG.get("dropout", 0.3)
            ).to(device)
            
            # Train
            history = train_with_early_stopping(
                model, train_loader, val_loader,
                epochs=epochs, learning_rate=learning_rate,
                patience=patience, device=device, model_type="BiLSTM"
            )
            
            # Evaluate
            y_pred, y_true = get_predictions_rnn(model, test_loader, device)
            metrics = compute_metrics(y_true, y_pred)
            
            train_time = time.time() - start_time
            
            result = {
                "model": "BiLSTM",
                "representation": "vocab",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": metrics["per_class_f1"],
                "train_time": train_time,
            }
            
            st.session_state["dl_results"].append(result)
            st.session_state["dl_fitted_models"]["BiLSTM"] = {
                "model": model,
                "vocab": vocab,
                "history": history,
            }
            
            st.success(f"✓ BiLSTM trained in {train_time:.1f}s")
    
    # Display results
    if "BiLSTM" in st.session_state["dl_fitted_models"]:
        model_data = st.session_state["dl_fitted_models"]["BiLSTM"]
        history = model_data["history"]
        
        st.divider()
        st.subheader("Training History")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(1, len(history["train_loss"]) + 1)
        
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs_range, history["train_acc"], label="Train Acc", marker="o")
        ax2.plot(epochs_range, history["val_acc"], label="Val Acc", marker="s")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Test Set Metrics")
        bilstm_results = [r for r in st.session_state["dl_results"] if r["model"] == "BiLSTM"]
        if bilstm_results:
            latest = bilstm_results[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{latest['accuracy']:.4f}")
            col2.metric("Macro F1", f"{latest['macro_f1']:.4f}")
            col3.metric("Train Time", f"{latest['train_time']:.1f}s")
            
            per_class_df = pd.DataFrame({
                "Class": ["Negative (0)", "Neutral (1)", "Positive (2)"],
                "F1 Score": [
                    latest["per_class_f1"][0],
                    latest["per_class_f1"][1],
                    latest["per_class_f1"][2],
                ]
            })
            st.dataframe(per_class_df, hide_index=True, width='stretch')
    
    if save_button_bilstm and "BiLSTM" in st.session_state["dl_fitted_models"]:
        save_dir = Path(CONFIG["saved_models_dir"]) / "dl"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(st.session_state["dl_fitted_models"]["BiLSTM"]["model"].state_dict(),
                   save_dir / "bilstm.pt")
        st.success(f"✓ Model saved to {save_dir / 'bilstm.pt'}")

# =====================================================================
# TAB 4: Attention-LSTM
# =====================================================================

with tab_attention:
    st.subheader("LSTM with Self-Attention")
    st.caption("LSTM enhanced with attention mechanism for interpretability.")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        train_button_attn = st.button("Train Attention-LSTM", key="train_attn", type="primary")
    
    with col2:
        save_button_attn = st.button("Save Model", key="save_attn")
    
    with col3:
        load_button_attn = st.button("Load Model", key="load_attn")
    
    if train_button_attn:
        with st.spinner("Building vocabulary and training Attention-LSTM..."):
            start_time = time.time()
            
            # Build vocabulary
            train_texts_full = df_train["processed_text"].tolist()
            vocab = Vocabulary(min_freq=CONFIG.get("min_word_freq", 2))
            vocab.build(train_texts_full)
            
            st.info(f"Vocabulary size: {vocab.vocab_size:,}")
            
            # Create datasets
            max_len = CONFIG.get("max_seq_len", 256)
            
            train_idx, val_idx = train_test_split(
                list(range(len(train_texts_full))), test_size=0.1,
                random_state=CONFIG["random_seed"]
            )
            
            train_texts_split = [train_texts_full[i] for i in train_idx]
            val_texts_split = [train_texts_full[i] for i in val_idx]
            y_train_full = df_train["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).values
            y_train_split = y_train_full[train_idx]
            y_val_split = y_train_full[val_idx]
            
            train_dataset = ReviewDataset(train_texts_split, y_train_split, vocab, max_len)
            val_dataset = ReviewDataset(val_texts_split, y_val_split, vocab, max_len)
            test_dataset = ReviewDataset(test_texts, y_test, vocab, max_len)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Create model
            model = SentimentLSTMWithAttention(
                vocab_size=vocab.vocab_size,
                embedding_dim=CONFIG.get("embedding_dim", 128),
                hidden_dim=CONFIG.get("hidden_dim", 256),
                num_layers=CONFIG.get("num_layers", 2),
                dropout=CONFIG.get("dropout", 0.3)
            ).to(device)
            
            # Train
            history = train_with_early_stopping(
                model, train_loader, val_loader,
                epochs=epochs, learning_rate=learning_rate,
                patience=patience, device=device, model_type="Attention"
            )
            
            # Evaluate
            y_pred, y_true = get_predictions_rnn(model, test_loader, device)
            metrics = compute_metrics(y_true, y_pred)
            
            train_time = time.time() - start_time
            
            result = {
                "model": "Attention-LSTM",
                "representation": "vocab",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": metrics["per_class_f1"],
                "train_time": train_time,
            }
            
            st.session_state["dl_results"].append(result)
            
            # Get attention visualizations
            sample_texts = test_texts[:5]
            sample_labels = y_test[:5].tolist()
            
            attention_samples = get_attention_weights_for_samples(
                model, sample_texts, sample_labels, vocab, device, max_len, n_samples=5
            )
            
            st.session_state["dl_fitted_models"]["Attention"] = {
                "model": model,
                "vocab": vocab,
                "history": history,
                "attention_samples": attention_samples,
            }
            
            st.success(f"✓ Attention-LSTM trained in {train_time:.1f}s")
    
    # Display results
    if "Attention" in st.session_state["dl_fitted_models"]:
        model_data = st.session_state["dl_fitted_models"]["Attention"]
        history = model_data["history"]
        
        st.divider()
        st.subheader("Training History")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(1, len(history["train_loss"]) + 1)
        
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs_range, history["train_acc"], label="Train Acc", marker="o")
        ax2.plot(epochs_range, history["val_acc"], label="Val Acc", marker="s")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Test Set Metrics")
        attn_results = [r for r in st.session_state["dl_results"] if r["model"] == "Attention-LSTM"]
        if attn_results:
            latest = attn_results[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{latest['accuracy']:.4f}")
            col2.metric("Macro F1", f"{latest['macro_f1']:.4f}")
            col3.metric("Train Time", f"{latest['train_time']:.1f}s")
            
            per_class_df = pd.DataFrame({
                "Class": ["Negative (0)", "Neutral (1)", "Positive (2)"],
                "F1 Score": [
                    latest["per_class_f1"][0],
                    latest["per_class_f1"][1],
                    latest["per_class_f1"][2],
                ]
            })
            st.dataframe(per_class_df, hide_index=True, width='stretch')
        
        # Attention visualization
        if "attention_samples" in model_data:
            st.divider()
            st.subheader("Attention Visualizations")
            st.caption("Attention weights for sample predictions (darker = higher attention)")
            
            for i, sample in enumerate(model_data["attention_samples"]):
                with st.expander(f"Sample {i+1}: {sample['text'][:80]}..."):
                    st.write(f"**True Label:** {sample['true_label']} | **Predicted:** {sample['predicted_label']}")
                    
                    # Create heatmap
                    words = sample["words"]
                    attn = sample["attention"]
                    
                    # Use only the diagonal or mean attention
                    attn_weights = np.mean(attn, axis=0)  # Average attention across all positions
                    
                    fig, ax = plt.subplots(figsize=(12, 2))
                    
                    # Plot as horizontal bar
                    y_pos = np.arange(len(words))
                    ax.barh(y_pos, attn_weights[:len(words)])
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(words, fontsize=8)
                    ax.set_xlabel("Attention Weight")
                    ax.set_title(f"Word Attention Weights")
                    ax.invert_yaxis()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    if save_button_attn and "Attention" in st.session_state["dl_fitted_models"]:
        save_dir = Path(CONFIG["saved_models_dir"]) / "dl"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(st.session_state["dl_fitted_models"]["Attention"]["model"].state_dict(),
                   save_dir / "attention_lstm.pt")
        st.success(f"✓ Model saved to {save_dir / 'attention_lstm.pt'}")

# =====================================================================
# TAB 5: DistilBERT
# =====================================================================

with tab_bert:
    st.subheader("DistilBERT Fine-Tuning")
    st.caption("Transformer-based model fine-tuned on sentiment classification.")
    
    st.info(f"⚠️ Training on reduced sample: {CONFIG.get('bert_sample_size', 20000):,} rows (CPU/RAM friendly)")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        train_button_bert = st.button("Train DistilBERT", key="train_bert", type="primary")
    
    with col2:
        save_button_bert = st.button("Save Model", key="save_bert")
    
    with col3:
        load_button_bert = st.button("Load Model", key="load_bert")
    
    if train_button_bert:
        with st.spinner("Fine-tuning DistilBERT... This may take several minutes."):
            start_time = time.time()
            
            # Sample data
            bert_sample_size = CONFIG.get("bert_sample_size", 20000)
            
            train_texts_full = df_train["processed_text"].tolist()
            y_train_full = df_train["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).values
            
            # Sample if needed
            if len(train_texts_full) > bert_sample_size:
                indices = np.random.RandomState(CONFIG["random_seed"]).choice(
                    len(train_texts_full), bert_sample_size, replace=False
                )
                train_texts_sampled = [train_texts_full[i] for i in indices]
                y_train_sampled = y_train_full[indices]
            else:
                train_texts_sampled = train_texts_full
                y_train_sampled = y_train_full
            
            # Train/val split
            train_texts_bert, val_texts_bert, y_train_bert, y_val_bert = train_test_split(
                train_texts_sampled, y_train_sampled, test_size=0.1,
                random_state=CONFIG["random_seed"], stratify=y_train_sampled
            )
            
            # Fine-tune
            output_dir = str(Path(CONFIG["saved_models_dir"]) / "bert_finetuned")
            
            model, tokenizer, history = fine_tune_bert(
                train_texts_bert, y_train_bert,
                val_texts_bert, y_val_bert,
                model_name=CONFIG.get("bert_model", "distilbert-base-uncased"),
                epochs=CONFIG.get("bert_epochs", 3),
                batch_size=CONFIG.get("bert_batch_size", 32),
                learning_rate=CONFIG.get("bert_lr", 2e-5),
                output_dir=output_dir
            )
            
            # Evaluate on test set
            y_pred = predict_bert(model, tokenizer, test_texts, device, batch_size=32)
            metrics = compute_metrics(y_test, y_pred)
            
            train_time = time.time() - start_time
            
            result = {
                "model": "DistilBERT",
                "representation": "transformer",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": metrics["per_class_f1"],
                "train_time": train_time,
            }
            
            st.session_state["dl_results"].append(result)
            st.session_state["dl_fitted_models"]["BERT"] = {
                "model": model,
                "tokenizer": tokenizer,
                "history": history,
            }
            
            st.success(f"✓ DistilBERT fine-tuned in {train_time:.1f}s ({train_time/60:.1f} min)")
    
    # Display results
    if "BERT" in st.session_state["dl_fitted_models"]:
        model_data = st.session_state["dl_fitted_models"]["BERT"]
        history = model_data["history"]
        
        st.divider()
        st.subheader("Training History")
        
        # Plot loss
        if history["train_loss"] and history["val_loss"]:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            epochs_range = range(1, len(history["train_loss"]) + 1)
            
            ax.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
            ax.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Fine-Tuning Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.subheader("Test Set Metrics")
        bert_results = [r for r in st.session_state["dl_results"] if r["model"] == "DistilBERT"]
        if bert_results:
            latest = bert_results[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{latest['accuracy']:.4f}")
            col2.metric("Macro F1", f"{latest['macro_f1']:.4f}")
            col3.metric("Train Time", f"{latest['train_time']:.1f}s")
            
            per_class_df = pd.DataFrame({
                "Class": ["Negative (0)", "Neutral (1)", "Positive (2)"],
                "F1 Score": [
                    latest["per_class_f1"][0],
                    latest["per_class_f1"][1],
                    latest["per_class_f1"][2],
                ]
            })
            st.dataframe(per_class_df, hide_index=True, width='stretch')
            
            # Compare with best ML model
            st.divider()
            st.subheader("Comparison with Best Classical ML Model")
            
            if st.session_state.get("ml_results_classification"):
                ml_results = st.session_state["ml_results_classification"]
                if ml_results:
                    best_ml = max(ml_results, key=lambda x: x.get("macro_f1", 0))
                    
                    comparison_df = pd.DataFrame({
                        "Model": ["Best ML (Classical)", "DistilBERT (Transformer)"],
                        "Accuracy": [best_ml.get("accuracy", 0), latest["accuracy"]],
                        "Macro F1": [best_ml.get("macro_f1", 0), latest["macro_f1"]],
                    })
                    
                    st.dataframe(comparison_df, hide_index=True, width='stretch')
                    
                    improvement = (latest["macro_f1"] - best_ml.get("macro_f1", 0)) * 100
                    if improvement > 0:
                        st.success(f"✓ BERT improves macro F1 by {improvement:.2f} percentage points")
                    else:
                        st.info(f"BERT macro F1 change: {improvement:.2f} percentage points")
    
    if save_button_bert and "BERT" in st.session_state["dl_fitted_models"]:
        st.info("BERT model already saved during training in saved_models/bert_finetuned/")

# =====================================================================
# SUMMARY
# =====================================================================

if st.session_state["dl_results"]:
    st.divider()
    st.header("📊 All Deep Learning Results")
    
    results_df = pd.DataFrame(st.session_state["dl_results"])
    
    # Display summary table
    display_df = results_df[["model", "representation", "accuracy", "macro_f1", "train_time"]].copy()
    display_df.columns = ["Model", "Representation", "Accuracy", "Macro F1", "Train Time (s)"]
    
    st.dataframe(display_df, hide_index=True, width='stretch')
    
    # Best model
    best_model = results_df.loc[results_df["macro_f1"].idxmax()]
    st.success(f"🏆 Best Model: **{best_model['model']}** — Macro F1: {best_model['macro_f1']:.4f}")
