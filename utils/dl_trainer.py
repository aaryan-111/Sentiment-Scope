"""
Deep Learning utilities for Sentiment Classification.
Phases 13a–d: ANN, LSTM, BiLSTM, Self-Attention, DistilBERT.
"""
from __future__ import annotations

import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Suppress HuggingFace warnings
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

# =====================================================================
# VOCABULARY & DATASET (for LSTM/BiLSTM/Attention)
# =====================================================================


class Vocabulary:
    """Build word2idx mapping from corpus. Includes PAD, UNK tokens."""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def build(self, texts: List[str]):
        """Build vocabulary from list of text strings."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        idx = 2
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        self.vocab_size = len(self.word2idx)

    def encode(self, text: str) -> List[int]:
        """Convert text to list of indices."""
        return [self.word2idx.get(word, 1) for word in text.split()]


class ReviewDataset(Dataset):
    """Dataset for LSTM/BiLSTM/Attention. Tokenize, encode, pad."""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        vocab: Vocabulary,
        max_len: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        # Encode and pad
        encoded = self.vocab.encode(text)
        length = min(len(encoded), self.max_len)
        encoded = encoded[: self.max_len]

        # Pad
        if len(encoded) < self.max_len:
            encoded = encoded + [0] * (self.max_len - len(encoded))

        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


# =====================================================================
# ANN (Phase 13a)
# =====================================================================


class SentimentANN(nn.Module):
    """Feedforward neural network for sentiment classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 3)  # 3 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =====================================================================
# LSTM (Phase 13b)
# =====================================================================


class SentimentLSTM(nn.Module):
    """LSTM model for sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, input_ids, lengths):
        # Embedding
        embedded = self.embedding(input_ids)

        # Pack sequence
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        output, (hidden, _) = self.lstm(packed)

        # Use last hidden state
        hidden = self.dropout(hidden[-1])
        logits = self.fc(hidden)
        return logits


class SentimentBiLSTM(nn.Module):
    """Bidirectional LSTM for sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 3)  # *2 for bidirectional

    def forward(self, input_ids, lengths):
        # Embedding
        embedded = self.embedding(input_ids)

        # Pack sequence
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # BiLSTM
        output, (hidden, _) = self.lstm(packed)

        # Concatenate last hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


# =====================================================================
# SELF-ATTENTION (Phase 13c)
# =====================================================================


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention mechanism."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - 1 for valid tokens, 0 for padding
        Returns:
            context: (batch, seq_len, hidden_dim)
            attention_weights: (batch, seq_len, seq_len)
        """
        Q = self.query(x)  # (batch, seq_len, hidden_dim)
        K = self.key(x)  # (batch, seq_len, hidden_dim)
        V = self.value(x)  # (batch, seq_len, hidden_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        return context, attention_weights


class SentimentLSTMWithAttention(nn.Module):
    """LSTM with self-attention for sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = SelfAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, input_ids, lengths, return_attention=False):
        # Embedding
        embedded = self.embedding(input_ids)

        # Pack sequence
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # Create mask
        batch_size, seq_len, _ = output.shape
        mask = torch.arange(seq_len, device=output.device).expand(
            batch_size, seq_len
        ) < lengths.unsqueeze(1)

        # Self-attention
        context, attention_weights = self.attention(output, mask)

        # Pool: mean over sequence (weighted by attention)
        # Use last non-padded position for simplicity
        indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, context.size(2))
        pooled = torch.gather(context, 1, indices).squeeze(1)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        if return_attention:
            return logits, attention_weights
        return logits


# =====================================================================
# BERT DATASET (Phase 13d)
# =====================================================================


class BERTDataset(Dataset):
    """Dataset for BERT fine-tuning."""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_len: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# =====================================================================
# TRAINING UTILITIES
# =====================================================================


def train_epoch_ann(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Train one epoch for ANN."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        inputs = batch["input_ids"].float().to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_ann(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate ANN."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].float().to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_epoch_rnn(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Train one epoch for LSTM/BiLSTM/Attention."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_rnn(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate LSTM/BiLSTM/Attention."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int,
    device: str,
    model_type: str,
) -> Dict[str, List[float]]:
    """Train with early stopping. Returns history of losses and accuracies."""
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    train_fn = train_epoch_ann if model_type == "ANN" else train_epoch_rnn
    eval_fn = evaluate_ann if model_type == "ANN" else lambda m, d, c, dev: evaluate_rnn(m, d, c, dev)[:2]

    for epoch in range(epochs):
        train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_fn(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return history


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute accuracy and F1 scores."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": {i: f1 for i, f1 in enumerate(per_class_f1)},
    }


def get_predictions_rnn(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions for LSTM/BiLSTM/Attention."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, lengths)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def get_predictions_ann(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions for ANN."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].float().to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def get_attention_weights_for_samples(
    model: SentimentLSTMWithAttention,
    texts: List[str],
    labels: List[int],
    vocab: Vocabulary,
    device: str,
    max_len: int = 256,
    n_samples: int = 5,
) -> List[Dict[str, Any]]:
    """Get attention weights for sample predictions."""
    model.eval()
    results = []

    for i in range(min(n_samples, len(texts))):
        text = texts[i]
        label = labels[i]

        # Encode
        encoded = vocab.encode(text)
        length = min(len(encoded), max_len)
        encoded = encoded[:max_len]

        if len(encoded) < max_len:
            encoded = encoded + [0] * (max_len - len(encoded))

        input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
        lengths = torch.tensor([length], dtype=torch.long).to(device)

        with torch.no_grad():
            logits, attention_weights = model(input_ids, lengths, return_attention=True)
            _, predicted = torch.max(logits, 1)

        # Get attention weights for first sequence
        attn = attention_weights[0, :length, :length].cpu().numpy()

        # Get words
        words = text.split()[:length]

        results.append({
            "text": text,
            "words": words,
            "true_label": label,
            "predicted_label": predicted.item(),
            "attention": attn,
        })

    return results


# =====================================================================
# BERT FINE-TUNING
# =====================================================================


def fine_tune_bert(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    output_dir: str = "saved_models/bert",
) -> Tuple[Any, Any, Dict[str, List[float]]]:
    """Fine-tune BERT model."""
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Datasets
    train_dataset = BERTDataset(train_texts, train_labels, tokenizer, max_len=128)
    val_dataset = BERTDataset(val_texts, val_labels, tokenizer, max_len=128)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    trainer.train()

    # Extract history
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for log in trainer.state.log_history:
        if "loss" in log:
            history["train_loss"].append(log["loss"])
        if "eval_loss" in log:
            history["val_loss"].append(log["eval_loss"])

    return model, tokenizer, history


def predict_bert(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Get predictions from BERT model."""
    model.eval()
    model.to(device)

    predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encodings = tokenizer(
            batch_texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, 1)

        predictions.extend(preds.cpu().numpy())

    return np.array(predictions)


# =====================================================================
# DATASET UTILITIES for ANN
# =====================================================================


class ANNDataset(Dataset):
    """Dataset for ANN - dense features."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_ids": self.X[idx],
            "label": self.y[idx],
        }
