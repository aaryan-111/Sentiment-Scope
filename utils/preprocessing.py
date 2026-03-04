"""
Phase 5 — NLP preprocessing: lemmatization, stemming, ML/DL-ready outputs.
Uses ai_toolkit.nlp.preprocessor where applicable; SpaCy + NLTK for lemma/stem.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from utils.config import CONFIG

try:
    import ai_toolkit.nlp as nlp_toolkit
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False
    nlp_toolkit = None

# SpaCy and NLTK (optional)
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    spacy = None
try:
    from nltk.stem import PorterStemmer
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    PorterStemmer = None


def _get_nlp():
    """Load SpaCy model once (small)."""
    if not _SPACY_AVAILABLE or spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        return None


def lemmatize_text(text: str, nlp: Any = None) -> str:
    """SpaCy-based lemmatization; returns space-joined lemmas."""
    if nlp is None:
        nlp = _get_nlp()
    if nlp is None:
        return " ".join(re.findall(r"\b\w+\b", text.lower()))
    doc = nlp(text[:1_000_000])
    return " ".join(t.lemma_.strip() for t in doc if not t.is_space and t.lemma_.strip())


def stem_text(text: str, stemmer: Any = None) -> str:
    """NLTK PorterStemmer; returns space-joined stems."""
    if stemmer is None and _NLTK_AVAILABLE and PorterStemmer is not None:
        stemmer = PorterStemmer()
    if stemmer is None:
        return " ".join(re.findall(r"\b\w+\b", text.lower()))
    tokens = re.findall(r"\b\w+\b", text.lower())
    return " ".join(stemmer.stem(t) for t in tokens)


def compare_stem_vs_lemma(sample_texts: list[str], n_samples: int = 5) -> list[dict[str, str]]:
    """Side-by-side stem vs lemma for sample texts. Returns list of {text, stem, lemma}."""
    nlp = _get_nlp()
    stemmer = PorterStemmer() if _NLTK_AVAILABLE and PorterStemmer else None
    out = []
    for t in sample_texts[:n_samples]:
        out.append({
            "text": t[:200],
            "stem": stem_text(t, stemmer),
            "lemma": lemmatize_text(t, nlp),
        })
    return out


def preprocess_for_ml(texts: list[str], use_lemma: bool = True, nlp: Any = None) -> list[str]:
    """Final token string per document for sklearn (e.g. CountVectorizer/TfidfVectorizer)."""
    if use_lemma and _SPACY_AVAILABLE:
        if nlp is None:
            nlp = _get_nlp()
        return [lemmatize_text(t, nlp) for t in texts]
    stemmer = PorterStemmer() if _NLTK_AVAILABLE and PorterStemmer else None
    return [stem_text(t, stemmer) for t in texts]


class SimpleVocab:
    """Vocabulary for DL: word2idx, idx2word, <UNK>, <PAD>."""

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._built = False

    def build(self, tokenized_texts: list[list[str]]) -> "SimpleVocab":
        from collections import Counter
        c = Counter()
        for toks in tokenized_texts:
            c.update(toks)
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for w, cnt in c.most_common():
            if cnt >= self.min_freq and w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._built = True
        return self

    def encode(self, tokens: list[str], max_len: Optional[int] = None) -> list[int]:
        max_len = max_len or CONFIG.get("max_seq_len", 256)
        ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.word2idx)


def tokenize_simple(text: str) -> list[str]:
    """Simple tokenization: lower, split on non-alpha."""
    return re.findall(r"\b\w+\b", text.lower())


def preprocess_for_dl(
    texts: list[str],
    vocab: Optional[SimpleVocab] = None,
    max_len: Optional[int] = None,
) -> tuple[list[list[int]], SimpleVocab]:
    """
    Integer-encoded sequences for PyTorch. Returns (list of sequences, vocab).
    If vocab is None, builds from tokenized texts.
    """
    tokenized = [tokenize_simple(t) for t in texts]
    max_len = max_len or CONFIG.get("max_seq_len", 256)
    if vocab is None:
        vocab = SimpleVocab(min_freq=CONFIG.get("min_word_freq", 2)).build(tokenized)
    sequences = [vocab.encode(toks, max_len) for toks in tokenized]
    return sequences, vocab


def run_preprocessing(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    use_lemma: bool = True,
) -> tuple[pd.DataFrame, Optional[SimpleVocab], dict[str, Any]]:
    """
    Phase 5 pipeline: add processed_text, optional vocab and token stats.
    Returns (df with processed_text, vocab or None, token_stats dict).
    """
    df = df.copy()
    texts = df[text_col].astype(str).tolist()
    df["processed_text"] = preprocess_for_ml(texts, use_lemma=use_lemma)
    vocab = SimpleVocab(min_freq=CONFIG.get("min_word_freq", 2))
    tokenized = [t.split() for t in df["processed_text"].tolist()]
    vocab.build(tokenized)
    token_stats = {
        "vocab_size": len(vocab),
        "n_docs": len(texts),
        "avg_tokens_per_doc": sum(len(t) for t in tokenized) / len(tokenized) if tokenized else 0,
    }
    return df, vocab, token_stats
