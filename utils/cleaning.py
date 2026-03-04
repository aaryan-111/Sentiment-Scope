"""
Phase 2 — Data cleaning. Uses ai_toolkit.nlp (preprocessor) when available;
fallback implementations so the app runs without the toolkit.
"""
import re
from typing import Optional

import pandas as pd

from utils.config import RATING_TO_SENTIMENT

# Prefer toolkit; fallback to local
try:
    from ai_toolkit.nlp import preprocessor  # type: ignore
    _TOOLKIT_AVAILABLE = True
except ImportError:
    _TOOLKIT_AVAILABLE = False
    preprocessor = None

if not _TOOLKIT_AVAILABLE:
    try:
        import contractions
    except ImportError:
        contractions = None
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None


def _remove_html(text: str) -> str:
    if BeautifulSoup is None:
        return re.sub(r"<[^>]+>", "", text)
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def _expand_contractions(text: str) -> str:
    if contractions is None:
        return text
    return contractions.fix(text)


def _remove_noise(text: str) -> str:
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # numbers
    text = re.sub(r"\d+", " ", text)
    # punctuation and extra symbols (keep apostrophe for contractions if done before)
    text = re.sub(r"[^\w\s']", " ", text)
    return text


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def clean_pipeline(text: str, use_toolkit: bool = True) -> str:
    """
    Compose: remove HTML → expand contractions → remove noise → normalize.
    Uses ai_toolkit.nlp.preprocessor when use_toolkit=True and available.
    """
    if use_toolkit and _TOOLKIT_AVAILABLE and hasattr(preprocessor, "clean_pipeline"):
        return preprocessor.clean_pipeline(text)
    if use_toolkit and _TOOLKIT_AVAILABLE and hasattr(preprocessor, "normalize"):
        # If toolkit has separate steps, chain them
        t = _remove_html(str(text))
        t = _expand_contractions(t)
        t = _remove_noise(t)
        return preprocessor.normalize(t)
    # Fallback
    t = _remove_html(str(text))
    t = _expand_contractions(t)
    t = _remove_noise(t)
    return _normalize_text(t)


def handle_nulls(
    df: pd.DataFrame,
    text_column: str,
    drop_threshold: float = 1.0,
    fill_value: str = "",
) -> pd.DataFrame:
    """
    Handle nulls in text column: drop rows where text is null if drop_threshold=1.0;
    otherwise fill with fill_value. Returns copy of DataFrame.
    """
    out = df.copy()
    null_mask = out[text_column].isna()
    if drop_threshold >= 1.0:
        out = out[~null_mask].reset_index(drop=True)
    else:
        out.loc[null_mask, text_column] = fill_value
    return out


def create_sentiment_label(rating_series: pd.Series) -> pd.Series:
    """Map numeric rating 1–5 to sentiment: 1–2 → negative, 3 → neutral, 4–5 → positive."""
    def _map(x):
        if pd.isna(x):
            return "neutral"
        try:
            return RATING_TO_SENTIMENT.get(int(x), "neutral")
        except (ValueError, TypeError):
            return "neutral"
    return rating_series.map(_map)


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "review_text",
    rating_column: str = "rating",
    use_toolkit: bool = True,
) -> pd.DataFrame:
    """
    Full cleaning: handle nulls, run clean_pipeline on text, add sentiment label.
    Returns df with columns: original text, clean_text, sentiment (and original rating if present).
    """
    out = handle_nulls(df, text_column)
    out = out.copy()
    out["clean_text"] = out[text_column].astype(str).map(lambda t: clean_pipeline(t, use_toolkit=use_toolkit))
    if rating_column in out.columns:
        out["sentiment"] = create_sentiment_label(out[rating_column])
    else:
        out["sentiment"] = "neutral"
    return out
