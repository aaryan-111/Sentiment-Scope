"""
Phase 3 & 4 — EDA pipeline: analyses + visualizations via ai_toolkit.eda.
Produces 8–10 findings, each with a summary and a figure.
Uses local fallback for class distribution plots (see docs/toolkit-issues.md).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from utils.config import CONFIG

try:
    import ai_toolkit.eda as eda

    _TOOLKIT_AVAILABLE = True
except ImportError:
    _TOOLKIT_AVAILABLE = False
    eda = None


def _plot_class_distribution_fallback(
    df: pd.DataFrame,
    target_col: str,
    order: Optional[Sequence[Any]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
):
    """
    Fallback bar chart: categorical x-axis, count y-axis.
    Use when ai_toolkit.eda.plot_class_distribution misrenders (numeric x, 0–1 y).
    """
    counts = df[target_col].value_counts()
    if order is not None:
        counts = counts.reindex([x for x in order if x in counts.index]).dropna()
    else:
        counts = counts.sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = [str(x) for x in counts.index]
    ax.bar(x_labels, counts.values, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, ax


def _plot_bar_fallback(
    agg: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
):
    """
    Fallback bar chart: categorical x, numeric y (raw values, not normalized).
    Use when ai_toolkit.eda.plot_bar misrenders (y-axis 0–1).
    """
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = [str(x) for x in agg[x_col]]
    ax.bar(x_labels, agg[y_col].values, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, ax


def _plot_wordcloud_fallback(
    texts: list[str],
    label: Optional[str] = None,
    max_words: int = 100,
    figsize: tuple[float, float] = (12, 7),
    dpi: int = 100,
):
    """
    Fallback word cloud: build with wordcloud lib, display at readable size.
    Use when ai_toolkit.eda.plot_wordcloud renders too small in Streamlit.
    """
    from wordcloud import WordCloud

    combined = " ".join(t.strip() for t in texts if isinstance(t, str) and t.strip())
    if not combined:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No text", ha="center", va="center")
        if label:
            ax.set_title(label)
        return fig, ax
    wc = WordCloud(width=800, height=400, max_words=max_words, background_color="white").generate(combined)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=14)
    plt.tight_layout()
    return fig, ax


def _get_work_df(
    df: pd.DataFrame,
    text_col: str,
    sentiment_col: str,
    rating_col: str,
    max_sample: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Subset and ensure required columns; drop rows with missing text."""
    out = df[[c for c in [text_col, sentiment_col, rating_col] if c in df.columns]].copy()
    if text_col not in out.columns:
        return pd.DataFrame()
    out = out.loc[out[text_col].notna() & (out[text_col].astype(str).str.strip() != "")].copy()
    out["_char_len"] = out[text_col].astype(str).str.len()
    out["_word_len"] = out[text_col].astype(str).str.split().str.len()
    if max_sample and len(out) > max_sample:
        out = out.sample(n=max_sample, random_state=seed).reset_index(drop=True)
    return out


def run_eda(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    sentiment_col: str = "sentiment",
    rating_col: str = "rating",
    max_sample: Optional[int] = 50_000,
    top_ngram: int = 20,
    wordcloud_sample_per_class: int = 5_000,
    seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Run full EDA: Phase 3 analyses + Phase 4 visualizations using ai_toolkit.eda.
    Returns a list of {"name", "summary", "figure"} for Streamlit display.
    """
    if not _TOOLKIT_AVAILABLE or eda is None:
        return [{"name": "Error", "summary": "ai_toolkit.eda not available. Install aaryan-ai-toolkit[full].", "figure": None}]

    seed = seed or CONFIG.get("random_seed", 42)
    work = _get_work_df(df, text_col, sentiment_col, rating_col, max_sample, seed)
    if work.empty:
        return [{"name": "Error", "summary": "No data with non-empty text column.", "figure": None}]

    results: list[dict[str, Any]] = []
    figsize = (10, 5)

    # 1. Class distribution (sentiment) — fallback: toolkit plots categorical as numeric
    try:
        counts = work[sentiment_col].value_counts()
        summary = counts.to_dict()
        fig, _ = _plot_class_distribution_fallback(
            work, target_col=sentiment_col,
            order=["negative", "neutral", "positive"],
            title="Sentiment class distribution", figsize=figsize,
        )
        results.append({"name": "Sentiment class distribution", "summary": summary, "figure": fig})
    except Exception as e:
        results.append({"name": "Sentiment class distribution", "summary": {"error": str(e)}, "figure": None})

    # 2. Class distribution (raw rating) — fallback: same toolkit issue
    if rating_col in work.columns:
        try:
            summary = work[rating_col].value_counts().sort_index().to_dict()
            fig, _ = _plot_class_distribution_fallback(
                work, target_col=rating_col,
                order=[1, 2, 3, 4, 5],
                title="Rating distribution (1–5)", figsize=figsize,
            )
            results.append({"name": "Rating distribution", "summary": summary, "figure": fig})
        except Exception as e:
            results.append({"name": "Rating distribution", "summary": {"error": str(e)}, "figure": None})

    # 3. Text length (char-level) — cap x-axis so main body is visible (scale fix)
    try:
        fig, ax = eda.plot_text_length_distribution(
            work, text_col=text_col, kind="char", n_bins=50, kde=True,
            figsize=figsize, title="Review length (characters)", show=False,
        )
        s = work["_char_len"]
        x_max = max(s.quantile(0.99), 100)  # 99th percentile, at least 100
        ax.set_xlim(left=0, right=x_max)
        summary = {"min": int(s.min()), "max": int(s.max()), "mean": round(s.mean(), 1), "median": float(s.median())}
        results.append({"name": "Text length (characters)", "summary": summary, "figure": fig})
    except Exception as e:
        results.append({"name": "Text length (characters)", "summary": {"error": str(e)}, "figure": None})

    # 4. Text length (word-level) — cap x-axis so main body is visible (scale fix)
    try:
        fig, ax = eda.plot_text_length_distribution(
            work, text_col=text_col, kind="word", n_bins=50, kde=True,
            figsize=figsize, title="Review length (words)", show=False,
        )
        s = work["_word_len"]
        x_max = max(s.quantile(0.99), 50)  # 99th percentile, at least 50
        ax.set_xlim(left=0, right=x_max)
        summary = {"min": int(s.min()), "max": int(s.max()), "mean": round(s.mean(), 1), "median": float(s.median())}
        results.append({"name": "Text length (words)", "summary": summary, "figure": fig})
    except Exception as e:
        results.append({"name": "Text length (words)", "summary": {"error": str(e)}, "figure": None})

    # 5. Null / duplicate analysis (summary only)
    null_counts = df.isnull().sum()
    n_dup = df.duplicated(subset=[text_col] if text_col in df.columns else None).sum()
    summary = {"null_counts": null_counts.to_dict(), "n_duplicates": int(n_dup), "n_rows": len(df)}
    results.append({"name": "Nulls & duplicates", "summary": summary, "figure": None})

    # 6. Review length vs rating — fallback: toolkit plot_bar uses y-axis 0–1
    if rating_col in work.columns:
        try:
            agg = work.groupby(rating_col)["_word_len"].agg(["mean", "median", "count"]).reset_index()
            agg.columns = [rating_col, "mean_words", "median_words", "count"]
            fig, _ = _plot_bar_fallback(
                agg, x_col=rating_col, y_col="mean_words",
                title="Mean review length (words) by rating",
                xlabel="Rating", ylabel="Mean word count", figsize=figsize,
            )
            results.append({"name": "Length vs rating", "summary": agg.to_dict("records"), "figure": fig})
        except Exception as e:
            results.append({"name": "Length vs rating", "summary": {"error": str(e)}, "figure": None})

    # 7 & 8. Top unigrams and bigrams (all data) + per-class top-5 in summary
    for n, label in [(1, "Unigrams"), (2, "Bigrams")]:
        try:
            texts_list = work[text_col].astype(str).tolist()
            fig = eda.plot_ngram_frequency(
                texts_list, n=n, top_k=top_ngram,
                title=f"Top {top_ngram} {label.lower()} (all data)", backend="matplotlib", show=False,
            )
            if isinstance(fig, tuple):
                fig = fig[0]
            per_class: dict[str, list] = {}
            for cls in work[sentiment_col].dropna().unique():
                sub = work[work[sentiment_col] == cls][text_col].astype(str).tolist()
                tokens = []
                for t in sub:
                    if n == 1:
                        tokens.extend(re.findall(r"\b\w+\b", t.lower()))
                    else:
                        words = re.findall(r"\b\w+\b", t.lower())
                        tokens.extend(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
                top = Counter(tokens).most_common(5)
                per_class[str(cls)] = [str(x) for x in top]
            results.append({"name": f"Top {label}", "summary": {"per_class_top5": per_class}, "figure": fig})
        except Exception as e:
            results.append({"name": f"Top {label}", "summary": {"error": str(e)}, "figure": None})

    # 9. Vocabulary size
    all_tokens = []
    for t in work[text_col].astype(str):
        all_tokens.extend(t.lower().split())
    vocab_size = len(set(all_tokens))
    n_tokens = len(all_tokens)
    summary = {"vocab_size": vocab_size, "total_tokens": n_tokens, "avg_tokens_per_doc": round(n_tokens / len(work), 1)}
    results.append({"name": "Vocabulary size", "summary": summary, "figure": None})

    # 10. WordCloud per sentiment class — fallback: toolkit renders too small in Streamlit
    for cls in work[sentiment_col].dropna().unique():
        sub = work[work[sentiment_col] == cls][text_col].astype(str).tolist()
        if len(sub) > wordcloud_sample_per_class:
            import random
            rng = random.Random(seed)
            sub = rng.sample(sub, wordcloud_sample_per_class)
        if not sub:
            continue
        try:
            fig, _ = _plot_wordcloud_fallback(
                sub, label=str(cls), max_words=100, figsize=(12, 7), dpi=100,
            )
            results.append({"name": f"WordCloud — {cls}", "summary": {"n_docs": len(sub)}, "figure": fig})
        except Exception as e:
            results.append({"name": f"WordCloud — {cls}", "summary": {"error": str(e)}, "figure": None})

    return results
