"""
Phase 1 — Data gathering: download, load, sample, and initial inspection.
Download/load/sample implemented here; inspection uses ai_toolkit where available.
Supports both file path and in-memory upload (Streamlit file_uploader).
"""
import gzip
import io
import json
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve

import pandas as pd


def download_dataset(url: str, save_path: str) -> Path:
    """
    Download dataset from URL with basic caching (skip if file exists).
    Handles .jsonl.gz; save_path should be the full path for the saved file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    urlretrieve(url, path)
    return path


def _read_jsonl_gz(
    path: Path, n: Optional[int], seed: int, max_rows: int = 500_000
) -> pd.DataFrame:
    """Read JSONL.gz and return DataFrame. Reads up to max_rows lines, then samples n if requested."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def load_and_sample(
    path: str,
    n: Optional[int] = None,
    seed: int = 42,
    text_column: str = "review_text",
    rating_column: str = "rating",
) -> pd.DataFrame:
    """
    Load JSON/JSONL (or .jsonl.gz) from path and optionally sample n rows with seed.
    Returns DataFrame. Tries to normalize column names (review_text / text, rating).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    if p.suffix == ".gz" or (p.suffix == ".jsonl" and str(p).endswith(".jsonl.gz")):
        df = _read_jsonl_gz(p, n, seed)
    elif p.suffix in (".json", ".jsonl"):
        df = pd.read_json(path, lines=(p.suffix == ".jsonl"), encoding="utf-8", encoding_errors="replace")
        if n is not None and len(df) > n:
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", nrows=n)
        if n is not None and len(df) > n:
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    _normalize_columns(df, text_column, rating_column)
    return df


def load_from_upload(
    file_or_bytes: Union[io.BytesIO, bytes],
    filename: Optional[str] = None,
    n: Optional[int] = None,
    seed: int = 42,
    text_column: str = "review_text",
    rating_column: str = "rating",
) -> pd.DataFrame:
    """
    Load from an uploaded file (e.g. Streamlit st.file_uploader).
    file_or_bytes: BytesIO or bytes. filename optional (used to infer format).
    Supports: .csv, .json, .jsonl, .jsonl.gz
    """
    if hasattr(file_or_bytes, "read"):
        raw = file_or_bytes.read()
        if hasattr(file_or_bytes, "name"):
            filename = filename or getattr(file_or_bytes, "name", "")
    else:
        raw = file_or_bytes
    filename = filename or ""
    name_lower = filename.lower()

    if name_lower.endswith(".jsonl.gz") or name_lower.endswith(".gz"):
        buf = io.BytesIO(raw)
        rows = []
        with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
            for line in io.TextIOWrapper(gz, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        df = pd.DataFrame(rows)
    elif name_lower.endswith(".jsonl") or name_lower.endswith(".json"):
        text = raw.decode("utf-8", errors="replace")
        lines = text.strip().split("\n")
        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        df = pd.DataFrame(rows)
    elif name_lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", encoding_errors="replace")
    else:
        # Try JSONL by default (common for reviews)
        text = raw.decode("utf-8", errors="replace")
        lines = text.strip().split("\n")
        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        df = pd.DataFrame(rows) if rows else pd.read_csv(io.BytesIO(raw), encoding="utf-8", encoding_errors="replace")

    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    _normalize_columns(df, text_column, rating_column)
    return df


def _normalize_columns(
    df: pd.DataFrame, text_column: str, rating_column: str
) -> None:
    """Normalize column names in place (text/review_text, overall/rating)."""
    if "text" in df.columns and text_column not in df.columns:
        df.rename(columns={"text": text_column}, inplace=True)
    if "overall" in df.columns and rating_column not in df.columns:
        df.rename(columns={"overall": rating_column}, inplace=True)


def initial_inspection(df: pd.DataFrame) -> dict:
    """
    Basic inspection: shape, dtypes, null counts, sample rows.
    Returns a dict for display. Use ai_toolkit.eda for richer inspection when available.
    """
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": null_counts.to_dict(),
        "null_pct": null_pct.to_dict(),
        "sample": df.head(10),
    }
