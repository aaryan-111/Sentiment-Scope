"""
Train/val/test split — single seed-based split for reproducibility.
Reused by Preprocessing, ML, and DL. CONFIG test_size, val_size, random_seed.
"""
from __future__ import annotations

import pandas as pd

from utils.config import CONFIG


def get_train_val_test_split(
    df: pd.DataFrame,
    seed: int | None = None,
    test_size: float | None = None,
    val_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, val, test using config defaults.
    train = (1 - test_size - val_size), val = val_size, test = test_size.
    Returns (df_train, df_val, df_test).
    """
    seed = seed if seed is not None else CONFIG["random_seed"]
    test_size = test_size if test_size is not None else CONFIG["test_size"]
    val_size = val_size if val_size is not None else CONFIG["val_size"]
    train_ratio = 1.0 - test_size - val_size
    if train_ratio <= 0:
        raise ValueError("test_size + val_size must be < 1")
    import numpy as np
    n = len(df)
    idx = df.index.to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(idx)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val
    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val :]
    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )
