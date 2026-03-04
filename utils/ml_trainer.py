"""
Phase 7, 8, 9, 10 — Scaling, encoding, pipelines, training, tuning.
Uses ai_toolkit.ml (trainer, evaluator) when available; sklearn fallbacks otherwise.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.config import CONFIG

try:
    from ai_toolkit import ml as ml_toolkit
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    ml_toolkit = None

# Sklearn fallbacks
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    mean_squared_error,
    mean_absolute_error,
)

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    lgb = None


# --- Phase 7: Scaling & encoding ---

def scale_features(
    train: np.ndarray | pd.DataFrame,
    test: np.ndarray | pd.DataFrame,
    method: str = "standard",
) -> tuple[Any, np.ndarray, np.ndarray]:
    """
    Scale numerical features (e.g. text_stats, or W2V/SBERT when combined).
    Returns (fitted_scaler, X_train_scaled, X_test_scaled).
    """
    if _ML_AVAILABLE and hasattr(ml_toolkit, "scale_features"):
        return ml_toolkit.scale_features(train, test, method=method)
    # Fallback: StandardScaler
    scaler = StandardScaler()
    X_train = np.asarray(train) if not isinstance(train, np.ndarray) else train
    X_test = np.asarray(test) if not isinstance(test, np.ndarray) else test
    if hasattr(X_train, "values"):
        X_train = train.values
    if hasattr(X_test, "values"):
        X_test = test.values
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def encode_labels(series: pd.Series | np.ndarray, method: str = "label") -> tuple[Any, np.ndarray]:
    """Encode labels to integers. Returns (encoder, encoded_array)."""
    if _ML_AVAILABLE and hasattr(ml_toolkit, "encode_labels"):
        return ml_toolkit.encode_labels(series, method=method)
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    arr = np.asarray(series)
    encoded = enc.fit_transform(arr)
    return enc, encoded


# --- Phase 8: Pipeline building ---

def build_classification_pipeline(vectorizer: Any, classifier: Any) -> Pipeline:
    """Build sklearn Pipeline: vectorizer -> classifier (or just classifier if vectorizer is None)."""
    if _ML_AVAILABLE and hasattr(ml_toolkit, "build_classification_pipeline"):
        return ml_toolkit.build_classification_pipeline(vectorizer, classifier)
    if vectorizer is None:
        return Pipeline([("clf", classifier)])
    return Pipeline([("vec", vectorizer), ("clf", classifier)])


def build_regression_pipeline(vectorizer: Any, regressor: Any) -> Pipeline:
    """Build sklearn Pipeline: vectorizer -> regressor (or just regressor if vectorizer is None)."""
    if _ML_AVAILABLE and hasattr(ml_toolkit, "build_regression_pipeline"):
        return ml_toolkit.build_regression_pipeline(vectorizer, regressor)
    if vectorizer is None:
        return Pipeline([("reg", regressor)])
    return Pipeline([("vec", vectorizer), ("reg", regressor)])


def run_gridsearch(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int | None = None,
    n_jobs: int | None = None,
    scoring: str = "f1_macro",
) -> tuple[Pipeline, dict, float]:
    """
    Run GridSearchCV. Returns (best_estimator, best_params, best_score), and logs time.
    """
    cv = cv or CONFIG.get("cv_folds", 5)
    n_jobs = n_jobs if n_jobs is not None else CONFIG.get("n_jobs", -1)
    if _ML_AVAILABLE and hasattr(ml_toolkit, "run_gridsearch"):
        return ml_toolkit.run_gridsearch(
            pipeline, param_grid, X_train, y_train, cv=cv
        )
    start = time.perf_counter()
    search = GridSearchCV(
        pipeline, param_grid, cv=cv, n_jobs=n_jobs, scoring=scoring, verbose=0
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return search.best_estimator_, search.best_params_, float(search.best_score_)


# --- Metrics (fixed contract: Phase 8) ---

def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro F1, per-class F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    # per_class: (precision, recall, f1, support)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class_f1 = dict(zip([int(l) for l in labels], per_class[2].tolist()))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# --- Phase 9: Train and evaluate ---

def _cast_X(X: Any) -> Any:
    """
    Ensure feature matrix has float dtype required by LightGBM (and safe for all estimators).
    CountVectorizer/BoW produces int64 sparse matrices; cast to float32.
    """
    from scipy import sparse as _sparse
    if _sparse.issparse(X):
        if X.dtype not in (np.float32, np.float64):
            return X.astype(np.float32)
        return X
    arr = np.asarray(X)
    if arr.dtype not in (np.float32, np.float64):
        return arr.astype(np.float32)
    return arr


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: Any,
    X_test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str = "classification",
) -> dict[str, Any]:
    """
    Fit pipeline on (X_train, y_train), predict on X_test, return metrics dict.
    task in ('classification', 'regression').
    Always uses the local sklearn implementation for reliable, consistent metrics.
    """
    # TODO: restore toolkit — see toolkit-issues.md #5 (remove casts once toolkit handles dtype internally)
    # Cast feature dtype to float32 — LightGBM rejects int64 sparse data
    X_train = _cast_X(X_train)
    X_test = _cast_X(X_test)

    # TODO: restore toolkit — see toolkit-issues.md #4 (restore ml_toolkit.train_and_evaluate once return keys are standardized)
    pipeline = pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if task == "classification":
        metrics = _classification_metrics(y_test, y_pred)
    else:
        metrics = _regression_metrics(y_test, y_pred)
    metrics["_fitted_pipeline"] = pipeline
    return metrics


def build_results_table(all_results: list[dict]) -> pd.DataFrame:
    """Build sortable DataFrame from list of result dicts (from train_and_evaluate)."""
    # TODO: restore toolkit — see toolkit-issues.md (toolkit includes per_class_f1 as raw dict column with int keys — breaks Arrow/PyArrow serialization in Streamlit)
    rows = []
    for r in all_results:
        row = {
            "model": r.get("model", ""),
            "representation": r.get("representation", ""),
            **{k: v for k, v in r.items() if k in ("accuracy", "macro_f1", "rmse", "mae") and isinstance(v, (int, float))},
        }
        if "per_class_f1" in r and isinstance(r["per_class_f1"], dict):
            for k, v in r["per_class_f1"].items():
                row[f"f1_class_{k}"] = round(float(v), 4)  # str key, float value — Arrow-safe
        rows.append(row)
    return pd.DataFrame(rows)


def get_xy_for_representation(
    features: dict[str, Any],
    representation: str,
    scale_combined: bool = True,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """
    Get (X_train, X_test, vectorizer_or_None) for a given representation.
    representation: 'bow', 'tfidf', 'w2v', 'sbert', 'tfidf_stats'.
    For tfidf_stats we scale text_stats and hstack with TF-IDF (showcases scaling).
    For w2v/sbert we return as-is (dense); scale when combining with stats if needed.
    """
    rep = representation.lower().replace(" ", "_")
    if rep == "bow":
        vec, Xt, Xte = features["bow"]
        return Xt, Xte, vec
    if rep == "tfidf":
        vec, Xt, Xte = features["tfidf"]
        return Xt, Xte, vec
    if rep == "w2v":
        _, Xt, Xte = features["w2v"]
        return np.asarray(Xt), np.asarray(Xte), None
    if rep == "sbert":
        Xt, Xte = features["sbert"]
        return np.asarray(Xt), np.asarray(Xte), None
    if rep == "tfidf_stats":
        _, X_tfidf_train, X_tfidf_test = features["tfidf"]
        df_st_train, df_st_test = features["text_stats"]
        if scale_combined:
            _, st_train, st_test = scale_features(df_st_train, df_st_test)
        else:
            st_train = np.asarray(df_st_train)
            st_test = np.asarray(df_st_test)
        from scipy import sparse
        if sparse.issparse(X_tfidf_train):
            X_train = np.hstack([X_tfidf_train.toarray(), st_train])
            X_test = np.hstack([X_tfidf_test.toarray(), st_test])
        else:
            X_train = np.hstack([X_tfidf_train, st_train])
            X_test = np.hstack([X_tfidf_test, st_test])
        return X_train, X_test, None
    raise ValueError(f"Unknown representation: {representation}")


# --- Phase 10: Tuning ---

def get_param_grid(model_name: str, task: str = "classification") -> dict:
    """
    Predefined param grid per model type (in-repo).
    model_name: e.g. 'LogisticRegression', 'ComplementNB', 'RandomForest', 'LightGBM',
                'Ridge', 'RandomForestRegressor', 'LightGBMRegressor'.
    """
    model_name = model_name.lower().replace(" ", "")
    if task == "regression":
        if "ridge" in model_name:
            return {"reg__alpha": [0.1, 1.0, 10.0]}
        if "linear" in model_name:  # LinearRegression has no hyperparams to tune
            return {}
        if "forest" in model_name or "randomforest" in model_name:
            return {
                "reg__n_estimators": [50, 100],
                "reg__max_depth": [10, 20],
                "reg__min_samples_leaf": [2, 5],
            }
        if "lightgbm" in model_name or "lgb" in model_name:
            return {
                "reg__n_estimators": [50, 100],
                "reg__max_depth": [5, 10],
                "reg__learning_rate": [0.05, 0.1],
            }
        return {}
    # Classification
    if "logistic" in model_name or "lr" in model_name:
        return {"clf__C": [0.1, 1.0, 10.0], "clf__max_iter": [500, 1000]}
    if "complement" in model_name or "nb" in model_name:
        return {"clf__alpha": [0.1, 0.5, 1.0]}
    if "forest" in model_name or "randomforest" in model_name:
        return {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [10, 20],
            "clf__min_samples_leaf": [2, 5],
        }
    if "lightgbm" in model_name or "lgb" in model_name:
        return {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [5, 10],
            "clf__learning_rate": [0.05, 0.1],
        }
    return {}


def tune_pipeline(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "classification",
    cv: int | None = None,
    n_iter: int = 10,
) -> tuple[Pipeline, dict, float, float]:
    """
    RandomizedSearchCV wrapper with timing.
    Returns (best_estimator, best_params, best_score, elapsed_seconds).
    """
    cv = cv or CONFIG.get("cv_folds", 5)
    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    # TODO: restore toolkit — see toolkit-issues.md #5 (remove cast once toolkit handles dtype internally)
    X_train = _cast_X(X_train)
    start = time.perf_counter()
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=min(n_iter, 1 + sum(len(v) if hasattr(v, "__len__") else 1 for v in param_grid.values())),
        cv=cv,
        n_jobs=CONFIG.get("n_jobs", -1),
        scoring=scoring,
        random_state=CONFIG.get("random_seed", 42),
        verbose=0,
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    score = float(search.best_score_)
    if task == "regression":
        score = -score  # neg_root_mean_squared_error
    return search.best_estimator_, search.best_params_, score, elapsed


# --- Model constructors (for dropdown) ---

def get_classification_estimator(name: str):
    """Return a classifier instance by name."""
    name = name.lower().replace(" ", "")
    if "logistic" in name or name == "lr":
        return LogisticRegression(max_iter=1000, random_state=CONFIG.get("random_seed", 42))
    if "complement" in name or "nb" in name:
        return ComplementNB()
    if "forest" in name or "random" in name:
        return RandomForestClassifier(random_state=CONFIG.get("random_seed", 42))
    if "lightgbm" in name or "lgb" in name:
        if not _LGB_AVAILABLE:
            raise RuntimeError("LightGBM not installed. pip install lightgbm")
        return lgb.LGBMClassifier(random_state=CONFIG.get("random_seed", 42), verbosity=-1)
    raise ValueError(f"Unknown classifier: {name}")


def get_regression_estimator(name: str):
    """Return a regressor instance by name."""
    name = name.lower().replace(" ", "")
    if "linear" in name and "ridge" not in name:
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if "ridge" in name:
        return Ridge()  # Ridge does not accept random_state
    if "forest" in name or "random" in name:
        return RandomForestRegressor(random_state=CONFIG.get("random_seed", 42))
    if "lightgbm" in name or "lgb" in name:
        if not _LGB_AVAILABLE:
            raise RuntimeError("LightGBM not installed. pip install lightgbm")
        return lgb.LGBMRegressor(random_state=CONFIG.get("random_seed", 42), verbosity=-1)
    raise ValueError(f"Unknown regressor: {name}")


# --- Saved model naming (industry-grade) ---

def saved_model_path(
    task: str,
    model_name: str,
    representation: str,
    tuned: bool = True,
    saved_models_dir: str | Path | None = None,
) -> Path:
    """
    Consistent naming: {task}/{model_slug}_{representation}.joblib
    e.g. classification/lr_tfidf.joblib, regression/ridge_sbert.joblib
    """
    saved_models_dir = Path(saved_models_dir or CONFIG["saved_models_dir"])
    slug = model_name.lower().replace(" ", "_")
    if "logistic" in slug or slug == "lr":
        slug = "lr"
    elif "complement" in slug:
        slug = "complement_nb"
    elif "random" in slug or "forest" in slug:
        slug = "rf"
    elif "lightgbm" in slug or slug == "lgb":
        slug = "lgbm"
    elif "ridge" in slug:
        slug = "ridge"
    elif "linear" in slug:
        slug = "linear"
    rep = representation.lower().replace(" ", "_")
    filename = f"{slug}_{rep}.joblib"
    subdir = saved_models_dir / task
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / filename


def representation_from_saved_path(path: Path) -> str:
    """Parse representation from saved model filename, e.g. lr_tfidf.joblib -> tfidf."""
    stem = path.stem
    for slug in ("lgbm", "complement_nb", "ridge", "linear", "rf", "lr"):
        prefix = slug + "_"
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return "tfidf"
