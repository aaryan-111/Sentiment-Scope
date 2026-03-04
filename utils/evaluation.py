"""
Evaluation utilities for model analysis, error analysis, and explainability.
Phases 11, 12, 14: Classification reports, confusion matrix, ROC, SHAP, LIME.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# Try to import SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


# =====================================================================
# CLASSIFICATION METRICS & REPORTS
# =====================================================================


def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        class_names: Names of classes
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1 (macro and per-class)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics["precision_macro"] = np.mean(precision)
    metrics["recall_macro"] = np.mean(recall)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(precision))]
    
    metrics["per_class"] = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for name, p, r, f, s in zip(class_names, precision, recall, f1, support)
    }
    
    # ROC AUC (if probabilities provided)
    if y_pred_proba is not None:
        try:
            n_classes = y_pred_proba.shape[1]
            if n_classes == 2:
                # Binary classification
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Multiclass (one-vs-rest)
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_pred_proba, multi_class="ovr", average="macro"
                )
        except Exception:
            metrics["roc_auc"] = None
    
    return metrics


def get_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Get classification report as DataFrame.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        DataFrame with classification report
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in np.unique(y_true)]
    
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Round numeric columns
    numeric_cols = ["precision", "recall", "f1-score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)
    
    # Format support as integer
    if "support" in df.columns:
        df["support"] = df["support"].astype(int)
    
    return df


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize counts
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names or np.unique(y_true),
        yticklabels=class_names or np.unique(y_true),
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    plt.tight_layout()
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot ROC curves for multiclass classification (one-vs-rest).
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        class_names: Names of classes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_classes = y_pred_proba.shape[1]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        
        ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =====================================================================
# ERROR ANALYSIS
# =====================================================================


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    max_samples: int = 100,
) -> pd.DataFrame:
    """
    Analyze misclassified samples.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        texts: Original texts
        class_names: Names of classes
        max_samples: Maximum number of error samples to return
        
    Returns:
        DataFrame with error samples
    """
    # Find misclassified indices
    errors_idx = np.where(y_true != y_pred)[0]
    
    if len(errors_idx) == 0:
        return pd.DataFrame()
    
    # Limit number of samples
    if len(errors_idx) > max_samples:
        errors_idx = np.random.choice(errors_idx, max_samples, replace=False)
    
    error_data = []
    
    for idx in errors_idx:
        row = {
            "index": int(idx),
            "true_label": int(y_true[idx]),
            "predicted_label": int(y_pred[idx]),
        }
        
        if class_names is not None:
            row["true_class"] = class_names[y_true[idx]]
            row["predicted_class"] = class_names[y_pred[idx]]
        
        if y_pred_proba is not None:
            row["confidence"] = float(y_pred_proba[idx, y_pred[idx]])
            row["true_class_prob"] = float(y_pred_proba[idx, y_true[idx]])
        
        if texts is not None:
            row["text"] = texts[idx][:200]  # Truncate for display
        
        error_data.append(row)
    
    df = pd.DataFrame(error_data)
    
    # Sort by confidence (most confident errors first)
    if "confidence" in df.columns:
        df = df.sort_values("confidence", ascending=False)
    
    return df


def get_error_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Get statistics about errors per class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        DataFrame with error statistics
    """
    stats = []
    
    unique_labels = np.unique(y_true)
    
    for label in unique_labels:
        mask = y_true == label
        errors = np.sum((y_true == label) & (y_pred != label))
        total = np.sum(mask)
        error_rate = errors / total if total > 0 else 0
        
        stats.append({
            "label": int(label),
            "class_name": class_names[label] if class_names else f"Class {label}",
            "total": int(total),
            "errors": int(errors),
            "error_rate": float(error_rate),
            "accuracy": float(1 - error_rate),
        })
    
    return pd.DataFrame(stats)


# =====================================================================
# REGRESSION METRICS
# =====================================================================


def get_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }
    
    return metrics


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot predicted vs actual values for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
    
    # Metrics annotation
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    ax.text(
        0.05, 0.95,
        f'RMSE: {rmse:.3f}\nR²: {r2:.3f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =====================================================================
# SHAP EXPLAINABILITY
# =====================================================================


def explain_with_shap(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
) -> Tuple[Optional[Any], Optional[plt.Figure]]:
    """
    Generate SHAP explanations.
    
    Args:
        model: Trained model
        X_train: Training data (for background)
        X_test: Test data
        feature_names: Names of features
        max_display: Maximum features to display
        
    Returns:
        Tuple of (explainer, summary plot figure)
    """
    if not SHAP_AVAILABLE:
        return None, None
    
    try:
        # Try TreeExplainer first (for tree-based models)
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            # Fall back to KernelExplainer
            # Use a subset of training data as background
            background = shap.sample(X_train, min(100, len(X_train)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test[:100])  # Limit to 100 samples
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            # Multi-class: use first class
            shap_values = shap_values[0]
        
        shap.summary_plot(
            shap_values,
            X_test[:100],
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        return explainer, plt.gcf()
        
    except Exception as e:
        print(f"SHAP error: {e}")
        return None, None


# =====================================================================
# LIME EXPLAINABILITY
# =====================================================================


def explain_with_lime(
    model: Any,
    texts: List[str],
    class_names: List[str],
    num_samples: int = 5,
    num_features: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate LIME explanations for text samples.
    
    Args:
        model: Trained model with predict_proba method
        texts: Text samples
        class_names: Names of classes
        num_samples: Number of samples to explain
        num_features: Number of features to show
        
    Returns:
        List of explanation dictionaries
    """
    if not LIME_AVAILABLE:
        return []
    
    try:
        explainer = LimeTextExplainer(class_names=class_names)
        
        explanations = []
        
        for i in range(min(num_samples, len(texts))):
            text = texts[i]
            
            # Generate explanation
            exp = explainer.explain_instance(
                text,
                model.predict_proba,
                num_features=num_features,
                num_samples=500
            )
            
            # Get prediction
            pred_proba = model.predict_proba([text])[0]
            pred_class = np.argmax(pred_proba)
            
            explanations.append({
                "text": text,
                "predicted_class": class_names[pred_class],
                "confidence": float(pred_proba[pred_class]),
                "explanation": exp,
                "top_features": exp.as_list(),
            })
        
        return explanations
        
    except Exception as e:
        print(f"LIME error: {e}")
        return []


# =====================================================================
# MODEL COMPARISON
# =====================================================================


def create_comparison_table(
    ml_results: List[Dict[str, Any]],
    dl_results: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Create master comparison table of all models.
    
    Args:
        ml_results: Results from ML models
        dl_results: Results from DL models
        
    Returns:
        DataFrame with comparison
    """
    all_results = []
    
    # Add ML results
    for result in ml_results:
        row = {
            "Model Type": "Classical ML",
            "Model": result.get("model", "Unknown"),
            "Representation": result.get("representation", "Unknown"),
            "Accuracy": result.get("accuracy", 0),
            "Macro F1": result.get("macro_f1", 0),
            "Train Time (s)": result.get("train_time", 0),
        }
        all_results.append(row)
    
    # Add DL results if provided
    if dl_results:
        for result in dl_results:
            row = {
                "Model Type": "Deep Learning",
                "Model": result.get("model", "Unknown"),
                "Representation": result.get("representation", "Unknown"),
                "Accuracy": result.get("accuracy", 0),
                "Macro F1": result.get("macro_f1", 0),
                "Train Time (s)": result.get("train_time", 0),
            }
            all_results.append(row)
    
    df = pd.DataFrame(all_results)
    
    # Sort by Macro F1 descending
    if len(df) > 0:
        df = df.sort_values("Macro F1", ascending=False).reset_index(drop=True)
    
    return df


def get_best_models(
    comparison_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Get top N best models by Macro F1.
    
    Args:
        comparison_df: Comparison DataFrame
        top_n: Number of top models to return
        
    Returns:
        DataFrame with top models
    """
    return comparison_df.head(top_n)
