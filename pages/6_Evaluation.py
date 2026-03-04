"""
Phase 11, 12, 14 — Evaluation, SHAP/LIME, final comparison. Uses ai_toolkit.ml (evaluator).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.sparse import issparse

from utils.config import CONFIG
from utils.evaluation import (
    analyze_errors,
    create_comparison_table,
    explain_with_lime,
    explain_with_shap,
    get_best_models,
    get_classification_metrics,
    get_classification_report_df,
    get_error_statistics,
    plot_confusion_matrix,
    plot_roc_curves,
    SHAP_AVAILABLE,
    LIME_AVAILABLE,
)

st.set_page_config(page_title="Evaluation — SentimentScope", layout="wide")
st.title("Evaluation & Explainability")
st.caption("Phase 11, 12, 14: Classification reports, confusion matrix, ROC, error analysis, SHAP/LIME, model comparison")

# --- Check prerequisites ---
def _features_ready():
    return (
        st.session_state.get("features_done") is True
        and "features" in st.session_state
        and "y_train" in st.session_state
        and "y_test" in st.session_state
    )

def _models_available():
    ml_results = st.session_state.get("ml_results_classification", [])
    dl_results = st.session_state.get("dl_results", [])
    return len(ml_results) > 0 or len(dl_results) > 0

# --- Sidebar ---
with st.sidebar:
    st.header("Evaluation Config")
    
    if _features_ready() and _models_available():
        st.success("Data & models ready")
        
        # Model selector
        ml_results = st.session_state.get("ml_results_classification", [])
        dl_results = st.session_state.get("dl_results", [])
        
        all_models = []
        
        # Build model list
        for result in ml_results:
            model_name = f"ML: {result['model']} ({result['representation']})"
            all_models.append(("ml", result, model_name))
        
        for result in dl_results:
            model_name = f"DL: {result['model']} ({result['representation']})"
            all_models.append(("dl", result, model_name))
        
        if all_models:
            model_names = [m[2] for m in all_models]
            selected_idx = st.selectbox(
                "Select Model for Analysis",
                range(len(model_names)),
                format_func=lambda i: model_names[i],
                key="eval_model_select"
            )
            
            selected_model_type, selected_result, selected_name = all_models[selected_idx]
            st.session_state["eval_selected_model"] = (selected_model_type, selected_result)
        else:
            st.warning("No trained models found")
    else:
        if not _features_ready():
            st.warning("Run **Preprocessing** first")
        if not _models_available():
            st.warning("Train models on **ML Models** or **DL Models** pages")

# --- Gate check ---
if not _features_ready():
    st.info("⚠️ Load data and build features on the **Preprocessing** page first.")
    st.stop()

if not _models_available():
    st.info("⚠️ Train at least one model on the **ML Models** or **DL Models** page first.")
    st.stop()

# --- Get data ---
features = st.session_state["features"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]
df_test = st.session_state.get("df_test")
test_texts = features.get("test_texts", df_test["processed_text"].tolist() if df_test is not None else [])

CLASS_NAMES = ["Negative", "Neutral", "Positive"]

# Create tabs
tab_comparison, tab_metrics, tab_errors, tab_shap, tab_lime = st.tabs([
    "📊 Model Comparison", "📈 Detailed Metrics", "🔍 Error Analysis", "🧠 SHAP", "💡 LIME"
])

# =====================================================================
# TAB 1: MODEL COMPARISON
# =====================================================================

with tab_comparison:
    st.subheader("Master Model Comparison")
    st.caption("Compare all trained models across representations and architectures.")
    
    ml_results = st.session_state.get("ml_results_classification", [])
    dl_results = st.session_state.get("dl_results", [])
    
    # Create comparison table
    comparison_df = create_comparison_table(ml_results, dl_results)
    
    if len(comparison_df) > 0:
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", len(comparison_df))
        
        with col2:
            best_acc = comparison_df["Accuracy"].max()
            st.metric("Best Accuracy", f"{best_acc:.4f}")
        
        with col3:
            best_f1 = comparison_df["Macro F1"].max()
            st.metric("Best Macro F1", f"{best_f1:.4f}")
        
        with col4:
            avg_time = comparison_df["Train Time (s)"].mean()
            st.metric("Avg Train Time", f"{avg_time:.1f}s")
        
        st.divider()
        
        # Full comparison table
        st.subheader("All Models")
        
        # Format display
        display_df = comparison_df.copy()
        display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x:.4f}")
        display_df["Macro F1"] = display_df["Macro F1"].apply(lambda x: f"{x:.4f}")
        display_df["Train Time (s)"] = display_df["Train Time (s)"].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(display_df, hide_index=True, width='stretch')
        
        # Top 5 models
        st.divider()
        st.subheader("🏆 Top 5 Models (by Macro F1)")
        
        top_models = get_best_models(comparison_df, top_n=5)
        
        for idx, row in top_models.iterrows():
            with st.expander(
                f"#{idx+1}: {row['Model']} ({row['Representation']}) - F1: {row['Macro F1']:.4f}",
                expanded=(idx == 0)
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{row['Accuracy']:.4f}")
                col2.metric("Macro F1", f"{row['Macro F1']:.4f}")
                col3.metric("Train Time", f"{row['Train Time (s)']:.1f}s")
                
                st.caption(f"**Type:** {row['Model Type']} | **Model:** {row['Model']} | **Features:** {row['Representation']}")
        
        # Visualization: Bar chart of top models
        st.divider()
        st.subheader("Performance Comparison")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Top 10 models by F1
        top_10 = comparison_df.head(10).copy()
        top_10["Model_Label"] = top_10["Model"] + "\n(" + top_10["Representation"] + ")"
        
        # F1 scores
        ax1.barh(range(len(top_10)), top_10["Macro F1"].values)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels(top_10["Model_Label"].values, fontsize=8)
        ax1.set_xlabel("Macro F1 Score")
        ax1.set_title("Top 10 Models by Macro F1")
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Accuracy vs Train Time scatter
        ax2.scatter(
            comparison_df["Train Time (s)"],
            comparison_df["Accuracy"],
            c=comparison_df["Macro F1"],
            cmap="viridis",
            s=100,
            alpha=0.6
        )
        ax2.set_xlabel("Train Time (seconds)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs Train Time")
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(
            vmin=comparison_df["Macro F1"].min(),
            vmax=comparison_df["Macro F1"].max()
        ))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label("Macro F1", rotation=270, labelpad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("No models trained yet. Train models on ML Models or DL Models pages.")

# =====================================================================
# TAB 2: DETAILED METRICS
# =====================================================================

with tab_metrics:
    st.subheader("Detailed Classification Metrics")
    st.caption("In-depth analysis of selected model performance.")
    
    if "eval_selected_model" not in st.session_state:
        st.info("Select a model from the sidebar.")
    else:
        model_type, result = st.session_state["eval_selected_model"]
        
        st.info(f"**Analyzing:** {result['model']} ({result['representation']})")
        
        # Get predictions for this model
        # We need to re-predict, so we'll need to access the fitted model
        fitted_model = None
        X_test = None
        
        # Try to get fitted model
        if model_type == "ml":
            pipelines = st.session_state.get("ml_fitted_pipelines_classification", {})
            model_key = f"{result['model']}_{result['representation']}"
            fitted_model = pipelines.get(model_key)
            
            # Get features for this representation
            if result['representation'] == "tfidf":
                _, X_test, _ = features["tfidf"]
            elif result['representation'] == "sbert":
                _, X_test = features["sbert"]
            elif result['representation'] == "bow":
                _, X_test, _ = features.get("bow", (None, None, None))
            elif result['representation'] == "w2v":
                _, X_test = features.get("w2v", (None, None))
            elif result['representation'] == "tfidf_stats":
                _, X_test = features.get("tfidf_stats", (None, None))
        
        elif model_type == "dl":
            # For DL models, we'd need to re-run inference
            # This is more complex, so we'll show stored results
            st.warning("Detailed metrics for DL models require re-inference. Showing stored results.")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{result['accuracy']:.4f}")
        
        with col2:
            st.metric("Macro F1", f"{result['macro_f1']:.4f}")
        
        with col3:
            st.metric("Train Time", f"{result['train_time']:.1f}s")
        
        st.divider()
        
        # Per-class metrics
        st.subheader("Per-Class Performance")
        
        if "per_class_f1" in result:
            per_class_data = []
            for class_idx, f1 in result["per_class_f1"].items():
                per_class_data.append({
                    "Class": CLASS_NAMES[class_idx],
                    "F1 Score": f1,
                })
            
            per_class_df = pd.DataFrame(per_class_data)
            st.dataframe(per_class_df, hide_index=True, width='stretch')
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(per_class_df["Class"], per_class_df["F1 Score"])
            ax.set_ylabel("F1 Score")
            ax.set_title("Per-Class F1 Scores")
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
        
        # If we have the fitted model, show more detailed metrics
        if fitted_model is not None and X_test is not None:
            st.divider()
            st.subheader("Classification Report")
            
            try:
                # Convert sparse to dense if needed
                if issparse(X_test):
                    X_test_dense = X_test.toarray()
                else:
                    X_test_dense = X_test
                
                # Get predictions
                y_pred = fitted_model.predict(X_test_dense)
                
                # Get probabilities if available
                try:
                    y_pred_proba = fitted_model.predict_proba(X_test_dense)
                except Exception:
                    y_pred_proba = None
                
                # Classification report
                report_df = get_classification_report_df(y_test, y_pred, CLASS_NAMES)
                st.dataframe(report_df, width='stretch')
                
                st.divider()
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_cm = plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, normalize=True)
                    st.pyplot(fig_cm)
                
                with col2:
                    fig_cm_raw = plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, normalize=False)
                    st.pyplot(fig_cm_raw)
                
                # ROC Curves
                if y_pred_proba is not None:
                    st.divider()
                    st.subheader("ROC Curves")
                    
                    fig_roc = plot_roc_curves(y_test, y_pred_proba, CLASS_NAMES)
                    st.pyplot(fig_roc)
                
            except Exception as e:
                st.error(f"Error generating detailed metrics: {e}")

# =====================================================================
# TAB 3: ERROR ANALYSIS
# =====================================================================

with tab_errors:
    st.subheader("Error Analysis")
    st.caption("Analyze misclassified samples to understand model weaknesses.")
    
    if "eval_selected_model" not in st.session_state:
        st.info("Select a model from the sidebar.")
    else:
        model_type, result = st.session_state["eval_selected_model"]
        
        st.info(f"**Analyzing:** {result['model']} ({result['representation']})")
        
        # Get fitted model and predictions
        fitted_model = None
        X_test = None
        
        if model_type == "ml":
            pipelines = st.session_state.get("ml_fitted_pipelines_classification", {})
            model_key = f"{result['model']}_{result['representation']}"
            fitted_model = pipelines.get(model_key)
            
            # Get features
            if result['representation'] == "tfidf":
                _, X_test, _ = features["tfidf"]
            elif result['representation'] == "sbert":
                _, X_test = features["sbert"]
            elif result['representation'] == "bow":
                _, X_test, _ = features.get("bow", (None, None, None))
            elif result['representation'] == "w2v":
                _, X_test = features.get("w2v", (None, None))
            elif result['representation'] == "tfidf_stats":
                _, X_test = features.get("tfidf_stats", (None, None))
        
        if fitted_model is not None and X_test is not None:
            try:
                # Convert sparse to dense if needed
                if issparse(X_test):
                    X_test_dense = X_test.toarray()
                else:
                    X_test_dense = X_test
                
                # Get predictions
                y_pred = fitted_model.predict(X_test_dense)
                
                # Get probabilities if available
                try:
                    y_pred_proba = fitted_model.predict_proba(X_test_dense)
                except Exception:
                    y_pred_proba = None
                
                # Error statistics
                st.subheader("Error Statistics by Class")
                
                error_stats = get_error_statistics(y_test, y_pred, CLASS_NAMES)
                st.dataframe(error_stats, hide_index=True, width='stretch')
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Error rate by class
                ax1.bar(error_stats["class_name"], error_stats["error_rate"])
                ax1.set_ylabel("Error Rate")
                ax1.set_title("Error Rate by Class")
                ax1.set_ylim([0, 1])
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Total samples and errors
                x = np.arange(len(error_stats))
                width = 0.35
                ax2.bar(x - width/2, error_stats["total"], width, label="Total", alpha=0.7)
                ax2.bar(x + width/2, error_stats["errors"], width, label="Errors", alpha=0.7)
                ax2.set_xticks(x)
                ax2.set_xticklabels(error_stats["class_name"])
                ax2.set_ylabel("Count")
                ax2.set_title("Samples vs Errors by Class")
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.divider()
                
                # Error samples
                st.subheader("Misclassified Samples")
                
                max_errors = st.slider("Number of error samples to display", 5, 100, 20, 5)
                
                error_df = analyze_errors(
                    y_test, y_pred, y_pred_proba, test_texts, CLASS_NAMES, max_samples=max_errors
                )
                
                if len(error_df) > 0:
                    st.write(f"**Total misclassifications:** {len(np.where(y_test != y_pred)[0]):,}")
                    st.write(f"**Showing:** {len(error_df)} samples (sorted by confidence)")
                    
                    # Display errors
                    for idx, row in error_df.iterrows():
                        with st.expander(
                            f"Error {idx+1}: Predicted {row.get('predicted_class', row['predicted_label'])} "
                            f"(True: {row.get('true_class', row['true_label'])}) "
                            f"- Confidence: {row.get('confidence', 0):.2%}"
                        ):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("True Label", row.get('true_class', row['true_label']))
                            
                            with col2:
                                st.metric("Predicted", row.get('predicted_class', row['predicted_label']))
                            
                            with col3:
                                if 'confidence' in row:
                                    st.metric("Confidence", f"{row['confidence']:.2%}")
                            
                            if 'text' in row:
                                st.text_area("Text", row['text'], height=100, key=f"error_text_{idx}")
                else:
                    st.success("🎉 No misclassifications found! Perfect accuracy.")
                
            except Exception as e:
                st.error(f"Error in error analysis: {e}")
        else:
            st.warning("Fitted model not available for detailed error analysis.")

# =====================================================================
# TAB 4: SHAP EXPLANATIONS
# =====================================================================

with tab_shap:
    st.subheader("SHAP Explanations")
    st.caption("Global feature importance using SHAP values.")
    
    if not SHAP_AVAILABLE:
        st.error("SHAP is not installed. Install with: `pip install shap`")
    else:
        if "eval_selected_model" not in st.session_state:
            st.info("Select a model from the sidebar.")
        else:
            model_type, result = st.session_state["eval_selected_model"]
            
            st.info(f"**Analyzing:** {result['model']} ({result['representation']})")
            
            if model_type == "ml":
                pipelines = st.session_state.get("ml_fitted_pipelines_classification", {})
                model_key = f"{result['model']}_{result['representation']}"
                fitted_model = pipelines.get(model_key)
                
                # Get features
                X_train = None
                X_test = None
                feature_names = None
                
                if result['representation'] == "tfidf":
                    vectorizer, X_train, X_test = features["tfidf"]
                    if issparse(X_train):
                        X_train = X_train.toarray()
                    if issparse(X_test):
                        X_test = X_test.toarray()
                    # Get feature names
                    try:
                        feature_names = vectorizer.get_feature_names_out()
                    except Exception:
                        feature_names = None
                
                elif result['representation'] in ["sbert", "w2v"]:
                    if result['representation'] == "sbert":
                        X_train, X_test = features["sbert"]
                    else:
                        X_train, X_test = features.get("w2v", (None, None))
                    
                    # For dense embeddings, use dimension indices as feature names
                    if X_train is not None:
                        feature_names = [f"dim_{i}" for i in range(X_train.shape[1])]
                
                if fitted_model is not None and X_train is not None and X_test is not None:
                    if st.button("Generate SHAP Explanations", type="primary"):
                        with st.spinner("Computing SHAP values... This may take a while."):
                            try:
                                explainer, fig = explain_with_shap(
                                    fitted_model,
                                    X_train[:500],  # Use subset for background
                                    X_test[:100],   # Explain subset
                                    feature_names=feature_names[:20] if feature_names is not None else None,
                                    max_display=20
                                )
                                
                                if fig is not None:
                                    st.pyplot(fig)
                                    st.success("✓ SHAP analysis complete")
                                    
                                    st.info(
                                        "**How to interpret:** Features are ranked by importance (average absolute SHAP value). "
                                        "Color indicates feature value (red = high, blue = low). "
                                        "Position on x-axis shows impact on prediction."
                                    )
                                else:
                                    st.error("Failed to generate SHAP explanations. Model might not be compatible with SHAP.")
                            
                            except Exception as e:
                                st.error(f"SHAP error: {e}")
                                st.info("SHAP works best with tree-based models (RandomForest, LightGBM). "
                                       "For other models, it may be slow or fail.")
                else:
                    st.warning("Model or features not available for SHAP analysis.")
            else:
                st.warning("SHAP analysis currently only available for ML models.")

# =====================================================================
# TAB 5: LIME EXPLANATIONS
# =====================================================================

with tab_lime:
    st.subheader("LIME Explanations")
    st.caption("Local interpretable model-agnostic explanations for individual predictions.")
    
    if not LIME_AVAILABLE:
        st.error("LIME is not installed. Install with: `pip install lime`")
    else:
        if "eval_selected_model" not in st.session_state:
            st.info("Select a model from the sidebar.")
        else:
            model_type, result = st.session_state["eval_selected_model"]
            
            st.info(f"**Analyzing:** {result['model']} ({result['representation']})")
            
            if model_type == "ml":
                pipelines = st.session_state.get("ml_fitted_pipelines_classification", {})
                model_key = f"{result['model']}_{result['representation']}"
                fitted_model = pipelines.get(model_key)
                
                if fitted_model is not None and len(test_texts) > 0:
                    num_samples = st.slider("Number of samples to explain", 1, 10, 3)
                    num_features = st.slider("Number of features to show", 5, 20, 10)
                    
                    if st.button("Generate LIME Explanations", type="primary"):
                        with st.spinner("Computing LIME explanations..."):
                            try:
                                # Create a wrapper that takes raw text
                                class TextClassifier:
                                    def __init__(self, pipeline, representation, features_dict):
                                        self.pipeline = pipeline
                                        self.representation = representation
                                        self.features_dict = features_dict
                                    
                                    def predict_proba(self, texts):
                                        # Transform texts using the same vectorizer
                                        if self.representation == "tfidf":
                                            vectorizer = self.features_dict["tfidf"][0]
                                            X = vectorizer.transform(texts)
                                            if issparse(X):
                                                X = X.toarray()
                                        else:
                                            # For other representations, this is more complex
                                            # For now, we'll skip
                                            raise NotImplementedError(
                                                f"LIME not yet supported for {self.representation}"
                                            )
                                        
                                        return self.pipeline.predict_proba(X)
                                
                                text_clf = TextClassifier(fitted_model, result['representation'], features)
                                
                                explanations = explain_with_lime(
                                    text_clf,
                                    test_texts[:10],  # Sample from first 10
                                    CLASS_NAMES,
                                    num_samples=num_samples,
                                    num_features=num_features
                                )
                                
                                if explanations:
                                    for i, exp_data in enumerate(explanations):
                                        with st.expander(
                                            f"Sample {i+1}: Predicted {exp_data['predicted_class']} "
                                            f"({exp_data['confidence']:.2%} confidence)",
                                            expanded=(i == 0)
                                        ):
                                            st.text_area("Text", exp_data['text'], height=100, key=f"lime_text_{i}")
                                            
                                            st.subheader("Top Contributing Features")
                                            
                                            # Create DataFrame of features
                                            features_df = pd.DataFrame(
                                                exp_data['top_features'],
                                                columns=["Feature", "Weight"]
                                            )
                                            
                                            # Sort by absolute weight
                                            features_df = features_df.reindex(
                                                features_df['Weight'].abs().sort_values(ascending=False).index
                                            )
                                            
                                            st.dataframe(features_df, hide_index=True, width='stretch')
                                            
                                            # Visualization
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            
                                            colors = ['green' if w > 0 else 'red' for w in features_df['Weight']]
                                            ax.barh(range(len(features_df)), features_df['Weight'], color=colors, alpha=0.6)
                                            ax.set_yticks(range(len(features_df)))
                                            ax.set_yticklabels(features_df['Feature'])
                                            ax.set_xlabel('Weight (Green = supports prediction, Red = contradicts)')
                                            ax.set_title(f'LIME Explanation for "{exp_data["predicted_class"]}" prediction')
                                            ax.grid(True, alpha=0.3, axis='x')
                                            ax.invert_yaxis()
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                    
                                    st.success("✓ LIME analysis complete")
                                    
                                    st.info(
                                        "**How to interpret:** Green bars show words supporting the prediction. "
                                        "Red bars show words contradicting it. "
                                        "The magnitude shows the strength of the effect."
                                    )
                                else:
                                    st.error("Failed to generate LIME explanations.")
                            
                            except NotImplementedError as e:
                                st.warning(str(e))
                                st.info("LIME text explanations currently only supported for TF-IDF representation.")
                            
                            except Exception as e:
                                st.error(f"LIME error: {e}")
                else:
                    st.warning("Model or text data not available for LIME analysis.")
            else:
                st.warning("LIME analysis currently only available for ML models.")

# =====================================================================
# FOOTER
# =====================================================================

st.divider()
st.caption(
    "💡 **Tip:** Use Model Comparison to identify best performers, "
    "Detailed Metrics for per-class analysis, "
    "Error Analysis to find weaknesses, "
    "and SHAP/LIME to understand why models make specific predictions."
)
