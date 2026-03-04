"""
Phase 8, 9, 10 — Sklearn pipelines, training, tuning.
Uses ai_toolkit.ml (trainer) when available; utils.ml_trainer fallbacks.
Two tabs: Classification and Regression. One-by-one actions; caching via session_state.
"""
from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.config import CONFIG
from utils.ml_trainer import (
    build_classification_pipeline,
    build_regression_pipeline,
    build_results_table,
    get_classification_estimator,
    get_regression_estimator,
    get_param_grid,
    get_xy_for_representation,
    representation_from_saved_path,
    saved_model_path,
    train_and_evaluate,
    tune_pipeline,
)

st.set_page_config(page_title="ML Models — SentimentScope", layout="wide")
st.title("Classical ML Models")
st.caption("Phase 8–10: Pipelines, train & compare, tune & save. Two tabs: Classification | Regression.")

# --- Session state init ---
for key in (
    "ml_baseline_classification",
    "ml_baseline_regression",
    "ml_results_classification",
    "ml_results_regression",
    "ml_fitted_pipelines_classification",
    "ml_fitted_pipelines_regression",
):
    if key not in st.session_state:
        if "results" in key:
            st.session_state[key] = []
        elif "pipelines" in key:
            st.session_state[key] = {}
        else:
            st.session_state[key] = {}

def _features_ready():
    return (
        st.session_state.get("features_done") is True
        and "features" in st.session_state
        and "y_train" in st.session_state
        and "y_test" in st.session_state
    )

def _get_features():
    return st.session_state["features"]

def _get_y_train_test(task: str):
    if task == "classification":
        return st.session_state["y_train"], st.session_state["y_test"]
    # Regression: rating from df_train, df_test
    df_train = st.session_state.get("df_train")
    df_test = st.session_state.get("df_test")
    if df_train is None or df_test is None or "rating" not in df_train.columns:
        return None, None
    return df_train["rating"].values.astype(np.float64), df_test["rating"].values.astype(np.float64)

# --- Sidebar ---
with st.sidebar:
    st.header("Config")
    st.write("CV folds:", CONFIG.get("cv_folds", 5), "| n_jobs:", CONFIG.get("n_jobs", -1))
    if _features_ready():
        st.success("Features ready")
        feats = _get_features()
        st.caption(f"Train: {feats['tfidf'][1].shape[0]:,} | Test: {feats['tfidf'][2].shape[0]:,}")
    else:
        st.warning("Run **Preprocessing** and build features first.")

REPRESENTATIONS_CLF = ["bow", "tfidf", "w2v", "sbert", "tfidf_stats"]
REPRESENTATIONS_REG = ["tfidf", "w2v", "sbert", "tfidf_stats"]
MODELS_CLF = ["LogisticRegression", "ComplementNB", "RandomForest", "LightGBM"]
MODELS_REG = ["LinearRegression", "Ridge", "RandomForest", "LightGBM"]

# --- Tabs ---
tab_clf, tab_reg = st.tabs(["Classification", "Regression"])

# ========== Classification tab ==========
with tab_clf:
    if not _features_ready():
        st.info("Load data and build features on the **Preprocessing** page first.")
    else:
        features = _get_features()
        y_train, y_test = _get_y_train_test("classification")
        X_tfidf_train, X_tfidf_test, vec_tfidf = get_xy_for_representation(features, "tfidf")

        st.subheader("Baseline (reference)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Compute majority-class baseline", key="bl_maj"):
                from sklearn.dummy import DummyClassifier
                dummy = DummyClassifier(strategy="most_frequent", random_state=CONFIG.get("random_seed"))
                dummy.fit(np.zeros((len(y_train), 1)), y_train)  # no features needed
                y_pred = dummy.predict(np.zeros((len(y_test), 1)))
                acc = float(np.mean(y_pred == y_test))
                macro_f1 = float(
                    __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    )
                )
                st.session_state["ml_baseline_classification"]["majority"] = {
                    "accuracy": acc, "macro_f1": macro_f1,
                    "per_class_f1": dict(zip(
                        sorted(np.unique(y_test)),
                        __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                            y_test, y_pred, average=None, zero_division=0
                        ).tolist(),
                    )),
                }
                st.rerun()
        with col2:
            if st.button("Compute TF-IDF + LR baseline", key="bl_tfidf_lr"):
                # X_tfidf_train is already vectorized — use estimator-only pipeline (None vec)
                pipe = build_classification_pipeline(None, get_classification_estimator("LogisticRegression"))
                with st.spinner("Fitting baseline…"):
                    res = train_and_evaluate(pipe, X_tfidf_train, X_tfidf_test, y_train, y_test, task="classification")
                st.session_state["ml_baseline_classification"]["tfidf_lr"] = {
                    "accuracy": res.get("accuracy", res.get("acc", 0.0)),
                    "macro_f1": res.get("macro_f1", res.get("f1_macro", res.get("f1", 0.0))),
                    "per_class_f1": res.get("per_class_f1", {}),
                }
                st.rerun()

        bl = st.session_state.get("ml_baseline_classification") or {}
        if bl:
            st.caption("Baseline metrics (every model is compared to these)")
            rows = []
            for name, m in bl.items():
                rows.append({"baseline": name, "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1")})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.subheader("Train a model")
        model_clf = st.selectbox("Model", MODELS_CLF, key="sel_model_clf")
        rep_clf = st.selectbox("Representation", REPRESENTATIONS_CLF, key="sel_rep_clf")
        if model_clf == "ComplementNB" and rep_clf in ("w2v", "sbert", "tfidf_stats"):
            st.warning(
                "ComplementNB requires non-negative input. "
                "Choose **bow** or **tfidf** representation when using ComplementNB."
            )
        if st.button("Train selected (classification)", key="train_clf"):
            try:
                if model_clf == "ComplementNB" and rep_clf in ("w2v", "sbert", "tfidf_stats"):
                    st.error("ComplementNB cannot be used with w2v / sbert / tfidf_stats (negative values). Select bow or tfidf.")
                    st.stop()
                X_train, X_test, vec = get_xy_for_representation(features, rep_clf)
                clf = get_classification_estimator(model_clf)
                # X_train is already vectorized — pass None so pipeline is estimator-only
                pipe = build_classification_pipeline(None, clf)
                with st.spinner("Training…"):
                    res = train_and_evaluate(pipe, X_train, X_test, y_train, y_test, task="classification")
                pipe_fitted = res.pop("_fitted_pipeline", None)
                row = {
                    "model": model_clf,
                    "representation": rep_clf,
                    "accuracy": res["accuracy"],
                    "macro_f1": res["macro_f1"],
                    "per_class_f1": res.get("per_class_f1", {}),
                }
                st.session_state["ml_results_classification"].append(row)
                if pipe_fitted is not None:
                    st.session_state["ml_fitted_pipelines_classification"][(model_clf, rep_clf)] = pipe_fitted
                st.success(f"Done. Accuracy: {res['accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f}")
                st.rerun()
            except Exception as e:
                st.error(str(e))
                import traceback
                st.code(traceback.format_exc())

        st.subheader("Results table")
        if st.session_state["ml_results_classification"]:
            df_res = build_results_table(st.session_state["ml_results_classification"])
            st.dataframe(df_res, width="stretch", hide_index=True)

        st.subheader("Tune & save")
        if st.session_state["ml_results_classification"]:
            opts = [f"{r['model']} | {r['representation']}" for r in st.session_state["ml_results_classification"]]
            sel_idx = st.selectbox("Select run to tune", range(len(opts)), format_func=lambda i: opts[i], key="tune_sel_clf")
            if st.button("Tune selected (classification)", key="tune_clf"):
                r = st.session_state["ml_results_classification"][sel_idx]
                model_clf, rep_clf = r["model"], r["representation"]
                X_train, X_test, vec = get_xy_for_representation(features, rep_clf)
                pipe = st.session_state["ml_fitted_pipelines_classification"].get((model_clf, rep_clf))
                if pipe is None:
                    # X_train is already vectorized — estimator-only pipeline
                    pipe = build_classification_pipeline(None, get_classification_estimator(model_clf))
                    pipe.fit(X_train, y_train)
                param_grid = get_param_grid(model_clf, task="classification")
                if not param_grid:
                    st.warning("No param grid for this model.")
                else:
                    with st.spinner("Tuning (RandomizedSearchCV)…"):
                        best_pipe, best_params, best_score, elapsed = tune_pipeline(
                            pipe, param_grid, X_train, y_train, task="classification"
                        )
                    st.session_state["ml_fitted_pipelines_classification"][(model_clf, rep_clf)] = best_pipe
                    st.write("Best params:", best_params)
                    st.write("Best CV score (macro F1):", round(best_score, 4), "| Time (s):", round(elapsed, 1))
                    st.session_state["_tune_result_clf"] = (model_clf, rep_clf, best_pipe)
                    st.rerun()

            if st.button("Save tuned model (classification)", key="save_clf"):
                tune_result = st.session_state.get("_tune_result_clf")
                if tune_result:
                    model_clf, rep_clf, best_pipe = tune_result
                    path = saved_model_path("classification", model_clf, rep_clf, tuned=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(best_pipe, path)
                    st.success(f"Saved to `{path}`")
                else:
                    st.info("Tune a model first, then save.")

        st.subheader("Load tuned model")
        saved_dir = Path(CONFIG["saved_models_dir"]) / "classification"
        if saved_dir.exists():
            saved_files = list(saved_dir.glob("*.joblib"))
            if saved_files:
                f = st.selectbox("File", saved_files, format_func=lambda p: p.name, key="load_clf_file")
                if st.button("Load and evaluate (classification)", key="load_clf"):
                    pipe = joblib.load(f)
                    rep_load = representation_from_saved_path(Path(f))
                    X_train, X_test, vec = get_xy_for_representation(features, rep_load)
                    # Pipelines with vectorizer expect raw text; estimator-only expect X
                    # All saved pipelines are estimator-only (features pre-computed on Page 3)
                    y_pred = pipe.predict(X_test)
                    acc = float(np.mean(y_pred == y_test))
                    macro_f1 = float(
                        __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                            y_test, y_pred, average="macro", zero_division=0
                        )
                    )
                    st.metric("Accuracy", round(acc, 4))
                    st.metric("Macro F1", round(macro_f1, 4))
            else:
                st.caption("No saved models in classification/")
        else:
            st.caption("No saved_models/classification/ folder yet.")

# ========== Regression tab ==========
with tab_reg:
    if not _features_ready():
        st.info("Load data and build features on the **Preprocessing** page first.")
    else:
        y_rating_train, y_rating_test = _get_y_train_test("regression")
        if y_rating_train is None:
            st.warning("Regression needs `rating` in train/test. Use Data page data that includes rating.")
        else:
            features = _get_features()

            st.subheader("Baseline (reference)")
            if st.button("Compute mean-rating baseline", key="bl_mean"):
                mean_pred = np.full_like(y_rating_test, float(np.mean(y_rating_train)))
                rmse = float(np.sqrt(np.mean((y_rating_test - mean_pred) ** 2)))
                mae = float(np.mean(np.abs(y_rating_test - mean_pred)))
                st.session_state["ml_baseline_regression"]["mean"] = {"rmse": rmse, "mae": mae}
                st.rerun()
            bl_reg = st.session_state.get("ml_baseline_regression") or {}
            if bl_reg:
                st.dataframe(pd.DataFrame([bl_reg.get("mean", {})]).T.rename(columns={0: "value"}), hide_index=True)

            st.subheader("Train a model")
            model_reg = st.selectbox("Model", MODELS_REG, key="sel_model_reg")
            rep_reg = st.selectbox("Representation", REPRESENTATIONS_REG, key="sel_rep_reg")
            if st.button("Train selected (regression)", key="train_reg"):
                try:
                    X_train, X_test, vec = get_xy_for_representation(features, rep_reg)
                    reg = get_regression_estimator(model_reg)
                    # X_train is already vectorized — pass None so pipeline is estimator-only
                    pipe = build_regression_pipeline(None, reg)
                    with st.spinner("Training…"):
                        res = train_and_evaluate(
                            pipe, X_train, X_test, y_rating_train, y_rating_test, task="regression"
                        )
                    pipe_fitted = res.pop("_fitted_pipeline", None)
                    row = {
                        "model": model_reg,
                        "representation": rep_reg,
                        "rmse": res["rmse"],
                        "mae": res["mae"],
                    }
                    st.session_state["ml_results_regression"].append(row)
                    if pipe_fitted is not None:
                        st.session_state["ml_fitted_pipelines_regression"][(model_reg, rep_reg)] = pipe_fitted
                    st.success(f"Done. RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                    import traceback
                    st.code(traceback.format_exc())

            st.subheader("Results table")
            if st.session_state["ml_results_regression"]:
                df_res = build_results_table(st.session_state["ml_results_regression"])
                st.dataframe(df_res, width="stretch", hide_index=True)

            st.subheader("Tune & save")
            if st.session_state["ml_results_regression"]:
                opts = [f"{r['model']} | {r['representation']}" for r in st.session_state["ml_results_regression"]]
                sel_idx = st.selectbox("Select run to tune", range(len(opts)), format_func=lambda i: opts[i], key="tune_sel_reg")
                if st.button("Tune selected (regression)", key="tune_reg"):
                    r = st.session_state["ml_results_regression"][sel_idx]
                    model_reg, rep_reg = r["model"], r["representation"]
                    X_train, X_test, vec = get_xy_for_representation(features, rep_reg)
                    pipe = st.session_state["ml_fitted_pipelines_regression"].get((model_reg, rep_reg))
                    if pipe is None:
                        # X_train is already vectorized — estimator-only pipeline
                        pipe = build_regression_pipeline(None, get_regression_estimator(model_reg))
                        pipe.fit(X_train, y_rating_train)
                    param_grid = get_param_grid(model_reg, task="regression")
                    if not param_grid:
                        st.warning("No param grid for this model.")
                    else:
                        with st.spinner("Tuning (RandomizedSearchCV)…"):
                            best_pipe, best_params, best_score, elapsed = tune_pipeline(
                                pipe, param_grid, X_train, y_rating_train, task="regression"
                            )
                        st.session_state["ml_fitted_pipelines_regression"][(model_reg, rep_reg)] = best_pipe
                        st.write("Best params:", best_params)
                        st.write("Best CV RMSE:", round(best_score, 4), "| Time (s):", round(elapsed, 1))
                        st.session_state["_tune_result_reg"] = (model_reg, rep_reg, best_pipe)
                        st.rerun()

                if st.button("Save tuned model (regression)", key="save_reg"):
                    tune_result = st.session_state.get("_tune_result_reg")
                    if tune_result:
                        model_reg, rep_reg, best_pipe = tune_result
                        path = saved_model_path("regression", model_reg, rep_reg, tuned=True)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(best_pipe, path)
                        st.success(f"Saved to `{path}`")
                    else:
                        st.info("Tune a model first, then save.")

            st.subheader("Load tuned model")
            saved_dir_reg = Path(CONFIG["saved_models_dir"]) / "regression"
            if saved_dir_reg.exists():
                saved_files_reg = list(saved_dir_reg.glob("*.joblib"))
                if saved_files_reg:
                    f = st.selectbox("File", saved_files_reg, format_func=lambda p: p.name, key="load_reg_file")
                    if st.button("Load and evaluate (regression)", key="load_reg"):
                        pipe = joblib.load(f)
                        rep_load = representation_from_saved_path(Path(f))
                        X_train, X_test, vec = get_xy_for_representation(features, rep_load)
                        # All saved pipelines are estimator-only (features pre-computed on Page 3)
                        y_pred = pipe.predict(X_test)
                        rmse = float(np.sqrt(np.mean((y_rating_test - y_pred) ** 2)))
                        mae = float(np.mean(np.abs(y_rating_test - y_pred)))
                        st.metric("RMSE", round(rmse, 4))
                        st.metric("MAE", round(mae, 4))
                else:
                    st.caption("No saved models in regression/")
            else:
                st.caption("No saved_models/regression/ folder yet.")
