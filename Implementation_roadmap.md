# SentimentScope — Complete Implementation Roadmap

**Purpose:** This project is designed to cover the full stack of AI technologies for textual data: **data analysis**, **classical ML**, **deep learning**, **attention mechanisms**, **transformers**, and every related technique for **text analysis** — so that you work end-to-end with real text data from ingestion and EDA through classical models, sequence models (e.g. LSTM), attention, and transformer-based models (e.g. BERT).

---

## How to use this file (instruction prompt for agents)

**Audience:** An AI agent or developer implementing the SentimentScope project in a **separate folder/repository** (not inside the ai_toolkit repo).

**Mandatory dependency:** You **must** use the **aaryan-ai-toolkit** library for NLP, EDA, ML, and DL logic. Do not reimplement those functions; import and call them from the toolkit.

- **Install:** `pip install aaryan-ai-toolkit[full]` (or `aaryan-ai-toolkit[viz]` for EDA-only). Use the `[full]` extra for Streamlit + DL + SHAP/LIME.
- **Import:** Use the `ai_toolkit` package, e.g. `from ai_toolkit.nlp import ...`, `from ai_toolkit.eda import ...`, `from ai_toolkit.ml import ...`, `from ai_toolkit.dl import ...`.
- **Your repo:** You are building the SentimentScope app (Streamlit pages, config, data, saved models) in your own project folder. That project depends on `aaryan-ai-toolkit`; the toolkit code lives in its own package, not in your repo.
- **When a phase lists "Use aaryan-ai-toolkit: …"** — implement the workflow (data loading, pipelines, UI) in your project, but call the listed `ai_toolkit` modules/functions for the actual logic (preprocessing, plots, training, evaluation, etc.).

Treat this document as the single source of instructions: follow phases in order, use aaryan-ai-toolkit where indicated, and structure your project as described below.

---

## Pre-Phase: Environment & Resource Setup

Before writing a single line of project code, get this done once.

### Local Setup

- **Python:** 3.10+ recommended (for PyTorch, Transformers, Streamlit).
- **GPU (optional):** CUDA-capable GPU speeds up LSTM and DistilBERT; CPU is fine for classical ML and smaller samples.
- **Virtual environment:** Create and use a venv (or conda) in this project folder so dependencies stay isolated.

### Dependencies

- **Toolkit (required):** `pip install aaryan-ai-toolkit[full]` — provides NLP, EDA, ML, and DL functions. Your project uses these; do not reimplement them.
- **App & extras:** In your project's `requirements.txt` also include: `streamlit`, `pandas`, `numpy`, and any extras (e.g. `datasets` for HuggingFace). The `[full]` extra of aaryan-ai-toolkit already brings in `scikit-learn`, `spacy`, `sentence-transformers`, `gensim`, `lightgbm`, `shap`, `lime`, `transformers`, `torch`, `captum`, `wordcloud`, `plotly`, `beautifulsoup4`, `contractions`.
- **SpaCy model:** Run once: `python -m spacy download en_core_web_sm`.

### Project Paths (Local)

- **Data:** e.g. `data/raw/`, `data/processed/` under project root.
- **Outputs:** `outputs/` for figures, reports, exports.
- **Models:** `saved_models/` or `checkpoints/` for fitted pipelines, PyTorch state dicts, and HuggingFace checkpoints. Commit only configs and code — not large binaries (use `.gitignore`).

### Dataset Download

Amazon Reviews 2023 (McAuley Lab) — direct URL download, no Kaggle login needed. Pick "Software" or "Digital Music" category. Target ~100k–200k reviews for a good balance of training signal and local runtimes; smaller samples (e.g. 50k) are fine for faster iteration.

### Repo Structure (SentimentScope project — your folder)

This is the layout for **your** SentimentScope project (the app repo). NLP/EDA/ML/DL logic comes from **aaryan-ai-toolkit**; your repo contains the app, config, data, and outputs.

```
sentimentscope/              # or your project root
├── app.py                    # Streamlit entry point (multi-page app)
├── pages/                    # Streamlit pages (one per major workflow)
│   ├── 1_Data.py             # Load, inspect, clean — uses ai_toolkit
│   ├── 2_EDA.py              # Explore & visualize — uses ai_toolkit.eda
│   ├── 3_Preprocessing.py    # NLP pipeline, features — uses ai_toolkit.nlp
│   ├── 4_ML_Models.py        # Train, compare, tune — uses ai_toolkit.ml
│   ├── 5_DL_Models.py        # ANN, LSTM, Attention, BERT — uses ai_toolkit.dl
│   └── 6_Evaluation.py       # Metrics, SHAP/LIME — uses ai_toolkit.ml
├── utils/
│   └── config.py             # Paths, hyperparams (your config only)
├── data/                     # .gitignore raw/processed or use DVC
│   ├── raw/
│   └── processed/
├── saved_models/             # .gitignore or track only small configs
├── outputs/                  # plots, tables, exports
├── requirements.txt          # streamlit, aaryan-ai-toolkit[full], ...
└── Implementation_roadmap.md # this file (instruction prompt)
```

**Do not** duplicate `nlp/`, `eda/`, `ml/`, `dl/` from the toolkit in your repo — import from `ai_toolkit` instead.

### Streamlit UI — Design Principles

- **When to build:** Build the app shell and page stubs early (Pre-Phase or right after Phase 2). As each phase is implemented, wire one or two key outputs into the app. Avoid “all backend first, then all UI” — better UX and earlier catch of API issues.
- **Single entry:** `streamlit run app.py` from project root.
- **Pages:** Each phase’s “key outputs” surface in the corresponding page (data table, plots, model comparison table, error samples, SHAP/LIME widgets).
- **State:** Use `st.session_state` for loaded data, fitted models, and selected config so users don’t re-run heavy steps unnecessarily. Optionally cache with `@st.cache_data` / `@st.cache_resource` for data load and model load.
- **Progressive disclosure:** Sidebar for config (paths, sample size, model choice); main area for results. Buttons like “Run EDA”, “Train selected model”, “Run SHAP” call **aaryan-ai-toolkit** functions and display outputs (tables, `st.pyplot`, `st.plotly_chart`, etc.).
- **No business logic in UI:** All NLP/EDA/ML/DL logic comes from **aaryan-ai-toolkit**. Your app only imports from `ai_toolkit`, calls those functions, and displays results.

### Train/val/test split (reproducibility)

Define a **single seed-based split** in config and reuse it everywhere (EDA, ML, DL). No ad-hoc splits per phase. Document the split (seed, test_size, val_size) in config and surface it on the Streamlit **Data** page so users see exactly which data is used for train/val/test.

### Master Config

Live in `utils/config.py` (or a single dict in code). Same keys as below; paths point to local dirs.

```python
CONFIG = {
    # Data
    "data_url": "...",
    "data_path": "data/raw/",
    "processed_path": "data/processed/",
    "sample_size": 150000,
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,

    # Text
    "max_features": 50000,
    "ngram_range": (1, 2),
    "max_seq_len": 256,
    "min_word_freq": 2,

    # Classical ML
    "cv_folds": 5,
    "n_jobs": -1,

    # Deep Learning
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-3,
    "patience": 3,

    # BERT
    "bert_model": "distilbert-base-uncased",
    "bert_epochs": 3,
    "bert_lr": 2e-5,
    "bert_batch_size": 32,

    # Sentence Transformers
    "sbert_model": "all-MiniLM-L6-v2",

    # Paths (local)
    "outputs_dir": "outputs/",
    "saved_models_dir": "saved_models/",
}
```

---

## Phase 1 — Data Gathering & Initial Inspection

**Goal:** Programmatically download, load, and understand the raw data structure.

**What you build:**

- `download_dataset(url, save_path)` — handles download + caching
- `load_and_sample(path, n, seed)` — loads JSON/CSV, samples consistently
- `initial_inspection(df)` — shape, dtypes, null counts, sample rows

**Key outputs:** raw dataframe, null report, column understanding.

**Streamlit:** Page `1_Data.py` — data path/URL, sample size, “Load” button; display shape, dtypes, null counts, sample rows.

**Use aaryan-ai-toolkit:** `ai_toolkit.utils`, `ai_toolkit.eda` (inspection/visualizer). Implement download/load/sample in your repo; use the toolkit for inspection and viz.

---

## Phase 2 — Data Cleaning

**Goal:** Transform raw messy text into clean, processable text.

**What you build:**

- `remove_html(text)` — BeautifulSoup
- `expand_contractions(text)` — contractions library
- `remove_noise(text)` — punctuation, special chars, numbers, URLs
- `normalize_text(text)` — lowercase, strip whitespace
- `clean_pipeline(text)` — composes all above in order
- `handle_nulls(df)` — drop / fill strategy with justification
- `create_sentiment_label(rating)` — maps 1-2→negative, 3→neutral, 4-5→positive

**Key outputs:** `df_clean` with `clean_text` column and `sentiment` label column.

**Use aaryan-ai-toolkit:** `ai_toolkit.nlp` (preprocessor). Call toolkit functions for cleaning, normalizing, and pipeline; do not reimplement.

---

## Phase 3 — EDA

**Goal:** Understand the data distribution before modeling.

**What you build:**

- Class distribution analysis (sentiment + raw rating)
- Text length distribution (char level + word level)
- Null / duplicate analysis
- Review length vs rating correlation
- Top unigrams + bigrams per sentiment class
- Vocabulary size analysis

**Key outputs:** 8–10 findings documented in markdown, each with a visualization.

**Use aaryan-ai-toolkit:** `ai_toolkit.eda` (visualizer). Use toolkit functions for all EDA plots and summaries.

---

## Phase 4 — Visualization

**Goal:** Produce all visual outputs for EDA findings and later model results.

**What you build:**

- `plot_class_distribution(df)` — Seaborn countplot
- `plot_text_length_distribution(df)` — histogram + KDE
- `plot_ngram_frequency(texts, n, top_k)` — Plotly bar chart
- `plot_wordcloud(texts, label)` — WordCloud per class
- `plot_confusion_matrix(y_true, y_pred)` — Seaborn heatmap
- `plot_roc_curves(models, X_test, y_test)` — multi-model ROC
- `plot_model_comparison(results_dict)` — Plotly grouped bar

**Note:** Phases 3 and 4 run together — every EDA finding is immediately visualized. Each finding is surfaced in the Streamlit **EDA** page.

**Streamlit:** Page `2_EDA.py` — sidebar filters (e.g. sample, column), main area: summary stats + interactive Plotly/Seaborn figures via `eda.visualizer`.

**Use aaryan-ai-toolkit:** `ai_toolkit.eda` — all visualization functions (class distribution, length, n-grams, wordcloud, confusion matrix, ROC, model comparison).

---

## Phase 5 — Text Preprocessing (NLP Pipeline)

**Goal:** Build a reusable, configurable NLP preprocessing pipeline.

**What you build:**

- SpaCy pipeline: tokenization → POS tagging → lemmatization → stopword removal → NER extraction
- `lemmatize_text(text, nlp)` — SpaCy-based
- `stem_text(text, stemmer)` — NLTK PorterStemmer
- `compare_stem_vs_lemma(sample_texts)` — side-by-side output, documents the difference
- `preprocess_for_ml(texts)` — final cleaned token string for sklearn
- `preprocess_for_dl(texts, vocab)` — integer-encoded sequences for PyTorch

**Key outputs:** `df['processed_text']` column, vocabulary object, token stats.

**Use aaryan-ai-toolkit:** `ai_toolkit.nlp` (preprocessor). Use for lemmatization, stemming, and ML/DL preprocessing pipelines.

---

## Phase 6 — Feature Engineering (Text Representations)

**Goal:** Build all five representations and understand what each captures.

**What you build:**

**BoW**

- `build_bow(train_texts, test_texts, max_features)` — CountVectorizer, return sparse matrices

**TF-IDF**

- `build_tfidf(train_texts, test_texts, ngram_range, max_features)` — TfidfVectorizer, your primary classical features

**Word2Vec**

- `train_word2vec(tokenized_texts, config)` — Gensim, train on corpus
- `get_tfidf_weighted_w2v(texts, w2v_model, tfidf_vectorizer)` — weighted average, not plain average

**Sentence Transformers**

- `get_sbert_embeddings(texts, model_name)` — batch encode, return numpy array
- Cache embeddings to disk (e.g. `data/processed/sbert_embeddings.npy`) — this call is slow, avoid recomputing every run

**Statistical Text Features** (supplement to all above)

- `extract_text_stats(df)` — char count, word count, avg word length, punctuation count, capital ratio

**Key outputs:** 5 feature matrices ready for model input, text stats dataframe.

**Use aaryan-ai-toolkit:** `ai_toolkit.nlp` (embeddings). Use for BoW, TF-IDF, Word2Vec, SBERT, and text stats; do not reimplement.

---

## Phase 7 — Scaling & Encoding

**Goal:** Prepare all non-text features for ML models.

**Ownership:** Scaling and encoding are owned by `ai_toolkit.ml` (trainer/utils). In your app, a small `utils/features.py` may wrap or call these and pipelines use them. The Streamlit **Preprocessing** or **ML** page should show “Scaled + encoded” as an explicit step so the workflow is clear.

**What you build:**

- `scale_features(train, test, method='standard')` — StandardScaler on text stats features
- `encode_labels(series, method='label')` — sentiment classes to integers
- Document why you don't scale TF-IDF/BoW (already normalized) but do scale Word2Vec and SBERT embeddings when combining with text stats

**Key outputs:** scaled numerical features, encoded label arrays.

**Use aaryan-ai-toolkit:** `ai_toolkit.ml` (trainer/utils). Use for scaling and encoding; keep pipeline construction and splits in your app or config.

---

## Phase 8 — Sklearn Pipelines

**Goal:** Wrap everything into proper sklearn Pipelines so preprocessing + modeling is one atomic unit.

**Baseline and metrics contract:**

- **Baseline:** Add an explicit baseline in Phase 8 (e.g. majority-class predictor or simple TF-IDF + LogisticRegression). Every model in the results table is compared against this baseline.
- **Fixed metric sets:** Use the same metrics everywhere so the Phase 14 master comparison is meaningful.
  - **Classification:** accuracy, macro F1, and at least one per-class metric (e.g. per-class F1 or recall).
  - **Regression:** RMSE, MAE.

**What you build:**

- **Baseline:** e.g. majority class or TF-IDF → LogisticRegression (fixed as reference).
- **Pipeline 1:** TF-IDF → LogisticRegression
- **Pipeline 2:** TF-IDF → ComplementNB
- **Pipeline 3:** TF-IDF → LightGBM
- **Pipeline 4:** BoW → LogisticRegression (comparison)
- **ColumnTransformer:** TF-IDF on text + StandardScaler on stats → LightGBM
- `build_classification_pipeline(vectorizer, classifier)` — generic builder
- `build_regression_pipeline(vectorizer, regressor)` — same pattern for regression
- `run_gridsearch(pipeline, param_grid, X_train, y_train, cv)` — wrapper with timing

**Key outputs:** fitted pipeline objects, best params per model, baseline score and comparison.

**Use aaryan-ai-toolkit:** `ai_toolkit.ml` (trainer). Use for pipeline building, GridSearch, and training.

---

## Phase 9 — Classical Model Training & Comparison

**Goal:** Train all classifiers and regressors across all representations, build comparison table.

**Classification models × representations matrix:**

- LR, ComplementNB, Random Forest, LightGBM  
- Across: BoW, TF-IDF, Word2Vec, SBERT

**Regression models on TF-IDF + Word2Vec + SBERT:**

- Linear Regression, Ridge, Random Forest Regressor, LightGBM Regressor

**What you build:**

- `train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, task)` — returns metrics dict
- `build_results_table(all_results)` — pandas DataFrame, sortable by F1/RMSE

**Key outputs:** full model × representation comparison table.

**Use aaryan-ai-toolkit:** `ai_toolkit.ml` (trainer, evaluator). Use for train/evaluate and results table construction.

---

## Phase 10 — Hyperparameter Tuning

**Goal:** Tune your top 2 performing pipelines, not all of them.

**Strategy:** After Phase 9, identify best performing pipeline per task. Run GridSearchCV (or RandomizedSearchCV) on those only — tuning everything wastes time and adds little learning value.

**What you build:**

- `get_param_grid(model_name)` — returns predefined param grid per model type
- `tune_pipeline(pipeline, param_grid, X_train, y_train)` — RandomizedSearchCV wrapper with timing + best score logging

**Key outputs:** tuned best models saved to `saved_models/`, before/after comparison. Expose “Load tuned model” and comparison table in Streamlit **ML** page.

---

## Phase 11 — Evaluation & Error Analysis

**Goal:** Go beyond accuracy numbers — understand where and why models fail.

**What you build:**

- `full_classification_report(model, X_test, y_test)` — precision, recall, F1, support (using the fixed metric set from Phase 8)
- `plot_confusion_matrix(...)` — normalized heatmap
- `plot_roc_auc(...)` — one-vs-rest multiclass
- `error_analysis(model, X_test, y_test, original_texts)` — samples misclassified examples with original text, predicted label, true label, confidence score
- **Data quality view:** A short view tying EDA to model failures — e.g. text length vs. label distribution, or where nulls/contractions are concentrated. This habit helps in production and makes the narrative clearer.
- For regression: `plot_predicted_vs_actual(y_pred, y_test)` — scatter with RMSE annotation

**Note:** The error analysis function is the most valuable output here — reading misclassified examples teaches you more about your data than any metric.

**Use aaryan-ai-toolkit:** `ai_toolkit.ml` (evaluator). Use for classification report, confusion matrix, ROC, error analysis, SHAP/LIME.

---

## Phase 12 — Explainability (SHAP + LIME)

**Goal:** Explain model decisions at global and instance level.

**What you build:**

- `explain_with_shap(model, X_train, X_test, feature_names)` — TreeExplainer on LightGBM, summary plot + force plot
- `explain_with_lime(model, X_test, texts, class_names, n_samples)` — LimeTextExplainer, visualize top 5 instances
- **Document:** SHAP = global understanding of what features matter, LIME = local understanding of why this specific prediction was made

**Use aaryan-ai-toolkit:** `ai_toolkit.ml` (evaluator). Use for classification report, confusion matrix, ROC, error analysis, SHAP/LIME.

---

## Phase 13 — Deep Learning Arc (PyTorch)

This is the largest phase. Break it into 4 sequential sub-phases.

### 13a — ANN (Feedforward Baseline)

Simple 3-layer network on TF-IDF or SBERT embeddings. Purpose: introduce PyTorch training loop, loss functions, optimizers before adding sequence complexity.

**What you build:**

- `SentimentANN(nn.Module)` — Linear → ReLU → Dropout → Linear
- `train_epoch(model, loader, optimizer, criterion)`
- `evaluate_model(model, loader)`
- `plot_training_curves(train_losses, val_losses)`

### 13b — LSTM from Scratch

**Optional (learning):** A simple **vanilla RNN** step before LSTM — one layer, no gating — helps appreciate why LSTM’s gating matters. Not required if time is tight; LSTM is the main sequence model.

**What you build:**

- **Vocabulary class** — build from corpus, word2idx, idx2word, handle `<UNK>` and `<PAD>`
- `ReviewDataset(Dataset)` — tokenize, encode, pad sequences
- `SentimentLSTM(nn.Module)` — Embedding → LSTM → Dropout → Linear
- `SentimentBiLSTM(nn.Module)` — same with `bidirectional=True`
- Packed sequences — `pack_padded_sequence` for variable length handling
- Full training loop with early stopping

### 13c — Self-Attention from Scratch

Build a single-head self-attention layer and plug it on top of your LSTM output. This is the bridge to transformers.

**What you build:**

- `SelfAttention(nn.Module)` — Q, K, V projections, scaled dot-product, softmax
- `SentimentLSTMWithAttention(nn.Module)` — LSTM → Attention → Linear
- Visualize attention weights on sample predictions — which words got attended to

### 13d — DistilBERT Fine-tuning

**Sample size:** When running DistilBERT, always use a **reduced sample** (e.g. `bert_sample_size` from config, typically 15k–20k). This keeps runs feasible on CPU and limited RAM; full dataset fine-tuning is optional for GPU setups.

**What you build:**

- `BERTDataset(Dataset)` — HuggingFace tokenizer, attention masks, token type ids
- Fine-tune distilbert-base-uncased using HuggingFace Trainer (on the reduced sample)
- `extract_bert_embeddings(texts, model, tokenizer)` — for using BERT as a feature extractor separately from fine-tuning
- Compare fine-tuned BERT against sklearn best model

**Use aaryan-ai-toolkit:** `ai_toolkit.dl` (lstm_model, attention, bert_finetuner). Use for ANN, LSTM, attention, and BERT; wire training loops and Streamlit in your app.

---

## Phase 14 — Final Summary & Model Comparison

**Goal:** Tell the complete story of what you built and what you learned.

**What you build:**

- Master comparison table: every model × representation × metric (same metrics as Phase 8)
- Learning curve plots for top models
- **Deliverable: final narrative** — Commit a `FINDINGS.md` (or a dedicated section in the README): what representation mattered most, where DL helped, where classical was sufficient, and one paragraph of business takeaway for a product team using these reviews. This completes the story for reviewers or interviewers.

---

## Streamlit Page ↔ Phase Mapping

| Streamlit Page       | Phases Covered        | Main UI Elements |
|----------------------|------------------------|------------------|
| 1_Data.py            | 1, 2                   | File path / URL, sample size, inspect table, null report, clean preview |
| 2_EDA.py             | 3, 4                   | Class dist, length dist, n-grams, wordclouds, filters |
| 3_Preprocessing.py   | 5, 6, 7                | Pipeline config, stem vs lemma sample, feature matrices status |
| 4_ML_Models.py       | 8, 9, 10               | Model/rep choice, train button, results table, tune & save |
| 5_DL_Models.py       | 13a–d                  | Model type (ANN/LSTM/BERT), train/infer, attention viz |
| 6_Evaluation.py      | 11, 12, 14             | Report, confusion matrix, ROC, error samples, SHAP/LIME |

## Phase → aaryan-ai-toolkit mapping

| Project Phase | Use aaryan-ai-toolkit module | Your Streamlit surface |
|---------------|-----------------------------|-------------------------|
| Phase 2       | `ai_toolkit.nlp` (preprocessor) | 1_Data (clean preview) |
| Phase 3–4     | `ai_toolkit.eda` (visualizer) | 2_EDA (all plots) |
| Phase 6       | `ai_toolkit.nlp` (embeddings) | 3_Preprocessing (feature build) |
| Phase 8–9     | `ai_toolkit.ml` (trainer)    | 4_ML_Models (train, table) |
| Phase 11–12   | `ai_toolkit.ml` (evaluator)  | 6_Evaluation (reports, SHAP/LIME) |
| Phase 13a–b   | `ai_toolkit.dl` (lstm_model) | 5_DL_Models (ANN, LSTM) |
| Phase 13c     | `ai_toolkit.dl` (attention) | 5_DL_Models (attention viz) |
| Phase 13d     | `ai_toolkit.dl` (bert_finetuner) | 5_DL_Models (BERT) |

**Workflow:** Install and use **aaryan-ai-toolkit**; implement data loading, config, and Streamlit UI in your project. For each phase, call the listed `ai_toolkit` modules and wire their outputs into the corresponding Streamlit page.

---

## Complete Phase Summary

| Phase   | Title                  | Primary Output           |
|---------|------------------------|--------------------------|
| Pre     | Setup                  | Environment, repo, config |
| 1       | Data Gathering         | Raw dataframe            |
| 2       | Cleaning               | df_clean                 |
| 3       | EDA                    | 10 findings              |
| 4       | Visualization          | All plots                |
| 5       | NLP Preprocessing      | Processed text, vocabulary |
| 6       | Feature Engineering    | 5 feature matrices       |
| 7       | Scaling & Encoding     | ML-ready arrays          |
| 8       | Sklearn Pipelines      | Pipeline objects         |
| 9       | Model Training         | Results table            |
| 10      | Hyperparameter Tuning  | Best models              |
| 11      | Evaluation             | Error analysis           |
| 12      | Explainability         | SHAP + LIME outputs      |
| 13a     | ANN                    | PyTorch baseline         |
| 13b     | LSTM + BiLSTM          | Sequence models          |
| 13c     | Self-Attention         | Attention layer + visualization |
| 13d     | DistilBERT             | Fine-tuned transformer   |
| 14      | Final Summary          | Master comparison, insights |

---

## Senior AI Engineer Evaluation

### Verdict: **Strong learning plan — ready to execute with minor gaps**

This roadmap is well-suited for learning end-to-end text handling: from raw data and EDA through classical ML and deep learning (RNN/LSTM → attention → transformers). Moving to a **local setup + Streamlit UI** improves reproducibility and portfolio value. Using **aaryan-ai-toolkit** for logic keeps a clean separation: toolkit = NLP/EDA/ML/DL, your repo = app, config, data, and wiring. Below is a concise assessment and what to improve.

---

### How well it covers the learning goals

| Learning area | Coverage | Notes |
|---------------|----------|--------|
| **Data: gather, extract** | ✅ Strong | Phase 1–2: download, load, sample, clean, null handling, sentiment mapping. Clear and sufficient. |
| **Analyze & visualize** | ✅ Strong | Phase 3–4: distributions, n-grams, wordclouds, correlations. EDA → viz coupling is good. |
| **ML: classification** | ✅ Strong | Phase 8–9: pipelines (LR, NB, RF, LightGBM) × representations (BoW, TF-IDF, W2V, SBERT). Builds real intuition. |
| **ML: regression** | ✅ Good | Same representations + Linear, Ridge, RF, LightGBM regressors. Slightly lighter than classification but enough. |
| **DL: RNN/LSTM** | ✅ Strong | Phase 13b: vocab, Dataset, LSTM/BiLSTM, packed sequences. Core sequence modeling is covered. |
| **Attention** | ✅ Strong | Phase 13c: single-head self-attention from scratch, LSTM+attention, weight visualization. Good bridge to transformers. |
| **Transformers** | ✅ Strong | Phase 13d: BERT dataset, fine-tuning with Trainer, embedding extraction. Industry-relevant. |
| **Explainability** | ✅ Good | SHAP (global) + LIME (local). Teaches interpretability; Streamlit can expose both. |
| **Engineering practice** | ✅ Good | Pipelines, config, modular layout, Streamlit for demos. No Colab lock-in. |

**Summary:** The plan covers the stated learning goals well. It progresses logically: data → features → classical ML → DL (ANN → LSTM → attention → BERT). For a learning portfolio or interview story, it’s coherent and complete.

---

### What can be done better (incorporated)

The following improvements are **accepted and reflected** in the roadmap above (see the referenced phases and Pre-Phase):

1. **Explicit train/val/test split and reproducibility** — Pre-Phase and config: single seed-based split, documented and surfaced on the Data page.
2. **Baseline and metrics contract** — Phase 8: explicit baseline, fixed metric sets (classification: accuracy, macro F1, per-class; regression: RMSE, MAE).
3. **Scaling and encoding ownership** — Phase 7: ownership assigned (ai_toolkit.ml + optional utils/features.py), “Scaled + encoded” shown in UI.
4. **DL: optional vanilla RNN** — Phase 13b: optional step before LSTM described.
5. **Streamlit: when to build it** — Pre-Phase: build app shell and stubs early; wire outputs incrementally (already the intended workflow).
6. **Error analysis and data quality** — Phase 11: data quality view (e.g. length vs. label, null/contras) added.
7. **Dependencies and environment** — *Left to implementer’s choice* (pip vs uv, pinning, conda, etc.). Not mandated in this roadmap.
8. **Phase 14 narrative** — Phase 14: deliverable `FINDINGS.md` or README section with narrative and business takeaway.

---

### Bottom line

- **As a learning plan:** It is strong. You will touch data ops, EDA, visualization, classical ML (classification + regression), and the full DL arc (ANN → LSTM → attention → BERT), with explainability and a local, UI-driven workflow.
- **As an executable project:** With the above tweaks (splits, baselines, metrics, and incremental Streamlit wiring), it becomes a solid portfolio piece and a clear narrative for “how I handle text from raw data to transformers.”
- **Verdict:** Proceed with implementation. Incorporate the improvements above as you go rather than blocking on them; the plan is already good enough to start.
