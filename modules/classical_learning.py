"""
modules/classical_learning.py
==============================
Classical Machine Learning Pipeline — huấn luyện, đánh giá, tuning, visualize.

Module này phục vụ notebook 03_classical_pipeline.ipynb và được import
vào 05_main.ipynb.

Cấu trúc
--------
  Phần 1 — Feature Extraction
      to_dense_array, extract_features_pca, plot_pca_variance

  Phần 2 — Baseline Training
      get_baseline_models, run_baseline, summarize_baseline

  Phần 3 — Hyperparameter Tuning
      get_tuning_configs, run_tuning

  Phần 4 — Evaluation & Visualization
      print_best_model_summary, plot_f1_comparison,
      plot_confusion_matrix, print_classification_report,
      plot_roc_curves, plot_metric_comparison,
      compute_improvement

  Phần 5 — Feature Importance
      plot_feature_importance

  Phần 6 — Save / Load
      save_features, save_results, save_model, load_model

Cách dùng
---------
    import modules.classical_learning as cl

    # Feature extraction
    feature_sets, y_train_model, y_test_model = cl.extract_features_pca(
        X_train_processed, X_test_processed, y_train, y_test
    )

    # Baseline
    baseline_models = cl.get_baseline_models()
    trained_models, baseline_results_df, baseline_predictions = cl.run_baseline(
        baseline_models, feature_sets, y_train_model, y_test_model
    )

    # Tuning
    tuning_results_df, tuned_models, tuned_predictions = cl.run_tuning(
        feature_sets, y_train_model, y_test_model
    )
"""

import os
import time
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, make_scorer,
)

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# Phần 1 — Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def to_dense_array(X) -> np.ndarray:
    """Chuyển sparse matrix sang dense numpy array nếu cần."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def encode_labels(y_train, y_test) -> tuple:
    """
    Chuẩn hóa nhãn chuỗi \'<=50K\' / \'>50K\' sang số nguyên 0 / 1.

    Parameters
    ----------
    y_train, y_test : pd.Series hoặc array-like

    Returns
    -------
    (y_train_model, y_test_model) : np.ndarray int
    """
    label_map = {"<=50K": 0, ">50K": 1}

    y_train_clean = (
        pd.Series(y_train).astype(str).str.strip().str.replace(".", "", regex=False)
    )
    y_test_clean = (
        pd.Series(y_test).astype(str).str.strip().str.replace(".", "", regex=False)
    )

    print("Train label distribution:")
    print(y_train_clean.value_counts().to_string())
    print("\nTest label distribution:")
    print(y_test_clean.value_counts().to_string())

    if set(y_train_clean.unique()).issubset(set(label_map.keys())):
        y_train_model = y_train_clean.map(label_map).astype(int).to_numpy()
        y_test_model  = y_test_clean.map(label_map).astype(int).to_numpy()
        print("\nLabels encoded: <=50K → 0, >50K → 1")
    else:
        y_train_model = y_train_clean.to_numpy()
        y_test_model  = y_test_clean.to_numpy()
        print("\nWarning: unexpected labels found, kept as original.")

    return y_train_model, y_test_model


def extract_features_pca(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    pca_variance: float = 0.95,
) -> tuple:
    """
    Tạo hai nhánh đặc trưng: full features và PCA giữ lại pca_variance phương sai.

    PCA chỉ được fit trên tập train rồi transform tập test để tránh data leakage.

    Parameters
    ----------
    X_train_processed, X_test_processed : sparse hoặc dense array
        Output của ColumnTransformer.
    y_train, y_test : array-like
        Nhãn gốc (chuỗi hoặc số).
    pca_variance : float, default=0.95
        Tỉ lệ phương sai cần giữ lại.

    Returns
    -------
    (feature_sets, y_train_model, y_test_model)
        feature_sets : dict với keys \'full_features\' và \'pca_95\'
        y_train_model, y_test_model : np.ndarray int
    """
    # Dense conversion
    X_train_full = to_dense_array(X_train_processed).astype(np.float32)
    X_test_full  = to_dense_array(X_test_processed).astype(np.float32)

    assert X_train_full.shape[0] == len(y_train), "Mismatch X_train / y_train"
    assert X_test_full.shape[0]  == len(y_test),  "Mismatch X_test / y_test"

    print("Full feature branch")
    print("  X_train_full:", X_train_full.shape)
    print("  X_test_full :", X_test_full.shape)

    # Encode labels
    y_train_model, y_test_model = encode_labels(y_train, y_test)

    # PCA branch
    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=42)
    X_train_pca = pca.fit_transform(X_train_full)
    X_test_pca  = pca.transform(X_test_full)

    print(f"\nPCA {int(pca_variance*100)}% feature branch")
    print(f"  Components  : {pca.n_components_}")
    print(f"  Var retained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  X_train_pca : {X_train_pca.shape}")
    print(f"  X_test_pca  : {X_test_pca.shape}")

    feature_sets = {
        "full_features": {"X_train": X_train_full, "X_test": X_test_full, "pca": None},
        f"pca_{int(pca_variance*100)}":  {"X_train": X_train_pca,  "X_test": X_test_pca,  "pca": pca},
    }

    return feature_sets, y_train_model, y_test_model


def plot_pca_variance(feature_sets: dict, pca_key: str = "pca_95"):
    """
    Vẽ biểu đồ cumulative explained variance của PCA.

    Parameters
    ----------
    feature_sets : dict
        Output của extract_features_pca.
    pca_key : str
        Key tương ứng với nhánh PCA trong feature_sets.
    """
    pca = feature_sets[pca_key]["pca"]
    if pca is None:
        print(f"No PCA object found for key \'{pca_key}\'.")
        return

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    threshold = pca.n_components  # float 0.95 hoặc int

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    if isinstance(threshold, float):
        plt.axhline(y=threshold, linestyle="--", label=f"{int(threshold*100)}% variance")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Phần 2 — Baseline Training
# ══════════════════════════════════════════════════════════════════════════════

def get_baseline_models() -> dict:
    """
    Trả về dict các mô hình baseline mặc định.

    Gồm: Logistic Regression, Linear SVM, Random Forest, Gaussian Naive Bayes.
    Linear SVM được dùng thay SVM-RBF để đảm bảo thời gian chạy phù hợp
    với môi trường Colab khi Run all.

    Returns
    -------
    dict[str, estimator]
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight="balanced", random_state=42,
            max_iter=10000, dual=False
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def _evaluate_one_model(model, model_name, feature_set_name,
                         X_train, X_test, y_train, y_test) -> tuple:
    """Train một model và trả về (trained_model, y_pred, result_dict)."""
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan

    result = {
        "feature_set":   feature_set_name,
        "model":         model_name,
        "accuracy":      accuracy_score(y_test, y_pred),
        "precision":     precision_score(y_test, y_pred, zero_division=0),
        "recall":        recall_score(y_test, y_pred, zero_division=0),
        "f1_score":      f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":       roc_auc,
        "train_time_sec": elapsed,
    }
    return model, y_pred, result


def run_baseline(
    baseline_models: dict,
    feature_sets: dict,
    y_train_model: np.ndarray,
    y_test_model: np.ndarray,
) -> tuple:
    """
    Huấn luyện và đánh giá các mô hình baseline trên tất cả feature branches.

    Parameters
    ----------
    baseline_models : dict[str, estimator]
    feature_sets : dict
        Output của extract_features_pca.
    y_train_model, y_test_model : np.ndarray

    Returns
    -------
    (trained_models, baseline_results_df, baseline_predictions)
        trained_models : dict[str, estimator]
        baseline_results_df : pd.DataFrame sắp xếp theo f1_score giảm dần
        baseline_predictions : dict[str, np.ndarray]
    """
    trained_models = {}
    baseline_predictions = {}
    results = []

    for fs_name, data in feature_sets.items():
        print("=" * 70)
        print(f"Feature set: {fs_name}")
        print("=" * 70)

        X_tr = data["X_train"]
        X_te = data["X_test"]

        for model_name, model in baseline_models.items():
            print(f"\n  Training: {model_name}")
            trained_model, y_pred, result = _evaluate_one_model(
                model, model_name, fs_name, X_tr, X_te, y_train_model, y_test_model
            )
            key = f"{fs_name}__{model_name}"
            trained_models[key] = trained_model
            baseline_predictions[key] = y_pred
            results.append(result)

            print(f"    F1-score : {result['f1_score']:.4f}")
            print(f"    Accuracy : {result['accuracy']:.4f}")
            print(f"    ROC-AUC  : {result['roc_auc']:.4f}")
            print(f"    Time     : {result['train_time_sec']:.2f}s")

    baseline_results_df = (
        pd.DataFrame(results)
        .sort_values(by=["f1_score", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    return trained_models, baseline_results_df, baseline_predictions


def print_best_model_summary(results_df: pd.DataFrame, stage: str = "Baseline"):
    """In thông tin mô hình tốt nhất từ bảng kết quả."""
    best = results_df.iloc[0]
    print(f"Best {stage} model:")
    for col in ["feature_set", "model", "accuracy", "precision",
                "recall", "f1_score", "roc_auc"]:
        val = best[col]
        print(f"  {col:20s}: {round(val, 4) if isinstance(val, float) else val}")


# ══════════════════════════════════════════════════════════════════════════════
# Phần 3 — Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════

def get_tuning_configs() -> dict:
    """
    Trả về dict cấu hình tuning cho 4 mô hình.

    Mỗi entry gồm: model, search_type ('grid' / 'random'), params, [n_iter].

    Returns
    -------
    dict
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(
                class_weight="balanced", random_state=42, max_iter=2000
            ),
            "search_type": "grid",
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["liblinear"],
            },
        },
        "Linear SVM": {
            "model": LinearSVC(
                class_weight="balanced", random_state=42,
                max_iter=10000, dual=False
            ),
            "search_type": "grid",
            "params": {"C": [0.01, 0.1, 1.0, 10.0]},
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "search_type": "random",
            "params": {
                "n_estimators":    [100, 150, 200],
                "max_depth":       [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
            },
            "n_iter": 15,
        },
        "Gaussian Naive Bayes": {
            "model": GaussianNB(),
            "search_type": "grid",
            "params": {
                "var_smoothing": [1e-12, 1e-10, 1e-9, 1e-8, 1e-7]
            },
        },
    }


def run_tuning(
    feature_sets: dict,
    y_train_model: np.ndarray,
    y_test_model: np.ndarray,
    main_feature_set: str = "full_features",
    n_splits: int = 5,
    tuning_configs: dict = None,
) -> tuple:
    """
    Chạy hyperparameter tuning bằng GridSearch / RandomizedSearch
    với StratifiedKFold cross-validation.

    Parameters
    ----------
    feature_sets : dict
    y_train_model, y_test_model : np.ndarray
    main_feature_set : str, default='full_features'
    n_splits : int, default=5
    tuning_configs : dict hoặc None
        Nếu None sẽ dùng get_tuning_configs().

    Returns
    -------
    (tuning_results_df, tuned_models, tuned_predictions)
    """
    if tuning_configs is None:
        tuning_configs = get_tuning_configs()

    X_train_tune = feature_sets[main_feature_set]["X_train"]
    X_test_tune  = feature_sets[main_feature_set]["X_test"]

    print(f"Tuning on feature set : {main_feature_set}")
    print(f"X_train_tune shape    : {X_train_tune.shape}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, pos_label=1)

    tuned_models = {}
    tuned_predictions = {}
    results = []

    for model_name, config in tuning_configs.items():
        print("\n" + "=" * 70)
        print(f"Tuning: {model_name}")
        print("=" * 70)

        t0 = time.time()

        if config["search_type"] == "grid":
            search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                scoring=f1_scorer, cv=cv, n_jobs=-1, verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["params"],
                n_iter=config.get("n_iter", 10),
                scoring=f1_scorer, cv=cv, n_jobs=-1,
                random_state=42, verbose=1,
            )

        search.fit(X_train_tune, y_train_model)
        elapsed = time.time() - t0

        best_model  = search.best_estimator_
        best_params = search.best_params_
        best_cv_f1  = search.best_score_

        y_pred = best_model.predict(X_test_tune)

        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test_tune)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_test_tune)
        else:
            y_score = None

        roc_auc = roc_auc_score(y_test_model, y_score) if y_score is not None else np.nan

        result = {
            "feature_set":    main_feature_set,
            "model":          model_name,
            "best_params":    best_params,
            "best_cv_f1":     best_cv_f1,
            "test_accuracy":  accuracy_score(y_test_model, y_pred),
            "test_precision": precision_score(y_test_model, y_pred, zero_division=0),
            "test_recall":    recall_score(y_test_model, y_pred, zero_division=0),
            "test_f1_score":  f1_score(y_test_model, y_pred, zero_division=0),
            "test_roc_auc":   roc_auc,
            "tuning_time_sec": elapsed,
        }

        tuned_models[model_name] = best_model
        tuned_predictions[model_name] = y_pred
        results.append(result)

        print(f"  Best params  : {best_params}")
        print(f"  CV F1        : {best_cv_f1:.4f}")
        print(f"  Test F1      : {result['test_f1_score']:.4f}")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Test ROC-AUC : {result['test_roc_auc']:.4f}")
        print(f"  Time         : {elapsed:.1f}s")

    tuning_results_df = (
        pd.DataFrame(results)
        .sort_values(by=["test_f1_score", "test_roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    return tuning_results_df, tuned_models, tuned_predictions


# ══════════════════════════════════════════════════════════════════════════════
# Phần 4 — Evaluation & Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_f1_comparison(results_df: pd.DataFrame,
                        score_col: str = "f1_score",
                        title: str = "Model Comparison by F1-score"):
    """Vẽ horizontal bar chart so sánh F1-score các mô hình."""
    labels = results_df["feature_set"] + " | " + results_df["model"]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, results_df[score_col])
    plt.xlabel(score_col.replace("_", " ").title())
    plt.ylabel("Model")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix"):
    """Vẽ confusion matrix với nhãn <=50K / >50K."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["<=50K", ">50K"]
    )
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, title: str = ""):
    """In classification report với nhãn <=50K / >50K."""
    if title:
        print(f"Classification report — {title}")
        print()
    print(classification_report(
        y_true, y_pred,
        target_names=["<=50K", ">50K"],
        zero_division=0,
    ))


def plot_roc_curves(tuned_models: dict, X_test, y_test,
                     title: str = "ROC Curves of Tuned Models"):
    """Vẽ ROC curve của tất cả tuned models trên cùng một biểu đồ."""
    plt.figure(figsize=(8, 6))
    for model_name, model in tuned_models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_improvement(best_baseline: pd.Series, best_tuned: pd.Series) -> pd.DataFrame:
    """
    Tính mức cải thiện từ baseline sang tuned model.

    Returns
    -------
    pd.DataFrame với các chỉ số và mức thay đổi.
    """
    metrics = {
        "f1_score":  ("f1_score",      "test_f1_score"),
        "accuracy":  ("accuracy",       "test_accuracy"),
        "roc_auc":   ("roc_auc",        "test_roc_auc"),
    }
    rows = []
    for metric, (b_col, t_col) in metrics.items():
        b_val = best_baseline[b_col]
        t_val = best_tuned[t_col]
        diff  = t_val - b_val
        rows.append({
            "metric":    metric,
            "baseline":  round(b_val, 4),
            "tuned":     round(t_val, 4),
            "delta":     round(diff, 4),
            "relative%": round(diff / b_val * 100, 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def build_final_comparison(best_baseline: pd.Series,
                             best_tuned: pd.Series) -> pd.DataFrame:
    """
    Tạo bảng so sánh mô hình baseline tốt nhất và tuned tốt nhất.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame([
        {
            "stage":      "Best Baseline",
            "feature_set": best_baseline["feature_set"],
            "model":      best_baseline["model"],
            "accuracy":   best_baseline["accuracy"],
            "precision":  best_baseline["precision"],
            "recall":     best_baseline["recall"],
            "f1_score":   best_baseline["f1_score"],
            "roc_auc":    best_baseline["roc_auc"],
            "time_sec":   best_baseline["train_time_sec"],
            "params":     "default",
        },
        {
            "stage":      "Best Tuned",
            "feature_set": best_tuned["feature_set"],
            "model":      best_tuned["model"],
            "accuracy":   best_tuned["test_accuracy"],
            "precision":  best_tuned["test_precision"],
            "recall":     best_tuned["test_recall"],
            "f1_score":   best_tuned["test_f1_score"],
            "roc_auc":    best_tuned["test_roc_auc"],
            "time_sec":   best_tuned["tuning_time_sec"],
            "params":     best_tuned["best_params"],
        },
    ])


def plot_metric_comparison(final_comparison: pd.DataFrame,
                             title: str = "Best Baseline vs Best Tuned Model"):
    """Vẽ grouped bar chart so sánh các chỉ số giữa baseline và tuned model."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    plot_df = final_comparison.set_index("stage")[metrics].T
    plot_df.plot(kind="bar", figsize=(10, 6))
    plt.title(title)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.legend(title="Stage")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Phần 5 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    tuned_models: dict,
    best_preprocessor,
    num_features: list,
    cat_features: list,
    top_n: int = 15,
    model_key: str = "Random Forest",
):
    """
    Vẽ biểu đồ feature importance của Random Forest (hoặc model có feature_importances_).

    Parameters
    ----------
    tuned_models : dict
    best_preprocessor : fitted ColumnTransformer
        Dùng để lấy tên cột sau OHE.
    num_features, cat_features : list
    top_n : int, default=15
    model_key : str, default='Random Forest'
    """
    rf = tuned_models.get(model_key)
    if rf is None or not hasattr(rf, "feature_importances_"):
        print(f"Model '{model_key}' không có feature_importances_.")
        return

    cat_encoder      = best_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    feature_names    = np.concatenate([np.array(num_features), cat_feature_names])

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    display_df = fi_df.head(top_n)
    print(display_df.to_string(index=False))

    top = display_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances — Tuned {model_key}")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()

    return fi_df


# ══════════════════════════════════════════════════════════════════════════════
# Phần 6 — Save / Load
# ══════════════════════════════════════════════════════════════════════════════

def save_features(feature_sets: dict, y_train_model, y_test_model,
                   output_dir: str = "features"):
    """Lưu toàn bộ feature branches và labels vào thư mục output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for name, data in feature_sets.items():
        np.save(os.path.join(output_dir, f"X_train_{name}.npy"), data["X_train"])
        np.save(os.path.join(output_dir, f"X_test_{name}.npy"),  data["X_test"])
    np.save(os.path.join(output_dir, "y_train.npy"), y_train_model)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_test_model)
    print(f"Features saved to {output_dir}/:", os.listdir(output_dir))


def save_results(df: pd.DataFrame, filename: str, output_dir: str = "results"):
    """Lưu DataFrame kết quả ra file CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_model(model, model_name: str, output_dir: str = "models") -> str:
    """Lưu model ra file .pkl bằng joblib."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"final_{model_name.replace(' ', '_').lower()}_model.pkl"
    path = os.path.join(output_dir, filename)
    joblib.dump(model, path)
    print(f"Model saved: {path}")
    return path


def load_model(path: str):
    """Load model từ file .pkl."""
    model = joblib.load(path)
    print(f"Model loaded: {path}")
    return model
