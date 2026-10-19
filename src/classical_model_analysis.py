import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# Allow sibling imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import preprocess_fer_directory, EMOTION_LABELS

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "fer2013")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "classical_analysis")

NUM_CLASSES = len(EMOTION_LABELS)
SEED = 42


# ============================================================
# DATA LOADING
# ============================================================
def load_flattened_data():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_DIR}\n"
            "Expected: data/fer2013/train/<emotion>/*.png and test/<emotion>/*.png"
        )

    X_train, y_train, X_test, y_test = preprocess_fer_directory(DATA_DIR)

    # Flatten 48x48x1 -> 2304
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    print(f"\nFlattened train shape: {X_train.shape}")
    print(f"Flattened test shape : {X_test.shape}")

    return X_train, y_train, X_test, y_test


# ============================================================
# MODELS
# ============================================================
def get_models():
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=SEED,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto"
            ))
        ]),

        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                random_state=SEED,
                class_weight="balanced",
                max_iter=5000
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
            ))
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=3,
            random_state=SEED
        ),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=80,
                random_state=SEED
            ))
        ]),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=-1
        )

    return models


# ============================================================
# HELPERS
# ============================================================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_probabilities(model, X):
    """
    Return prediction probabilities if available.
    Fallback for LinearSVC using decision_function + softmax-like normalization.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    return None


def classification_error(acc):
    return 1.0 - acc


def save_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


def save_bar_chart(df, x_col, y_col, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


def save_cv_chart(df, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(df["Model"], df["CV Mean"], yerr=df["CV Std"], capsize=5)
    plt.title("Cross-Validation Accuracy")
    plt.ylabel("CV Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


def save_feature_importance(model_name, model, feature_dim):
    """
    Save feature importance graph only for tree-based models that support it.
    Since flattened image pixels are features, this is pixel importance.
    """
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("clf", model)

    if not hasattr(estimator, "feature_importances_"):
        return

    importances = estimator.feature_importances_
    top_k = min(30, len(importances))
    top_idx = np.argsort(importances)[-top_k:][::-1]
    top_vals = importances[top_idx]

    plt.figure(figsize=(12, 6))
    plt.bar(range(top_k), top_vals)
    plt.title(f"Top {top_k} Feature Importances - {model_name}")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.xticks(range(top_k), top_idx, rotation=90)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{model_name}_feature_importance.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # Also save as heatmap 48x48 if possible
    if feature_dim == 48 * 48:
        heatmap = np.array(importances).reshape(48, 48)
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap="hot")
        plt.title(f"Pixel Importance Heatmap - {model_name}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{model_name}_importance_heatmap.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()


# ============================================================
# MAIN ANALYSIS
# ============================================================
def run_classical_analysis():
    ensure_output_dir()

    X_train, y_train, X_test, y_test = load_flattened_data()
    models = get_models()

    y_test_bin = label_binarize(y_test, classes=np.arange(NUM_CLASSES))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    all_results = []
    best_model_name = None
    best_model = None
    best_accuracy = -1.0
    best_cm = None
    best_report = None

    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {model_name}")
        print(f"{'=' * 60}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        err = classification_error(acc)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        probs = get_probabilities(model, X_test)
        if probs is not None and probs.shape[1] == NUM_CLASSES:
            try:
                roc_auc = roc_auc_score(
                    y_test_bin, probs, multi_class="ovr", average="weighted"
                )
            except Exception:
                roc_auc = np.nan
        else:
            roc_auc = np.nan

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=EMOTION_LABELS,
            digits=4,
            zero_division=0
        )

        print(f"Accuracy       : {acc:.4f}")
        print(f"Error Rate     : {err:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1 Score       : {f1:.4f}")
        print(f"ROC-AUC        : {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC        : N/A")
        print(f"CV Mean        : {cv_mean:.4f}")
        print(f"CV Std         : {cv_std:.4f}")

        all_results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Error Rate": err,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc if not np.isnan(roc_auc) else None,
            "CV Mean": cv_mean,
            "CV Std": cv_std,
        })

        # Save prediction values
        pred_df = pd.DataFrame({
            "true_label": y_test,
            "predicted_label": y_pred,
            "true_emotion": [EMOTION_LABELS[i] for i in y_test],
            "predicted_emotion": [EMOTION_LABELS[i] for i in y_pred],
        })
        pred_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{model_name}_predictions.csv"),
            index=False
        )

        with open(os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        save_confusion_matrix(
            cm,
            EMOTION_LABELS,
            title=f"Confusion Matrix - {model_name}",
            filename=os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
        )

        save_feature_importance(model_name, model, feature_dim=X_train.shape[1])

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = model_name
            best_model = model
            best_cm = cm
            best_report = report

    # Results dataframe
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    results_csv = os.path.join(OUTPUT_DIR, "classical_models_summary.csv")
    results_df.to_csv(results_csv, index=False)

    # Save comparison graphs
    save_bar_chart(
        results_df,
        x_col="Model",
        y_col="Accuracy",
        title="Model Accuracy Comparison",
        ylabel="Accuracy",
        filename=os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    )

    save_bar_chart(
        results_df,
        x_col="Model",
        y_col="Error Rate",
        title="Classification Error Rate Comparison",
        ylabel="Error Rate",
        filename=os.path.join(OUTPUT_DIR, "error_rate_comparison.png")
    )

    roc_df = results_df.copy()
    roc_df["ROC-AUC"] = pd.to_numeric(roc_df["ROC-AUC"], errors="coerce")
    roc_df = roc_df.dropna(subset=["ROC-AUC"])
    if not roc_df.empty:
        save_bar_chart(
            roc_df,
            x_col="Model",
            y_col="ROC-AUC",
            title="ROC-AUC Comparison",
            ylabel="ROC-AUC",
            filename=os.path.join(OUTPUT_DIR, "roc_auc_comparison.png")
        )

    save_cv_chart(
        results_df,
        filename=os.path.join(OUTPUT_DIR, "cross_validation_comparison.png")
    )

    # Save best model report
    best_summary = {
        "best_model": best_model_name,
        "best_accuracy": float(best_accuracy),
        "results_csv": results_csv,
        "output_dir": OUTPUT_DIR,
    }

    with open(os.path.join(OUTPUT_DIR, "best_model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    if best_report is not None:
        with open(os.path.join(OUTPUT_DIR, "best_model_report.txt"), "w", encoding="utf-8") as f:
            f.write(best_report)

    if best_cm is not None:
        save_confusion_matrix(
            best_cm,
            EMOTION_LABELS,
            title=f"Best Model Confusion Matrix - {best_model_name}",
            filename=os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png")
        )

    print(f"\nBest model: {best_model_name} | Accuracy: {best_accuracy:.4f}")
    print(f"Saved all outputs to: {OUTPUT_DIR}")
    print("\nFinal Summary:")
    print(results_df)


if __name__ == "__main__":
    run_classical_analysis()