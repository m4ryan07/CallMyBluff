"""
src/evaluate.py — CallMyBluff Phase 4
Rigorous model evaluation with classification report,
confusion matrix, and ROC-AUC curve.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import StylometricTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = "models"
PROCESSED_DIR = os.path.join("data", "processed")
PLOTS_DIR     = os.path.join("data", "plots")
TEXT_COL      = "text"
LABEL_COL     = "label"


# ── Load Artifacts ─────────────────────────────────────────────────────────────
def load_artifacts():
    model  = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    meta   = joblib.load(os.path.join(MODELS_DIR, "model_meta.joblib"))
    return model, scaler, meta


# ── Build Features ─────────────────────────────────────────────────────────────
def build_features(texts: pd.Series, scaler, fit: bool = False):
    st = StylometricTransformer()
    X  = st.fit_transform(texts)
    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X


# ── Plot Confusion Matrix ──────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, save_path: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("CallMyBluff — Confusion Matrix", fontsize=14, fontweight="bold")

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fig.text(
        0.5, 0.01,
        f"False Positive Rate: {fpr:.2%}  |  False Negative Rate: {fnr:.2%}",
        ha="center", fontsize=10, color="gray"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved → {save_path}")


# ── Plot ROC Curve ─────────────────────────────────────────────────────────────
def plot_roc_curve(y_test, y_proba, save_path: str):
    fpr_vals, tpr_vals, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_vals, tpr_vals, color="#2563EB", lw=2,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1,
            linestyle="--", label="Random Classifier")
    ax.fill_between(fpr_vals, tpr_vals, alpha=0.1, color="#2563EB")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("CallMyBluff — ROC-AUC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved → {save_path}")


# ── Plot Feature Importance ────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, save_path: str):
    """
    For LogisticRegression: uses coefficients.
    Shows which stylometric features push toward Human vs AI.
    """
    if not hasattr(model, "coef_"):
        print("  ⚠️  Feature importance only available for LogisticRegression")
        return

    coefs = model.coef_[0]
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "coefficient": coefs,
    }).sort_values("coefficient", ascending=True)

    colors = ["#EF4444" if c > 0 else "#22C55E"
              for c in importance_df["coefficient"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(importance_df["feature"], importance_df["coefficient"],
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("Feature Importance\n(Red = pushes toward AI, Green = pushes toward Human)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved → {save_path}")


# ── Benchmark Stats ────────────────────────────────────────────────────────────
def compute_benchmarks(train_df: pd.DataFrame) -> dict:
    """
    Computes average stylometric feature values per class
    from the training set. Used by the Streamlit app for
    comparison charts.
    """
    st = StylometricTransformer()
    feature_names = st.get_feature_names_out()
    from src.features import extract_stylometric_features
    stylo_df = extract_stylometric_features(train_df[TEXT_COL])
    stylo_df["label"] = train_df[LABEL_COL].values
    benchmarks = stylo_df.groupby("label").mean().to_dict()
    joblib.dump(benchmarks, os.path.join(MODELS_DIR, "benchmarks.joblib"))
    print(f"  ✅ Saved → models/benchmarks.joblib")
    return benchmarks


# ── Main ───────────────────────────────────────────────────────────────────────
def evaluate():
    print("\n📊 CallMyBluff — Phase 4: Evaluation")
    print("=" * 55)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load
    model, scaler, meta = load_artifacts()
    print(f"\n  Model loaded : {meta['model_name']}")

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    # Features
    print("\n  Building test features...")
    X_test  = build_features(test_df[TEXT_COL], scaler, fit=False)
    y_test  = test_df[LABEL_COL].values
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification Report
    print("\n── Classification Report ────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

    # Key metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(y_test, y_proba)

    print(f"  ROC-AUC Score      : {auc:.4f}")
    print(f"  False Positive Rate: {fpr:.4f} ({fpr:.2%})")
    print(f"  → We wrongly flag {fp} real human(s) as AI out of {tn+fp}")

    # Plots
    print("\n── Generating Plots ─────────────────────────────────────")
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png")
    )
    plot_roc_curve(
        y_test, y_proba,
        save_path=os.path.join(PLOTS_DIR, "roc_curve.png")
    )
    plot_feature_importance(
        model,
        meta["feature_names"],
        save_path=os.path.join(PLOTS_DIR, "feature_importance.png")
    )

    # Benchmarks for Streamlit app
    print("\n── Computing Training Benchmarks ────────────────────────")
    benchmarks = compute_benchmarks(train_df)

    print("\n── Per-feature Averages by Class ────────────────────────")
    print(f"  {'Feature':<25} {'Human':>10} {'AI':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10}")
    for feat in meta["feature_names"]:
        h_val = benchmarks[feat].get(0, 0)
        a_val = benchmarks[feat].get(1, 0)
        print(f"  {feat:<25} {h_val:>10.4f} {a_val:>10.4f}")

    print(f"\n✅ Phase 4 complete. Plots saved to data/plots/")
    print(f"   Ready for Phase 5: Streamlit App.\n")


if __name__ == "__main__":
    evaluate()