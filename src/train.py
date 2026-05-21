"""
src/train.py — CallMyBluff Phase 3
Trains three classifiers on the combined stylometric + TF-IDF
feature matrix and serializes the best-performing model.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import ComplementNB
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import MaxAbsScaler
from sklearn.metrics         import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

# Add project root to path so src imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_matrix


# ── Configuration ──────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models")
RANDOM_STATE  = 42
TEXT_COL      = "text"
LABEL_COL     = "label"


# ── Model Definitions ──────────────────────────────────────────────────────────
def get_models() -> dict:
    """
    Returns the three classifiers we are benchmarking.

    Why these three:
    - LogisticRegression : Fast, interpretable, strong on high-dim sparse data.
                           Its coefficients directly tell us which features
                           push toward Human vs AI.
    - ComplementNB       : Superior to MultinomialNB for imbalanced text data.
                           Works natively with TF-IDF frequency distributions.
    - RandomForestClassifier: Captures non-linear interactions between our
                           stylometric features (e.g. high burstiness AND
                           low TTR together = strong human signal).
    """
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0,                    # regularization strength
            max_iter=1000,
            class_weight="balanced",  # handles class imbalance automatically
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ComplementNB": ComplementNB(
            alpha=0.1,                # smoothing — lower = less smoothing
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,           # let trees grow fully
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ── Evaluation Helper ──────────────────────────────────────────────────────────
def evaluate_model(
    name: str,
    model,
    X_test,
    y_test: np.ndarray,
) -> dict:
    """
    Runs prediction and prints a clean report for one model.
    Returns a dict of scalar metrics for comparison.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (AI)

    report = classification_report(
        y_test, y_pred,
        target_names=["Human", "AI"],
        output_dict=True,
    )
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    # Extract key metrics
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)   # False Positive Rate — accusing humans of being AI

    print(f"\n{'─'*55}")
    print(f"  Model : {name}")
    print(f"{'─'*55}")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
    print(f"  ROC-AUC Score  : {auc:.4f}")
    print(f"  False Pos Rate : {fpr:.4f}  ← (lower = better, don't accuse humans)")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    return {
        "name":      name,
        "model":     model,
        "f1_ai":     report["AI"]["f1-score"],
        "f1_human":  report["Human"]["f1-score"],
        "precision": report["weighted avg"]["precision"],
        "recall":    report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "auc":       auc,
        "fpr":       fpr,
    }


# ── Main Training Pipeline ─────────────────────────────────────────────────────
def train():
    print("\n🏋️  CallMyBluff — Phase 3: Model Training Pipeline")
    print("=" * 55)

    # ── 1. Load Data ───────────────────────────────────────────────────────────
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test.csv")

    print(f"\n  Loading data...")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"  Train : {len(train_df):,} samples")
    print(f"  Test  : {len(test_df):,} samples")

    X_train_text = train_df[TEXT_COL]
    y_train      = train_df[LABEL_COL].values
    X_test_text  = test_df[TEXT_COL]
    y_test       = test_df[LABEL_COL].values

    # ── 2. Build Feature Matrices ──────────────────────────────────────────────
    from src.features import StylometricTransformer
    import numpy as np

    print("\n  Building TRAIN feature matrix (stylometric only)...")
    st = StylometricTransformer()
    X_train = st.fit_transform(X_train_text)
    tfidf = None

    print("\n  Building TEST feature matrix (stylometric only)...")
    X_test = st.transform(X_test_text)

    # ── 3. Scale Features ──────────────────────────────────────────────────────
    # MaxAbsScaler: scales to [-1, 1] without centering
    # Critical: preserves sparsity of the TF-IDF block
    print("\n  Scaling features (MaxAbsScaler)...")
    scaler  = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("  ✓ Scaling complete")

    # ── 4. Train & Evaluate All Models ────────────────────────────────────────
    print("\n\n📊 Training & Evaluating Models")
    print("=" * 55)

    models  = get_models()
    results = []

    for name, model in models.items():
        print(f"\n  ⏳ Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  ✓ Trained in {elapsed:.1f}s")

        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

    # ── 5. Select Best Model ───────────────────────────────────────────────────
    # We rank by F1-weighted but break ties using lowest FPR.
    # Low FPR = we don't falsely accuse real humans of being AI.
    print("\n\n🏆 Model Comparison")
    print("=" * 55)
    print(f"\n  {'Model':<25} {'F1-Weighted':>12} {'AUC':>8} {'FPR':>8}")
    print(f"  {'─'*25} {'─'*12} {'─'*8} {'─'*8}")
    for r in results:
        print(f"  {r['name']:<25} {r['f1_weighted']:>12.4f} "
              f"{r['auc']:>8.4f} {r['fpr']:>8.4f}")

    best = max(results, key=lambda r: (r["f1_weighted"], -r["fpr"]))
    print(f"\n  ✅ Best model: {best['name']}")
    print(f"     F1-Weighted : {best['f1_weighted']:.4f}")
    print(f"     ROC-AUC     : {best['auc']:.4f}")
    print(f"     FPR         : {best['fpr']:.4f}")

    # ── 6. Serialize Artifacts ─────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path  = os.path.join(MODELS_DIR, "best_model.joblib")
    tfidf_path  = os.path.join(MODELS_DIR, "tfidf.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    meta_path   = os.path.join(MODELS_DIR, "model_meta.joblib")

    joblib.dump(best["model"], model_path)
    joblib.dump(tfidf,         tfidf_path)
    joblib.dump(scaler,        scaler_path)

    # Save metadata so the app knows what it's loading
    meta = {
        "model_name":    best["name"],
        "f1_weighted":   best["f1_weighted"],
        "auc":           best["auc"],
        "fpr":           best["fpr"],
        "feature_names": [
            "burstiness", "avg_sentence_length", "avg_word_length",
            "type_token_ratio", "stopword_ratio", "punctuation_density",
            "comma_rate", "semicolon_rate", "exclamation_rate",
            "question_rate", "contraction_rate",
        ],
        "class_labels": {0: "Human", 1: "AI"},
    }
    joblib.dump(meta, meta_path)

    print(f"\n  💾 Saved artifacts:")
    print(f"     {model_path}")
    print(f"     {tfidf_path}")
    print(f"     {scaler_path}")
    print(f"     {meta_path}")
    print(f"\n🎯 Phase 3 complete. Ready for Phase 4: Evaluation.\n")

    return results, best


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()