"""
src/features.py — CallMyBluff Phase 2
Stylometric feature engineering pipeline.
Extracts statistical writing-style signals from raw text.
"""

import re
import string
import numpy as np
import pandas as pd
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix

STOPWORDS = set(stopwords.words("english"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Raw Feature Extraction Functions
# Each function takes a single string and returns a scalar float.
# Keeping them as pure functions makes them easy to test individually.
# ══════════════════════════════════════════════════════════════════════════════

def get_sentence_lengths(text: str) -> list[int]:
    """Tokenize into sentences and return list of word-counts per sentence."""
    sentences = sent_tokenize(text)
    # Filter out empty/trivial sentences (less than 2 chars)
    sentences = [s for s in sentences if len(s.strip()) > 2]
    if not sentences:
        return [0]
    return [len(word_tokenize(s)) for s in sentences]


def burstiness(text: str) -> float:
    """
    Standard deviation of sentence lengths.
    HIGH value = human (varied rhythm)
    LOW value  = AI (robotic uniformity)
    """
    lengths = get_sentence_lengths(text)
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths))


def avg_sentence_length(text: str) -> float:
    """Mean number of words per sentence."""
    lengths = get_sentence_lengths(text)
    return float(np.mean(lengths))


def avg_word_length(text: str) -> float:
    """
    Mean character length of words (excluding punctuation tokens).
    AI tends to use slightly longer, more formal words.
    """
    words = word_tokenize(text)
    words = [w for w in words if w.isalpha()]
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def type_token_ratio(text: str) -> float:
    """
    TTR = unique_words / total_words
    Measures vocabulary diversity.
    Ranges from 0.0 (no diversity) to 1.0 (every word unique).
    Note: sensitive to text length — best used on similarly-sized texts.
    """
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def stopword_ratio(text: str) -> float:
    """
    Proportion of tokens that are stopwords.
    Human writing tends to have higher stopword density
    due to natural conversational connectors.
    """
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    if not words:
        return 0.0
    sw_count = sum(1 for w in words if w in STOPWORDS)
    return sw_count / len(words)


def punctuation_density(text: str) -> float:
    """
    Ratio of punctuation characters to total characters.
    """
    if not text:
        return 0.0
    punct_count = sum(1 for ch in text if ch in string.punctuation)
    return punct_count / len(text)


def comma_rate(text: str) -> float:
    """Commas per 100 words — humans use more comma-spliced clauses."""
    words = word_tokenize(text)
    word_count = len([w for w in words if w.isalpha()])
    if word_count == 0:
        return 0.0
    return (text.count(",") / word_count) * 100


def semicolon_rate(text: str) -> float:
    """Semicolons per 100 words — a surprisingly strong human signal."""
    words = word_tokenize(text)
    word_count = len([w for w in words if w.isalpha()])
    if word_count == 0:
        return 0.0
    return (text.count(";") / word_count) * 100


def exclamation_rate(text: str) -> float:
    """Exclamation marks per 100 words — emotional expressiveness marker."""
    words = word_tokenize(text)
    word_count = len([w for w in words if w.isalpha()])
    if word_count == 0:
        return 0.0
    return (text.count("!") / word_count) * 100


def question_rate(text: str) -> float:
    """Question marks per 100 words — rhetorical device usage."""
    words = word_tokenize(text)
    word_count = len([w for w in words if w.isalpha()])
    if word_count == 0:
        return 0.0
    return (text.count("?") / word_count) * 100


def contraction_count(text: str) -> int:
    """
    Raw count of English contractions (don't, I've, they're...).
    AI models historically avoid contractions in formal writing.
    """
    pattern = r"\b\w+'\w+\b"
    return len(re.findall(pattern, text))


def contraction_rate(text: str) -> float:
    """Contractions per 100 words."""
    words = word_tokenize(text)
    word_count = len([w for w in words if w.isalpha()])
    if word_count == 0:
        return 0.0
    return (contraction_count(text) / word_count) * 100


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Master Feature Extractor
# Applies all functions above to every row in a DataFrame.
# ══════════════════════════════════════════════════════════════════════════════

# Registry: feature_name → function
# Add new features here without touching any other code.
FEATURE_FUNCTIONS = {
    "burstiness":          burstiness,
    "avg_sentence_length": avg_sentence_length,
    "avg_word_length":     avg_word_length,
    "type_token_ratio":    type_token_ratio,
    "stopword_ratio":      stopword_ratio,
    "punctuation_density": punctuation_density,
    "comma_rate":          comma_rate,
    "semicolon_rate":      semicolon_rate,
    "exclamation_rate":    exclamation_rate,
    "question_rate":       question_rate,
    "contraction_rate":    contraction_rate,
}


def extract_stylometric_features(texts: pd.Series) -> pd.DataFrame:
    """
    Applies all feature functions to a Series of texts.
    Returns a DataFrame where each column is one stylometric feature.
    """
    print(f"  Extracting stylometric features from {len(texts):,} texts...")
    feature_df = pd.DataFrame(index=texts.index)

    for name, fn in FEATURE_FUNCTIONS.items():
        feature_df[name] = texts.apply(fn)
        print(f"    ✓ {name}")

    print(f"  Done. Feature matrix shape: {feature_df.shape}")
    return feature_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Sklearn-Compatible Transformer
# Wraps the extractor above so it plugs directly into sklearn Pipelines.
# ══════════════════════════════════════════════════════════════════════════════

class StylometricTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for stylometric features.
    Input : pandas Series or list of strings
    Output: numpy array of shape (n_samples, n_features)
    """

    def fit(self, X, y=None):
        # Stateless transformer — nothing to learn from training data
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            texts = X.reset_index(drop=True)
        else:
            texts = pd.Series(X)
        df = extract_stylometric_features(texts)
        return df.values.astype(np.float64)

    def get_feature_names_out(self):
        return list(FEATURE_FUNCTIONS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Combined Feature Matrix Builder
# Merges stylometric features + TF-IDF into one unified sparse matrix.
# ══════════════════════════════════════════════════════════════════════════════

def build_tfidf(max_features: int = 3000) -> TfidfVectorizer:
    """
    Constrained TF-IDF vectorizer.
    max_features=3000 keeps it lightweight — the heavy lifting is done
    by our stylometric features, not raw word frequencies.
    We use character n-grams (analyzer='char_wb') as a secondary signal
    because AI models have subtle subword pattern differences.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),       # unigrams, bigrams, trigrams
        analyzer="word",
        sublinear_tf=True,        # log-scale TF dampens extreme frequencies
        min_df=3,                 # ignore very rare terms
        strip_accents="unicode",
        decode_error="replace",
    )


def build_feature_matrix(
    texts: pd.Series,
    tfidf: TfidfVectorizer = None,
    fit: bool = True,
) -> tuple:
    """
    Builds the full feature matrix by combining:
      - Stylometric features (dense, ~11 columns)
      - TF-IDF features     (sparse, ~3000 columns)

    Args:
        texts : pd.Series of raw text strings
        tfidf : a TfidfVectorizer instance (pass fitted one for test set)
        fit   : if True, fit the TF-IDF on this data (use for training set)

    Returns:
        X     : sparse matrix of shape (n_samples, n_stylo + n_tfidf)
        tfidf : the (fitted) TfidfVectorizer — save this for the test set
    """
    # --- Stylometric block (dense) ---
    stylo_array = StylometricTransformer().fit_transform(texts)
    stylo_sparse = csr_matrix(stylo_array)

    # --- TF-IDF block (sparse) ---
    if tfidf is None:
        tfidf = build_tfidf()
    if fit:
        tfidf_sparse = tfidf.fit_transform(texts)
    else:
        tfidf_sparse = tfidf.transform(texts)

    # --- Horizontal stack ---
    X = hstack([stylo_sparse, tfidf_sparse])
    print(f"\n  ✅ Combined feature matrix: {X.shape} "
          f"({stylo_array.shape[1]} stylometric + "
          f"{tfidf_sparse.shape[1]} TF-IDF)")
    return X, tfidf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Standalone Test Runner
# Run this file directly to validate features on the processed data.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    TRAIN_PATH = os.path.join("data", "processed", "train.csv")

    print("\n🔬 CallMyBluff — Phase 2: Feature Engineering Validation")
    print("=" * 60)

    # Load training data
    df = pd.read_csv(TRAIN_PATH)
    print(f"\n  Loaded {len(df):,} training samples")

    # Quick sanity check on a single text
    sample_human = df[df["label"] == 0]["text"].iloc[0]
    sample_ai    = df[df["label"] == 1]["text"].iloc[0]

    print("\n── Single-text feature comparison ──────────────────────────")
    print(f"{'Feature':<25} {'Human':>10} {'AI':>10}")
    print("-" * 47)
    for name, fn in FEATURE_FUNCTIONS.items():
        h_val = fn(sample_human)
        a_val = fn(sample_ai)
        print(f"  {name:<23} {h_val:>10.4f} {a_val:>10.4f}")

    # Build full feature matrix on a 500-sample subset (speed)
    print("\n── Building feature matrix (500-sample subset) ─────────────")
    subset = df.sample(500, random_state=42)
    X, tfidf = build_feature_matrix(subset["text"], fit=True)

    print("\n── Per-feature averages by class ────────────────────────────")
    stylo_df = extract_stylometric_features(subset["text"].reset_index(drop=True))
    stylo_df["label"] = subset["label"].values
    print(stylo_df.groupby("label").mean().T.rename(
        columns={0: "Human (avg)", 1: "AI (avg)"}
    ).to_string())

    print("\n\n✅ Phase 2 validation complete. Ready for Phase 3: Training.\n")