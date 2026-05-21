"""
ingest.py — CallMyBluff Phase 1
Loads raw dataset, inspects it, cleans it, and produces
stratified train/test splits saved to data/processed/.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
RANDOM_STATE  = 42
TEST_SIZE     = 0.2   # 80/20 split

# Expected column names after normalization
TEXT_COL  = "text"
LABEL_COL = "label"   # 0 = Human, 1 = AI


# ── Loaders (one per dataset format) ───────────────────────────────────────────
def load_daigt(filepath: str) -> pd.DataFrame:
    """
    Loads the DAIGT v4 Kaggle CSV.
    Kaggle schema: 'text', 'generated' (0=human, 1=AI)
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={"generated": LABEL_COL})
    df = df[[TEXT_COL, LABEL_COL]].copy()
    return df


def load_hc3(split: str = "train") -> pd.DataFrame:
    """
    Loads HC3 from HuggingFace datasets.
    Requires: pip install datasets
    """
    from datasets import load_dataset
    ds = load_dataset("Hello-SimpleAI/HC3", "all")[split]
    records = []
    for row in ds:
        for ans in row["human_answers"]:
            if ans.strip():
                records.append({TEXT_COL: ans, LABEL_COL: 0})
        for ans in row["chatgpt_answers"]:
            if ans.strip():
                records.append({TEXT_COL: ans, LABEL_COL: 1})
    return pd.DataFrame(records)


# ── Inspection ─────────────────────────────────────────────────────────────────
def inspect(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Prints a clean summary of the loaded dataframe."""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"\n  Label distribution:")
    counts = df[LABEL_COL].value_counts().sort_index()
    for label, count in counts.items():
        tag = "Human" if label == 0 else "AI   "
        pct = count / len(df) * 100
        print(f"    [{label}] {tag} → {count:,} ({pct:.1f}%)")
    print(f"\n  Text length stats (characters):")
    lengths = df[TEXT_COL].str.len()
    print(f"    Min    : {lengths.min():,}")
    print(f"    Max    : {lengths.max():,}")
    print(f"    Mean   : {lengths.mean():,.0f}")
    print(f"    Median : {lengths.median():,.0f}")
    print(f"\n  Missing values: {df.isnull().sum().to_dict()}")
    print(f"{'='*55}\n")


# ── Cleaning ───────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame, min_chars: int = 10) -> pd.DataFrame:
    """
    Removes nulls, duplicates, and texts too short for
    meaningful stylometric analysis.
    """
    original_len = len(df)

    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].str.len() >= min_chars]
    df = df.drop_duplicates(subset=[TEXT_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df.reset_index(drop=True)

    print(f"  Cleaning: {original_len:,} → {len(df):,} rows "
          f"(removed {original_len - len(df):,})")
    return df


# ── Splitting ──────────────────────────────────────────────────────────────────
def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 train/test split.
    Stratification ensures class balance is preserved in both splits.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL]   # ← critical: preserves class ratio
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"\n  ✅ Saved → {train_path}  ({len(train_df):,} rows)")
    print(f"  ✅ Saved → {test_path}   ({len(test_df):,} rows)")
    return train_df, test_df


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    combined_path = os.path.join(RAW_DIR, "combined_dataset.csv")
    print("\n📂 Loading combined dataset...")
    df = pd.read_csv(combined_path)
    df = df[["text", "label"]].copy()
    dataset_name = "Combined (DAIGT Human + Groq AI)"

    inspect(df, name=dataset_name)
    df = clean(df)
    inspect(df, name=f"{dataset_name} (cleaned)")
    train_df, test_df = split_and_save(df)

    print("\n🎯 Phase 1 complete. Ready for Phase 2: Feature Engineering.\n")