"""
prepare.py — MILU Eris Challenge Dataset Preparation

Downloads MILU from HuggingFace and writes three files to ./dataset/public/:
  train.csv          — labelled questions (MILU validation split, all 11 languages)
  test.csv           — unlabelled questions (stratified sample from MILU test split)
  sample_submission.csv — all-A baseline in the required submission format

Usage:
    python prepare.py

Requirements:
    pip install datasets pandas pyarrow
"""

import os
import random
import pandas as pd
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
TEST_SAMPLE_SIZE = 8900   # rows sampled from MILU test split

PUBLIC_DIR   = os.path.join(os.path.dirname(__file__), "dataset", "public")
PRIVATE_DIR  = os.path.join(os.path.dirname(__file__), "dataset", "private")

LANGUAGES = [
    "ben", "guj", "hin", "kan", "mal",
    "mar", "ory", "pan", "tam", "tel", "eng",
]

LANG_NAMES = {
    "ben": "Bengali",  "guj": "Gujarati", "hin": "Hindi",
    "kan": "Kannada",  "mal": "Malayalam","mar": "Marathi",
    "ory": "Odia",     "pan": "Punjabi",  "tam": "Tamil",
    "tel": "Telugu",   "eng": "English",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_milu(lang: str, split: str) -> pd.DataFrame:
    """Load one language config from MILU; return a normalised flat DataFrame."""
    ds = load_dataset("ai4bharat/MILU", lang, split=split, trust_remote_code=True)
    df = ds.to_pandas()

    # Identify option columns and rename to option_a … option_d
    option_cols = sorted(c for c in df.columns if c.lower().startswith("option"))
    rename = {col: f"option_{chr(ord('a') + i)}" for i, col in enumerate(option_cols)}
    df = df.rename(columns=rename)

    df["language"]      = lang
    df["language_name"] = LANG_NAMES.get(lang, lang)

    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.upper()

    df = df.reset_index(drop=True)
    df["question_id"] = lang + "_" + df.index.astype(str).str.zfill(6)

    return df


def select_cols(df: pd.DataFrame, include_answer: bool) -> pd.DataFrame:
    base = ["question_id", "language", "language_name", "domain", "subject",
            "question", "option_a", "option_b", "option_c", "option_d"]
    if include_answer:
        base.append("answer")
    return df[[c for c in base if c in df.columns]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PUBLIC_DIR,  exist_ok=True)
    os.makedirs(PRIVATE_DIR, exist_ok=True)

    print("=== MILU — Dataset Preparation ===\n")

    # ── Train split (MILU validation, labelled) ───────────────────────────────
    print("Loading train split (MILU validation)…")
    train_frames = []
    for lang in LANGUAGES:
        try:
            df = load_milu(lang, "validation")
            train_frames.append(df)
            print(f"  {lang}: {len(df):,} rows")
        except Exception as exc:
            print(f"  {lang}: SKIPPED — {exc}")

    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_out = select_cols(train_df, include_answer=True)
    train_out.to_csv(os.path.join(PUBLIC_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    print(f"\ntrain.csv → {len(train_out):,} rows\n")

    # ── Test split (MILU test, labels held out) ───────────────────────────────
    print("Loading test split (MILU test)…")
    test_frames = []
    for lang in LANGUAGES:
        try:
            df = load_milu(lang, "test")
            test_frames.append(df)
            print(f"  {lang}: {len(df):,} rows")
        except Exception as exc:
            print(f"  {lang}: SKIPPED — {exc}")

    test_full_df = pd.concat(test_frames, ignore_index=True)

    # Stratified sample preserving language distribution
    test_df = (
        test_full_df
        .groupby("language", group_keys=False)
        .apply(lambda g: g.sample(
            max(1, round(TEST_SAMPLE_SIZE * len(g) / len(test_full_df))),
            random_state=SEED,
        ))
        .reset_index(drop=True)
    )
    # Reassign question_ids for the test partition
    test_df["question_id"] = (
        "test_" + test_df["language"] + "_"
        + test_df.groupby("language").cumcount().astype(str).str.zfill(5)
    )

    # Save private copy (with answers) for grader
    private_out = select_cols(test_df, include_answer=True)
    private_out.to_csv(os.path.join(PRIVATE_DIR, "private.csv"), index=False, encoding="utf-8-sig")

    # Public test file — no answer column
    test_out = select_cols(test_df, include_answer=False)
    test_out.to_csv(os.path.join(PUBLIC_DIR, "test.csv"), index=False, encoding="utf-8-sig")
    print(f"\ntest.csv         → {len(test_out):,} rows")
    print(f"private.csv      → {len(private_out):,} rows (answers withheld from agents)\n")

    # ── Sample submission (all-A baseline) ────────────────────────────────────
    sample_sub = pd.DataFrame({
        "question_id": test_out["question_id"],
        "answer": "A",
    })
    sample_sub.to_csv(os.path.join(PUBLIC_DIR, "sample_submission.csv"), index=False)
    print(f"sample_submission.csv → {len(sample_sub):,} rows (all-A baseline)\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("Language distribution (test):")
    print(test_df["language"].value_counts().to_string())
    if "domain" in test_df.columns:
        print("\nDomain distribution (test):")
        print(test_df["domain"].value_counts().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
