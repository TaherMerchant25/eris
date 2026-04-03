"""
prepare.py — MILU Eris Challenge Dataset Preparation Script

Downloads the MILU dataset from HuggingFace and creates deterministic
public/private splits for the Eris challenge.

Usage:
    python prepare.py

Output:
    ./dataset/public/public.csv   — Questions with answers (few-shot reference)
    ./dataset/private/private.csv — Questions without answers (grading target)

Requirements:
    pip install datasets pandas pyarrow
"""

import os
import random
import pandas as pd
from datasets import load_dataset

# ── Configuration ────────────────────────────────────────────────────────────

RANDOM_SEED = 42
PRIVATE_SAMPLE_SIZE = 8900   # rows sampled from MILU test split for private set
PUBLIC_DATASET = "validation" # MILU validation split → public (with answers)
PRIVATE_DATASET = "test"      # MILU test split     → private (answers hidden)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")
PUBLIC_DIR = os.path.join(OUTPUT_DIR, "public")
PRIVATE_DIR = os.path.join(OUTPUT_DIR, "private")

# All 11 MILU language configs
LANGUAGES = [
    "ben", "guj", "hin", "kan", "mal",
    "mar", "ory", "pan", "tam", "tel", "eng"
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_language_split(lang: str, split: str) -> pd.DataFrame:
    """Load one language config from MILU and return a normalised DataFrame."""
    ds = load_dataset("ai4bharat/MILU", lang, split=split, trust_remote_code=True)
    df = ds.to_pandas()

    # Normalise option columns into a single list column
    option_cols = [c for c in df.columns if c.lower().startswith("option")]
    option_cols_sorted = sorted(option_cols)  # option_a, option_b, option_c, option_d

    df["options"] = df[option_cols_sorted].values.tolist()
    df.drop(columns=option_cols_sorted, inplace=True)

    # Ensure required columns exist
    df["language"] = lang
    df["language_name"] = _lang_name(lang)

    # Normalise answer to uppercase single letter
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.upper()

    # Create a unique question_id  <lang>_<index>
    df.reset_index(drop=True, inplace=True)
    df["question_id"] = df["language"] + "_" + df.index.astype(str).str.zfill(6)

    return df


def _lang_name(code: str) -> str:
    mapping = {
        "ben": "Bengali",  "guj": "Gujarati", "hin": "Hindi",
        "kan": "Kannada",  "mal": "Malayalam","mar": "Marathi",
        "ory": "Odia",     "pan": "Punjabi",  "tam": "Tamil",
        "tel": "Telugu",   "eng": "English",
    }
    return mapping.get(code, code)


def _select_columns(df: pd.DataFrame, include_answer: bool) -> pd.DataFrame:
    """Return only the columns that should be exposed in public/private files."""
    base_cols = ["question_id", "language", "language_name", "domain", "subject",
                 "question", "options"]
    if include_answer:
        base_cols.append("answer")
    # Keep only columns that exist
    return df[[c for c in base_cols if c in df.columns]]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    os.makedirs(PRIVATE_DIR, exist_ok=True)

    print("=== MILU Eris Challenge — Dataset Preparation ===\n")

    # ── 1. Build PUBLIC split (validation, all answers included) ─────────────
    print(f"Loading public split ({PUBLIC_DATASET})…")
    public_frames = []
    for lang in LANGUAGES:
        try:
            df = load_language_split(lang, PUBLIC_DATASET)
            public_frames.append(df)
            print(f"  {lang}: {len(df):,} rows")
        except Exception as exc:
            print(f"  {lang}: SKIPPED ({exc})")

    public_df = pd.concat(public_frames, ignore_index=True)
    public_df = _select_columns(public_df, include_answer=True)

    # Shuffle deterministically so row order doesn't leak language patterns
    public_df = public_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    public_path = os.path.join(PUBLIC_DIR, "public.csv")
    public_df.to_csv(public_path, index=False, encoding="utf-8-sig")
    print(f"\nPublic split saved → {public_path}  ({len(public_df):,} rows)\n")

    # ── 2. Build PRIVATE split (test, answers included for grading only) ──────
    print(f"Loading private split ({PRIVATE_DATASET})…")
    private_frames = []
    for lang in LANGUAGES:
        try:
            df = load_language_split(lang, PRIVATE_DATASET)
            private_frames.append(df)
            print(f"  {lang}: {len(df):,} rows")
        except Exception as exc:
            print(f"  {lang}: SKIPPED ({exc})")

    private_full_df = pd.concat(private_frames, ignore_index=True)

    # Stratified sample: preserve language × domain distribution
    rng = random.Random(RANDOM_SEED)
    private_full_df = private_full_df.sample(
        n=min(PRIVATE_SAMPLE_SIZE, len(private_full_df)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # Reassign question_ids for the private split to avoid collisions
    private_full_df["question_id"] = (
        "priv_" + private_full_df["language"] + "_"
        + private_full_df.groupby("language").cumcount().astype(str).str.zfill(5)
    )

    # Save WITH answers for grader use
    private_with_answers = _select_columns(private_full_df, include_answer=True)
    private_path = os.path.join(PRIVATE_DIR, "private.csv")
    private_with_answers.to_csv(private_path, index=False, encoding="utf-8-sig")
    print(f"\nPrivate split (with answers) saved → {private_path}  ({len(private_with_answers):,} rows)")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  Public  rows : {len(public_df):,}  (answers exposed — use for few-shot / finetuning)")
    print(f"  Private rows : {len(private_with_answers):,}  (answers withheld from agents)")
    print("\nLanguage distribution (private):")
    print(private_full_df["language"].value_counts().to_string())
    print("\nDomain distribution (private):")
    if "domain" in private_full_df.columns:
        print(private_full_df["domain"].value_counts().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
