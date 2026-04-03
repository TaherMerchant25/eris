"""
prepare.py — MILU Eris Challenge Dataset Preparation

The Eris platform calls:  prepare(raw, public, private)

Raw input structure (pre-extracted from HuggingFace by the platform):
  raw/extracted_datasets/
    bengali/
      test/        ← parquet shard(s)
      validation/  ← parquet shard(s)
      dataset_dict.json
    english/
    gujarati/
    ...  (11 language folders)

Outputs:
  public/train.csv            — labelled questions (from validation splits)
  public/test.csv             — unlabelled questions (from test splits, answers removed)
  public/sample_submission.csv — all-A baseline
  private/private.csv         — test split WITH answers (used by grade.py)
"""

from pathlib import Path
import random
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42

# Map raw folder names → ISO 639-3 codes and display names
LANG_MAP = {
    "bengali":   ("ben", "Bengali"),
    "english":   ("eng", "English"),
    "gujarati":  ("guj", "Gujarati"),
    "hindi":     ("hin", "Hindi"),
    "kannada":   ("kan", "Kannada"),
    "malayalam": ("mal", "Malayalam"),
    "marathi":   ("mar", "Marathi"),
    "odia":      ("ory", "Odia"),
    "punjabi":   ("pan", "Punjabi"),
    "tamil":     ("tam", "Tamil"),
    "telugu":    ("tel", "Telugu"),
}

TEST_SAMPLE_SIZE = 8900   # rows to sample from test splits across all languages


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_parquet_dir(folder: Path) -> pd.DataFrame:
    """Read all parquet shards in a directory into one DataFrame."""
    shards = sorted(folder.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def normalise(df: pd.DataFrame, lang_code: str, lang_name: str) -> pd.DataFrame:
    """Flatten MILU columns into the flat schema used by this challenge."""
    df = df.copy()

    # ── options list → flat option_a … option_d columns ──────────────────────
    if "options" in df.columns:
        opts = df["options"].tolist()
        for i, letter in enumerate("abcd"):
            df[f"option_{letter}"] = [
                (row[i] if isinstance(row, (list, tuple)) and len(row) > i else "")
                for row in opts
            ]
        df = df.drop(columns=["options"])

    # ── domain / subject normalisation ───────────────────────────────────────
    # MILU uses "subject" and "category" (or "topic"); map to our schema
    if "category" in df.columns and "domain" not in df.columns:
        df = df.rename(columns={"category": "domain"})
    if "topic" in df.columns and "domain" not in df.columns:
        df = df.rename(columns={"topic": "domain"})
    if "domain" not in df.columns:
        df["domain"] = "General"
    if "subject" not in df.columns:
        df["subject"] = df["domain"]

    # ── answer normalisation → uppercase letter ───────────────────────────────
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.upper()
        # Handle integer answers (0→A, 1→B, 2→C, 3→D)
        int_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
        df["answer"] = df["answer"].replace(int_map)

    # ── language columns ──────────────────────────────────────────────────────
    df["language"]      = lang_code
    df["language_name"] = lang_name

    return df.reset_index(drop=True)


def assign_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["question_id"] = (
        prefix + "_" + df["language"] + "_"
        + df.groupby("language").cumcount().astype(str).str.zfill(5)
    )
    return df


def select_cols(df: pd.DataFrame, include_answer: bool) -> pd.DataFrame:
    base = [
        "question_id", "language", "language_name", "domain", "subject",
        "question", "option_a", "option_b", "option_c", "option_d",
    ]
    if include_answer:
        base.append("answer")
    return df[[c for c in base if c in df.columns]]


# ── Main entry point (called by the Eris platform) ───────────────────────────

def _find_datasets_dir(raw: Path) -> Path:
    """
    Locate the directory that contains per-language subfolders.
    Handles two layouts the platform may provide:
      A)  raw/extracted_datasets/bengali/...
      B)  raw/bengali/...
    """
    candidate = raw / "extracted_datasets"
    if candidate.is_dir():
        return candidate

    # Check if raw itself directly contains language folders
    for name in LANG_MAP:
        if (raw / name).is_dir():
            return raw

    # Last resort: walk one level deep for any directory that holds language folders
    for child in raw.iterdir():
        if child.is_dir():
            for name in LANG_MAP:
                if (child / name).is_dir():
                    return child

    raise FileNotFoundError(
        f"Could not locate language subfolders under {raw}.\n"
        f"Directory tree:\n" +
        "\n".join(f"  {p}" for p in sorted(raw.rglob("*")) if p.is_dir())
    )


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Transform raw MILU parquet files into public/private CSV splits.

    Args:
        raw:     Path to raw input directory  (contains extracted_datasets/)
        public:  Path to write public outputs (train.csv, test.csv, sample_submission.csv)
        private: Path to write private output (private.csv with ground-truth answers)
    """
    raw     = Path(raw)
    public  = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    datasets_dir = _find_datasets_dir(raw)
    print(f"Reading language folders from: {datasets_dir}")

    train_frames = []
    test_frames  = []

    for lang_folder in sorted(datasets_dir.iterdir()):
        if not lang_folder.is_dir():
            continue

        folder_name = lang_folder.name.lower()
        if folder_name not in LANG_MAP:
            print(f"  Skipping unknown folder: {lang_folder.name}")
            continue

        lang_code, lang_name = LANG_MAP[folder_name]

        # ── Validation split → train ──────────────────────────────────────────
        val_dir = lang_folder / "validation"
        if val_dir.exists():
            try:
                df = read_parquet_dir(val_dir)
                df = normalise(df, lang_code, lang_name)
                train_frames.append(df)
                print(f"  train  {lang_folder.name}: {len(df):,} rows")
            except Exception as exc:
                print(f"  train  {lang_folder.name}: SKIPPED — {exc}")
        else:
            print(f"  train  {lang_folder.name}: no validation/ dir, skipping")

        # ── Test split → test / private ───────────────────────────────────────
        test_dir = lang_folder / "test"
        if test_dir.exists():
            try:
                df = read_parquet_dir(test_dir)
                df = normalise(df, lang_code, lang_name)
                test_frames.append(df)
                print(f"  test   {lang_folder.name}: {len(df):,} rows")
            except Exception as exc:
                print(f"  test   {lang_folder.name}: SKIPPED — {exc}")
        else:
            print(f"  test   {lang_folder.name}: no test/ dir, skipping")

    # ── Guard against empty frames ────────────────────────────────────────────
    if not train_frames:
        raise RuntimeError(
            "No validation splits found. Check that language folders contain a "
            f"'validation/' subdirectory with parquet files. Searched: {datasets_dir}"
        )
    if not test_frames:
        raise RuntimeError(
            "No test splits found. Check that language folders contain a "
            f"'test/' subdirectory with parquet files. Searched: {datasets_dir}"
        )

    # ── Build train split ─────────────────────────────────────────────────────
    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_df = assign_ids(train_df, "train")
    select_cols(train_df, include_answer=True).to_csv(
        public / "train.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\ntrain.csv          → {len(train_df):,} rows")

    # ── Build test / private splits ───────────────────────────────────────────
    test_full_df = pd.concat(test_frames, ignore_index=True)

    # Stratified sample preserving language proportions
    test_df = (
        test_full_df
        .groupby("language", group_keys=False)
        .apply(lambda g: g.sample(
            max(1, round(TEST_SAMPLE_SIZE * len(g) / len(test_full_df))),
            random_state=SEED,
        ))
        .reset_index(drop=True)
    )
    test_df = assign_ids(test_df, "test")

    # Public test (no answers)
    select_cols(test_df, include_answer=False).to_csv(
        public / "test.csv", index=False, encoding="utf-8-sig"
    )
    print(f"test.csv           → {len(test_df):,} rows")

    # Private (with answers — used only by grade.py)
    select_cols(test_df, include_answer=True).to_csv(
        private / "private.csv", index=False, encoding="utf-8-sig"
    )
    print(f"private/private.csv → {len(test_df):,} rows")

    # Sample submission (all-A baseline)
    sample_sub = pd.DataFrame({
        "question_id": select_cols(test_df, include_answer=False)["question_id"],
        "answer": "A",
    })
    sample_sub.to_csv(public / "sample_submission.csv", index=False)
    print(f"sample_submission.csv → {len(sample_sub):,} rows")

    print("\nLanguage distribution (test):")
    print(test_df["language"].value_counts().to_string())
    print("\nDone.")


# ── Local testing entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    raw_path     = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./raw")
    public_path  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./dataset/public")
    private_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./dataset/private")
    prepare(raw_path, public_path, private_path)
