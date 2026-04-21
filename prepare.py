"""
prepare.py — Zero-Shot Cross-Lingual Transfer Challenge

Eris platform calls:  prepare(raw, public, private)

Challenge framing
─────────────────
  SEEN   languages → train.csv (WITH answers, agents use for few-shot/finetuning)
  UNSEEN languages → test.csv  (WITHOUT answers, agents predict these)

Private ground truth
────────────────────
  private/answers.csv  —  ONLY  question_id, answer
  (grade.py reads this; no other columns needed)

Raw input layout
────────────────
  raw/extracted_datasets/<lang_folder>/
    dataset_dict.json
    validation/   data-*.arrow  ← seen languages use this
    test/         data-*.arrow  ← unseen languages use this
"""

from pathlib import Path
import random
import pandas as pd
from datasets import load_from_disk

# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42

SEEN_LANGS   = {"ben", "eng", "hin", "mar", "tam", "tel"}
UNSEEN_LANGS = {"guj", "kan", "mal", "ory", "pan"}

_LANG_LOOKUP = {
    "ben": ("ben", "Bengali"),   "bengali":   ("ben", "Bengali"),
    "eng": ("eng", "English"),   "english":   ("eng", "English"),
    "guj": ("guj", "Gujarati"),  "gujarati":  ("guj", "Gujarati"),
    "hin": ("hin", "Hindi"),     "hindi":     ("hin", "Hindi"),
    "kan": ("kan", "Kannada"),   "kannada":   ("kan", "Kannada"),
    "mal": ("mal", "Malayalam"), "malayalam": ("mal", "Malayalam"),
    "mar": ("mar", "Marathi"),   "marathi":   ("mar", "Marathi"),
    "ory": ("ory", "Odia"),      "odia":      ("ory", "Odia"),
    "pan": ("pan", "Punjabi"),   "punjabi":   ("pan", "Punjabi"),
    "tam": ("tam", "Tamil"),     "tamil":     ("tam", "Tamil"),
    "tel": ("tel", "Telugu"),    "telugu":    ("tel", "Telugu"),
}


def _lang_from_folder(name: str) -> tuple[str, str]:
    key = name.lower().strip()
    if key in _LANG_LOOKUP:
        return _LANG_LOOKUP[key]
    for k, v in _LANG_LOOKUP.items():
        if key.startswith(k) or k.startswith(key):
            return v
    return (key, name.title())


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_split(lang_folder: Path, split_name: str) -> pd.DataFrame | None:
    """Load one split from a language folder. Returns None if unavailable."""
    # Strategy A: HuggingFace DatasetDict (dataset_dict.json present)
    if (lang_folder / "dataset_dict.json").exists():
        try:
            ds_dict = load_from_disk(str(lang_folder))
            if split_name in ds_dict:
                df = ds_dict[split_name].to_pandas()
                print(f"    {lang_folder.name}/{split_name}: {len(df):,} rows  "
                      f"columns={list(df.columns)}")
                return df
            else:
                print(f"    {lang_folder.name}: split '{split_name}' not in DatasetDict "
                      f"(available: {list(ds_dict.keys())})")
                return None
        except Exception as exc:
            print(f"    {lang_folder.name}: load_from_disk failed — {exc}")

    # Strategy B: bare split subdirectory with Arrow/Parquet shards
    split_dir = lang_folder / split_name
    if not split_dir.exists():
        print(f"    {lang_folder.name}: no {split_name}/ directory")
        return None

    # Try load_from_disk on the split directory itself
    try:
        ds = load_from_disk(str(split_dir))
        df = ds.to_pandas()
        print(f"    {lang_folder.name}/{split_name}: {len(df):,} rows  columns={list(df.columns)}")
        return df
    except Exception:
        pass

    # Try raw parquet shards
    shards = sorted(split_dir.rglob("*.parquet"))
    if shards:
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        print(f"    {lang_folder.name}/{split_name}: {len(df):,} rows (parquet)  "
              f"columns={list(df.columns)}")
        return df

    # Try raw Arrow shards via pyarrow
    arrow_files = sorted(split_dir.rglob("*.arrow"))
    if arrow_files:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        tables = []
        for f in arrow_files:
            with pa.memory_map(str(f), "r") as src:
                tables.append(ipc.open_file(src).read_all())
        df = pa.concat_tables(tables).to_pandas()
        print(f"    {lang_folder.name}/{split_name}: {len(df):,} rows (arrow)  "
              f"columns={list(df.columns)}")
        return df

    print(f"    {lang_folder.name}/{split_name}: no readable data files found")
    return None


# ── Schema normalisation ──────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame, lang_code: str, lang_name: str) -> pd.DataFrame:
    """Flatten raw MILU columns into the challenge schema."""
    df = df.copy()

    # options list → flat option_a … option_d
    if "options" in df.columns:
        opts = df["options"].tolist()
        for i, letter in enumerate("abcd"):
            df[f"option_{letter}"] = [
                row[i] if isinstance(row, (list, tuple)) and len(row) > i else ""
                for row in opts
            ]
        df = df.drop(columns=["options"])

    # domain / subject
    for src in ("category", "topic"):
        if src in df.columns and "domain" not in df.columns:
            df = df.rename(columns={src: "domain"})
    if "domain" not in df.columns:
        df["domain"] = "General"
    if "subject" not in df.columns:
        df["subject"] = df["domain"]

    # answer → uppercase single letter (A/B/C/D)
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.upper()
        # Handle integer-encoded answers: 0→A, 1→B, 2→C, 3→D
        df["answer"] = df["answer"].replace({"0": "A", "1": "B", "2": "C", "3": "D"})
    else:
        print(f"    WARNING: no 'answer' column found for {lang_code}. "
              f"Available columns: {list(df.columns)}")

    df["language"]      = lang_code
    df["language_name"] = lang_name

    return df.reset_index(drop=True)


def _assign_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["question_id"] = (
        prefix + "_" + df["language"] + "_"
        + df.groupby("language").cumcount().astype(str).str.zfill(5)
    )
    return df


# ── Column selection ──────────────────────────────────────────────────────────

QUESTION_COLS = [
    "question_id", "language", "language_name", "domain", "subject",
    "question", "option_a", "option_b", "option_c", "option_d",
]

def _public_question_cols(df: pd.DataFrame) -> pd.DataFrame:
    """All question columns, NO answer — for test.csv."""
    return df[[c for c in QUESTION_COLS if c in df.columns]]

def _public_train_cols(df: pd.DataFrame) -> pd.DataFrame:
    """All question columns plus answer — for train.csv."""
    cols = QUESTION_COLS + ["answer"]
    present = [c for c in cols if c in df.columns]
    if "answer" not in present:
        raise RuntimeError(
            f"'answer' column missing from training data. "
            f"Available columns: {list(df.columns)}"
        )
    return df[present]

def _answers_only(df: pd.DataFrame) -> pd.DataFrame:
    """ONLY question_id + answer — for private/answers.csv."""
    if "answer" not in df.columns:
        raise RuntimeError(
            f"'answer' column missing — cannot write answers.csv. "
            f"Available columns: {list(df.columns)}"
        )
    return df[["question_id", "answer"]]


# ── Main entry point ──────────────────────────────────────────────────────────

def prepare(raw: Path, public: Path, private: Path) -> None:
    raw     = Path(raw)
    public  = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # Show full directory tree for diagnostics
    print(f"raw dir: {raw}")
    all_dirs = sorted(p for p in raw.rglob("*") if p.is_dir())
    print("Directory tree:")
    for d in all_dirs[:80]:
        print(f"  {d.relative_to(raw)}")
    if len(all_dirs) > 80:
        print(f"  ... ({len(all_dirs)} total dirs)")
    print()

    # Find the directory that holds language subfolders
    datasets_dir = raw / "extracted_datasets"
    if not (datasets_dir.is_dir() and any(datasets_dir.iterdir())):
        datasets_dir = raw

    print(f"Scanning: {datasets_dir}")
    lang_folders = sorted(p for p in datasets_dir.iterdir() if p.is_dir())
    print(f"Language folders: {[p.name for p in lang_folders]}\n")

    train_frames: list[pd.DataFrame] = []
    test_frames:  list[pd.DataFrame] = []

    for lang_folder in lang_folders:
        lang_code, lang_name = _lang_from_folder(lang_folder.name)

        if lang_code in SEEN_LANGS:
            df = _load_split(lang_folder, "validation")
            if df is not None:
                df = _normalise(df, lang_code, lang_name)
                train_frames.append(df)

        elif lang_code in UNSEEN_LANGS:
            df = _load_split(lang_folder, "test")
            if df is None:
                df = _load_split(lang_folder, "validation")
            if df is not None:
                df = _normalise(df, lang_code, lang_name)
                test_frames.append(df)

        else:
            print(f"  {lang_folder.name} ({lang_code}): not SEEN or UNSEEN — skipping")

    if not train_frames:
        raise RuntimeError("No seen-language (train) data loaded. Check directory tree above.")
    if not test_frames:
        raise RuntimeError("No unseen-language (test) data loaded. Check directory tree above.")

    # ── train.csv — seen languages WITH answers ───────────────────────────────
    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_df = _assign_ids(train_df, "train")
    _public_train_cols(train_df).to_csv(public / "train.csv", index=False, encoding="utf-8-sig")
    print(f"\ntrain.csv      → {len(train_df):,} rows  "
          f"languages={sorted(train_df['language'].unique())}  "
          f"has_answer={'answer' in train_df.columns}")

    # ── test.csv — unseen languages WITHOUT answers ───────────────────────────
    test_full = pd.concat(test_frames, ignore_index=True)
    test_df = (
        test_full
        .groupby("language", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 900), random_state=SEED))
        .reset_index(drop=True)
    )
    test_df = _assign_ids(test_df, "test")
    _public_question_cols(test_df).to_csv(public / "test.csv", index=False, encoding="utf-8-sig")
    print(f"test.csv       → {len(test_df):,} rows  "
          f"languages={sorted(test_df['language'].unique())}")

    # ── private/answers.csv — ONLY question_id + answer ──────────────────────
    _answers_only(test_df).to_csv(private / "answers.csv", index=False, encoding="utf-8-sig")
    print(f"answers.csv    → {len(test_df):,} rows  columns=[question_id, answer]")

    # ── sample_submission.csv — all-A baseline ────────────────────────────────
    pd.DataFrame({
        "question_id": test_df["question_id"],
        "answer": "A",
    }).to_csv(public / "sample_submission.csv", index=False)
    print(f"sample_submission.csv → {len(test_df):,} rows")

    # Verify answer distribution (catches bad normalization)
    print(f"\nAnswer distribution in answers.csv:")
    print(test_df["answer"].value_counts().sort_index().to_string())
    print(f"\nTrain answer distribution:")
    print(train_df["answer"].value_counts().sort_index().to_string())

    print("\nDone.")


# ── Local testing ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    prepare(
        raw     = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./raw"),
        public  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./dataset/public"),
        private = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./dataset/private"),
    )
