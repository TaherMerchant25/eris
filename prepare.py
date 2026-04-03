"""
prepare.py — MILU Eris Challenge Dataset Preparation

The Eris platform calls:  prepare(raw, public, private)

Raw input layout (pre-extracted from HuggingFace by the platform):
  raw/
    extracted_datasets/        ← or raw/ directly
      <lang_folder>/           ← any name: "ben", "bengali", etc.
        validation/
          *.parquet
        test/
          *.parquet
        dataset_dict.json

Outputs:
  public/train.csv             — labelled questions (from validation splits)
  public/test.csv              — unlabelled questions (answers removed)
  public/sample_submission.csv — all-A baseline
  private/private.csv          — test split WITH answers (for grade.py)
"""

from pathlib import Path
import random
import pandas as pd
from datasets import load_from_disk


# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42
TEST_SAMPLE_SIZE = 8900

# Lookup table: any folder name variant → (iso_code, display_name)
# Used when we CAN match a folder name; falls back gracefully when we can't.
_KNOWN = {
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
    """Return (iso_code, display_name) for a folder name, with fallback."""
    key = name.lower().strip()
    if key in _KNOWN:
        return _KNOWN[key]
    # Partial match — e.g. "bengali_v2" → "bengali"
    for k, v in _KNOWN.items():
        if key.startswith(k) or k.startswith(key):
            return v
    # Unknown folder: use the name itself as the code
    return (key, name.title())


# ── Split reading ─────────────────────────────────────────────────────────────

def _read_split(split_dir: Path) -> pd.DataFrame:
    """
    Read a HuggingFace split directory (Arrow or Parquet shards).
    split_dir is e.g.  .../bengali/validation/
    """
    # Prefer datasets.load_from_disk — handles Arrow IPC natively
    try:
        ds = load_from_disk(str(split_dir))
        df = ds.to_pandas()
        print(f"    {split_dir.parent.name}/{split_dir.name}: {len(df):,} rows (arrow)")
        return df
    except Exception:
        pass

    # Fallback: raw parquet shards
    shards = sorted(split_dir.rglob("*.parquet"))
    if shards:
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        print(f"    {split_dir.parent.name}/{split_dir.name}: {len(df):,} rows "
              f"({len(shards)} parquet shard(s))")
        return df

    # Fallback: raw arrow shards via pyarrow
    arrow_files = sorted(split_dir.rglob("*.arrow"))
    if arrow_files:
        import pyarrow.ipc as ipc
        import pyarrow as pa
        tables = []
        for f in arrow_files:
            with pa.memory_map(str(f), "r") as src:
                tables.append(ipc.open_file(src).read_all())
        df = pa.concat_tables(tables).to_pandas()
        print(f"    {split_dir.parent.name}/{split_dir.name}: {len(df):,} rows "
              f"({len(arrow_files)} arrow shard(s))")
        return df

    raise FileNotFoundError(
        f"No readable data files (.arrow or .parquet) found under {split_dir}"
    )


# ── Schema normalisation ──────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame, lang_code: str, lang_name: str) -> pd.DataFrame:
    df = df.copy()

    # options list/sequence → flat option_a … option_d
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

    # answer → uppercase letter
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.upper()
        df["answer"] = df["answer"].replace({"0": "A", "1": "B", "2": "C", "3": "D"})

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


def _select_cols(df: pd.DataFrame, include_answer: bool) -> pd.DataFrame:
    cols = ["question_id", "language", "language_name", "domain", "subject",
            "question", "option_a", "option_b", "option_c", "option_d"]
    if include_answer:
        cols.append("answer")
    return df[[c for c in cols if c in df.columns]]


# ── Directory scanning ────────────────────────────────────────────────────────

def _collect_splits(datasets_dir: Path) -> tuple[list, list]:
    """
    Walk datasets_dir and load validation + test splits for every language folder.
    Supports two strategies:
      A) DatasetDict per language folder (dataset_dict.json present) — load_from_disk
      B) Bare validation/ and test/ subdirs containing Arrow/Parquet shards
    """
    train_frames: list[pd.DataFrame] = []
    test_frames:  list[pd.DataFrame] = []

    lang_folders = sorted(p for p in datasets_dir.iterdir() if p.is_dir())
    print(f"  Language folders found: {[p.name for p in lang_folders]}\n")

    for lang_folder in lang_folders:
        lang_code, lang_name = _lang_from_folder(lang_folder.name)

        # Strategy A: DatasetDict saved with save_to_disk (dataset_dict.json present)
        if (lang_folder / "dataset_dict.json").exists():
            try:
                ds_dict = load_from_disk(str(lang_folder))
                if "validation" in ds_dict:
                    df = ds_dict["validation"].to_pandas()
                    df = _normalise(df, lang_code, lang_name)
                    train_frames.append(df)
                    print(f"    {lang_folder.name}/validation: {len(df):,} rows (DatasetDict)")
                if "test" in ds_dict:
                    df = ds_dict["test"].to_pandas()
                    df = _normalise(df, lang_code, lang_name)
                    test_frames.append(df)
                    print(f"    {lang_folder.name}/test:       {len(df):,} rows (DatasetDict)")
                continue
            except Exception as exc:
                print(f"    {lang_folder.name}: DatasetDict load failed ({exc}), trying splits...")

        # Strategy B: bare split subdirectories
        for split_name, frames in (("validation", train_frames), ("test", test_frames)):
            split_dir = lang_folder / split_name
            if split_dir.exists():
                try:
                    df = _read_split(split_dir)
                    df = _normalise(df, lang_code, lang_name)
                    frames.append(df)
                except Exception as exc:
                    print(f"    SKIPPED {lang_folder.name}/{split_name}: {exc}")
            else:
                print(f"    {lang_folder.name}/{split_name}: not found, skipping")

    return train_frames, test_frames


def _find_datasets_dir(raw: Path) -> Path:
    """Return the directory that directly contains per-language subfolders."""
    # Common layouts:
    #   raw/extracted_datasets/<lang>/...
    #   raw/<lang>/...
    candidate = raw / "extracted_datasets"
    if candidate.is_dir() and any(candidate.iterdir()):
        return candidate
    # Fall back to raw itself if it contains subdirs with validation/ or test/ inside
    if any(True for _ in raw.rglob("validation")):
        # Check if extracted_datasets is nested further
        for child in raw.iterdir():
            if child.is_dir() and any(child.rglob("validation")):
                # Return the immediate parent of the language folders
                val_examples = list(child.rglob("validation"))
                if val_examples:
                    return val_examples[0].parent.parent
    return raw


# ── Main entry point ──────────────────────────────────────────────────────────

def prepare(raw: Path, public: Path, private: Path) -> None:
    raw     = Path(raw)
    public  = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # Print full directory tree for diagnosis
    print(f"raw dir: {raw}")
    all_dirs = sorted(p for p in raw.rglob("*") if p.is_dir())
    print("Directory tree:")
    for d in all_dirs[:60]:
        print(f"  {d.relative_to(raw)}")
    if len(all_dirs) > 60:
        print(f"  ... ({len(all_dirs)} dirs total)")
    print()

    datasets_dir = _find_datasets_dir(raw)
    print(f"Scanning from: {datasets_dir}\n")

    train_frames, test_frames = _collect_splits(datasets_dir)

    if not train_frames:
        raise RuntimeError(
            f"No validation splits loaded from {datasets_dir}. "
            "See directory tree above."
        )
    if not test_frames:
        raise RuntimeError(
            f"No test splits loaded from {datasets_dir}. "
            "See directory tree above."
        )

    # ── train.csv ─────────────────────────────────────────────────────────────
    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_df = _assign_ids(train_df, "train")
    _select_cols(train_df, include_answer=True).to_csv(
        public / "train.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\ntrain.csv          → {len(train_df):,} rows")

    # ── test.csv + private.csv ────────────────────────────────────────────────
    test_full = pd.concat(test_frames, ignore_index=True)
    test_df = (
        test_full
        .groupby("language", group_keys=False)
        .apply(lambda g: g.sample(
            max(1, round(TEST_SAMPLE_SIZE * len(g) / len(test_full))),
            random_state=SEED,
        ))
        .reset_index(drop=True)
    )
    test_df = _assign_ids(test_df, "test")

    _select_cols(test_df, include_answer=False).to_csv(
        public / "test.csv", index=False, encoding="utf-8-sig"
    )
    print(f"test.csv           → {len(test_df):,} rows")

    _select_cols(test_df, include_answer=True).to_csv(
        private / "private.csv", index=False, encoding="utf-8-sig"
    )
    print(f"private.csv        → {len(test_df):,} rows")

    pd.DataFrame({
        "question_id": _select_cols(test_df, include_answer=False)["question_id"],
        "answer": "A",
    }).to_csv(public / "sample_submission.csv", index=False)
    print(f"sample_submission  → {len(test_df):,} rows")

    print("\nLanguage distribution (test):")
    print(test_df["language"].value_counts().to_string())
    print("\nDone.")


# ── Local testing ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    prepare(
        raw     = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./raw"),
        public  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./dataset/public"),
        private = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./dataset/private"),
    )
