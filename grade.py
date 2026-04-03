"""
grade.py — MILU Eris Challenge Grading Script

Scores a submission against the private ground truth using overall accuracy.

Usage:
    python grade.py \
        --private  ./dataset/private/private.csv \
        --solution ./working/submission.csv

Output (stdout, JSON):
    {"score": 0.7312, "details": {...}}

Exit codes:
    0 — success
    1 — file not found, missing columns, or format error
"""

import argparse
import json
import os
import sys

import pandas as pd

VALID_ANSWERS  = {"A", "B", "C", "D"}
DEFAULT_ANSWER = "A"


def load_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return df


def validate_submission(sub: pd.DataFrame) -> pd.DataFrame:
    missing = {"question_id", "answer"} - set(sub.columns)
    if missing:
        raise ValueError(f"submission.csv missing columns: {missing}")

    sub = sub[["question_id", "answer"]].copy()
    sub["answer"] = sub["answer"].astype(str).str.strip().str.upper()

    invalid = ~sub["answer"].isin(VALID_ANSWERS)
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"WARNING: {n_invalid} invalid answer value(s) — defaulting to '{DEFAULT_ANSWER}'",
              file=sys.stderr)
        sub.loc[invalid, "answer"] = DEFAULT_ANSWER

    dupes = sub.duplicated("question_id")
    n_dupes = int(dupes.sum())
    if n_dupes:
        print(f"WARNING: {n_dupes} duplicate question_id(s) — keeping first occurrence",
              file=sys.stderr)
        sub = sub.drop_duplicates("question_id", keep="first")

    return sub


def grade(private_path: str, solution_path: str) -> dict:
    private    = load_csv(private_path, "private.csv")
    submission = load_csv(solution_path, "submission.csv")
    submission = validate_submission(submission)

    for col in ("question_id", "answer"):
        if col not in private.columns:
            raise ValueError(f"private.csv missing column: '{col}'")

    private["answer"] = private["answer"].astype(str).str.strip().str.upper()

    merged = private.merge(
        submission.rename(columns={"answer": "predicted"}),
        on="question_id",
        how="left",
    )

    n_missing = int(merged["predicted"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing} question_id(s) absent from submission — "
              f"defaulting to '{DEFAULT_ANSWER}'", file=sys.stderr)
        merged["predicted"] = merged["predicted"].fillna(DEFAULT_ANSWER)

    merged["correct"] = merged["answer"] == merged["predicted"]
    total     = len(merged)
    n_correct = int(merged["correct"].sum())
    score     = n_correct / total if total > 0 else 0.0

    # Per-language breakdown
    per_language = {}
    if "language" in merged.columns:
        for lang, grp in merged.groupby("language"):
            per_language[str(lang)] = {
                "total":    len(grp),
                "correct":  int(grp["correct"].sum()),
                "accuracy": round(float(grp["correct"].mean()), 4),
            }

    # Per-domain breakdown
    per_domain = {}
    if "domain" in merged.columns:
        for dom, grp in merged.groupby("domain"):
            per_domain[str(dom)] = {
                "total":    len(grp),
                "correct":  int(grp["correct"].sum()),
                "accuracy": round(float(grp["correct"].mean()), 4),
            }

    return {
        "score": round(score, 6),
        "details": {
            "total_questions":    total,
            "correct_predictions": n_correct,
            "missing_predictions": n_missing,
            "per_language":        per_language,
            "per_domain":          per_domain,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private",  default="./dataset/private/private.csv")
    parser.add_argument("--solution", default="./working/submission.csv")
    args = parser.parse_args()

    try:
        result = grade(args.private, args.solution)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "score": 0.0}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
