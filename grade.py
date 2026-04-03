"""
grade.py — MILU Eris Challenge Grading Script

Computes the accuracy of an agent's predictions against the private ground truth.

Usage (Eris platform):
    python grade.py \
        --private  ./dataset/private/private.csv \
        --solution ./working/submission.csv

Exit codes:
    0 — grading succeeded; score printed to stdout as JSON
    1 — grading failed (file not found, format error, etc.)

Output (stdout, JSON):
    {"score": 0.7312, "details": {...}}
"""

import argparse
import json
import sys
import os
import ast

import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_ANSWERS = {"A", "B", "C", "D"}
DEFAULT_ANSWER = "A"   # applied when a question_id is missing from submission


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return df


def validate_submission(sub: pd.DataFrame) -> pd.DataFrame:
    """Check submission format; normalise answer column."""
    required = {"question_id", "answer"}
    missing = required - set(sub.columns)
    if missing:
        raise ValueError(f"submission.csv is missing columns: {missing}")

    sub = sub[["question_id", "answer"]].copy()
    sub["answer"] = sub["answer"].astype(str).str.strip().str.upper()

    # Coerce invalid values to DEFAULT_ANSWER
    invalid_mask = ~sub["answer"].isin(VALID_ANSWERS)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        print(
            f"  WARNING: {n_invalid} submission rows have invalid answer values "
            f"(not in A/B/C/D) — defaulting to '{DEFAULT_ANSWER}'.",
            file=sys.stderr,
        )
        sub.loc[invalid_mask, "answer"] = DEFAULT_ANSWER

    # Remove duplicate question_ids (keep first)
    n_dupes = int(sub.duplicated("question_id").sum())
    if n_dupes > 0:
        print(
            f"  WARNING: {n_dupes} duplicate question_id(s) in submission — keeping first occurrence.",
            file=sys.stderr,
        )
        sub = sub.drop_duplicates("question_id", keep="first")

    return sub


# ── Core grading ──────────────────────────────────────────────────────────────

def grade(private_path: str, solution_path: str) -> dict:
    # Load files
    private = load_csv(private_path, "private")
    submission = load_csv(solution_path, "submission")

    # Validate submission format
    submission = validate_submission(submission)

    # Required private columns
    for col in ("question_id", "answer"):
        if col not in private.columns:
            raise ValueError(f"private.csv is missing required column: '{col}'")

    private["answer"] = private["answer"].astype(str).str.strip().str.upper()

    # Merge submission onto private (left join — ensures every private row is graded)
    merged = private.merge(
        submission.rename(columns={"answer": "predicted"}),
        on="question_id",
        how="left",
    )

    # Fill missing predictions with DEFAULT_ANSWER
    n_missing = int(merged["predicted"].isna().sum())
    if n_missing > 0:
        print(
            f"  WARNING: {n_missing} question_id(s) from the private set are absent "
            f"from submission — defaulting to '{DEFAULT_ANSWER}'.",
            file=sys.stderr,
        )
        merged["predicted"] = merged["predicted"].fillna(DEFAULT_ANSWER)

    # Compute correctness
    merged["correct"] = merged["answer"] == merged["predicted"]

    total = len(merged)
    n_correct = int(merged["correct"].sum())
    overall_accuracy = n_correct / total if total > 0 else 0.0

    # Per-language breakdown
    per_language = {}
    if "language" in merged.columns:
        for lang, grp in merged.groupby("language"):
            per_language[str(lang)] = {
                "total": len(grp),
                "correct": int(grp["correct"].sum()),
                "accuracy": round(float(grp["correct"].mean()), 4),
            }

    # Per-domain breakdown
    per_domain = {}
    if "domain" in merged.columns:
        for dom, grp in merged.groupby("domain"):
            per_domain[str(dom)] = {
                "total": len(grp),
                "correct": int(grp["correct"].sum()),
                "accuracy": round(float(grp["correct"].mean()), 4),
            }

    return {
        "score": round(overall_accuracy, 6),
        "details": {
            "total_questions": total,
            "correct_predictions": n_correct,
            "missing_predictions": n_missing,
            "per_language": per_language,
            "per_domain": per_domain,
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grade MILU challenge submissions."
    )
    parser.add_argument(
        "--private",
        default="./dataset/private/private.csv",
        help="Path to private.csv (with ground-truth answers).",
    )
    parser.add_argument(
        "--solution",
        default="./working/submission.csv",
        help="Path to agent's submission.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = grade(args.private, args.solution)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)
    except Exception as exc:
        error_output = {"error": str(exc), "score": 0.0}
        print(json.dumps(error_output, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
