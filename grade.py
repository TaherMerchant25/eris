"""
grade.py — Zero-Shot Cross-Lingual Transfer Challenge

Compares agent submission against private/answers.csv and returns accuracy.

private/answers.csv contains ONLY:  question_id, answer

Eris platform calls:
    grade(solution, private_answers)  →  float score in [0, 1]

CLI usage:
    python grade.py --solution ./working/submission.csv \
                    --private  ./dataset/private/answers.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Union

import pandas as pd

VALID_ANSWERS  = {"A", "B", "C", "D"}
DEFAULT_ANSWER = "A"

TableInput = Union[str, os.PathLike, pd.DataFrame]


def _load(src: TableInput, label: str) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        df = src.copy()
    elif hasattr(src, "read"):
        # file-like object (StringIO, etc.)
        df = pd.read_csv(src, dtype=str)
    else:
        path = os.fspath(src)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return df


def _clean_answers(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "question_id" not in df.columns:
        raise ValueError(f"{label} missing 'question_id' column. Got: {list(df.columns)}")
    if "answer" not in df.columns:
        raise ValueError(f"{label} missing 'answer' column. Got: {list(df.columns)}")
    df = df[["question_id", "answer"]].copy()
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()
    df["answer"] = df["answer"].replace({"0": "A", "1": "B", "2": "C", "3": "D"})
    return df


def grade(solution: TableInput, private_answers: TableInput) -> float:
    """
    Returns accuracy as a float in [0.0, 1.0].
    Eris calls float(grade(solution, private_answers)).
    """
    answers    = _load(private_answers, "answers.csv")
    submission = _load(solution, "submission.csv")

    answers    = _clean_answers(answers,    "answers.csv")
    submission = _clean_answers(submission, "submission.csv")

    # Remove duplicate question_ids (keep first)
    answers    = answers.drop_duplicates("question_id", keep="first")
    submission = submission.drop_duplicates("question_id", keep="first")

    # Coerce invalid submission answers to default
    invalid = ~submission["answer"].isin(VALID_ANSWERS)
    if invalid.any():
        print(f"WARNING: {int(invalid.sum())} invalid answer(s) defaulted to '{DEFAULT_ANSWER}'",
              file=sys.stderr)
        submission.loc[invalid, "answer"] = DEFAULT_ANSWER

    # Left-join: every row in answers.csv must be evaluated
    merged = answers.rename(columns={"answer": "correct"}).merge(
        submission.rename(columns={"answer": "predicted"}),
        on="question_id",
        how="left",
    )

    n_missing = int(merged["predicted"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing} question_id(s) missing from submission — "
              f"defaulted to '{DEFAULT_ANSWER}'", file=sys.stderr)
        merged["predicted"] = merged["predicted"].fillna(DEFAULT_ANSWER)

    total    = len(merged)
    n_correct = int((merged["correct"] == merged["predicted"]).sum())
    return round(n_correct / total, 6) if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade MILU cross-lingual challenge.")
    parser.add_argument("--private",  default="./dataset/private/answers.csv",
                        help="Path to private/answers.csv (question_id, answer)")
    parser.add_argument("--solution", default="./working/submission.csv",
                        help="Path to agent submission.csv (question_id, answer)")
    args = parser.parse_args()

    try:
        score = grade(args.solution, args.private)
        print(json.dumps({
            "score":     score,
            "min_score": 0.0,
            "max_score": 1.0,
        }, indent=2))
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "score": 0.0}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
