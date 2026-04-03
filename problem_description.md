# Multilingual Indic Question Answering

## Overview

You are given a collection of multiple-choice questions drawn from competitive
examinations, textbooks, and government assessments across India. Questions span
11 Indic languages and 8 subject domains including STEM, Law & Governance,
Medicine, and Classical Literature.

Your task is to assign a single answer label — A, B, C, or D — to every
question in the test set.

Questions are written in their native script (Devanagari, Tamil, Bengali, etc.).
Generic English-only models perform substantially below multilingual models on
this benchmark. Strategies that adapt to language and domain are likely to help.

## Evaluation

Submissions are scored using **overall accuracy**:

```
score = correct_predictions / total_questions
```

Accuracy is computed over all languages and domains jointly.
A model predicting all-A scores approximately 0.250 (random chance over 4 options).

Per-language and per-domain breakdowns are logged but do not affect the final score.

## Dataset

`public/train.csv` — labelled questions for training and few-shot examples:

| Column          | Type     | Description                                              |
|-----------------|----------|----------------------------------------------------------|
| `question_id`   | str      | Unique question identifier                               |
| `language`      | str      | ISO 639-3 code: ben, guj, hin, kan, mal, mar, ory, pan, tam, tel, eng |
| `language_name` | str      | Full language name (e.g., Hindi, Tamil, Bengali)         |
| `domain`        | str      | Subject domain (e.g., STEM, Law & Governance, Medicine)  |
| `subject`       | str      | Specific subject (e.g., Physics, Constitutional Law)     |
| `question`      | str      | Question text in the target language script              |
| `option_a`      | str      | Answer choice A                                          |
| `option_b`      | str      | Answer choice B                                          |
| `option_c`      | str      | Answer choice C                                          |
| `option_d`      | str      | Answer choice D                                          |
| `answer`        | str      | Correct answer letter: A, B, C, or D                     |

`public/test.csv` — unlabelled questions to predict:

| Column          | Type     | Description                                              |
|-----------------|----------|----------------------------------------------------------|
| `question_id`   | str      | Unique question identifier (use in submission)           |
| `language`      | str      | ISO 639-3 language code                                  |
| `language_name` | str      | Full language name                                       |
| `domain`        | str      | Subject domain                                           |
| `subject`       | str      | Specific subject                                         |
| `question`      | str      | Question text in the target language script              |
| `option_a`      | str      | Answer choice A                                          |
| `option_b`      | str      | Answer choice B                                          |
| `option_c`      | str      | Answer choice C                                          |
| `option_d`      | str      | Answer choice D                                          |

`public/sample_submission.csv` — expected output format (all-A baseline).

## Submission Format

Submit a CSV file: `submission.csv`

| Column        | Type | Description                        |
|---------------|------|------------------------------------|
| `question_id` | str  | Question ID from `test.csv`        |
| `answer`      | str  | Predicted answer letter            |

Requirements:

- Exactly one row per `question_id` in `test.csv`
- `answer` must be one of: `A`, `B`, `C`, `D` (uppercase)
- Include header row

## Notes

- Train and test sets are stratified across all 11 languages and 8 domains
- Questions are independent — no cross-question context is needed
- Options are provided as separate columns (`option_a` through `option_d`), not a list
- Language codes follow ISO 639-3: `hin` = Hindi, `tam` = Tamil, `ben` = Bengali, etc.
- Some questions reference India-specific legal, cultural, or historical knowledge not
  well-represented in general English pretraining corpora
