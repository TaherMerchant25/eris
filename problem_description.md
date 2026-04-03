# Zero-Shot Cross-Lingual Transfer for Indic Question Answering

## Overview

You are given labelled multiple-choice questions in six Indic languages. Your
task is to predict correct answers for questions in five **completely different**
languages for which you receive no labelled training examples.

The six training languages are: Bengali, English, Hindi, Marathi, Tamil, Telugu.
The five test languages are: Gujarati, Kannada, Malayalam, Odia, Punjabi.

Training and test languages share no overlap. Solutions that simply memorise
per-language patterns or fine-tune language-specific heads will fail on the test
set. Genuine cross-lingual generalisation — through multilingual representations,
translate-then-answer pipelines, or script-agnostic prompting — is required.

This is not a general multilingual benchmark. It is a transfer learning challenge:
how well can a system trained on six seen languages answer questions in five
unseen ones?

## Evaluation

Submissions are scored using **accuracy on unseen languages only**:

```
score = correct_predictions / total_test_questions
```

Only questions from the five test languages (Gujarati, Kannada, Malayalam, Odia,
Punjabi) are evaluated. Predictions for any other language ID are ignored.

A model predicting all-A scores approximately 0.250. A system that copies
per-language statistics from Hindi (the closest seen language) achieves roughly
0.35–0.40.

## Dataset

`public/train.csv` — labelled questions in six seen languages:

| Column          | Type | Description                                              |
|-----------------|------|----------------------------------------------------------|
| `question_id`   | str  | Unique question identifier                               |
| `language`      | str  | ISO 639-3 code: ben, eng, hin, mar, tam, tel             |
| `language_name` | str  | Full language name                                       |
| `domain`        | str  | Subject domain (e.g., STEM, Law & Governance, Medicine)  |
| `subject`       | str  | Specific subject (e.g., Physics, Constitutional Law)     |
| `question`      | str  | Question text in the target language script              |
| `option_a`      | str  | Answer choice A                                          |
| `option_b`      | str  | Answer choice B                                          |
| `option_c`      | str  | Answer choice C                                          |
| `option_d`      | str  | Answer choice D                                          |
| `answer`        | str  | Correct answer letter: A, B, C, or D                     |

`public/test.csv` — unlabelled questions in five unseen languages:

| Column          | Type | Description                                              |
|-----------------|------|----------------------------------------------------------|
| `question_id`   | str  | Unique question identifier (use in submission)           |
| `language`      | str  | ISO 639-3 code: guj, kan, mal, ory, pan                  |
| `language_name` | str  | Full language name                                       |
| `domain`        | str  | Subject domain                                           |
| `subject`       | str  | Specific subject                                         |
| `question`      | str  | Question text in the target language script              |
| `option_a`      | str  | Answer choice A                                          |
| `option_b`      | str  | Answer choice B                                          |
| `option_c`      | str  | Answer choice C                                          |
| `option_d`      | str  | Answer choice D                                          |

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

- `train.csv` contains no examples from Gujarati, Kannada, Malayalam, Odia, or Punjabi
- `test.csv` contains only Gujarati, Kannada, Malayalam, Odia, and Punjabi questions
- All five test languages use distinct scripts: Gujarati (Gujarati script), Kannada
  (Kannada script), Malayalam (Malayalam script), Odia (Odia script), Punjabi (Gurmukhi)
- Domain and subject distributions are consistent across train and test languages —
  cross-domain generalisation is not required, only cross-lingual
- Questions are independent — no cross-question context is needed
