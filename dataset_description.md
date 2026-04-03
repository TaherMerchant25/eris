# MILU: Multi-task Indic Language Understanding Benchmark

## Overview

MILU (Multi-task Indic Language Understanding) is a large-scale multiple-choice question answering benchmark designed to evaluate language models across 11 Indic languages. It contains **79,617 questions** spanning 8 broad domains and 41 subjects, making it one of the most comprehensive Indic language evaluation benchmarks available.

- **Source**: [ai4bharat/MILU on HuggingFace](https://huggingface.co/datasets/ai4bharat/MILU)
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — permits commercial use with attribution
- **Paper**: "MILU: A Multi-task Indic Language Understanding Benchmark" (AI4Bharat)

---

## Languages Covered

| Language    | ISO Code | Script     |
|-------------|----------|------------|
| Bengali     | ben      | Bengali    |
| Gujarati    | guj      | Gujarati   |
| Hindi       | hin      | Devanagari |
| Kannada     | kan      | Kannada    |
| Malayalam   | mal      | Malayalam  |
| Marathi     | mar      | Devanagari |
| Odia        | ory      | Odia       |
| Punjabi     | pan      | Gurmukhi   |
| Tamil       | tam      | Tamil      |
| Telugu      | tel      | Telugu     |
| English     | eng      | Latin      |

---

## Domains and Subjects

| Domain                       | Example Subjects                                          |
|------------------------------|-----------------------------------------------------------|
| STEM                         | Physics, Chemistry, Biology, Mathematics, Computer Science |
| Social Science               | Economics, History, Political Science, Geography, Sociology |
| Humanities                   | Literature, Philosophy, Language/Grammar                  |
| Arts & Humanities            | Fine Arts, Music, Film Studies                            |
| Law & Governance             | Constitutional Law, Civics, Legal Aptitude                |
| Medicine & Health            | Medicine, Nursing, Pharmacy, Public Health                |
| Agriculture                  | Agricultural Science, Animal Husbandry                    |
| General                      | Current Affairs, General Knowledge, Logical Reasoning     |

---

## Dataset Schema

### `train.csv` — labelled questions (provided to agents)

| Column          | Type   | Description                                                             |
|-----------------|--------|-------------------------------------------------------------------------|
| `question_id`   | str    | Unique question identifier (e.g., `"hin_000042"`)                       |
| `language`      | str    | ISO 639-3 code (e.g., `"hin"`, `"ben"`, `"tam"`)                        |
| `language_name` | str    | Full language name (e.g., `"Hindi"`, `"Bengali"`, `"Tamil"`)            |
| `domain`        | str    | Subject domain (e.g., `"STEM"`, `"Law & Governance"`)                   |
| `subject`       | str    | Specific subject (e.g., `"Physics"`, `"Constitutional Law"`)            |
| `question`      | str    | Question text in the target language script                             |
| `option_a`      | str    | Answer choice A                                                         |
| `option_b`      | str    | Answer choice B                                                         |
| `option_c`      | str    | Answer choice C                                                         |
| `option_d`      | str    | Answer choice D                                                         |
| `answer`        | str    | Correct answer letter: `A`, `B`, `C`, or `D`                            |

### `test.csv` — unlabelled questions (agents must predict answers)

Same schema as `train.csv` but **without** the `answer` column.

### `sample_submission.csv` — all-A baseline

| Column        | Type | Description                     |
|---------------|------|---------------------------------|
| `question_id` | str  | Question ID from `test.csv`     |
| `answer`      | str  | Predicted answer (`A` baseline) |

---

## Split Sizes

| File                   | Rows   | Purpose                                       |
|------------------------|--------|-----------------------------------------------|
| `train.csv`            | ~8,900 | Labelled examples for few-shot / finetuning   |
| `test.csv`             | ~8,900 | Unlabelled questions — agents predict answers |
| `sample_submission.csv`| ~8,900 | All-A baseline in required submission format  |

> `train.csv` is sourced from the MILU `validation` split (~8,933 examples).  
> `test.csv` is a stratified sample from the MILU `test` split (~79,617 examples).

---

## Usage Notes

1. **Language distribution**: Questions are distributed across all 11 languages. Agents should handle non-Latin scripts correctly (ensure UTF-8 encoding throughout).
2. **Answer format**: Each question has exactly 4 options (A, B, C, D). The target output is a single letter per question.
3. **Cultural specificity**: Some questions reference Indian cultural, legal, and historical context. Generic English-trained LLMs may underperform; multilingual and India-specific models often do better.
4. **Few-shot learning**: The public split contains representative examples for all language/domain combinations and can be used for in-context learning or finetuning.

---

## Attribution

```
@misc{milu2024,
  title={MILU: A Multi-task Indic Language Understanding Benchmark},
  author={AI4Bharat},
  year={2024},
  url={https://huggingface.co/datasets/ai4bharat/MILU}
}
```
