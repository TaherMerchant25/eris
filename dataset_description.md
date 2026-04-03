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

### Public Split (provided to agents)

| Column          | Type     | Description                                                                 |
|-----------------|----------|-----------------------------------------------------------------------------|
| `question_id`   | string   | Unique identifier for each question (e.g., `"hindi_0001"`)                  |
| `language`      | string   | ISO language code (e.g., `"hin"`, `"ben"`, `"tam"`)                         |
| `language_name` | string   | Human-readable language name (e.g., `"Hindi"`, `"Bengali"`, `"Tamil"`)     |
| `domain`        | string   | Broad subject domain (e.g., `"STEM"`, `"Law & Governance"`)                |
| `subject`       | string   | Specific subject (e.g., `"Physics"`, `"Constitutional Law"`)               |
| `question`      | string   | The full question text in the target language                               |
| `options`       | list[str]| List of 4 answer choices: `[A, B, C, D]`                                   |
| `answer`        | string   | **Correct answer letter** — one of `"A"`, `"B"`, `"C"`, `"D"` (**public split only**) |

> **Note**: In the **private split** (test set), the `answer` column is withheld. Agents must predict the answer using only `question_id`, `language`, `domain`, `subject`, `question`, and `options`.

### Private Split (ground truth — withheld from agents)

Same schema as public split, including `answer`.

---

## Split Sizes

| Split   | Rows   | Purpose                                          |
|---------|--------|--------------------------------------------------|
| Public  | ~8,900 | Agent training, few-shot examples, validation    |
| Private | ~8,900 | Ground truth for evaluation (answers withheld)   |

> The original MILU dataset provides a `validation` split (~8,933 examples) used as the **public split** for few-shot learning, and a `test` split (~79,617 examples) used to create the **private evaluation set** by subsampling.

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
