# MILU: Multilingual Indic LLM Benchmark Challenge

## Background

Large language models are increasingly deployed in multilingual settings, yet their performance on low-resource and culturally-specific languages lags far behind English. India alone has over 22 scheduled languages and more than 1.4 billion speakers — yet most LLM benchmarks remain English-centric.

**MILU** (Multi-task Indic Language Understanding) is a rigorous multiple-choice QA benchmark covering **11 Indic languages**, **8 domains**, and **41 subjects** ranging from Physics and Constitutional Law to Classical Literature and Agricultural Science. Questions were sourced from competitive exams, textbooks, and government assessments — making them genuinely culturally grounded, not just translated from English.

Your task: build an LLM pipeline that answers as many questions correctly as possible across all 11 languages.

---

## The Task

Given a set of multiple-choice questions in various Indic languages, predict the correct answer (A, B, C, or D) for each question.

Each question has:
- A question text in the target language
- Four answer options (A, B, C, D) in the target language
- Language and domain metadata

---

## Dataset

The dataset lives in `./dataset/public/public.csv`.

| Column          | Description                                                           |
|-----------------|-----------------------------------------------------------------------|
| `question_id`   | Unique identifier for the question                                    |
| `language`      | ISO language code (`hin`, `ben`, `tam`, etc.)                         |
| `language_name` | Full language name (`Hindi`, `Bengali`, `Tamil`, etc.)               |
| `domain`        | Subject domain (`STEM`, `Law & Governance`, `Social Science`, etc.)  |
| `subject`       | Specific subject (`Physics`, `Constitutional Law`, etc.)             |
| `question`      | Question text in the target language                                  |
| `options`       | List of 4 answer choices: `["option_a_text", "option_b_text", ...]`  |
| `answer`        | **Correct answer letter** — `"A"`, `"B"`, `"C"`, or `"D"`           |

> The `answer` column is present in the public split — use it freely for few-shot examples, finetuning, or validation. The private evaluation set withholds answers.

**Languages**: Bengali (ben), Gujarati (guj), Hindi (hin), Kannada (kan), Malayalam (mal), Marathi (mar), Odia (ory), Punjabi (pan), Tamil (tam), Telugu (tel), English (eng)

---

## Submission Format

Your solution must write a CSV file to `./working/submission.csv` with exactly **two columns**:

| Column        | Type   | Description                              |
|---------------|--------|------------------------------------------|
| `question_id` | string | Must match the IDs in the evaluation set |
| `answer`      | string | Your predicted answer: `A`, `B`, `C`, or `D` |

Example:
```
question_id,answer
priv_hin_00001,B
priv_ben_00002,A
priv_tam_00003,D
```

**Rules:**
- Every `question_id` in the private set must appear exactly once in your submission.
- `answer` must be one of: `A`, `B`, `C`, `D` (uppercase).
- Missing predictions default to `A` (penalised as likely wrong).
- Extra rows beyond the private set are ignored.

---

## Scoring

Your score is the **overall accuracy** across all languages and domains:

```
score = (number of correct predictions) / (total private questions)
```

Scores range from 0.0 to 1.0. A random baseline scores ~0.25 (4 choices). The current SOTA (GPT-4o) scores approximately **0.74** on the full MILU benchmark.

### Per-Language Breakdown (informational)

The leaderboard shows your overall score. The evaluation script also logs per-language and per-domain accuracy to help you identify where to improve.

| Language    | GPT-4o (approx.) | Harder domains              |
|-------------|------------------|-----------------------------|
| English     | ~85%             | Law & Governance            |
| Hindi       | ~80%             | Arts & Humanities           |
| Bengali     | ~72%             |                             |
| Tamil       | ~70%             | Law & Governance, Medicine  |
| Malayalam   | ~68%             |                             |
| Kannada     | ~67%             |                             |
| Odia        | ~65%             |                             |
| Gujarati    | ~66%             |                             |
| Punjabi     | ~63%             |                             |
| Marathi     | ~70%             |                             |
| Telugu      | ~69%             |                             |

---

## Approach Guidance

You are encouraged to explore any of the following strategies (not exhaustive):

### 1. Zero-Shot / Few-Shot Prompting
Use a multilingual LLM (e.g., `google/gemma-3`, `meta-llama/Llama-3.1`, `ai4bharat/indic-bert`) with carefully crafted prompts. Include language-specific instructions and representative few-shot examples sampled from the public split.

### 2. Retrieval-Augmented Prompting
Build a per-language example store from the public split. At inference time, retrieve the most semantically similar examples using multilingual embeddings and include them as few-shot context.

### 3. Finetuning
Finetune a multilingual LLM on the public split using instruction-following format. Recommended base models:
- `google/mt5-large` or `google/mt5-xl`
- `ai4bharat/IndicBART`
- `meta-llama/Llama-3.2-3B` with LoRA/QLoRA
- `microsoft/mdeberta-v3-base` (for discriminative finetuning)

### 4. Ensemble / Self-Consistency
Run multiple inference passes with temperature > 0, then take the majority vote. Works especially well across languages where the model is uncertain.

### 5. Chain-of-Thought
Prompt the model to produce a step-by-step reasoning trace before committing to a final answer. Particularly effective for STEM and Law & Governance questions.

### 6. Language-Specific Routing
Train a lightweight language/domain classifier and route each question to the best-performing model configuration for that language+domain combination.

---

## Constraints

- Your solution notebook (`solution.ipynb`) must run end-to-end in under **30 minutes** on a standard Kaggle CPU/GPU environment.
- Use only libraries available in the Kaggle Docker image (PyPI packages pre-installed on Kaggle).
- Read data exclusively from `./dataset/public/` (do not hard-code external URLs at inference time).
- Write your final predictions to `./working/submission.csv`.
- Your notebook must be reproducible: fix all random seeds and avoid non-deterministic external API calls in the prediction loop.

---

## Evaluation Timeline

| Phase       | Description                                         |
|-------------|-----------------------------------------------------|
| Development | Use `public.csv` answers to build and validate your pipeline |
| Submission  | Submit `submission.csv` — graded against private set |

---

## Tips for a Strong Score

1. **Don't ignore lower-resource languages** (Odia, Punjabi, Gujarati) — they have the most headroom for improvement over a naive baseline.
2. **Use native-script prompts** — passing the question in its original script (not transliterated) significantly improves performance for most multilingual models.
3. **Domain-aware few-shot selection** — picking examples from the same domain as the test question outperforms random few-shot selection.
4. **Validate per-language** — the public split has enough examples to compute per-language accuracy; use it to detect and fix regressions before submission.
5. **Calibrate your answer extraction** — LLMs sometimes output `"B)"`, `"(B)"`, or `"option B"` instead of `"B"`. Robust post-processing of model output is essential.
