# Zero-Shot Cross-Lingual Transfer — Rubrics

---

## Rubric 1: No Test-Language Leakage in Training
**Type**: DATA_HANDLING  
**Importance**: REQUIRED

The solution must not use any labelled or unlabelled data from the five test
languages (Gujarati, Kannada, Malayalam, Odia, Punjabi) during model training
or few-shot construction.

**Pass**:
- Only `train.csv` rows are used to build the few-shot pool or fine-tune the model
- No external datasets specific to guj, kan, mal, ory, or pan are loaded
- The solution does not hardcode expected answer distributions for test languages

**Fail**:
- Test-language questions from an external source are used as training examples
- Few-shot examples are drawn from the test split
- Label statistics from test languages are used to bias predictions

---

## Rubric 2: Cross-Lingual Transfer Strategy
**Type**: MODELING  
**Importance**: REQUIRED

The solution implements a genuine strategy for transferring knowledge from seen
languages to unseen ones. Predicting a constant label or copying seen-language
statistics does not satisfy this rubric.

**Pass** (any one):
- Uses a multilingual model capable of representing all 11 Indic scripts
  (e.g., `google/mt5-*`, `ai4bharat/*`, `intfloat/multilingual-e5-*`,
  `meta-llama/Llama-3*`, GPT-4o, Claude)
- OR translates test questions into a seen language before inference, using
  a translation model or API
- OR uses language-agnostic semantic embeddings to retrieve cross-lingual
  few-shot examples from `train.csv`

**Fail**:
- English-only model with no translation or adaptation step
- Model selected purely because it works for seen languages with no consideration
  of unseen script support
- Prediction is identical for all test languages (constant output)

---

## Rubric 3: Cross-Lingual Few-Shot Retrieval
**Type**: FEATURE_ENGINEERING  
**Importance**: RECOMMENDED

Because no labelled examples exist for test languages, the solution uses
cross-lingual semantic similarity to retrieve relevant examples from `train.csv`
rather than relying on exact language matching.

**Pass**:
- Few-shot examples are retrieved using multilingual embeddings
  (e.g., `intfloat/multilingual-e5-large`, `LaBSE`, `paraphrase-multilingual-*`)
- Retrieved examples come from `train.csv` and are from a seen language, not the
  test language
- Retrieval is based on question content or domain similarity, not random sampling

**Fail**:
- Zero-shot only — no training examples used at all
- Few-shot examples are selected by language match (impossible for unseen languages)
  without a cross-lingual fallback

---

## Rubric 4: Seen-Language Validation
**Type**: CODE_QUALITY  
**Importance**: RECOMMENDED

The solution validates its transfer approach on a held-out seen language before
running on the unseen test set, demonstrating that the cross-lingual method
actually works.

**Pass**:
- One seen language is withheld from the few-shot pool and used as a proxy
  for unseen-language evaluation (e.g., treat Telugu as "unseen" during development)
- Per-language accuracy on this proxy is reported and used to tune the approach
- Proxy validation accuracy is meaningfully above the all-A baseline (>0.30)

**Fail**:
- Validation is done only on languages the model has explicitly trained on
  (in-language validation does not measure cross-lingual transfer ability)
- No validation is performed before generating test predictions

---

## Rubric 5: Answer Extraction
**Type**: FEATURE_ENGINEERING  
**Importance**: RECOMMENDED

The solution includes post-processing to extract a clean A/B/C/D letter from
raw model output, with a fallback for unparseable outputs.

**Pass**:
- Handles at least: `"A"`, `"(A)"`, `"A."`, `"Answer: A"`, `"The answer is A"`
- A fallback (e.g., `"A"`) is applied when no letter can be extracted
- All rows in `submission.csv` contain only `A`, `B`, `C`, or `D`

**Fail**:
- Raw model output written directly to submission
- Any NaN or empty string in the answer column
- No fallback causes a crash or NaN

---

## Rubric 6: Reproducibility
**Type**: CODE_QUALITY  
**Importance**: UNIVERSAL

The notebook runs end-to-end without errors, produces `./working/submission.csv`,
and completes within 30 minutes.

**Pass**:
- All random seeds are set before any stochastic operation
- Running all cells from a fresh kernel produces a valid submission
- No hard-coded absolute paths outside `./dataset/` and `./working/`
- Required packages listed in a pip install cell or comment at the top

**Fail**:
- Notebook errors on re-run from scratch
- Runtime exceeds 30 minutes on a Kaggle T4 GPU
- A cell requires manual edits to run

---

## Rubric 7: Submission File Format
**Type**: AGENT_BEHAVIOR  
**Importance**: REQUIRED

`./working/submission.csv` matches the required format exactly.

**Pass**:
- Exactly two columns: `question_id`, `answer`
- All `answer` values are in `{A, B, C, D}`
- No duplicate `question_id` values
- Row count matches `test.csv` (within ±1%)

**Fail**:
- Missing `question_id` or `answer` column
- Any value outside `{A, B, C, D}` in the answer column
- Fewer than 90% of `test.csv` question IDs present
