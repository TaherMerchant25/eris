# MILU Challenge — Rubrics

---

## Rubric 1: Multilingual Data Loading
**Type**: DATA_HANDLING  
**Importance**: REQUIRED

The solution loads `train.csv` and `test.csv` without encoding errors and
correctly handles all 11 Indic language scripts.

**Pass**:
- All 11 language codes appear in the loaded DataFrame (`df['language'].nunique() == 11`)
- No NaN values in `question`, `option_a`, `option_b`, `option_c`, `option_d`
- Non-ASCII text (Devanagari, Tamil, Bengali, etc.) renders correctly when printed
- Files read with UTF-8 encoding (`encoding='utf-8-sig'` or equivalent)

**Fail**:
- Any language is missing entirely from the loaded data
- Non-Latin script columns contain `\u` escape sequences instead of rendered characters
- Encoding errors raised during file read

---

## Rubric 2: Multilingual Model Selection
**Type**: MODELING  
**Importance**: REQUIRED

The solution uses a model capable of processing text in Indic language scripts.
An English-only model applied naively without translation or adaptation does not satisfy this rubric.

**Pass** (any one):
- Model checkpoint is multilingual: `google/mt5-*`, `ai4bharat/*`, `meta-llama/Llama-3*`,
  `google/gemma-*`, `intfloat/multilingual-e5-*`, GPT-4o, Claude, or equivalent
- OR: solution translates or transliterates questions to English before passing to
  an English model, and this step is clearly implemented in the notebook

**Fail**:
- Model is `bert-base-uncased`, `gpt2`, or another English-only checkpoint with no
  language adaptation
- No model or API call is identifiable in the notebook
- Solution always predicts the same label regardless of input

---

## Rubric 3: Use of Labelled Training Data
**Type**: TRAINING  
**Importance**: REQUIRED

The solution makes meaningful use of `train.csv` labels to improve predictions.
Loading the file without using it does not satisfy this rubric.

**Pass** (any one):
- **Few-shot**: prompts include ≥1 labelled example from `train.csv` for each test question
- **RAG**: an embedding index is built from `train.csv`; similar examples are retrieved
  at inference time
- **Finetuning**: a model is trained or LoRA-adapted on `train.csv` before inference

**Fail**:
- `train.csv` is loaded but the labels are never referenced during inference
- Zero-shot inference only, with no use of available labelled examples
- Few-shot examples are all in English regardless of the test question's language

---

## Rubric 4: Answer Extraction
**Type**: FEATURE_ENGINEERING  
**Importance**: RECOMMENDED

The solution includes a post-processing step that reliably extracts a single
`A/B/C/D` letter from raw model output.

**Pass**:
- Extraction handles at least these formats: `"A"`, `"(A)"`, `"A."`, `"A)"`,
  `"Option A"`, `"The answer is A"`, `"answer: A"`
- A fallback is in place for unparseable outputs (e.g., default to `"A"`)
- All rows in the final `submission.csv` contain only `A`, `B`, `C`, or `D`

**Fail**:
- Raw model output written directly to submission without extraction
- Submission contains empty strings, `None`, or multi-word outputs in the answer column
- No fallback causes a KeyError or NaN in the submission

---

## Rubric 5: Per-Language Validation
**Type**: CODE_QUALITY  
**Importance**: RECOMMENDED

The solution evaluates per-language accuracy on a held-out portion of `train.csv`
and uses the breakdown to inform at least one modelling decision.

**Pass**:
- A table or printed output showing per-language accuracy on the training split is present
- At least one decision in the notebook cites the per-language results
  (e.g., increased few-shot examples for lowest-accuracy language)

**Fail**:
- Only overall accuracy is reported
- Per-language breakdown is computed but not referenced anywhere in subsequent cells

---

## Rubric 6: Reproducibility
**Type**: CODE_QUALITY  
**Importance**: UNIVERSAL

The notebook runs end-to-end without errors or manual edits, produces
`./working/submission.csv`, and completes in under 30 minutes.

**Pass**:
- All random seeds are set (`random.seed`, `np.random.seed`, `torch.manual_seed`)
- Running all cells from a fresh kernel produces `./working/submission.csv`
- No hard-coded absolute paths outside `./dataset/` and `./working/`
- Required packages are listed in a pip install cell or comment at the top

**Fail**:
- Notebook errors on re-run from scratch
- No random seeds set anywhere
- A cell requires manual parameter changes to run
- Runtime exceeds 30 minutes on a Kaggle T4 GPU environment

---

## Rubric 7: Submission File Format
**Type**: AGENT_BEHAVIOR  
**Importance**: REQUIRED

The output file `./working/submission.csv` exactly matches the required format.

**Pass**:
- Exactly two columns: `question_id`, `answer`
- All `answer` values are in `{A, B, C, D}`
- No duplicate `question_id` values
- Row count matches `test.csv` (within ±1%)

**Fail**:
- Missing `question_id` or `answer` column
- Any value outside `{A, B, C, D}` in the answer column
- Fewer than 90% of `test.csv` question IDs are present
