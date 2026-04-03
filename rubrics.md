# MILU Challenge — Grading Rubrics

The following rubrics define the criteria by which agent solutions are evaluated beyond raw accuracy score. Each rubric assesses a specific quality dimension of the submitted solution.

---

## Rubric 1: Multilingual Data Handling
**Type**: DATA_HANDLING  
**Importance**: REQUIRED

**Description**:  
The solution correctly loads and processes questions across all 11 Indic language scripts without encoding errors, garbled text, or dropped rows.

**How to assess**:
- Open `solution.ipynb` and verify the data loading cell.
- Run `df['language'].value_counts()` — all 11 languages should appear.
- Inspect a sample of non-Latin script questions (e.g., Tamil, Malayalam) to confirm text is not corrupted.
- Check that the `options` column is correctly parsed from list-formatted strings into usable Python lists.

**Pass criteria**:
- All 11 language codes appear in the loaded DataFrame.
- No NaN values in `question` or `options` columns.
- Non-ASCII characters render correctly when printed/displayed.
- `options` is parsed as a Python `list` of 4 strings, not a raw string representation.

**Fail criteria**:
- Any language is missing entirely from predictions.
- `question` column contains `\u` escape sequences instead of rendered Unicode.
- `options` remains as a string (e.g., `"['A', 'B', 'C', 'D']"`) and is never parsed.

---

## Rubric 2: Model Selection and Multilingual Capability
**Type**: MODELING  
**Importance**: REQUIRED

**Description**:  
The solution uses a model genuinely capable of multilingual inference. It must go beyond an English-only model applied naively, and the model choice must be justified.

**How to assess**:
- Identify which model(s) are used (look for `from_pretrained`, API calls, or model name strings).
- Verify the model supports at least the major Indic languages (Hindi, Bengali, Tamil, Telugu).
- Check that the solution does not simply pass all questions to an English-only model without any adaptation.

**Pass criteria**:
- Model is explicitly multilingual (e.g., `google/mt5-*`, `ai4bharat/*`, `meta-llama/Llama-3.*`, `google/gemma-*`, GPT-4o, Claude, `intfloat/multilingual-e5-*`).
- OR: solution includes transliteration/translation as a preprocessing step before an English model, and this is clearly documented.
- Model name or API identifier is visible in the notebook.

**Fail criteria**:
- Solution uses `bert-base-uncased` or another English-only model with no adaptation.
- Model is described as "multilingual" but handles only English questions correctly.
- No model is used at all (e.g., always predicts "A").

---

## Rubric 3: Few-Shot or Finetuning Strategy
**Type**: TRAINING  
**Importance**: REQUIRED

**Description**:  
The solution makes meaningful use of the public split (labeled examples) to improve performance. This could be few-shot prompting, RAG-based example retrieval, finetuning, or a combination.

**How to assess**:
- Trace how `public.csv` is used in the pipeline.
- Verify that labeled examples are incorporated into the prediction process (not just loaded and ignored).
- Check that few-shot examples are drawn from the same language or domain as the test question, or that a principled selection strategy is described.

**Pass criteria** (any one is sufficient):
- **Few-shot**: Notebook constructs per-question prompts that include ≥1 labeled example from `public.csv`.
- **RAG**: Notebook builds an embedding index from `public.csv` and retrieves similar examples at inference time.
- **Finetuning**: Notebook trains or adapts a model on `public.csv` before running inference.

**Fail criteria**:
- `public.csv` is loaded but never used in building predictions.
- The notebook uses zero-shot inference with no reference to the labeled data.
- Few-shot examples are all in English regardless of the test question's language.

---

## Rubric 4: Answer Extraction and Robustness
**Type**: FEATURE_ENGINEERING  
**Importance**: RECOMMENDED

**Description**:  
The solution includes a robust post-processing step to extract a clean `A/B/C/D` answer from raw model outputs, handling the variety of formats LLMs produce.

**How to assess**:
- Find the answer extraction function/cell.
- Check that it handles common LLM output patterns.
- Verify the submission CSV has only `A`, `B`, `C`, `D` values (no `None`, empty strings, or verbose text).

**Pass criteria**:
- Post-processing handles at least 3 of these formats: `"A"`, `"A."`, `"(A)"`, `"A)"`, `"Option A"`, `"The answer is A"`, `"A: <text>"`.
- A fallback (e.g., default to `"A"` or random choice) is in place for unparseable outputs.
- Final `submission.csv` contains only valid `A/B/C/D` values in the `answer` column.

**Fail criteria**:
- Raw model output is written directly to the submission without extraction.
- Many rows have `None` or blank answers.
- No fallback for unparseable output (causes KeyError or NaN in submission).

---

## Rubric 5: Per-Language Validation and Analysis
**Type**: CODE_QUALITY  
**Importance**: RECOMMENDED

**Description**:  
The solution includes a validation step that measures per-language accuracy on the public split, and the author uses this breakdown to inform modeling decisions.

**How to assess**:
- Look for a cell that computes accuracy grouped by `language` on `public.csv`.
- Check whether the analysis is used to adapt the pipeline (e.g., extra few-shot examples for low-accuracy languages, or a note explaining an observed pattern).

**Pass criteria**:
- A table or chart showing accuracy per language on the public split is present.
- At least one modeling decision references the per-language breakdown (e.g., "Odia accuracy was lowest at 45%, so we added domain-specific examples for Odia").

**Fail criteria**:
- Only overall accuracy is reported; no per-language breakdown exists.
- Per-language breakdown is computed but completely ignored in all subsequent decisions.

---

## Rubric 6: Reproducibility and Code Clarity
**Type**: CODE_QUALITY  
**Importance**: UNIVERSAL

**Description**:  
The notebook is reproducible end-to-end: random seeds are fixed, external dependencies are clearly documented, and the notebook runs without modification from top to bottom.

**How to assess**:
- Restart the kernel and run all cells from scratch. The notebook should complete without errors in ≤30 minutes.
- Check that all random seeds are set (`random.seed`, `np.random.seed`, `torch.manual_seed`, etc.).
- Verify that no hard-coded absolute paths outside `./dataset/` or `./working/` are used.

**Pass criteria**:
- All random seeds are explicitly set before any stochastic operation.
- Running all cells produces `./working/submission.csv` without manual intervention.
- No hard-coded paths to `/home/user/...` or `/kaggle/input/...` (use `./dataset/public/public.csv`).
- Required packages are listed in a `requirements` comment or pip install cell at the top.

**Fail criteria**:
- Notebook errors on re-run (e.g., variable not defined, file not found).
- No random seed is set anywhere.
- A cell requires manual parameter editing to run.
- Notebook takes >30 minutes to run end-to-end on Kaggle T4 GPU.

---

## Rubric 7: Submission File Integrity
**Type**: AGENT_BEHAVIOR  
**Importance**: REQUIRED

**Description**:  
The submission CSV contains exactly the required columns and the correct number of predictions.

**How to assess**:
- Load `./working/submission.csv` in Python.
- Check shape, column names, and value distribution.

```python
import pandas as pd
sub = pd.read_csv("./working/submission.csv")
assert list(sub.columns) == ["question_id", "answer"], "Wrong columns"
assert sub["answer"].isin(["A","B","C","D"]).all(), "Invalid answer values"
assert sub["question_id"].nunique() == len(sub), "Duplicate question_ids"
print(f"Submission has {len(sub)} rows — expected ~8,900")
```

**Pass criteria**:
- Exactly 2 columns: `question_id` and `answer`.
- All answers are one of `A`, `B`, `C`, `D`.
- No duplicate `question_id` values.
- Row count matches private set size (within ±1%).

**Fail criteria**:
- Missing `question_id` or `answer` column.
- Any non-A/B/C/D value in `answer` column.
- Fewer than 90% of private question IDs are present.
