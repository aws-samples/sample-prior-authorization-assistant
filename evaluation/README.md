# Extraction model comparison

Compares Bedrock models on the **medical data extraction** step: the payor, CPT codes and ICD-10 codes the pipeline extracts from each document in `samples/`. It runs the real extraction Lambda handler locally and does not deploy anything.

## What it runs

- **Cases.** One per document in `samples/`, stored in `cases/`. Expected values are what each synthetic document states:
  - FHIR bundles: read from the FHIR resources.
  - PDFs: read from the PDF text, with CPT and ICD-10 values confirmed by a reviewer.
- **Rows.**
  - *As-shipped:* the pipeline's default model and sampling for this step.
  - *One row per candidate model,* applied with `PA_MODEL_ID_MEDICAL_EXTRACTION`. Candidates run at the step's default temperature (0.1) unless the model rejects non-default sampling.
- **Runtime.**
  - S3, DynamoDB and CloudWatch are mocked with `moto`.
  - Bedrock calls are real.
  - Textract is replayed from `fixtures/textract/`, recorded through the production code path.
- **Scope.** Only the 3 PDFs exercise a model. The FHIR route parses codes from the bundle, so the 4 FHIR cases are model-independent controls.

## Run it

From the repository root, with the dev dependencies and `requirements.txt` installed:

```bash
pip install -e '.[dev]' -r requirements.txt

# Plan and cost estimate. No AWS calls.
PYTHONPATH=src python -m evaluation.extraction.runner --dry-run

# Real run. Uses your AWS credentials and needs Bedrock access to each model.
AWS_PROFILE=<profile> PYTHONPATH=src python -m evaluation.extraction.runner --trials 3
```

Results go to `evaluation/results/<timestamp>/`:
- `report.md`: summary, limits and every trial
- `trials.jsonl` and `trials.csv`: per-trial data (the committed example run includes `trials.jsonl` only)
- `manifest.json`: commit, models, sampling, lifecycle checks and price table date

Claude Haiku 4.5 is included only if Bedrock reports it `ACTIVE` when the run starts, and the report records that check. Use `--models` to choose the candidates; the as-shipped row always runs.

The committed run (`results/2026-10-16-run2/`: 4 rows, 3 trials per PDF case) had an estimated model cost of $0.27, computed from token counts and the price table. The dry run prints an estimate before you spend anything.

## Tests

```bash
pytest tests/evaluation -q
```

The tests need no AWS credentials. They use a fake Bedrock client.

## Limits

- Three synthetic PDFs are a small sample. Results describe these documents only, and the report makes no ranking claims.
- Scores reflect what the pipeline returns after its own post-processing. Two pipeline behaviors affected every row equally in the committed run, and the report notes both:
  - Payor normalization drops "Anthem" from "Anthem Blue Cross Blue Shield".
  - The FHIR route keeps only the first CPT code in a bundle.
- Prices come from the AWS Price List API, dated in `pricing.json`. Re-check them before relying on cost figures.

## Updating the inputs

- **Rebuild cases:** `python -m evaluation.ground_truth build`. Confirmed PDF cases are left unchanged.
- **Validate them:** `python -m evaluation.ground_truth check`.
- **New or changed PDFs** need new Textract fixtures. Recording requires a temporary S3 bucket, because Textract reads multi-page PDFs only from S3.
