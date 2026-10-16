# Extraction model comparison

- Git commit: a113c8bcec2ff4ca82aa219584bf9a63e6002702
- Started: 2026-10-16T19:45:44+00:00
- Finished: 2026-10-16T19:52:43+00:00

## Limits

- 3 PDF cases exercise a model; each runs 3 trials per row.
- 4 FHIR cases are model-independent controls (codes are parsed from the bundle with no model call); each runs once.
- Only the `medical_extraction` step runs; no other workflow step, Knowledge Base or deployed stack is used.
- The cases are synthetic sample documents from `samples/`. Expected values are what those documents state, not clinical ground truth.
- PDF expected payor values were not separately reviewed (per the case files' `review` notes): aetna-physical-therapy-pa-approved.pdf, anthem-knee-surgery-pa-approved.pdf, humana-cardiac-cath-pa-approved.pdf.
- Failed or truncated trials score zero on every field and count against their row.
- Latency is the model latency Bedrock reports (`metrics.latencyMs`), summed over a trial's model calls; it is not wall-clock time.
- Costs are estimates from token counts and the dated price table, not billed amounts.
- With this few cases and trials, the results describe these documents only. This report does not name a winner or rank models beyond the numbers shown.

## PDF results (model-dependent)

| Row | Model ID | Sampling | Cases | Trials | Payor accuracy | CPT P | CPT R | ICD P | ICD R | Failure rate | Truncation rate | Latency p50 (ms) | Latency p95 (ms) | Mean cost per case | Total cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as-shipped | amazon.nova-pro-v1:0 | temperature 0.1, top_p 0.9 (role defaults) | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 2904 | 3241 | $0.0037 | $0.0330 |
| us.anthropic.claude-sonnet-4-6 | us.anthropic.claude-sonnet-4-6 | temperature 0.1 | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 4104 | 5253 | $0.0190 | $0.1707 |
| openai.gpt-oss-120b-1:0 | openai.gpt-oss-120b-1:0 | temperature 0.1 | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 6488 | 14069 | $0.0012 | $0.0108 |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | us.anthropic.claude-haiku-4-5-20251001-v1:0 | temperature 0.1 | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 4334 | 4734 | $0.0065 | $0.0589 |

## FHIR controls (model-independent)

| Row | Model ID | Sampling | Cases | Trials | Payor accuracy | CPT P | CPT R | ICD P | ICD R | Failure rate | Truncation rate | Latency p50 (ms) | Latency p95 (ms) | Mean cost per case | Total cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as-shipped | amazon.nova-pro-v1:0 | temperature 0.1, top_p 0.9 (role defaults) | 4 | 4 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | n/a | $0.0000 | $0.0000 |
| us.anthropic.claude-sonnet-4-6 | us.anthropic.claude-sonnet-4-6 | temperature 0.1 | 4 | 4 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | n/a | $0.0000 | $0.0000 |
| openai.gpt-oss-120b-1:0 | openai.gpt-oss-120b-1:0 | temperature 0.1 | 4 | 4 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | n/a | $0.0000 | $0.0000 |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | us.anthropic.claude-haiku-4-5-20251001-v1:0 | temperature 0.1 | 4 | 4 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | n/a | $0.0000 | $0.0000 |

## Unavailable models

| Model ID | Reason |
|---|---|
| us.anthropic.claude-sonnet-5-5 | AccessDeniedException: not available for this account (2026-10-16) |
| us.anthropic.claude-haiku-5-5 | AccessDeniedException: not available for this account (2026-10-16) |

## Lifecycle checks

| Model ID | Status | Source | Checked at |
|---|---|---|---|
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | ACTIVE | bedrock:GetFoundationModel(anthropic.claude-haiku-4-5-20251001-v1:0) | 2026-10-16T19:45:44+00:00 |

## Price table

As of 2026-10-16, source: AWS Price List API (pricing:GetProducts), us-east-1, OnDemand terms. Service codes: AmazonBedrockFoundationModels (Anthropic models), AmazonBedrock (Amazon and OpenAI models). (USD, per 1K tokens). Models missing from the table show cost n/a.

| Model ID | Input | Output |
|---|---|---|
| amazon.nova-pro-v1:0 | 0.0008 | 0.0032 |
| openai.gpt-oss-120b-1:0 | 0.00015 | 0.0006 |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | 0.0011 | 0.0055 |
| us.anthropic.claude-sonnet-4-6 | 0.0033 | 0.0165 |

## Notes

- Only the medical_extraction step runs; no other pipeline step is evaluated.
- Textract input is replayed from fixtures recorded 2026-10-16 through the production DocumentProcessor path. On all three PDFs, synchronous AnalyzeDocument (TABLES, FORMS) fails with UnsupportedDocumentException and the code continues on text detection only, as it does in production.
- FHIR cases make no model call (the FHIR route parses codes from the bundle). They run once per row as a control.
- Payor results pass through MedicalCodeValidator.normalize_payor_name (src/common/validators.py), which maps any name containing 'Blue Cross Blue Shield' to 'Blue Cross Blue Shield' and drops 'Anthem'. The Anthem PDF's payor scored as wrong for every row in this run.
- The FHIR route keeps only the first CPT code in a bundle (src/extractors/fhir_extractor.py, single service_info.cpt_code), so multi-procedure bundles lose codes in the FHIR control rows.
- amazon.nova-pro-v1:0 is not run as a separate row: with this override it sends the same model and sampling as the as-shipped row (an override equal to the default model keeps the default sampling).

## Every trial (52)

| Row | Case | Trial | Status | Payor ok | CPT P | CPT R | ICD P | ICD R | Stop reasons | Input tokens | Output tokens | Latency (ms) | Cost | Error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as-shipped | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| as-shipped | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 326 | 2129 | $0.0040 |  |
| as-shipped | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 326 | 2089 | $0.0040 |  |
| as-shipped | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 326 | 3241 | $0.0040 |  |
| as-shipped | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| as-shipped | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| as-shipped | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 329 | 2904 | $0.0033 |  |
| as-shipped | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 331 | 2961 | $0.0033 |  |
| as-shipped | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 329 | 2928 | $0.0033 |  |
| as-shipped | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| as-shipped | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 356 | 3050 | $0.0037 |  |
| as-shipped | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 356 | 2124 | $0.0037 |  |
| as-shipped | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 347 | 2878 | $0.0037 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 540 | 4890 | $0.0212 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 542 | 4834 | $0.0213 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 540 | 4818 | $0.0212 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-sonnet-4-6 | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-sonnet-4-6 | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 403 | 4052 | $0.0163 |  |
| us.anthropic.claude-sonnet-4-6 | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 403 | 4104 | $0.0163 |  |
| us.anthropic.claude-sonnet-4-6 | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 403 | 4083 | $0.0163 |  |
| us.anthropic.claude-sonnet-4-6 | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-sonnet-4-6 | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 466 | 3592 | $0.0189 |  |
| us.anthropic.claude-sonnet-4-6 | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 425 | 3241 | $0.0182 |  |
| us.anthropic.claude-sonnet-4-6 | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 589 | 5253 | $0.0209 |  |
| openai.gpt-oss-120b-1:0 | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| openai.gpt-oss-120b-1:0 | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3300 | 994 | 5619 | $0.0011 |  |
| openai.gpt-oss-120b-1:0 | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3300 | 1530 | 7317 | $0.0014 |  |
| openai.gpt-oss-120b-1:0 | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3300 | 950 | 3304 | $0.0011 |  |
| openai.gpt-oss-120b-1:0 | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| openai.gpt-oss-120b-1:0 | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| openai.gpt-oss-120b-1:0 | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2563 | 1130 | 8081 | $0.0011 |  |
| openai.gpt-oss-120b-1:0 | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2563 | 1233 | 6258 | $0.0011 |  |
| openai.gpt-oss-120b-1:0 | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2563 | 1484 | 9122 | $0.0013 |  |
| openai.gpt-oss-120b-1:0 | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| openai.gpt-oss-120b-1:0 | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2956 | 1260 | 4242 | $0.0012 |  |
| openai.gpt-oss-120b-1:0 | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2956 | 1138 | 14069 | $0.0011 |  |
| openai.gpt-oss-120b-1:0 | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2956 | 1588 | 6488 | $0.0014 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 559 | 4672 | $0.0072 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 559 | 4734 | $0.0072 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 565 | 4587 | $0.0072 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 479 | 4147 | $0.0058 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 483 | 4192 | $0.0059 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 483 | 4310 | $0.0059 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 519 | 4324 | $0.0066 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 519 | 4347 | $0.0066 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 516 | 4334 | $0.0066 |  |
