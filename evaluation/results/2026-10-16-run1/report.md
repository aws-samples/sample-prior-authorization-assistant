# Extraction model comparison

- Git commit: f41d5e64a390bcc90d8dd81d56b7b1d95649ce43
- Started: 2026-10-16T19:28:56+00:00
- Finished: 2026-10-16T19:32:25+00:00

## Limits

- 3 PDF cases exercise a model; each runs 3 trials per row.
- 4 FHIR cases are model-independent controls (codes are parsed from the bundle with no model call); each runs once.
- Only the `medical_extraction` step runs; no other workflow step, Knowledge Base or deployed stack is used.
- The cases are synthetic sample documents from `samples/`. Expected values are what those documents state, not clinical ground truth.
- PDF expected payor values were not separately reviewed (per the case files' `review` notes): aetna-physical-therapy-pa-approved.pdf, anthem-knee-surgery-pa-approved.pdf, humana-cardiac-cath-pa-approved.pdf.
- Failed or truncated trials score zero on every field and count against their row.
- With this few cases and trials, the results describe these documents only. This report does not name a winner or rank models beyond the numbers shown.

## PDF results (model-dependent)

| Row | Model ID | Sampling | Cases | Trials | Payor accuracy | CPT P | CPT R | ICD P | ICD R | Failure rate | Truncation rate | Latency p50 (ms) | Latency p95 (ms) | Mean cost per case | Total cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as-shipped | amazon.nova-pro-v1:0 | temperature 0.1, top_p 0.9 (role defaults) | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 3040 | 3736 | $0.0037 | $0.0334 |
| us.anthropic.claude-sonnet-4-6 | us.anthropic.claude-sonnet-4-6 | temperature 0.1 | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 4738 | 5265 | $0.0190 | $0.1714 |
| amazon.nova-pro-v1:0 | amazon.nova-pro-v1:0 | temperature 0.1 | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 3156 | 3552 | $0.0037 | $0.0329 |
| openai.gpt-oss-120b-1:0 | openai.gpt-oss-120b-1:0 | temperature 0.1 | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 6044 | 17457 | $0.0012 | $0.0107 |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | us.anthropic.claude-haiku-4-5-20251001-v1:0 | temperature 0.1 | 3 | 9 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 4287 | 4899 | $0.0065 | $0.0589 |

## FHIR controls (model-independent)

| Row | Model ID | Sampling | Cases | Trials | Payor accuracy | CPT P | CPT R | ICD P | ICD R | Failure rate | Truncation rate | Latency p50 (ms) | Latency p95 (ms) | Mean cost per case | Total cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as-shipped | amazon.nova-pro-v1:0 | temperature 0.1, top_p 0.9 (role defaults) | 4 | 4 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | n/a | $0.0000 | $0.0000 |
| us.anthropic.claude-sonnet-4-6 | us.anthropic.claude-sonnet-4-6 | temperature 0.1 | 4 | 4 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | n/a | $0.0000 | $0.0000 |
| amazon.nova-pro-v1:0 | amazon.nova-pro-v1:0 | temperature 0.1 | 4 | 4 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | n/a | $0.0000 | $0.0000 |
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
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | ACTIVE | bedrock:GetFoundationModel(anthropic.claude-haiku-4-5-20251001-v1:0) | 2026-10-16T19:28:56+00:00 |

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
- Payor results pass through MedicalCodeValidator.normalize_payor_name (src/common/validators.py), which maps any name containing 'Blue Cross Blue Shield' to 'Blue Cross Blue Shield' and drops 'Anthem'. The Anthem PDF's payor therefore scores as wrong for every row, independent of the model.
- The FHIR route keeps only the first CPT code in a bundle (src/extractors/fhir_extractor.py, single service_info.cpt_code), so multi-procedure bundles lose codes in the FHIR control rows.

## Every trial (65)

| Row | Case | Trial | Status | Payor ok | CPT P | CPT R | ICD P | ICD R | Stop reasons | Input tokens | Output tokens | Latency (ms) | Cost | Error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as-shipped | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| as-shipped | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 362 | 3493 | $0.0041 |  |
| as-shipped | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 362 | 2959 | $0.0041 |  |
| as-shipped | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 377 | 3736 | $0.0041 |  |
| as-shipped | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| as-shipped | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| as-shipped | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 329 | 2664 | $0.0033 |  |
| as-shipped | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 329 | 3183 | $0.0033 |  |
| as-shipped | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 329 | 3040 | $0.0033 |  |
| as-shipped | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| as-shipped | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 356 | 2795 | $0.0037 |  |
| as-shipped | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 356 | 2463 | $0.0037 |  |
| as-shipped | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 356 | 3308 | $0.0037 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 542 | 4814 | $0.0213 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 544 | 4738 | $0.0213 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 542 | 4825 | $0.0213 |  |
| us.anthropic.claude-sonnet-4-6 | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-sonnet-4-6 | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-sonnet-4-6 | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 403 | 5265 | $0.0163 |  |
| us.anthropic.claude-sonnet-4-6 | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 403 | 4070 | $0.0163 |  |
| us.anthropic.claude-sonnet-4-6 | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 403 | 4100 | $0.0163 |  |
| us.anthropic.claude-sonnet-4-6 | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-sonnet-4-6 | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 462 | 3563 | $0.0188 |  |
| us.anthropic.claude-sonnet-4-6 | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 587 | 4866 | $0.0209 |  |
| us.anthropic.claude-sonnet-4-6 | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 466 | 3611 | $0.0189 |  |
| amazon.nova-pro-v1:0 | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| amazon.nova-pro-v1:0 | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 326 | 3536 | $0.0040 |  |
| amazon.nova-pro-v1:0 | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 311 | 2802 | $0.0039 |  |
| amazon.nova-pro-v1:0 | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3666 | 326 | 2813 | $0.0040 |  |
| amazon.nova-pro-v1:0 | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| amazon.nova-pro-v1:0 | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| amazon.nova-pro-v1:0 | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 331 | 2960 | $0.0033 |  |
| amazon.nova-pro-v1:0 | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 331 | 2964 | $0.0033 |  |
| amazon.nova-pro-v1:0 | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2809 | 331 | 3552 | $0.0033 |  |
| amazon.nova-pro-v1:0 | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| amazon.nova-pro-v1:0 | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 356 | 3348 | $0.0037 |  |
| amazon.nova-pro-v1:0 | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 349 | 3239 | $0.0037 |  |
| amazon.nova-pro-v1:0 | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3220 | 347 | 3156 | $0.0037 |  |
| openai.gpt-oss-120b-1:0 | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| openai.gpt-oss-120b-1:0 | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3300 | 1375 | 6953 | $0.0013 |  |
| openai.gpt-oss-120b-1:0 | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3300 | 1086 | 6149 | $0.0011 |  |
| openai.gpt-oss-120b-1:0 | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3300 | 1142 | 6204 | $0.0012 |  |
| openai.gpt-oss-120b-1:0 | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| openai.gpt-oss-120b-1:0 | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| openai.gpt-oss-120b-1:0 | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2563 | 1263 | 17457 | $0.0011 |  |
| openai.gpt-oss-120b-1:0 | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2563 | 1330 | 4865 | $0.0012 |  |
| openai.gpt-oss-120b-1:0 | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2563 | 1053 | 6044 | $0.0010 |  |
| openai.gpt-oss-120b-1:0 | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| openai.gpt-oss-120b-1:0 | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2956 | 1071 | 5152 | $0.0011 |  |
| openai.gpt-oss-120b-1:0 | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2956 | 1587 | 4706 | $0.0014 |  |
| openai.gpt-oss-120b-1:0 | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2956 | 1350 | 5508 | $0.0013 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-physical-therapy-pa-approved.fhir | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-physical-therapy-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 537 | 4476 | $0.0071 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-physical-therapy-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 559 | 4843 | $0.0072 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-physical-therapy-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3736 | 559 | 4899 | $0.0072 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | aetna-spinal-fusion-pa-denied.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | anthem-knee-surgery-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | anthem-knee-surgery-pa-approved.pdf | 1 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 483 | 4251 | $0.0059 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | anthem-knee-surgery-pa-approved.pdf | 2 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 483 | 4178 | $0.0059 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | anthem-knee-surgery-pa-approved.pdf | 3 | ok | no | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 2922 | 483 | 4166 | $0.0059 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | humana-cardiac-cath-pa-approved.fhir | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | none | 0 | 0 | n/a | $0.0000 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | humana-cardiac-cath-pa-approved.pdf | 1 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 544 | 4478 | $0.0067 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | humana-cardiac-cath-pa-approved.pdf | 2 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 516 | 4287 | $0.0066 |  |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | humana-cardiac-cath-pa-approved.pdf | 3 | ok | yes | 1.000 | 1.000 | 1.000 | 1.000 | end_turn | 3399 | 516 | 4243 | $0.0066 |  |
