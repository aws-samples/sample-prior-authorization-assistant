# Prior Authorization Assistant

An AI-powered system that automates prior authorization validation for healthcare providers using Amazon Bedrock, AWS Step Functions, and a React frontend. The application extracts clinical information from medical documents, matches it against payor policies, and provides detailed gap analysis with recommendations.

## Overview

Prior authorization is a utilization management process used by health insurance companies that requires healthcare providers to obtain advance approval before delivering specific services. This process is often manual, time-consuming, and error-prone.

This solution automates the workflow by:

1. Accepting prior authorization documents (PDF, FHIR JSON) through a web portal
2. Extracting medical information (CPT codes, ICD codes, payor details) using Amazon Textract and Amazon Bedrock
3. Matching requests against relevant payor policies stored in an Amazon Bedrock Knowledge Base
4. Extracting clinical criteria from matched policies
5. Reviewing medical records against policy requirements
6. Producing a detailed assessment with gap analysis and recommendations

## Architecture

![Architecture Diagram](docs/architecture-diagram.jpg)

The solution uses a serverless, event-driven architecture:

| Component | AWS Service | Purpose |
|-----------|-------------|---------|
| Frontend | Amazon S3 + Amazon CloudFront | React/TypeScript web application |
| API | Amazon API Gateway | REST API with rate limiting |
| Document Processing | AWS Lambda (Python 3.11) | Extract medical data from documents |
| AI/ML | Amazon Bedrock (Amazon Nova Pro, Claude Sonnet 4.6) | Clinical analysis and policy matching (see [Model configuration](#model-configuration)) |
| Document OCR | Amazon Textract | PDF text extraction |
| Policy Store | Amazon Bedrock Knowledge Base | Payor policy retrieval (RAG) |
| Orchestration | AWS Step Functions | Multi-step validation workflow |
| Document Storage | Amazon S3 | Uploaded documents and policies |
| State Management | Amazon DynamoDB | Request tracking and results |
| Encryption | AWS KMS | Customer-managed encryption keys |
| Infrastructure | AWS CDK (Python) | Infrastructure as code |

All resources follow the naming pattern `pa-assist-{resource}-{environment}`.

## Prerequisites

Before deploying this solution, ensure you have the following installed:

- [Python 3.11](https://www.python.org/downloads/) or later
- [Node.js 18.x](https://nodejs.org/) or later
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- [AWS CDK v2](https://docs.aws.amazon.com/cdk/v2/guide/getting-started.html) (`npm install -g aws-cdk`)
- [Docker](https://docs.docker.com/get-docker/) (required for Lambda bundling)
- An AWS account with permissions to create the resources listed above
- AWS CLI configured with valid credentials (`aws configure`)

### Supported AWS Regions

This solution requires Amazon Bedrock with Amazon Nova Pro and Claude Sonnet 4.6 model access. Verify [model availability](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) in your target region. The default deployment region is `us-east-1`.

### Amazon Bedrock Setup

Before deployment, you must [enable model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) for the following models in the Amazon Bedrock console:

- Amazon Nova Pro (for medical data extraction, policy selection, compliance assessment, evidence mapping, and gap analysis)
- Anthropic Claude Sonnet 4.6 (for clinical criteria extraction, policy metadata extraction, and Knowledge Base response generation)
- Amazon Titan Embeddings V2 (for the Knowledge Base vector store)

The CDK stack automatically creates the Amazon Bedrock Knowledge Base, S3 Vectors storage, and data source. No manual KB setup is required.

### Model configuration

Each pipeline step uses the Bedrock model and sampling settings below. They are defined in `src/common/model_config.py`. Generation calls use the Amazon Bedrock Converse API, except `kb_generation`, which uses Knowledge Base RetrieveAndGenerate.

| Step (role) | Lambda function | Default model | Sampling |
|-------------|-----------------|---------------|----------|
| `medical_extraction` | Extractor | `amazon.nova-pro-v1:0` | temperature 0.1, top_p 0.9 |
| `policy_selection` | Selector \* | `amazon.nova-pro-v1:0` | temperature 0.1, top_p 0.9 |
| `policy_criteria_extraction` | Selector \* | `amazon.nova-pro-v1:0` | temperature 0.1, top_p 0.9 |
| `compliance_assessment` | Reviewer | `amazon.nova-pro-v1:0` | temperature 0.1, top_p 0.9 |
| `evidence_mapping` | Reviewer | `amazon.nova-pro-v1:0` | temperature 0.1, top_p 0.9 |
| `gap_analysis` | Reviewer | `amazon.nova-pro-v1:0` | temperature 0.1, top_p 0.9 |
| `clinical_criteria_extraction` | ExtractClinicalCriteria | `us.anthropic.claude-sonnet-4-6` | temperature 0.0 |
| `policy_metadata_extraction` | MetadataExtractor, MetadataProcessor | `us.anthropic.claude-sonnet-4-6` | temperature 0.0 |
| `kb_generation` | IntelligentPolicySelector | `us.anthropic.claude-sonnet-4-6` | model default |

\* The Selector Lambda is deployed, but the Step Functions workflow does not invoke it. In the deployed workflow, policy selection runs in IntelligentPolicySelector (`kb_generation`).

**Default data routing.** The three Claude Sonnet 4.6 steps use `us.anthropic.claude-sonnet-4-6`, a cross-Region inference profile. In the default deployment, prompts for these steps, which include clinical content such as diagnosis codes and patient context, can be processed in other AWS Regions in the United States. The Nova Pro steps use the base model ID and run in the deployment Region.

#### Overriding a model

Overrides are environment variables on the Lambda function that runs the step:

- `PA_MODEL_ID_<ROLE>` (for example `PA_MODEL_ID_GAP_ANALYSIS`) overrides one step.
- `PA_MODEL_ID` overrides every step in that Lambda function. A step-specific variable takes precedence.
- `PA_TEMPERATURE_<ROLE>` and `PA_TOP_P_<ROLE>` set sampling for an overridden step, as numbers from 0 to 1. When a step's model is overridden, its default sampling settings are not carried over, because valid settings differ by model (for example, some models reject `temperature` and `top_p` together). An override equal to the default model ID keeps the default sampling. `kb_generation` uses only the model ID and sends no sampling settings.

Set overrides in the CDK stack, not in the Lambda console. Console changes aren't in the stack, so they drift from it and can be overwritten by a later deployment. Most functions share `common_env` in `infrastructure/stacks/consolidated_stack.py`. MetadataExtractor has its own `environment`, and MetadataProcessor is defined in `infrastructure/stacks/policy_extractor_stack.py`.

Before overriding a model, check the following:

- **IAM.** The Lambda execution role allows a fixed list of models (`infrastructure/stacks/consolidated_stack.py`). Besides the defaults, the list includes the base model IDs `amazon.nova-lite-v1:0`, `amazon.nova-micro-v1:0`, `anthropic.claude-3-haiku-20240307-v1:0`, and `anthropic.claude-3-5-haiku-20241022-v1:0`. Other models, including newer Claude models and any inference profile other than `us.anthropic.claude-sonnet-4-6`, need an IAM change. Changing it is an operator decision, and this solution does not make it for you. The foundation-model entries use a Region wildcard so the `us.` profile can reach other US Regions.
- **Data routing.** Model IDs that start with `us.` are cross-Region inference profiles. Because IAM already allows `us.anthropic.claude-sonnet-4-6`, setting `PA_MODEL_ID` to it on a function such as Extractor or Reviewer moves every step in that function, including full clinical document text, to cross-Region processing with no IAM change. Choose model IDs that match your data residency requirements.
- **Failures are silent in some steps.** If the overridden model is denied or fails, four steps log a warning and continue with a fallback instead of failing the workflow. Clinical criteria extraction uses a generic default criterion. Evidence mapping and gap analysis return no results. Policy metadata extraction writes minimal metadata. After changing a model, check each function's CloudWatch logs for Bedrock errors before relying on results.
- **Model access and Region.** Enable access to the new model in the Amazon Bedrock console, and check the model card for the Regions where the base model ID supports on-demand invocation. Elsewhere, use an inference profile ID, which also needs an IAM change.
- **Knowledge Base support.** `kb_generation` must use a model that supports Bedrock Knowledge Bases.

## Deployment

### 1. Clone the repository

```bash
git clone https://github.com/aws-samples/<repo-name>.git
cd <repo-name>
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
pip install -r infrastructure/requirements.txt
```

### 3. Install Lambda layer dependencies

```bash
pip install -r layers/python-deps/requirements.txt -t layers/python-deps/python/ --upgrade
```

### 4. Bootstrap AWS CDK (first time only)

```bash
cd infrastructure
cdk bootstrap aws://<ACCOUNT_ID>/<REGION>
```

### 5. Configure deployment parameters

From the `infrastructure/` directory, edit `cdk.json` and set the context parameters:

```json
{
  "context": {
    "environment": "dev"
  }
}
```

| Parameter | Description | Required |
|-----------|-------------|----------|
| `environment` | Deployment environment (`dev`, `staging`, `prod`) | Yes (default: `dev`) |

Or set environment variables:

```bash
export ENVIRONMENT=dev
```

### 6. Deploy the stack

Still in the `infrastructure/` directory:

```bash
cdk deploy --require-approval never --outputs-file deployment-outputs.json
```

Estimated deployment time: 10-15 minutes.

On successful deployment, CDK prints the following outputs:

| Output | Description |
|--------|-------------|
| `APIEndpointOutput` | API Gateway endpoint URL |
| `FrontendURLOutput` | CloudFront distribution URL for the web app |
| `FrontendBucketOutput` | S3 bucket name for frontend assets |
| `DocumentsBucketOutput` | S3 bucket name for uploaded documents |
| `PoliciesBucketOutput` | S3 bucket name for payor policies |
| `KnowledgeBaseIdOutput` | Bedrock Knowledge Base ID (auto-created) |
| `DataSourceIdOutput` | Bedrock Data Source ID (auto-created) |

### 7. Configure the frontend

Return to the project root and create a `.env` file from the example:

```bash
cd ..
cp frontend/.env.example frontend/.env
```

Update `frontend/.env` with the `APIEndpointOutput` value from step 6:

```
REACT_APP_API_BASE_URL=https://<your-api-id>.execute-api.<region>.amazonaws.com/prod
```

### 8. Build and deploy the frontend

```bash
cd frontend
npm install
npm run build
```

Upload the build output to the frontend S3 bucket (use the `FrontendBucketOutput` value from step 6):

```bash
aws s3 sync build/ s3://<FrontendBucketOutput> --delete
```

To serve locally for development instead:

```bash
npm start
```

## Usage

1. Open the web application URL (CloudFront distribution URL from CDK outputs)
2. Upload a prior authorization document (PDF or FHIR JSON)
3. The system automatically processes the document through the Step Functions workflow:
   - Extracts medical information (CPT/ICD codes, payor, provider details)
   - Scrapes and extracts the relevant payor policies automatically
   - Syncs the policies to the Bedrock Knowledge Base
   - Selects matching policies using RAG-powered intelligent search
   - Extracts clinical criteria from the matched policies
   - Reviews the medical record against policy requirements
4. On first upload, the policy extraction and KB sync may take a few minutes. Subsequent uploads for the same payor will be faster since policies are already indexed.
5. View the results page with the detailed assessment, gap analysis, and recommendations

## Sample Data

The `samples/` directory contains synthetic prior authorization request documents for testing:

- Aetna physical therapy (PDF and FHIR JSON)
- Aetna spinal fusion (FHIR JSON)
- Anthem knee surgery (PDF and FHIR JSON)
- Humana cardiac catheterization (PDF and FHIR JSON)

Upload any of these through the web UI to test the end-to-end workflow. These documents are entirely synthetic and do not contain any real patient information.

## Cleanup

To remove all deployed resources and avoid ongoing charges:

```bash
cd infrastructure
cdk destroy --force
```

This removes all AWS resources created by the stack, including S3 buckets, DynamoDB tables, Lambda functions, and the Step Functions state machine. S3 buckets with the `autoDeleteObjects` property will have their contents removed automatically.

> **Important:** Ensure you have downloaded any data you need from S3 buckets before running cleanup.

## Security

This application follows AWS security best practices:

- All data encrypted at rest using customer-managed AWS KMS keys
- All data encrypted in transit via TLS
- IAM roles follow the principle of least privilege
- API Gateway enforces HTTPS with rate limiting and throttling
- DynamoDB tables have point-in-time recovery enabled
- CloudWatch logging enabled for audit trails
- No hardcoded credentials — all secrets managed through environment variables and CDK context

See [CONTRIBUTING](CONTRIBUTING.md) for information on reporting security issues.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
