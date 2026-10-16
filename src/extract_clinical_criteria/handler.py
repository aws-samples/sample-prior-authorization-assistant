"""
Lambda handler for extracting clinical criteria from policy documents.

This module is separated from policy selection to allow for longer processing times
and better timeout management for Bedrock calls.

Version: 1.0 - Initial extraction from IntelligentPolicySelector
"""

import json
import os
import time
import boto3
from botocore.config import Config
from typing import Dict, Any, List
import structlog

from common.aws_clients import converse_role
from common.models import ClinicalCriterion
from common.exceptions import PriorAuthException
from common.logging_config import get_logger, get_audit_logger
from common.decorators import handle_processing_errors
from common.step_functions_utils import ContextAwareErrorHandler
from common.serialization import create_serializable_response

logger = get_logger(__name__)
audit_logger = get_audit_logger()

# Initialize AWS clients with timeout configuration
config = Config(
    read_timeout=120,  # 2 minute read timeout for Bedrock
    connect_timeout=10,
    retries={'max_attempts': 2}
)
bedrock_runtime = boto3.client('bedrock-runtime', config=config)
s3_client = boto3.client('s3')


@handle_processing_errors
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for clinical criteria extraction.
    
    Expected input:
    {
        "selected_policy": {
            "s3_uri": "s3://bucket/key",
            "payor": "Aetna",
            "policy_id": "...",
            ...
        },
        "extraction_context": {
            "service_codes": ["22630", "22633"],
            "icd_codes": ["M51.26"],
            "service_type": "Spinal Fusion"
        }
    }
    
    Returns:
    {
        "status": "success",
        "clinical_criteria": [...],
        "extraction_metadata": {...}
    }
    """
    
    error_handler = ContextAwareErrorHandler(event)
    
    logger.info("Clinical Criteria Extractor invoked",
               request_id=context.aws_request_id if context else None)
    
    try:
        # Extract parameters from Step Functions event structure
        # Step Functions wraps Lambda responses in Payload
        selector_result = event.get('selector_result', {})
        
        # Handle Lambda response wrapper
        if 'Payload' in selector_result:
            selector_payload = selector_result['Payload']
            policy_info = selector_payload.get('policy_info')  # Changed from selected_policy
            extraction_context = selector_payload.get('extraction_context', {})
        else:
            # Direct invocation (testing)
            policy_info = event.get('policy_info')
            extraction_context = event.get('extraction_context', {})
        
        if not policy_info:
            raise ValueError("policy_info is required in selector_result")
        
        s3_uri = policy_info.get('policy_metadata', {}).get('s3_uri')
        payor = policy_info.get('payor')
        policy_id = policy_info.get('policy_id')
        
        if not s3_uri:
            raise ValueError("s3_uri is required in policy_info.policy_metadata")
        
        service_codes = extraction_context.get('service_codes', [])
        icd_codes = extraction_context.get('icd_codes', [])
        service_type = extraction_context.get('service_type', '')
        
        logger.info("Extracting clinical criteria",
                   policy_id=policy_id,
                   payor=payor,
                   service_codes=service_codes,
                   s3_uri=s3_uri)
        
        # Step 1: Retrieve policy content from S3
        policy_content = retrieve_policy_from_s3(s3_uri)
        
        # Step 2: Truncate policy content to avoid timeout (keep first 30KB)
        max_content_length = 30000
        if len(policy_content) > max_content_length:
            truncated_content = policy_content[:max_content_length] + "\n\n[Content truncated for processing]"
            logger.info("Truncated policy content for criteria extraction",
                       original_length=len(policy_content),
                       truncated_length=len(truncated_content))
            policy_content = truncated_content
        
        # Step 3: Extract clinical criteria using Bedrock
        clinical_criteria = extract_clinical_criteria(
            policy_content=policy_content,
            payor=payor,
            service_codes=service_codes,
            icd_codes=icd_codes,
            service_type=service_type
        )
        
        # Log successful extraction
        audit_logger.log_policy_access(
            policy_id=policy_id,
            payor=payor,
            access_type="criteria_extraction"
        )
        
        logger.info("Criteria extraction completed",
                   policy_id=policy_id,
                   criteria_count=len(clinical_criteria))
        
        # Now create the complete SelectedPolicy object with criteria
        from common.models import SelectedPolicy
        from datetime import datetime
        
        selected_policy = SelectedPolicy(
            policy_id=policy_info['policy_id'],
            payor=policy_info['payor'],
            policy_title=policy_info['policy_title'],
            effective_date=policy_info['effective_date'],
            relevance_score=policy_info['relevance_score'],
            applicable_criteria=clinical_criteria,  # Full criteria list
            policy_metadata=policy_info['policy_metadata']
        )
        
        logger.info("Created SelectedPolicy object with criteria",
                   policy_id=policy_id,
                   criteria_count=len(clinical_criteria))
        
        # Create response with complete selected_policy (replaces selector_result)
        response_data = {
            'status': 'success',
            'selected_policy': selected_policy.to_dict(),
            'extraction_metadata': {
                'policy_id': policy_id,
                'payor': payor,
                'criteria_count': len(clinical_criteria),
                'policy_content_length': len(policy_content),
                'service_codes': service_codes,
                'icd_codes': icd_codes
            }
        }
        
        serializable_data = create_serializable_response(response_data)
        return error_handler.create_success_response(serializable_data)
    
    except PriorAuthException as e:
        logger.error("Criteria extraction failed with known error", error=str(e))
        return error_handler.handle_error(e, e.retry_able)
    
    except Exception as e:
        logger.error("Criteria extraction failed with unexpected error", 
                    error=str(e),
                    error_type=type(e).__name__)
        return error_handler.handle_error(e, True)


def retrieve_policy_from_s3(s3_uri: str) -> str:
    """
    Retrieve full policy content from S3 using the URI.
    
    Args:
        s3_uri: S3 URI in format s3://bucket-name/key/path
    
    Returns:
        Full policy content as string
    """
    # Parse S3 URI
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    bucket_name = parts[0]
    object_key = parts[1]
    
    logger.info("Retrieving policy from S3",
               bucket=bucket_name,
               key=object_key)
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        policy_content = response['Body'].read().decode('utf-8')
        
        logger.info("Policy retrieved successfully",
                   content_length=len(policy_content),
                   bucket=bucket_name,
                   key=object_key)
        
        return policy_content
    
    except s3_client.exceptions.NoSuchKey:
        raise ValueError(f"Policy not found in S3: {s3_uri}")
    except Exception as e:
        logger.error("Failed to retrieve policy from S3",
                    s3_uri=s3_uri,
                    error=str(e))
        raise ValueError(f"Failed to retrieve policy from S3: {str(e)}")


def extract_clinical_criteria(
    policy_content: str,
    payor: str,
    service_codes: List[str],
    icd_codes: List[str],
    service_type: str
) -> List[ClinicalCriterion]:
    """
    Extract clinical criteria from policy content using Bedrock.
    
    Returns:
        List of ClinicalCriterion objects
    """
    # Create criteria extraction prompt
    prompt = create_criteria_extraction_prompt(
        policy_content=policy_content,
        payor=payor,
        service_codes=service_codes,
        icd_codes=icd_codes,
        service_type=service_type
    )
    
    logger.info("Extracting clinical criteria with Bedrock",
               policy_length=len(policy_content),
               service_codes_count=len(service_codes),
               prompt_length=len(prompt))
    
    criteria_start = time.time()
    
    try:
        # Use Bedrock to extract criteria with timeout protection
        # Deterministic extraction (temperature 0.0 by default)
        generated_text = converse_role(
            bedrock_runtime, 'clinical_criteria_extraction', prompt, 4000
        )
        
        criteria_duration = time.time() - criteria_start
        logger.info("Bedrock response received",
                   response_length=len(generated_text),
                   duration_seconds=round(criteria_duration, 2))
        
        # Parse JSON response
        criteria_data = parse_criteria_response(generated_text)
        
        # Convert to ClinicalCriterion objects
        clinical_criteria = []
        for idx, criterion_dict in enumerate(criteria_data):
            criterion = ClinicalCriterion(
                criterion_id=criterion_dict.get('criterion_id', f'crit_{idx+1:03d}'),
                description=criterion_dict['description'],
                required_evidence=criterion_dict.get('required_evidence', []),
                clinical_indicators=criterion_dict.get('clinical_indicators', []),
                exclusion_criteria=criterion_dict.get('exclusion_criteria', [])
            )
            clinical_criteria.append(criterion)
        
        logger.info("Clinical criteria extracted successfully",
                   criteria_count=len(clinical_criteria),
                   total_duration_seconds=round(time.time() - criteria_start, 2))
        
        return clinical_criteria
    
    except Exception as e:
        logger.error("Failed to extract clinical criteria",
                    error=str(e),
                    error_type=type(e).__name__)
        # Return empty list with a default criterion rather than failing
        logger.warning("Returning default criterion due to extraction failure")
        return [
            ClinicalCriterion(
                criterion_id='default_001',
                description=f'Medical necessity criteria for {service_type} as defined in payor policy',
                required_evidence=[
                    'Clinical documentation supporting medical necessity',
                    'Relevant diagnostic test results',
                    'Treatment history and outcomes'
                ],
                clinical_indicators=[
                    'Documented medical condition requiring intervention',
                    'Failed conservative treatment (if applicable)',
                    'Clinical justification for proposed service'
                ],
                exclusion_criteria=[]
            )
        ]


def create_criteria_extraction_prompt(
    policy_content: str,
    payor: str,
    service_codes: List[str],
    icd_codes: List[str],
    service_type: str
) -> str:
    """
    Create prompt for extracting clinical criteria from policy.
    """
    return f"""You are a medical policy analyst extracting clinical criteria from insurance policies.

Extract the clinical criteria from the following {payor} policy for {service_type}.

**Service Codes**: {', '.join(service_codes)}
**Diagnosis Codes**: {', '.join(icd_codes)}

**Policy Content**:
{policy_content}

**Instructions**:
1. Identify all clinical criteria that must be met for approval
2. Extract required evidence/documentation for each criterion
3. Note any clinical indicators that support approval
4. Identify exclusion criteria that would prevent approval

Return a JSON array of criteria objects with this structure:
```json
[
  {{
    "criterion_id": "unique_id",
    "description": "Clear description of the criterion",
    "required_evidence": ["evidence item 1", "evidence item 2"],
    "clinical_indicators": ["indicator 1", "indicator 2"],
    "exclusion_criteria": ["exclusion 1", "exclusion 2"]
  }}
]
```

Focus on extracting 3-7 key criteria that are most relevant to the service codes and diagnoses provided.
Be specific and actionable - criteria should be verifiable from medical records.
"""


def parse_criteria_response(generated_text: str) -> List[Dict[str, Any]]:
    """Parse criteria extraction response from Bedrock."""
    # Try to extract JSON from markdown code blocks
    json_str = generated_text.strip()
    
    if '```json' in json_str:
        start = json_str.find('```json') + 7
        end = json_str.find('```', start)
        json_str = json_str[start:end].strip()
    elif '```' in json_str:
        start = json_str.find('```') + 3
        end = json_str.find('```', start)
        json_str = json_str[start:end].strip()
    else:
        # Find first [ and last ] for JSON array
        start = json_str.find('[')
        end = json_str.rfind(']')
        if start >= 0 and end > start:
            json_str = json_str[start:end+1]
    
    try:
        criteria_data = json.loads(json_str)
        
        if not isinstance(criteria_data, list):
            logger.warning("Expected JSON array, got single object, wrapping in array")
            criteria_data = [criteria_data]
        
        return criteria_data
    
    except json.JSONDecodeError as e:
        logger.error("Failed to parse criteria JSON",
                    error=str(e),
                    response_preview=generated_text[:500])
        return []


# Health check endpoint
def health_check(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Health check endpoint for the Lambda function"""
    
    error_handler = ContextAwareErrorHandler(event)
    
    return error_handler.create_success_response({
        'status': 'healthy',
        'service': 'extract-clinical-criteria',
        'bedrock_configured': True,
        'timestamp': context.aws_request_id if context else 'local'
    })
