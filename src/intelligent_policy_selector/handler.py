"""
Lambda handler for Intelligent Policy Selection using Bedrock Knowledge Base.

This module replaces the separate SearchPolicies and SelectBestPolicy steps
with a single KB-powered selection that evaluates and selects the best policy
in one operation.

Version: 1.1 - Fixed SelectedPolicy model compatibility
"""

import json
import os
import time
import boto3
from botocore.config import Config
from typing import Dict, Any, Optional
import structlog

from .prompt_templates import create_policy_selection_prompt, create_criteria_extraction_prompt
from common.models import ExtractedData, SelectedPolicy, ClinicalCriterion
from common.model_config import resolve as resolve_model_role
from common.exceptions import PolicySelectionError, PriorAuthException
from common.logging_config import get_logger, get_audit_logger
from common.decorators import handle_processing_errors
from common.step_functions_utils import ContextAwareErrorHandler
from common.serialization import create_serializable_response

logger = get_logger(__name__)
audit_logger = get_audit_logger()

# Initialize AWS clients
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
bedrock_runtime = boto3.client('bedrock-runtime')
s3_client = boto3.client('s3')


@handle_processing_errors
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for intelligent policy selection.
    
    This function:
    1. Uses Bedrock KB RetrieveAndGenerate to select the best policy
    2. Retrieves the full policy content from S3
    3. Extracts clinical criteria from the policy
    4. Returns structured policy selection with criteria
    """
    
    error_handler = ContextAwareErrorHandler(event)
    
    logger.info("Intelligent Policy Selector invoked",
               request_id=context.aws_request_id if context else None)
    
    try:
        # Extract parameters from Step Functions workflow
        extracted_data_dict = extract_data_from_event(event)
        
        if not extracted_data_dict:
            raise ValueError("Could not extract medical data from event")
        
        # Convert to ExtractedData object
        extracted_data = ExtractedData.from_dict(extracted_data_dict)
        
        logger.info("Processing policy selection",
                   document_id=extracted_data.document_id,
                   payor=extracted_data.payor_name,
                   cpt_codes=extracted_data.cpt_codes,
                   service_type=extracted_data.service_type)
        
        # Step 1: Validate required parameters
        validate_search_parameters(extracted_data)
        
        # Step 2: Extract all service codes and descriptions
        all_service_codes, all_service_descriptions = extract_all_service_codes(extracted_data)
        
        # Step 3: Extract diagnosis descriptions
        icd_descriptions = extract_icd_descriptions(extracted_data)
        
        # Note: PayorValidator already checked that policies exist for service codes
        # If we reached this point, policies exist in S3
        
        # Step 4: Use KB to select best policy
        selection_result = select_policy_with_kb(
            extracted_data,
            all_service_codes,
            all_service_descriptions,
            icd_descriptions
        )
        
        # Step 2: Retrieve policy metadata (source_url, title, etc.)
        policy_metadata = retrieve_policy_metadata(selection_result['selected_policy_s3_uri'])
        
        # Step 3: Create policy info dict (NOT SelectedPolicy - that will be created by ExtractClinicalCriteria)
        # We don't create SelectedPolicy here because it requires criteria, which will be extracted in the next Lambda
        policy_id = extract_policy_id_from_uri(selection_result['selected_policy_s3_uri'])
        
        # Use the ACTUAL policy title from metadata, not Claude's potentially hallucinated title
        actual_policy_title = policy_metadata.get('title', f'Policy {policy_id}')
        
        logger.info("Using actual policy title from metadata",
                   policy_id=policy_id,
                   actual_title=actual_policy_title)
        
        policy_info = {
            'policy_id': policy_id,
            'payor': extracted_data.payor_name,
            'policy_title': actual_policy_title,  # From metadata, not Claude
            'effective_date': policy_metadata.get('effective_date', '2024-01-01'),
            'relevance_score': selection_result['confidence_score'],
            'policy_metadata': {
                's3_uri': selection_result['selected_policy_s3_uri'],
                'source_url': policy_metadata.get('source_url', policy_metadata.get('source', '')),
                'selection_rationale': selection_result['selection_rationale'],
                'alternative_policies': selection_result.get('alternative_policies_considered', []),
                'evaluation_summary': selection_result.get('evaluation_summary', {}),
                'payor': policy_metadata.get('payor', extracted_data.payor_name),
                'search_code': policy_metadata.get('search_code', ''),
                'service_category': policy_metadata.get('service_category', ''),
                'procedure_codes': policy_metadata.get('procedure_codes', '')
            }
        }
        
        # Log successful selection
        audit_logger.log_policy_access(
            policy_id=policy_id,
            payor=extracted_data.payor_name,
            access_type="intelligent_select"
        )
        
        logger.info("Policy selection completed",
                   document_id=extracted_data.document_id,
                   policy_id=policy_id,
                   confidence=selection_result['confidence_score'])
        
        # Create response with policy info and extraction context for next Lambda
        response_data = {
            'status': 'success',
            'policy_info': policy_info,  # Simple dict, not SelectedPolicy object
            'selection_metadata': {
                'method': 'kb_intelligent_selection',
                'confidence_score': selection_result['confidence_score'],
                'rationale': selection_result['selection_rationale'],
                'alternatives_considered': len(selection_result.get('alternative_policies_considered', [])),
                'total_policies_reviewed': selection_result.get('evaluation_summary', {}).get('total_policies_reviewed', 0)
            },
            'extraction_context': {
                'service_codes': all_service_codes,
                'icd_codes': extracted_data.icd_codes,
                'service_type': extracted_data.service_type
            }
        }
        
        serializable_data = create_serializable_response(response_data)
        return error_handler.create_success_response(serializable_data)
    
    except PriorAuthException as e:
        logger.error("Policy selection failed with known error", error=str(e))
        return error_handler.handle_error(e, e.retry_able)
    
    except Exception as e:
        logger.error("Policy selection failed with unexpected error", 
                    error=str(e),
                    error_type=type(e).__name__)
        return error_handler.handle_error(e, True)


def validate_search_parameters(extracted_data: ExtractedData) -> None:
    """
    Validate that required search parameters are present.
    
    Raises:
        ValueError: If required parameters are missing
    """
    missing_params = []
    
    # Check required fields
    if not extracted_data.payor_name:
        missing_params.append('payor_name')
    
    if not extracted_data.icd_codes:
        missing_params.append('icd_codes')
    
    if not extracted_data.service_type:
        missing_params.append('service_type')
    
    # Check if at least one procedure/medication code type is present
    structured_fields = extracted_data.structured_fields or {}
    service_details = structured_fields.get('service_details', {})
    
    has_procedure_codes = bool(
        extracted_data.cpt_codes or
        service_details.get('hcpcs_codes') or
        service_details.get('rxnorm_codes')
    )
    
    if not has_procedure_codes:
        missing_params.append('cpt_codes, hcpcs_codes, or rxnorm_codes')
    
    if missing_params:
        logger.error("Missing required search parameters",
                    missing=missing_params,
                    document_id=extracted_data.document_id)
        raise ValueError(f"Missing required search parameters: {', '.join(missing_params)}")
    
    logger.info("Search parameters validated successfully",
               document_id=extracted_data.document_id,
               payor=extracted_data.payor_name,
               has_cpt=bool(extracted_data.cpt_codes),
               has_hcpcs=bool(service_details.get('hcpcs_codes')),
               has_rxnorm=bool(service_details.get('rxnorm_codes')))


def extract_data_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract medical data from Step Functions event structure.
    
    Handles nested Payload structure from previous Lambda invocations.
    """
    # Try direct extraction_result first
    if 'extraction_result' in event:
        extraction_data = event['extraction_result']
        
        # Handle Lambda response format with Payload
        if isinstance(extraction_data, dict) and 'Payload' in extraction_data:
            payload = extraction_data['Payload']
            if isinstance(payload, dict) and 'extraction_result' in payload:
                return payload['extraction_result']
        
        # Direct format
        if isinstance(extraction_data, dict):
            return extraction_data
    
    # Try extracted_data key (for testing)
    if 'extracted_data' in event:
        return event['extracted_data']
    
    return None


def extract_all_service_codes(extracted_data: ExtractedData) -> tuple:
    """
    Extract all service codes (CPT, HCPCS, RxNorm) and their descriptions.
    
    Returns:
        Tuple of (all_codes_list, descriptions_dict)
        - all_codes_list: Combined list of all service codes
        - descriptions_dict: {code: description} for all codes with descriptions
    """
    all_codes = []
    descriptions = {}
    
    structured_fields = extracted_data.structured_fields or {}
    service_details = structured_fields.get('service_details', {})
    
    # Add CPT codes
    all_codes.extend(extracted_data.cpt_codes or [])
    
    # Add HCPCS codes (may be None)
    hcpcs_codes = service_details.get('hcpcs_codes') or []
    if hcpcs_codes:
        all_codes.extend(hcpcs_codes)
    
    # Add RxNorm codes (may be None)
    rxnorm_codes = service_details.get('rxnorm_codes') or []
    if rxnorm_codes:
        all_codes.extend(rxnorm_codes)
    
    # Extract descriptions for all service codes
    # CPT descriptions
    cpt_desc_list = service_details.get('all_cpt_descriptions') or []
    if cpt_desc_list:
        descriptions.update(parse_description_list(cpt_desc_list))
    
    # HCPCS descriptions
    hcpcs_desc_list = service_details.get('hcpcs_descriptions') or []
    if hcpcs_desc_list:
        descriptions.update(parse_description_list(hcpcs_desc_list))
    
    # RxNorm descriptions
    rxnorm_desc_list = service_details.get('rxnorm_descriptions') or []
    if rxnorm_desc_list:
        descriptions.update(parse_description_list(rxnorm_desc_list))
    
    logger.info("Extracted all service codes",
               total_codes=len(all_codes),
               with_descriptions=len(descriptions))
    
    return all_codes, descriptions


def extract_icd_descriptions(extracted_data: ExtractedData) -> Dict[str, str]:
    """
    Extract ICD code descriptions.
    
    Returns:
        Dictionary mapping ICD codes to descriptions
    """
    descriptions = {}
    
    structured_fields = extracted_data.structured_fields or {}
    diagnosis_details = structured_fields.get('diagnosis_details', {})
    
    # ICD descriptions are stored as a list matching the order of icd_codes
    icd_desc_list = diagnosis_details.get('diagnosis_descriptions') or []
    if icd_desc_list and extracted_data.icd_codes:
        # Match descriptions to codes by position
        descriptions = dict(zip(extracted_data.icd_codes, icd_desc_list))
    
    logger.info("Extracted ICD descriptions",
               icd_count=len(descriptions))
    
    return descriptions


def parse_description_list(desc_list: list) -> Dict[str, str]:
    """
    Parse description list in format "CODE: Description" to dictionary.
    
    Examples:
        ["12345: Procedure description", "67890: Another procedure"]
        -> {"12345": "Procedure description", "67890": "Another procedure"}
    """
    result = {}
    for desc_str in desc_list:
        if isinstance(desc_str, str) and ':' in desc_str:
            # Split on first colon only
            parts = desc_str.split(':', 1)
            if len(parts) == 2:
                code = parts[0].strip()
                description = parts[1].strip()
                result[code] = description
    return result


def build_metadata_filter(payor: str, service_codes: list) -> Dict[str, Any]:
    """
    Build metadata filter for Bedrock KB retrieval.
    
    Filters by payor only - let semantic search handle service code matching.
    
    Note: We don't filter by search-code because:
    1. A policy might be indexed under code A but also cover code B
    2. The procedure-codes field contains ALL codes, not just search-code
    3. Semantic search is better at finding relevant policies across codes
    
    This reduces search space while maintaining accuracy.
    """
    # Only filter by payor - semantic search will find relevant policies
    return {
        'equals': {
            'key': 'payor',
            'value': payor
        }
    }


def select_policy_with_kb(
    extracted_data: ExtractedData,
    all_service_codes: list,
    service_descriptions: Dict[str, str],
    icd_descriptions: Dict[str, str]
) -> Dict[str, Any]:
    """
    Use Bedrock Knowledge Base RetrieveAndGenerate to select the best policy.
    
    Returns:
        Dictionary with:
        - selected_policy_s3_uri: S3 URI of the selected policy
        - policy_title: Title of the policy
        - confidence_score: Confidence in selection (0.0-1.0)
        - selection_rationale: Detailed rationale for selection
        - alternative_policies_considered: List of alternatives
        - evaluation_summary: Summary of evaluation process
    """
    
    knowledge_base_id = os.environ.get('KNOWLEDGE_BASE_ID')
    if not knowledge_base_id:
        raise PolicySelectionError("KNOWLEDGE_BASE_ID environment variable not set")
    
    # Normalize payor name to match what's stored in KB metadata
    # KB metadata stores normalized payor (e.g., "Aetna" not "Aetna Better Health")
    # Use simple normalization: take first word
    normalized_payor = extracted_data.payor_name.split()[0]
    
    # Combine both names for comprehensive KB matching
    # Format: "Aetna Better Health (Aetna)"
    # This helps KB match whether metadata has full or normalized name
    payor_display = f"{extracted_data.payor_name} ({normalized_payor})" if normalized_payor != extracted_data.payor_name else extracted_data.payor_name
    
    logger.info("Using payor names for KB search",
               original_payor=extracted_data.payor_name,
               normalized_payor=normalized_payor,
               payor_display=payor_display)
    
    # Create selection prompt with all service codes and descriptions
    prompt = create_policy_selection_prompt(
        payor=payor_display,
        service_codes=all_service_codes,
        service_descriptions=service_descriptions,
        icd_codes=extracted_data.icd_codes,
        icd_descriptions=icd_descriptions,
        service_type=extracted_data.service_type,
        patient_age_range=extracted_data.patient_info.age_range,
        patient_gender=extracted_data.patient_info.gender,
        primary_diagnosis=extracted_data.patient_info.primary_diagnosis,
        comorbidities=extracted_data.patient_info.comorbidities
    )
    
    logger.info("Invoking Bedrock KB for policy selection",
               knowledge_base_id=knowledge_base_id,
               payor=extracted_data.payor_name,
               prompt_length=len(prompt))
    
    kb_start = time.time()
    
    try:
        # Build metadata filter for faster, more accurate retrieval
        # Filter by payor and service codes to narrow down search space
        metadata_filter = build_metadata_filter(
            payor=normalized_payor,
            service_codes=all_service_codes
        )
        
        logger.info("Using metadata filter for KB search",
                   payor=normalized_payor,
                   service_codes=all_service_codes,
                   filter_structure=metadata_filter)
        
        # Use RetrieveAndGenerate for intelligent selection
        # Note: PayorValidator already verified policies exist in S3 path
        # Metadata filter narrows search to relevant policies only
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={
                'text': prompt
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': resolve_model_role('kb_generation').model_id,
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 5,  # Reduced from 10 since we're using metadata filter
                            'overrideSearchType': 'SEMANTIC',  # Use SEMANTIC search
                            'filter': metadata_filter
                        }
                    }
                }
            }
        )
        
        # Extract generated text and citations
        generated_text = response['output']['text']
        citations = response.get('citations', [])
        kb_duration = time.time() - kb_start
        
        logger.info("KB response received",
                   response_length=len(generated_text),
                   citations_count=len(citations),
                   kb_duration_seconds=round(kb_duration, 2))
        
        # Parse JSON response from KB (for rationale and confidence only)
        # Note: Claude may hallucinate S3 URIs and titles, so we use citations instead
        try:
            selection_result = parse_kb_selection_response(generated_text)
            selection_rationale = selection_result.get('selection_rationale', 'Policy selected based on KB evaluation')
            confidence_score = selection_result.get('confidence_score', 0.85)
        except Exception as e:
            logger.warning("Could not parse Claude's JSON response, using defaults",
                          error=str(e),
                          response_preview=generated_text[:200])
            selection_rationale = "Policy selected based on semantic relevance to request"
            confidence_score = 0.80
        
        # Use the FIRST CITATION as the selected policy (most relevant by KB ranking)
        # This is more reliable than Claude's generated S3 URI which may be hallucinated
        if not citations:
            raise PolicySelectionError("No citations returned from Knowledge Base")
        
        first_citation = citations[0]
        retrieved_references = first_citation.get('retrievedReferences', [])
        
        if not retrieved_references:
            raise PolicySelectionError("No retrieved references in citations")
        
        # Extract S3 location from the first reference (most relevant policy)
        s3_location = retrieved_references[0].get('location', {}).get('s3Location', {})
        if not s3_location:
            raise PolicySelectionError("No S3 location in first citation")
        
        actual_s3_uri = s3_location.get('uri', '')
        if not actual_s3_uri:
            raise PolicySelectionError("No URI in S3 location")
        
        logger.info("Using first citation as selected policy",
                   s3_uri=actual_s3_uri,
                   citation_score=retrieved_references[0].get('metadata', {}).get('score', 'N/A'))
        
        # Create selection result with actual S3 URI from citation
        selection_result = {
            'selected_policy_s3_uri': actual_s3_uri,
            'confidence_score': confidence_score,
            'selection_rationale': selection_rationale,
            'alternative_policies_considered': [ref.get('location', {}).get('s3Location', {}).get('uri', '') 
                                               for ref in retrieved_references[1:] if ref.get('location')],
            'evaluation_summary': {
                'total_policies_reviewed': len(retrieved_references),
                'kb_duration_seconds': round(kb_duration, 2)
            }
        }
        
        # Validate S3 URI format
        if not selection_result['selected_policy_s3_uri'].startswith('s3://'):
            raise PolicySelectionError(
                f"Invalid S3 URI format: {selection_result['selected_policy_s3_uri']}"
            )
        
        return selection_result
    
    except Exception as e:
        logger.warning("KB policy selection failed",
                    error=str(e),
                    error_type=type(e).__name__)
        raise PolicySelectionError(
            f"Failed to select policy using Knowledge Base: {str(e)}",
            payor=extracted_data.payor_name
        )


def parse_kb_selection_response(generated_text: str) -> Dict[str, Any]:
    """
    Parse the JSON response from Bedrock KB.
    
    Handles cases where KB returns JSON wrapped in markdown code blocks or with text prefix.
    """
    json_str = generated_text.strip()
    
    # Try to extract JSON from markdown code blocks
    if '```json' in json_str:
        start = json_str.find('```json') + 7
        end = json_str.find('```', start)
        json_str = json_str[start:end].strip()
    elif '```' in json_str:
        start = json_str.find('```') + 3
        end = json_str.find('```', start)
        json_str = json_str[start:end].strip()
    else:
        # KB often returns text before JSON like "Based on the evaluation... {\n  \"selected_policy_s3_uri\": ..."
        # Find the first { and extract from there
        json_start = json_str.find('{')
        if json_start > 0:
            json_str = json_str[json_start:]
        
        # Find the last } to handle any trailing text
        json_end = json_str.rfind('}')
        if json_end > 0:
            json_str = json_str[:json_end + 1]
    
    try:
        result = json.loads(json_str)
        
        # Validate required fields
        required_fields = ['selected_policy_s3_uri', 'policy_title', 'confidence_score', 'selection_rationale']
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in KB response: {missing_fields}")
        
        # Fix S3 URI if it's using wrong bucket name
        # KB metadata might have old bucket names, replace with actual bucket
        s3_uri = result['selected_policy_s3_uri']
        if s3_uri.startswith('s3://'):
            # Extract the key (everything after first /)
            parts = s3_uri[5:].split('/', 1)
            if len(parts) == 2:
                old_bucket, key = parts
                # Use the actual policies bucket from environment
                actual_bucket = os.environ.get('POLICIES_BUCKET', 'prior-authorization-assistant-policies-dev')
                result['selected_policy_s3_uri'] = f"s3://{actual_bucket}/{key}"
                
                if old_bucket != actual_bucket:
                    logger.info("Corrected S3 bucket in URI",
                               original_bucket=old_bucket,
                               corrected_bucket=actual_bucket,
                               key=key)
        
        return result
    
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse KB JSON response",
                    error=str(e),
                    response_preview=generated_text[:500])
        raise PolicySelectionError(
            f"KB returned invalid JSON response: {str(e)}"
        )


# retrieve_policy_from_s3 moved to extract_clinical_criteria Lambda


def retrieve_policy_metadata(s3_uri: str) -> Dict[str, Any]:
    """
    Retrieve policy metadata from the .metadata.json file in S3.
    
    Args:
        s3_uri: S3 URI of the policy document
    
    Returns:
        Dictionary containing policy metadata (source_url, effective_date, etc.)
    """
    # Parse S3 URI
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    bucket_name = parts[0]
    object_key = parts[1]
    metadata_key = f"{object_key}.metadata.json"
    
    logger.info("Retrieving policy metadata from S3",
               bucket=bucket_name,
               metadata_key=metadata_key)
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=metadata_key)
        metadata_content = response['Body'].read().decode('utf-8')
        metadata = json.loads(metadata_content)
        
        # Extract metadataAttributes if present (Bedrock format)
        if 'metadataAttributes' in metadata:
            metadata = metadata['metadataAttributes']
        
        logger.info("Policy metadata retrieved successfully",
                   bucket=bucket_name,
                   metadata_key=metadata_key,
                   has_source_url='source_url' in metadata or 'source' in metadata)
        
        return metadata
    
    except s3_client.exceptions.NoSuchKey:
        logger.warning("Metadata file not found, returning empty metadata",
                      metadata_key=metadata_key)
        return {}
    except Exception as e:
        logger.warning("Failed to retrieve policy metadata, returning empty metadata",
                      metadata_key=metadata_key,
                      error=str(e))
        return {}


# Criteria extraction moved to separate Lambda: extract_clinical_criteria


def extract_policy_id_from_uri(s3_uri: str) -> str:
    """Extract policy ID from S3 URI."""
    # Extract filename from URI
    filename = s3_uri.split('/')[-1]
    # Remove extension
    policy_id = filename.rsplit('.', 1)[0]
    return policy_id


# Health check endpoint
def health_check(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Health check endpoint for the Lambda function"""
    
    error_handler = ContextAwareErrorHandler(event)
    
    # Test KB connectivity
    try:
        knowledge_base_id = os.environ.get('KNOWLEDGE_BASE_ID')
        kb_status = 'configured' if knowledge_base_id else 'not_configured'
    except Exception as e:
        kb_status = f'error: {str(e)}'
    
    return error_handler.create_success_response({
        'status': 'healthy',
        'service': 'intelligent-policy-selector',
        'knowledge_base_status': kb_status,
        'timestamp': context.aws_request_id if context else 'local'
    })
