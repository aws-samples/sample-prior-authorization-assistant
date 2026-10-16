"""
Metadata extraction from policy document text using Bedrock.

Extracts structured metadata including procedure codes, ICD codes,
policy numbers, effective dates, and service categories.
"""

import boto3
import json
from typing import Dict, Any, List
import structlog

from common.bedrock_converse import converse_role

logger = structlog.get_logger(__name__)


def extract_metadata_from_text(
    bedrock_client: boto3.client,
    text: str,
    payor: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Extract metadata from policy document text using Bedrock Claude.
    
    Args:
        bedrock_client: Boto3 Bedrock Runtime client
        text: Extracted text from policy document
        payor: Payor name extracted from S3 prefix
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary containing extracted metadata
        
    Raises:
        Exception: If extraction fails after all retries
    """
    logger.info(
        "Starting metadata extraction",
        payor=payor,
        text_length=len(text)
    )
    
    prompt = _build_extraction_prompt(text)
    
    for attempt in range(max_retries):
        try:
            response = _call_bedrock_claude(bedrock_client, prompt)
            metadata = _parse_bedrock_response(response, payor)
            
            logger.info(
                "Metadata extraction successful",
                payor=payor,
                attempt=attempt + 1,
                procedure_codes_count=len(metadata.get('procedure_codes', [])),
                icd_codes_count=len(metadata.get('icd_codes', []))
            )
            
            return metadata
            
        except Exception as e:
            logger.warning(
                "Metadata extraction attempt failed",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(e)
            )
            
            if attempt == max_retries - 1:
                # Last attempt failed, return minimal metadata
                logger.warning(
                    "All metadata extraction attempts failed",
                    payor=payor,
                    error=str(e)
                )
                return _create_minimal_metadata(payor)
    
    return _create_minimal_metadata(payor)


def _build_extraction_prompt(text: str) -> str:
    """
    Build the extraction prompt for Bedrock Claude.
    
    Args:
        text: Policy document text
        
    Returns:
        Formatted prompt string
    """
    # Truncate text if too long (keep first 50000 chars)
    truncated_text = text[:50000] if len(text) > 50000 else text
    
    prompt = f"""Extract the following information from this medical policy document:

1. Policy title (the main title or heading of the policy document)
2. All procedure codes (CPT, HCPCS, J-codes, RxNorm) - include ranges like "93600-93603"
3. All diagnosis codes (ICD-10/ICD-11) - include ranges like "I20.0-I20.9"
4. Policy effective date (ISO 8601 format)
5. Policy number
6. Service category (e.g., "Cardiac Catheterization", "Orthopedic Surgery")

Return ONLY a JSON object with this exact structure (no additional text):
{{
  "title": "Physical Therapy - Clinical Policy Bulletins",
  "procedure_codes": ["93458", "93600-93603", "E0100"],
  "icd_codes": ["I25.10", "I20.0-I20.9"],
  "effective_date": "2024-01-01",
  "policy_number": "HUM-1034-005",
  "service_category": "Cardiac Catheterization"
}}

If any field cannot be found, use an empty array [] for lists or null for single values.

Policy Document:
{truncated_text}

JSON Response:"""
    
    return prompt


def _call_bedrock_claude(
    bedrock_client: boto3.client,
    prompt: str
) -> str:
    """
    Call the policy metadata extraction model with the extraction prompt.
    
    Args:
        bedrock_client: Boto3 Bedrock Runtime client
        prompt: Extraction prompt
        
    Returns:
        Model response text
    """
    # Deterministic for extraction (temperature 0.0 by default)
    return converse_role(
        bedrock_client, "policy_metadata_extraction", prompt, 2000
    )


def _parse_bedrock_response(response_text: str, payor: str) -> Dict[str, Any]:
    """
    Parse Bedrock response and create metadata dictionary.
    
    Args:
        response_text: Response text from Bedrock
        payor: Payor name
        
    Returns:
        Metadata dictionary
        
    Raises:
        ValueError: If response cannot be parsed
    """
    # Try to extract JSON from response
    response_text = response_text.strip()
    
    # Handle markdown code blocks
    if response_text.startswith('```'):
        # Remove markdown code block markers
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        response_text = response_text.replace('```json', '').replace('```', '').strip()
    
    try:
        extracted_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse Bedrock response as JSON",
            response_text=response_text[:500],
            error=str(e)
        )
        raise ValueError(f"Invalid JSON response from Bedrock: {str(e)}")
    
    # Build metadata dictionary
    metadata = {
        'payor': payor,
        'title': extracted_data.get('title'),
        'procedure_codes': _format_code_list(extracted_data.get('procedure_codes', [])),
        'icd_codes': _format_code_list(extracted_data.get('icd_codes', [])),
        'effective_date': extracted_data.get('effective_date'),
        'policy_number': extracted_data.get('policy_number'),
        'service_category': extracted_data.get('service_category')
    }
    
    return metadata


def _format_code_list(codes: List[str]) -> List[str]:
    """
    Format and clean code list.
    
    Args:
        codes: List of codes
        
    Returns:
        Cleaned list of codes
    """
    if not codes:
        return []
    
    # Remove empty strings and strip whitespace
    cleaned = [code.strip() for code in codes if code and code.strip()]
    
    return cleaned


def _create_minimal_metadata(payor: str) -> Dict[str, Any]:
    """
    Create minimal metadata when extraction fails.
    
    Args:
        payor: Payor name
        
    Returns:
        Minimal metadata dictionary with only payor
    """
    return {
        'payor': payor,
        'title': None,
        'procedure_codes': [],
        'icd_codes': [],
        'effective_date': None,
        'policy_number': None,
        'service_category': None
    }


def format_metadata_for_s3(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Format metadata dictionary for S3 object metadata.
    
    S3 metadata values must be strings, so we convert lists to comma-separated strings.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Dictionary with string values suitable for S3 metadata
    """
    s3_metadata = {}
    
    # Add payor
    if metadata.get('payor'):
        s3_metadata['payor'] = metadata['payor']
    
    # Add procedure codes as comma-separated string
    if metadata.get('procedure_codes'):
        s3_metadata['procedure-codes'] = ','.join(metadata['procedure_codes'])
    
    # Add ICD codes as comma-separated string
    if metadata.get('icd_codes'):
        s3_metadata['icd-codes'] = ','.join(metadata['icd_codes'])
    
    # Add effective date
    if metadata.get('effective_date'):
        s3_metadata['effective-date'] = metadata['effective_date']
    
    # Add policy number
    if metadata.get('policy_number'):
        s3_metadata['policy-number'] = metadata['policy_number']
    
    # Add service category
    if metadata.get('service_category'):
        s3_metadata['service-category'] = metadata['service_category']
    
    # Add title
    if metadata.get('title'):
        s3_metadata['title'] = metadata['title']
    
    # Add source tracking fields (for dynamically extracted policies)
    if metadata.get('source'):
        s3_metadata['source'] = metadata['source']
    
    if metadata.get('search_code'):
        s3_metadata['search-code'] = metadata['search_code']
    
    if metadata.get('extracted_timestamp'):
        s3_metadata['extracted-timestamp'] = metadata['extracted_timestamp']
    
    logger.info(
        "Formatted metadata for S3",
        fields_count=len(s3_metadata)
    )
    
    return s3_metadata
