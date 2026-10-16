"""
Medical data extraction module using Amazon Bedrock.

This module uses AI to extract structured medical information from
prior authorization documents including CPT codes, ICD codes, payor
information, and patient demographics.
"""

import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from common.aws_clients import get_bedrock_service
from common.models import ExtractedData, PatientInfo
from common.exceptions import DocumentParsingException, AWSServiceError
from common.decorators import handle_processing_errors, monitor_performance, retry_with_backoff
from common.validators import validate_extracted_data, MedicalCodeValidator
from common.utils import (
    extract_medical_codes_from_text, 
    extract_payor_from_text,
    create_age_range,
    anonymize_patient_id,
    calculate_confidence_score,
    safe_json_loads
)
from common.logging_config import get_logger

logger = get_logger(__name__)


class MedicalDataExtractor:
    """AI-powered medical data extraction using Amazon Bedrock"""
    
    # US State code mapping for normalization
    STATE_CODES = {
        'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
        'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
        'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
        'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
        'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo',
        'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj',
        'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh',
        'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
        'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
        'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy',
        'district of columbia': 'dc', 'puerto rico': 'pr'
    }
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
    
    def _normalize_state_code(self, state_value: str) -> Optional[str]:
        """
        Normalize state code to lowercase 2-letter format.
        
        Args:
            state_value: State name or code (e.g., "CO", "Colorado", "california")
            
        Returns:
            Lowercase 2-letter state code (e.g., "co") or None if invalid
        """
        if not state_value:
            return None
        
        state_clean = state_value.strip().lower()
        
        # If already 2 letters, return as lowercase
        if len(state_clean) == 2 and state_clean.isalpha():
            return state_clean
        
        # Try to map full state name to code
        if state_clean in self.STATE_CODES:
            return self.STATE_CODES[state_clean]
        
        # Log warning if state cannot be normalized
        logger.warning("Could not normalize state code", state_value=state_value)
        return None
    
    @handle_processing_errors
    @monitor_performance
    @retry_with_backoff(max_attempts=3)
    def extract_medical_data(self, document_id: str, raw_text: str, 
                           structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive medical data from document text"""
        
        logger.info("Starting medical data extraction", document_id=document_id)
        
        # Pre-process text for better extraction
        processed_text = self._preprocess_text(raw_text, structured_data)
        
        # Extract data using Bedrock
        extraction_result = self._extract_with_bedrock(processed_text)
        
        # Validate and enhance extracted data
        validated_data = self._validate_and_enhance_extraction(
            extraction_result, processed_text, document_id
        )
        
        # Calculate overall confidence score
        confidence_factors = {
            'bedrock_confidence': extraction_result.get('confidence', 0.0),
            'code_validation': self._calculate_code_validation_confidence(validated_data),
            'payor_validation': 1.0 if MedicalCodeValidator.validate_payor_name(
                validated_data.get('payor_name', '')) else 0.5,
            'completeness': self._calculate_completeness_score(validated_data)
        }
        
        overall_confidence = calculate_confidence_score(confidence_factors)
        
        # Create final extracted data object
        try:
            patient_info = PatientInfo(
                patient_id=validated_data.get('patient_id', f"patient_{document_id[:8]}"),
                age_range=validated_data.get('age_range', 'unknown'),
                gender=validated_data.get('gender', 'unknown'),
                primary_diagnosis=validated_data.get('primary_diagnosis', ''),
                comorbidities=validated_data.get('comorbidities', [])
            )
            
            # Build structured_fields to match FHIR extractor format
            structured_fields = self._build_structured_fields(validated_data)
            
            # Normalize coverage state to lowercase 2-letter format
            coverage_state = validated_data.get('coverage_state')
            if coverage_state:
                coverage_state = self._normalize_state_code(coverage_state)
            
            extracted_data = ExtractedData(
                document_id=document_id,
                cpt_codes=validated_data.get('cpt_codes', []),
                icd_codes=validated_data.get('icd_codes', []),
                service_type=validated_data.get('service_type', ''),
                payor_name=validated_data.get('payor_name', ''),
                plan_type=validated_data.get('plan_type', ''),
                patient_info=patient_info,
                extraction_confidence=overall_confidence,
                raw_text=raw_text,  # Store complete medical record text
                coverage_state=coverage_state,  # Lowercase 2-letter state code from insurance coverage
                structured_fields=structured_fields
            )
            
            logger.info("Medical data extraction completed successfully",
                       document_id=document_id,
                       confidence=overall_confidence,
                       cpt_codes_count=len(extracted_data.cpt_codes),
                       icd_codes_count=len(extracted_data.icd_codes))
            
            return extracted_data.to_dict()
            
        except Exception as e:
            raise DocumentParsingException(
                f"Failed to create extracted data object: {str(e)}",
                document_id=document_id
            )
    
    def _preprocess_text(self, raw_text: str, structured_data: Dict[str, Any]) -> str:
        """Preprocess text to improve extraction accuracy"""
        
        # Check if we have actual document text
        if not raw_text or len(raw_text.strip()) == 0:
            raise DocumentParsingException(
                "Document text extraction failed - no actual document content available for processing"
            )
        
        # Combine raw text with structured data
        combined_text = raw_text
        
        # Add key-value pairs from structured data
        if structured_data.get('key_value_pairs'):
            kv_text = "\n\nStructured Fields:\n"
            for key, value in structured_data['key_value_pairs'].items():
                kv_text += f"{key}: {value}\n"
            combined_text += kv_text
        
        # Add table data
        if structured_data.get('tables'):
            table_text = "\n\nTable Data:\n"
            for i, table in enumerate(structured_data['tables']):
                table_text += f"Table {i+1}:\n"
                for row in table:
                    table_text += " | ".join(row) + "\n"
            combined_text += table_text
        
        return combined_text
    
    @retry_with_backoff(max_attempts=2)
    def _extract_with_bedrock(self, text: str) -> Dict[str, Any]:
        """Extract medical data using Bedrock Claude model"""
        
        prompt = self._create_extraction_prompt(text)
        
        try:
            response = self.bedrock_service.invoke_role(
                "medical_extraction",
                prompt=prompt,
                max_tokens=4000
            )
            
            # Parse JSON response
            extraction_result = safe_json_loads(response, {})
            
            if not extraction_result:
                # Try to extract JSON from response if it's wrapped in text
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    extraction_result = safe_json_loads(json_match.group(), {})
            
            return extraction_result
            
        except Exception as e:
            logger.warning("Bedrock extraction failed", error=str(e))
            raise AWSServiceError(f"Bedrock extraction failed: {str(e)}")
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create optimized prompt for medical data extraction"""
        
        return f"""
You are a medical coding specialist tasked with extracting key information from a prior authorization document. 
Analyze the following document text and extract the requested information with high accuracy.

DOCUMENT TEXT:
{text}

EXTRACTION REQUIREMENTS:
Extract the following information and return it as a valid JSON object:

CRITICAL: Only extract codes for REQUESTED SERVICES that need prior authorization.
DO NOT extract codes from patient history, past treatments, or conservative care already attempted.

Look for sections like:
- "Services Requested for Prior Authorization"
- "Requested Services"
- "Authorization Request"
- "Procedures Requested"
- "Treatment Plan" (future services, not past)

IGNORE sections like:
- "Past Medical History"
- "Conservative Treatment Attempted"
- "Previous Medications"
- "History of Present Illness"
- "Past Treatments"

1. CPT Codes: ONLY codes for procedures/services being REQUESTED for authorization (5-digit codes, may have modifiers)
2. HCPCS Codes: ONLY codes for equipment/services being REQUESTED for authorization (alphanumeric codes starting with letters like K, G, E, J, etc.)
3. RxNorm Codes: ONLY codes for medications being REQUESTED for authorization (numeric codes for drugs) - NOT medications from patient history
4. ICD Codes: All International Classification of Diseases codes for current diagnoses (format: Letter + 2 digits + optional decimal + more digits)
5. Payor Information: Insurance company name, plan type, and coverage state
6. Service Information: Type of medical service being requested
7. Patient Demographics: Age range, gender, primary diagnosis
8. Additional Clinical Information: Comorbidities, relevant medical history
9. Coverage State: State where the insurance coverage is issued (PRIORITY: extract from insurance/coverage section first, fallback to patient address if not found)

RESPONSE FORMAT:
Return ONLY a valid JSON object with this exact structure:
{{
    "cpt_codes": ["12345", "67890"],  // ONLY codes for REQUESTED services, NOT from patient history
    "icd_codes": ["I25.10", "I20.9"],
    "payor_name": "Insurance Company Name",
    "plan_type": "HMO/PPO/EPO/etc",
    "coverage_state": "co",
    "service_type": "Diagnostic/Therapeutic/Surgical/DME/Specialty Drug/etc",
    "patient_id": "anonymized_patient_identifier",
    "age_range": "XX-XX",
    "gender": "male/female/other/unknown",
    "primary_diagnosis": "Primary diagnosis description",
    "comorbidities": ["condition1", "condition2"],
    "additional_fields": {{
        "provider_name": "Healthcare Provider Name",
        "service_date": "YYYY-MM-DD",
        "urgency": "routine/urgent/emergent",
        "hcpcs_codes": ["K0005", "G0154"],  // ONLY REQUESTED equipment/services
        "hcpcs_descriptions": ["K0005: Ultralightweight wheelchair", "G0154: Home health nursing"],
        "rxnorm_codes": ["1551291", "224905"],  // ONLY REQUESTED medications, NOT from patient history
        "rxnorm_descriptions": ["1551291: adalimumab 40 MG", "224905: trastuzumab"],
        "all_cpt_codes": ["12345", "67890"],  // ONLY REQUESTED procedures
        "cpt_descriptions": ["12345: Procedure description", "67890: Another procedure"],
        "patient_address_state": "co"
    }},
    "confidence": 0.85
}}

CRITICAL REMINDER:
- cpt_codes, hcpcs_codes, rxnorm_codes should ONLY contain codes for services/medications being REQUESTED for authorization
- DO NOT include codes from "Conservative Treatment Attempted", "Past Medical History", "Previous Medications", etc.
- If the document shows "Patient is currently taking ibuprofen" - DO NOT extract ibuprofen's RxNorm code
- If the document shows "Requesting authorization for adalimumab" - DO extract adalimumab's RxNorm code

IMPORTANT GUIDELINES:
- ONLY extract codes for REQUESTED services that need prior authorization
- DO NOT extract codes from patient history, past medications, or conservative treatments already tried
- Look for explicit sections about "requested services", "authorization request", or "proposed treatment"
- Extract CPT codes (5-digit numeric), HCPCS codes (alphanumeric like K0005, G0154), and RxNorm codes (numeric medication codes)
- HCPCS codes are used for DME (Durable Medical Equipment), home health services, and some medications
- RxNorm codes are used for prescription medications and specialty drugs BEING REQUESTED (not from patient history)
- Use standard payor names (e.g., "Humana", "Aetna", "Blue Cross Blue Shield")
- For coverage_state: Extract the 2-letter state code in LOWERCASE where the insurance coverage is issued
  * PRIORITY 1: Look in insurance/coverage section for state information
  * PRIORITY 2: If not found in coverage section, use patient address state as fallback
  * Format: lowercase 2-letter code (e.g., "co", "ca", "ny", "tx")
  * Store patient address state separately in additional_fields.patient_address_state for reference
- For patient_id, create an anonymized identifier, never use real names or SSN
- Age ranges should be 10-year ranges for privacy (e.g., "XX-XX", "YY-YY")
- Set confidence between 0.0 and 1.0 based on clarity of information in document
- If information is not found, use empty arrays/strings or "unknown" as appropriate
- Ensure all extracted codes follow proper medical coding formats

EXAMPLES OF WHAT TO EXTRACT vs IGNORE:
✅ EXTRACT: "Services Requested: CPT 97110 (24 sessions), CPT 97140 (12 sessions)"
❌ IGNORE: "Conservative Treatment Attempted: NSAIDs (ibuprofen 600mg TID)"
✅ EXTRACT: "Requesting authorization for: RxNorm 1551291 (adalimumab 40 MG)"
❌ IGNORE: "Patient is currently taking: ibuprofen, cyclobenzaprine"
✅ EXTRACT: "Proposed Treatment Plan: Manual therapy (CPT 97140)"
❌ IGNORE: "Past Medical History: Previous physical therapy in 2020"

Return only the JSON object, no additional text or explanation.
"""
    
    def _validate_and_enhance_extraction(self, extraction_result: Dict[str, Any], 
                                       text: str, document_id: str) -> Dict[str, Any]:
        """Validate and enhance extracted data with fallback methods"""
        
        validated_data = extraction_result.copy()
        
        # Validate and clean CPT codes
        cpt_codes = validated_data.get('cpt_codes', [])
        if isinstance(cpt_codes, list):
            validated_cpts = []
            for code in cpt_codes:
                if MedicalCodeValidator.validate_cpt_code(str(code)):
                    validated_cpts.append(str(code).upper())
            validated_data['cpt_codes'] = validated_cpts
        
        # No fallback - if AI extraction fails, return empty list
        if not validated_data.get('cpt_codes'):
            validated_data['cpt_codes'] = []
        
        # Validate and clean ICD codes
        icd_codes = validated_data.get('icd_codes', [])
        if isinstance(icd_codes, list):
            validated_icds = []
            for code in icd_codes:
                if MedicalCodeValidator.validate_icd_code(str(code)):
                    validated_icds.append(str(code).upper())
            validated_data['icd_codes'] = validated_icds
        
        # No fallback - if AI extraction fails, return empty list  
        if not validated_data.get('icd_codes'):
            validated_data['icd_codes'] = []
        
        # Validate and normalize payor name
        payor_name = validated_data.get('payor_name', '')
        if payor_name:
            validated_data['payor_name'] = MedicalCodeValidator.normalize_payor_name(payor_name)
        else:
            # No fallback - if AI extraction fails, return empty string
            validated_data['payor_name'] = ''
        
        # Ensure patient_id is anonymized
        if not validated_data.get('patient_id'):
            validated_data['patient_id'] = anonymize_patient_id(document_id)
        
        # Validate gender
        gender = validated_data.get('gender', '').lower()
        if gender not in ['male', 'female', 'other']:
            validated_data['gender'] = 'unknown'
        
        # Ensure required fields have default values
        validated_data.setdefault('service_type', 'unknown')
        validated_data.setdefault('plan_type', 'unknown')
        validated_data.setdefault('primary_diagnosis', '')
        validated_data.setdefault('comorbidities', [])
        validated_data.setdefault('additional_fields', {})
        
        return validated_data
    
    def _calculate_code_validation_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score based on medical code validation"""
        
        total_codes = len(data.get('cpt_codes', [])) + len(data.get('icd_codes', []))
        if total_codes == 0:
            return 0.0
        
        valid_codes = 0
        
        # Check CPT codes
        for code in data.get('cpt_codes', []):
            if MedicalCodeValidator.validate_cpt_code(code):
                valid_codes += 1
        
        # Check ICD codes
        for code in data.get('icd_codes', []):
            if MedicalCodeValidator.validate_icd_code(code):
                valid_codes += 1
        
        return valid_codes / total_codes if total_codes > 0 else 0.0
    
    def _calculate_completeness_score(self, data: Dict[str, Any]) -> float:
        """Calculate completeness score based on extracted fields"""
        
        required_fields = [
            'cpt_codes', 'icd_codes', 'payor_name', 'service_type',
            'patient_id', 'gender', 'primary_diagnosis'
        ]
        
        completed_fields = 0
        for field in required_fields:
            value = data.get(field)
            if value and value != 'unknown' and value != '':
                if isinstance(value, list) and len(value) > 0:
                    completed_fields += 1
                elif not isinstance(value, list):
                    completed_fields += 1
        
        return completed_fields / len(required_fields)
    
    def _build_structured_fields(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build structured_fields dictionary to match FHIR extractor format.
        This ensures consistency between PDF and FHIR extraction outputs.
        """
        additional_fields = validated_data.get('additional_fields', {})
        
        # Extract patient details from validated_data and additional_fields
        patient_details = {
            'name': additional_fields.get('patient_name'),
            'mrn': additional_fields.get('medical_record_number') or additional_fields.get('mrn'),
            'dob': additional_fields.get('date_of_birth') or additional_fields.get('dob'),
            'address': additional_fields.get('patient_address') or additional_fields.get('address'),
            'phone': additional_fields.get('patient_phone') or additional_fields.get('phone')
        }
        
        # Extract coverage details
        coverage_details = {
            'member_id': additional_fields.get('member_id') or additional_fields.get('subscriber_id'),
            'group_number': additional_fields.get('group_number') or additional_fields.get('group_id'),
            'plan_name': additional_fields.get('plan_name') or additional_fields.get('insurance_plan'),
            'state': additional_fields.get('state') or additional_fields.get('coverage_state')
        }
        
        # Extract service details
        service_details = {
            'provider': additional_fields.get('provider_name') or additional_fields.get('provider'),
            'all_cpt_codes': additional_fields.get('all_cpt_codes'),
            'all_cpt_descriptions': additional_fields.get('cpt_descriptions'),
            'hcpcs_codes': additional_fields.get('hcpcs_codes') or additional_fields.get('all_hcpcs_codes'),
            'hcpcs_descriptions': additional_fields.get('hcpcs_descriptions'),
            'rxnorm_codes': additional_fields.get('rxnorm_codes') or additional_fields.get('all_rxnorm_codes'),
            'rxnorm_descriptions': additional_fields.get('rxnorm_descriptions'),
            'session_counts': additional_fields.get('session_counts') or additional_fields.get('treatment_sessions'),
            'treatment_duration_days': additional_fields.get('treatment_duration_days') or additional_fields.get('duration_days'),
            'treatment_start_date': additional_fields.get('service_start_date') or additional_fields.get('start_date'),
            'treatment_end_date': additional_fields.get('service_end_date') or additional_fields.get('end_date'),
            'service_location': additional_fields.get('service_location') or additional_fields.get('facility'),
            'supporting_info': additional_fields.get('supporting_info') or additional_fields.get('additional_notes'),
            'drug_category': additional_fields.get('drug_category') or additional_fields.get('medication_type'),
            'generic_drug_name': additional_fields.get('generic_drug_name') or additional_fields.get('medication_name')
        }
        
        # Extract diagnosis details
        diagnosis_details = {
            'diagnosis_codes': validated_data.get('icd_codes', []),
            'diagnosis_descriptions': additional_fields.get('diagnosis_descriptions') or []
        }
        
        # Add extraction metadata
        extraction_metadata = {
            'file_type': 'PDF',
            'extraction_method': 'AI_BEDROCK'
        }
        
        return {
            'patient_details': patient_details,
            'coverage_details': coverage_details,
            'service_details': service_details,
            'diagnosis_details': diagnosis_details,
            'extraction_metadata': extraction_metadata
        }


class FHIRDataExtractor:
    """Specialized extractor for FHIR-formatted prior authorization documents"""
    
    @staticmethod
    def extract_from_fhir(fhir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medical data from FHIR Bundle"""
        
        extracted = {
            'cpt_codes': [],
            'icd_codes': [],
            'payor_name': '',
            'plan_type': '',
            'service_type': '',
            'patient_id': '',
            'age_range': '',
            'gender': '',
            'primary_diagnosis': '',
            'comorbidities': [],
            'additional_fields': {}
        }
        
        if fhir_data.get('resourceType') != 'Bundle':
            return extracted
        
        entries = fhir_data.get('entry', [])
        
        for entry in entries:
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            
            if resource_type == 'Claim':
                FHIRDataExtractor._extract_from_claim(resource, extracted)
            elif resource_type == 'Patient':
                FHIRDataExtractor._extract_from_patient(resource, extracted)
            elif resource_type == 'Organization':
                FHIRDataExtractor._extract_from_organization(resource, extracted)
            elif resource_type == 'Coverage':
                FHIRDataExtractor._extract_from_coverage(resource, extracted)
        
        return extracted
    
    @staticmethod
    def _extract_from_claim(claim: Dict[str, Any], extracted: Dict[str, Any]):
        """Extract data from FHIR Claim resource"""
        
        # Extract CPT codes from items
        items = claim.get('item', [])
        for item in items:
            product_or_service = item.get('productOrService', {})
            coding = product_or_service.get('coding', [])
            for code in coding:
                if code.get('system') == 'http://www.ama-assn.org/go/cpt':
                    extracted['cpt_codes'].append(code.get('code', ''))
        
        # Extract ICD codes from diagnosis
        diagnoses = claim.get('diagnosis', [])
        for diagnosis in diagnoses:
            diagnosis_concept = diagnosis.get('diagnosisCodeableConcept', {})
            coding = diagnosis_concept.get('coding', [])
            for code in coding:
                if 'icd-10' in code.get('system', '').lower():
                    extracted['icd_codes'].append(code.get('code', ''))
                    if not extracted['primary_diagnosis'] and diagnosis.get('sequence') == 1:
                        extracted['primary_diagnosis'] = code.get('display', '')
    
    @staticmethod
    def _extract_from_patient(patient: Dict[str, Any], extracted: Dict[str, Any]):
        """Extract data from FHIR Patient resource"""
        
        # Extract patient ID (anonymized)
        patient_id = patient.get('id', '')
        if patient_id:
            extracted['patient_id'] = anonymize_patient_id(patient_id)
        
        # Extract gender
        extracted['gender'] = patient.get('gender', 'unknown')
        
        # Extract age range from birth date
        birth_date = patient.get('birthDate')
        if birth_date:
            extracted['age_range'] = create_age_range(birth_date)
    
    @staticmethod
    def _extract_from_organization(organization: Dict[str, Any], extracted: Dict[str, Any]):
        """Extract data from FHIR Organization resource"""
        
        org_type = organization.get('type', [])
        for type_info in org_type:
            coding = type_info.get('coding', [])
            for code in coding:
                if code.get('code') == 'ins':  # Insurance company
                    extracted['payor_name'] = organization.get('name', '')
    
    @staticmethod
    def _extract_from_coverage(coverage: Dict[str, Any], extracted: Dict[str, Any]):
        """Extract data from FHIR Coverage resource"""
        
        coverage_type = coverage.get('type', {})
        coding = coverage_type.get('coding', [])
        for code in coding:
            display = code.get('display', '')
            if display:
                extracted['plan_type'] = display
