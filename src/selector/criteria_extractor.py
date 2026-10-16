"""
Clinical criteria extraction system for the Policy Selector module.

This module extracts structured medical necessity criteria from selected
policies and creates actionable clinical requirements for medical record review.
"""

import json
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from common.aws_clients import get_bedrock_service
from common.models import ClinicalCriterion, ExtractedData
from common.exceptions import PolicySelectionError, AWSServiceError
from common.decorators import handle_processing_errors, monitor_performance, retry_with_backoff
from common.logging_config import get_logger
from common.utils import safe_json_loads, clean_text
from knowledge.policy_retrieval import PolicySearchResult

logger = get_logger(__name__)


class CriterionType(Enum):
    """Types of clinical criteria"""
    DIAGNOSTIC = "diagnostic"
    THERAPEUTIC = "therapeutic"
    PROCEDURAL = "procedural"
    CLINICAL_INDICATOR = "clinical_indicator"
    EXCLUSION = "exclusion"
    AGE_REQUIREMENT = "age_requirement"
    DOCUMENTATION = "documentation"


class CriteriaPriority(Enum):
    """Priority levels for clinical criteria"""
    CRITICAL = "critical"
    STANDARD = "standard"
    OPTIONAL = "optional"


@dataclass
class ExtractedCriterion:
    """Raw extracted criterion before processing"""
    raw_text: str
    criterion_type: CriterionType
    priority: CriteriaPriority
    medical_codes: List[str]
    keywords: List[str]
    context: str


class ClinicalCriteriaExtractor:
    """Extracts and structures clinical criteria from policy documents"""
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
    
    @handle_processing_errors
    @monitor_performance
    def extract_clinical_criteria(self, selected_policy: PolicySearchResult, 
                                extracted_data: ExtractedData) -> List[ClinicalCriterion]:
        """Extract structured clinical criteria from selected policy"""
        
        logger.info("Extracting clinical criteria",
                   policy_id=selected_policy.policy_id,
                   payor=selected_policy.payor)
        
        # Get full policy content
        policy_content = self._get_policy_content(selected_policy)
        
        # Extract raw criteria using AI
        raw_criteria = self._ai_extract_criteria(policy_content, extracted_data)
        
        # Process and structure criteria
        structured_criteria = self._process_raw_criteria(raw_criteria, extracted_data)
        
        # Validate and enhance criteria
        validated_criteria = self._validate_and_enhance_criteria(structured_criteria, extracted_data)
        
        logger.info("Clinical criteria extraction completed",
                   policy_id=selected_policy.policy_id,
                   criteria_count=len(validated_criteria))
        
        return validated_criteria
    
    def _get_policy_content(self, selected_policy: PolicySearchResult) -> str:
        """Get full policy content for criteria extraction
        
        FIX 2: Retrieve complete policy document from S3 for comprehensive criteria extraction.
        """
        
        # Get S3 URI from metadata
        s3_uri = selected_policy.metadata.get('x-amz-bedrock-kb-source-uri', '')
        
        if not s3_uri:
            logger.warning("No S3 URI found in policy metadata, using matched content only",
                          policy_id=selected_policy.policy_id)
            return self._get_policy_content_fallback(selected_policy)
        
        try:
            # Download full policy document from S3
            full_policy_content = self._download_policy_from_s3(s3_uri)
            
            logger.info("Retrieved full policy document from S3",
                       policy_id=selected_policy.policy_id,
                       s3_uri=s3_uri,
                       content_length=len(full_policy_content))
            
            # Add metadata header for context
            header = f"""Policy Title: {selected_policy.policy_title}
Payor: {selected_policy.payor}
Policy Number: {selected_policy.metadata.get('policy-number', 'unknown')}
Service Category: {selected_policy.metadata.get('service-category', 'unknown')}

=== FULL POLICY DOCUMENT ===

"""
            return header + full_policy_content
            
        except Exception as e:
            logger.error("Failed to retrieve full policy from S3, using fallback",
                        error=str(e),
                        s3_uri=s3_uri)
            return self._get_policy_content_fallback(selected_policy)
    
    def _download_policy_from_s3(self, s3_uri: str) -> str:
        """Download full policy document from S3"""
        import boto3
        
        # Parse S3 URI (format: s3://bucket-name/key/path)
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        
        # Remove s3:// prefix and split bucket/key
        s3_path = s3_uri[5:]  # Remove 's3://'
        bucket_name, object_key = s3_path.split('/', 1)
        
        logger.info("Downloading policy from S3",
                   bucket=bucket_name,
                   key=object_key)
        
        # Download from S3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        
        # Read content
        content = response['Body'].read()
        
        # Decode based on file type
        if object_key.endswith('.md'):
            return content.decode('utf-8')
        elif object_key.endswith('.txt'):
            return content.decode('utf-8')
        elif object_key.endswith('.json'):
            return content.decode('utf-8')
        else:
            # Try UTF-8 decode
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1
                return content.decode('latin-1')
    
    def _get_policy_content_fallback(self, selected_policy: PolicySearchResult) -> str:
        """Fallback method using matched content when S3 retrieval fails"""
        
        content_parts = [
            f"Policy Title: {selected_policy.policy_title}",
            f"Payor: {selected_policy.payor}",
            f"Matched Content: {selected_policy.matched_content}"
        ]
        
        # Add metadata information
        metadata = selected_policy.metadata
        if metadata.get('cpt_codes'):
            content_parts.append(f"Covered CPT Codes: {', '.join(metadata['cpt_codes'])}")
        
        if metadata.get('icd_codes'):
            content_parts.append(f"Covered ICD Codes: {', '.join(metadata['icd_codes'])}")
        
        if metadata.get('service_categories'):
            content_parts.append(f"Service Categories: {', '.join(metadata['service_categories'])}")
        
        return "\n\n".join(content_parts)
    
    @retry_with_backoff(max_attempts=3)
    def _ai_extract_criteria(self, policy_content: str, extracted_data: ExtractedData) -> List[Dict[str, Any]]:
        """Use AI to extract medical necessity criteria from policy content"""
        
        prompt = self._create_criteria_extraction_prompt(policy_content, extracted_data)
        
        try:
            response = self.bedrock_service.invoke_role(
                "policy_criteria_extraction",
                prompt=prompt,
                max_tokens=4000
            )
            
            # Parse JSON response
            criteria_data = safe_json_loads(response, {})
            
            if not criteria_data:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    criteria_data = safe_json_loads(json_match.group(), {})
            
            # Handle both old format and new pathway format
            if 'medical_necessity_pathways' in criteria_data:
                # New pathway-based format
                return self._convert_pathways_to_criteria(criteria_data)
            else:
                # Legacy format
                return criteria_data.get('criteria', [])
            
        except Exception as e:
            logger.warning("AI criteria extraction failed", error=str(e))
            raise AWSServiceError(f"Criteria extraction failed: {str(e)}")
    
    def _create_criteria_extraction_prompt(self, policy_content: str, extracted_data: ExtractedData) -> str:
        """Create patient-context driven prompt for detailed criteria extraction"""
        
        # Format patient context similar to your successful approach
        patient_context = f"""
Patient: {extracted_data.patient_info.age_range} year old {extracted_data.patient_info.gender} with {extracted_data.patient_info.primary_diagnosis}
Comorbidities: {', '.join(extracted_data.patient_info.comorbidities) if extracted_data.patient_info.comorbidities else 'None documented'}
CPT Codes: {', '.join(f'{code} ({self._get_cpt_description(code)})' for code in extracted_data.cpt_codes)}
ICD Codes: {', '.join(f'{code} ({self._get_icd_description(code)})' for code in extracted_data.icd_codes)}
Payor: {extracted_data.payor_name}
"""
        
        return f"""
Create detailed medical necessity requirements from attached policy only.

Patient context as follows:
{patient_context}

POLICY CONTENT:
{policy_content}

INSTRUCTIONS:
Extract specific, measurable medical necessity criteria from this policy that apply to this exact patient scenario. Create detailed clinical pathways with:

1. **Specific Quantitative Thresholds** (e.g., "≥10% ischemic myocardium", "LVEF ≤40%", "Duke Treadmill Score ≤-11")
2. **Alternative Qualifying Pathways** (patient may qualify through multiple routes)
3. **Required Diagnostic Test Results** with specific values/ranges
4. **Clinical Decision Points** with clear criteria
5. **Documentation Requirements** specific to this patient's condition

Focus on creating actionable criteria that a medical reviewer can verify against actual medical records.

RESPONSE FORMAT:
Return a JSON object with structured clinical pathways:
{{
    "medical_necessity_pathways": [
        {{
            "pathway_name": "Persistent Cardiac Symptoms Despite Treatment",
            "description": "For patients with ongoing symptoms despite guideline-directed medical therapy",
            "qualifying_criteria": [
                {{
                    "criterion_id": "moderate_severe_ischemia",
                    "description": "Moderate to severe ischemia confirmed by objective testing",
                    "required_evidence": [
                        "Nuclear perfusion (SPECT or PET): ≥10% ischemic myocardium",
                        "Echocardiogram: ≥3/16 segments with stress-induced severe hypokinesis or akinesis",
                        "Cardiac MRI perfusion: ≥12% ischemic myocardium"
                    ],
                    "clinical_indicators": [
                        "Chest pain/pressure/discomfort despite optimal medical therapy",
                        "Abnormal stress test results with clinical correlation"
                    ],
                    "priority": "critical",
                    "measurement_thresholds": {{
                        "nuclear_perfusion_threshold": "≥10%",
                        "echo_segments_threshold": "≥3/16",
                        "mri_perfusion_threshold": "≥12%"
                    }}
                }}
            ]
        }}
    ],
    "documentation_requirements": [
        "Complete medical history including previous interventions",
        "Results of all relevant diagnostic tests with specific values",
        "Evidence that conservative treatment has been attempted and failed",
        "Clinical assessment supporting medical necessity"
    ],
    "exclusion_criteria": [
        "Contraindications to the procedure",
        "Alternative less invasive options not yet attempted"
    ],
    "coverage_determination": {{
        "likely_covered": true,
        "confidence_level": 0.85,
        "key_requirements": [
            "Document moderate to severe ischemia with specific test results",
            "Show that symptoms persist despite guideline-directed medical therapy",
            "Demonstrate that results will guide further intervention decisions"
        ]
    }}
}}

CRITICAL REQUIREMENTS:
- Extract ONLY criteria explicitly stated in the policy
- Include specific numerical thresholds when mentioned
- Create multiple pathways if the policy allows different routes to approval
- Make criteria measurable and verifiable against medical records
- Focus on this specific patient's age, gender, diagnosis, and comorbidities
- Include timeframe requirements (e.g., "within 30 days", "recent testing")

Return only the JSON object with detailed, actionable medical necessity requirements.
"""
    
    def _process_raw_criteria(self, raw_criteria: List[Dict[str, Any]], 
                            extracted_data: ExtractedData) -> List[ClinicalCriterion]:
        """Process raw AI-extracted criteria into structured ClinicalCriterion objects"""
        
        processed_criteria = []
        
        for i, raw_criterion in enumerate(raw_criteria):
            try:
                # Generate criterion ID if not provided
                criterion_id = raw_criterion.get('criterion_id', f"criterion_{i+1}")
                
                # Extract and clean description
                description = clean_text(raw_criterion.get('description', ''))
                if not description:
                    continue  # Skip empty criteria
                
                # Process required evidence
                required_evidence = self._process_evidence_list(
                    raw_criterion.get('required_evidence', [])
                )
                
                # Process clinical indicators
                clinical_indicators = self._process_indicators_list(
                    raw_criterion.get('clinical_indicators', [])
                )
                
                # Process exclusion criteria
                exclusion_criteria = self._process_exclusions_list(
                    raw_criterion.get('exclusion_criteria', [])
                )
                
                # Determine priority
                priority = raw_criterion.get('priority', 'standard')
                if priority not in ['critical', 'standard', 'optional']:
                    priority = 'standard'
                
                # Create ClinicalCriterion object
                clinical_criterion = ClinicalCriterion(
                    criterion_id=criterion_id,
                    description=description,
                    required_evidence=required_evidence,
                    clinical_indicators=clinical_indicators,
                    exclusion_criteria=exclusion_criteria,
                    priority=priority
                )
                
                processed_criteria.append(clinical_criterion)
                
            except Exception as e:
                logger.warning("Failed to process criterion", 
                             criterion_index=i, error=str(e))
                continue
        
        return processed_criteria
    
    def _process_evidence_list(self, evidence_list: List[str]) -> List[str]:
        """Process and clean required evidence list"""
        
        processed_evidence = []
        
        for evidence in evidence_list:
            if isinstance(evidence, str) and evidence.strip():
                cleaned_evidence = clean_text(evidence.strip())
                if len(cleaned_evidence) > 5:  # Minimum meaningful length
                    processed_evidence.append(cleaned_evidence)
        
        return processed_evidence
    
    def _process_indicators_list(self, indicators_list: List[str]) -> List[str]:
        """Process and clean clinical indicators list"""
        
        processed_indicators = []
        
        for indicator in indicators_list:
            if isinstance(indicator, str) and indicator.strip():
                cleaned_indicator = clean_text(indicator.strip())
                if len(cleaned_indicator) > 3:  # Minimum meaningful length
                    processed_indicators.append(cleaned_indicator)
        
        return processed_indicators
    
    def _process_exclusions_list(self, exclusions_list: List[str]) -> List[str]:
        """Process and clean exclusion criteria list"""
        
        processed_exclusions = []
        
        for exclusion in exclusions_list:
            if isinstance(exclusion, str) and exclusion.strip():
                cleaned_exclusion = clean_text(exclusion.strip())
                if len(cleaned_exclusion) > 5:  # Minimum meaningful length
                    processed_exclusions.append(cleaned_exclusion)
        
        return processed_exclusions
    
    def _validate_and_enhance_criteria(self, criteria: List[ClinicalCriterion], 
                                     extracted_data: ExtractedData) -> List[ClinicalCriterion]:
        """Validate and enhance extracted criteria"""
        
        validated_criteria = []
        
        for criterion in criteria:
            # Validate criterion completeness
            if not self._is_criterion_valid(criterion):
                logger.warning("Invalid criterion skipped", 
                             criterion_id=criterion.criterion_id)
                continue
            
            # Enhance criterion with additional context
            enhanced_criterion = self._enhance_criterion(criterion, extracted_data)
            
            validated_criteria.append(enhanced_criterion)
        
        # Remove duplicate criteria
        validated_criteria = self._remove_duplicate_criteria(validated_criteria)
        
        # Sort by priority (critical first)
        validated_criteria.sort(key=lambda x: self._get_priority_order(x.priority))
        
        return validated_criteria
    
    def _is_criterion_valid(self, criterion: ClinicalCriterion) -> bool:
        """Validate if a criterion is complete and actionable"""
        
        # Check required fields
        if not criterion.description or len(criterion.description) < 10:
            return False
        
        # Check if criterion has actionable content
        if not criterion.required_evidence and not criterion.clinical_indicators:
            return False
        
        # Check for meaningful content
        if any(word in criterion.description.lower() for word in ['unknown', 'unclear', 'not specified']):
            return False
        
        return True
    
    def _enhance_criterion(self, criterion: ClinicalCriterion, 
                         extracted_data: ExtractedData) -> ClinicalCriterion:
        """Enhance criterion with additional context and specificity"""
        
        # Add context-specific enhancements
        enhanced_description = criterion.description
        
        # Add payor context if not present
        if extracted_data.payor_name.lower() not in enhanced_description.lower():
            enhanced_description = f"{enhanced_description} (per {extracted_data.payor_name} policy)"
        
        # Enhance evidence requirements with specific document types
        enhanced_evidence = criterion.required_evidence.copy()
        
        # Add standard evidence requirements if missing
        standard_evidence = [
            "Current medical records documenting the condition",
            "Physician's clinical assessment and treatment plan"
        ]
        
        for evidence in standard_evidence:
            if not any(evidence.lower() in existing.lower() for existing in enhanced_evidence):
                enhanced_evidence.append(evidence)
        
        # Create enhanced criterion
        enhanced_criterion = ClinicalCriterion(
            criterion_id=criterion.criterion_id,
            description=enhanced_description,
            required_evidence=enhanced_evidence,
            clinical_indicators=criterion.clinical_indicators,
            exclusion_criteria=criterion.exclusion_criteria,
            priority=criterion.priority
        )
        
        return enhanced_criterion
    
    def _remove_duplicate_criteria(self, criteria: List[ClinicalCriterion]) -> List[ClinicalCriterion]:
        """Remove duplicate or very similar criteria"""
        
        unique_criteria = []
        seen_descriptions = set()
        
        for criterion in criteria:
            # Create a normalized description for comparison
            normalized_desc = self._normalize_description(criterion.description)
            
            if normalized_desc not in seen_descriptions:
                seen_descriptions.add(normalized_desc)
                unique_criteria.append(criterion)
            else:
                logger.debug("Duplicate criterion removed", 
                           criterion_id=criterion.criterion_id)
        
        return unique_criteria
    
    def _normalize_description(self, description: str) -> str:
        """Normalize description for duplicate detection"""
        
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', description.lower().strip())
        
        # Remove common variations
        normalized = re.sub(r'\(per .+ policy\)', '', normalized)
        normalized = re.sub(r'must be|should be|shall be', 'be', normalized)
        
        return normalized
    
    def _get_priority_order(self, priority: str) -> int:
        """Get numeric order for priority sorting"""
        
        priority_order = {
            'critical': 0,
            'standard': 1,
            'optional': 2
        }
        
        return priority_order.get(priority, 1)
    
    def _get_cpt_description(self, cpt_code: str) -> str:
        """Get brief description for CPT code"""
        
        cpt_descriptions = {
            '93458': 'Left heart catheterization with coronary angiography',
            '93459': 'Left heart catheterization with coronary angiography and left ventriculography',
            '93460': 'Right heart catheterization',
            '93461': 'Right and left heart catheterization',
            '99213': 'Office visit, established patient, moderate complexity',
            '99214': 'Office visit, established patient, high complexity',
            '70553': 'MRI brain with and without contrast',
            '73721': 'MRI knee without contrast'
        }
        
        return cpt_descriptions.get(cpt_code, 'Medical procedure')
    
    def _get_icd_description(self, icd_code: str) -> str:
        """Get brief description for ICD code"""
        
        icd_descriptions = {
            'I25.10': 'Atherosclerotic heart disease of native coronary artery without angina pectoris',
            'I20.9': 'Angina pectoris, unspecified',
            'I25.9': 'Chronic ischemic heart disease, unspecified',
            'E11.9': 'Type 2 diabetes mellitus without complications',
            'I10': 'Essential hypertension',
            'E78.5': 'Hyperlipidemia, unspecified',
            'Z87.891': 'Personal history of nicotine dependence'
        }
        
        return icd_descriptions.get(icd_code, 'Medical condition')
    
    def _convert_pathways_to_criteria(self, pathways_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert pathway-based format to criteria list format"""
        
        criteria_list = []
        
        # Extract criteria from pathways
        pathways = pathways_data.get('medical_necessity_pathways', [])
        
        for pathway in pathways:
            pathway_name = pathway.get('pathway_name', 'Unknown Pathway')
            qualifying_criteria = pathway.get('qualifying_criteria', [])
            
            for criterion in qualifying_criteria:
                # Convert pathway criterion to standard format
                converted_criterion = {
                    'criterion_id': criterion.get('criterion_id', f"pathway_{len(criteria_list)}"),
                    'description': f"{pathway_name}: {criterion.get('description', '')}",
                    'criterion_type': 'clinical_indicator',
                    'priority': criterion.get('priority', 'standard'),
                    'required_evidence': criterion.get('required_evidence', []),
                    'clinical_indicators': criterion.get('clinical_indicators', []),
                    'exclusion_criteria': [],
                    'measurement_thresholds': criterion.get('measurement_thresholds', {}),
                    'pathway_context': pathway_name
                }
                
                criteria_list.append(converted_criterion)
        
        # Add general documentation requirements as criteria
        doc_requirements = pathways_data.get('documentation_requirements', [])
        if doc_requirements:
            doc_criterion = {
                'criterion_id': 'general_documentation',
                'description': 'General documentation requirements',
                'criterion_type': 'documentation',
                'priority': 'critical',
                'required_evidence': doc_requirements,
                'clinical_indicators': [],
                'exclusion_criteria': pathways_data.get('exclusion_criteria', [])
            }
            criteria_list.append(doc_criterion)
        
        return criteria_list


class CriteriaEnhancementService:
    """Service for enhancing and contextualizing clinical criteria"""
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
    
    @handle_processing_errors
    def enhance_criteria_with_context(self, criteria: List[ClinicalCriterion], 
                                    extracted_data: ExtractedData) -> List[ClinicalCriterion]:
        """Enhance criteria with patient-specific context"""
        
        enhanced_criteria = []
        
        for criterion in criteria:
            # Add patient-specific context
            contextualized_criterion = self._add_patient_context(criterion, extracted_data)
            
            # Add code-specific requirements
            code_enhanced_criterion = self._add_code_specific_requirements(
                contextualized_criterion, extracted_data
            )
            
            enhanced_criteria.append(code_enhanced_criterion)
        
        return enhanced_criteria
    
    def _add_patient_context(self, criterion: ClinicalCriterion, 
                           extracted_data: ExtractedData) -> ClinicalCriterion:
        """Add patient-specific context to criterion"""
        
        # Add age-specific considerations
        age_context = self._get_age_specific_context(
            extracted_data.patient_info.age_range, criterion.description
        )
        
        # Add gender-specific considerations
        gender_context = self._get_gender_specific_context(
            extracted_data.patient_info.gender, criterion.description
        )
        
        # Enhance clinical indicators with patient context
        enhanced_indicators = criterion.clinical_indicators.copy()
        
        if age_context:
            enhanced_indicators.append(f"Age-specific consideration: {age_context}")
        
        if gender_context:
            enhanced_indicators.append(f"Gender-specific consideration: {gender_context}")
        
        return ClinicalCriterion(
            criterion_id=criterion.criterion_id,
            description=criterion.description,
            required_evidence=criterion.required_evidence,
            clinical_indicators=enhanced_indicators,
            exclusion_criteria=criterion.exclusion_criteria,
            priority=criterion.priority
        )
    
    def _get_age_specific_context(self, age_range: str, description: str) -> Optional[str]:
        """Get age-specific context for criterion"""
        
        if not age_range or age_range == 'unknown':
            return None
        
        # Extract age information
        age_parts = age_range.split('-')
        if len(age_parts) == 2:
            try:
                min_age = int(age_parts[0])
                max_age = int(age_parts[1])
                avg_age = (min_age + max_age) // 2
                
                # Age-specific considerations
                if avg_age < 18:
                    return "Pediatric considerations may apply"
                elif avg_age > 65:
                    return "Geriatric considerations and comorbidities should be evaluated"
                elif 'cardiac' in description.lower() and avg_age > 50:
                    return "Age-related cardiovascular risk factors should be documented"
                
            except ValueError:
                pass
        
        return None
    
    def _get_gender_specific_context(self, gender: str, description: str) -> Optional[str]:
        """Get gender-specific context for criterion"""
        
        if not gender or gender == 'unknown':
            return None
        
        # Gender-specific considerations for certain procedures
        if gender == 'female':
            if any(term in description.lower() for term in ['cardiac', 'heart']):
                return "Consider pregnancy status and hormonal factors"
            elif 'imaging' in description.lower():
                return "Pregnancy screening may be required for certain imaging"
        
        return None
    
    def _add_code_specific_requirements(self, criterion: ClinicalCriterion, 
                                      extracted_data: ExtractedData) -> ClinicalCriterion:
        """Add CPT/ICD code-specific requirements to criterion"""
        
        enhanced_evidence = criterion.required_evidence.copy()
        
        # Add code-specific documentation requirements
        for cpt_code in extracted_data.cpt_codes:
            code_requirements = self._get_cpt_specific_requirements(cpt_code)
            enhanced_evidence.extend(code_requirements)
        
        for icd_code in extracted_data.icd_codes:
            code_requirements = self._get_icd_specific_requirements(icd_code)
            enhanced_evidence.extend(code_requirements)
        
        # Remove duplicates
        enhanced_evidence = list(set(enhanced_evidence))
        
        return ClinicalCriterion(
            criterion_id=criterion.criterion_id,
            description=criterion.description,
            required_evidence=enhanced_evidence,
            clinical_indicators=criterion.clinical_indicators,
            exclusion_criteria=criterion.exclusion_criteria,
            priority=criterion.priority
        )
    
    def _get_cpt_specific_requirements(self, cpt_code: str) -> List[str]:
        """Get CPT code-specific documentation requirements"""
        
        requirements = []
        
        # Basic CPT code range mappings
        if cpt_code.startswith('93'):  # Cardiovascular procedures
            requirements.extend([
                "Echocardiogram or cardiac imaging results",
                "Cardiac catheterization report if applicable",
                "Cardiology consultation notes"
            ])
        elif cpt_code.startswith('70'):  # Head and neck imaging
            requirements.extend([
                "Clinical indication for imaging",
                "Previous imaging results for comparison",
                "Neurological examination findings"
            ])
        elif cpt_code.startswith('27'):  # Musculoskeletal procedures
            requirements.extend([
                "Orthopedic examination findings",
                "Imaging studies (X-ray, MRI, CT)",
                "Conservative treatment history"
            ])
        
        return requirements
    
    def _get_icd_specific_requirements(self, icd_code: str) -> List[str]:
        """Get ICD code-specific documentation requirements"""
        
        requirements = []
        
        # ICD-10 code range mappings
        if icd_code.startswith('I'):  # Circulatory system diseases
            requirements.extend([
                "Cardiovascular examination findings",
                "Diagnostic test results supporting diagnosis",
                "Risk factor assessment"
            ])
        elif icd_code.startswith('M'):  # Musculoskeletal diseases
            requirements.extend([
                "Physical examination of affected area",
                "Functional assessment",
                "Pain assessment documentation"
            ])
        elif icd_code.startswith('G'):  # Nervous system diseases
            requirements.extend([
                "Neurological examination",
                "Cognitive or functional assessment",
                "Diagnostic imaging if indicated"
            ])
        
        return requirements
    
    def _get_cpt_description(self, cpt_code: str) -> str:
        """Get brief description for CPT code"""
        
        cpt_descriptions = {
            '93458': 'Left heart catheterization with coronary angiography',
            '93459': 'Left heart catheterization with coronary angiography and left ventriculography',
            '93460': 'Right heart catheterization',
            '93461': 'Right and left heart catheterization',
            '99213': 'Office visit, established patient, moderate complexity',
            '99214': 'Office visit, established patient, high complexity',
            '70553': 'MRI brain with and without contrast',
            '73721': 'MRI knee without contrast'
        }
        
        return cpt_descriptions.get(cpt_code, 'Medical procedure')
    
    def _get_icd_description(self, icd_code: str) -> str:
        """Get brief description for ICD code"""
        
        icd_descriptions = {
            'I25.10': 'Atherosclerotic heart disease of native coronary artery without angina pectoris',
            'I20.9': 'Angina pectoris, unspecified',
            'I25.9': 'Chronic ischemic heart disease, unspecified',
            'E11.9': 'Type 2 diabetes mellitus without complications',
            'I10': 'Essential hypertension',
            'E78.5': 'Hyperlipidemia, unspecified',
            'Z87.891': 'Personal history of nicotine dependence'
        }
        
        return icd_descriptions.get(icd_code, 'Medical condition')
    
    def _convert_pathways_to_criteria(self, pathways_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert pathway-based format to criteria list format"""
        
        criteria_list = []
        
        # Extract criteria from pathways
        pathways = pathways_data.get('medical_necessity_pathways', [])
        
        for pathway in pathways:
            pathway_name = pathway.get('pathway_name', 'Unknown Pathway')
            qualifying_criteria = pathway.get('qualifying_criteria', [])
            
            for criterion in qualifying_criteria:
                # Convert pathway criterion to standard format
                converted_criterion = {
                    'criterion_id': criterion.get('criterion_id', f"pathway_{len(criteria_list)}"),
                    'description': f"{pathway_name}: {criterion.get('description', '')}",
                    'criterion_type': 'clinical_indicator',
                    'priority': criterion.get('priority', 'standard'),
                    'required_evidence': criterion.get('required_evidence', []),
                    'clinical_indicators': criterion.get('clinical_indicators', []),
                    'exclusion_criteria': [],
                    'measurement_thresholds': criterion.get('measurement_thresholds', {}),
                    'pathway_context': pathway_name
                }
                
                criteria_list.append(converted_criterion)
        
        # Add general documentation requirements as criteria
        doc_requirements = pathways_data.get('documentation_requirements', [])
        if doc_requirements:
            doc_criterion = {
                'criterion_id': 'general_documentation',
                'description': 'General documentation requirements',
                'criterion_type': 'documentation',
                'priority': 'critical',
                'required_evidence': doc_requirements,
                'clinical_indicators': [],
                'exclusion_criteria': pathways_data.get('exclusion_criteria', [])
            }
            criteria_list.append(doc_criterion)
        
        return criteria_list

# Force deployment update - methods moved to ClinicalCriteriaExtractor class