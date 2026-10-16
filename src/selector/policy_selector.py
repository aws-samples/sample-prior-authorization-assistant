"""
Intelligent policy selection algorithm for the Policy Selector module.

This module implements AI-powered policy selection using extracted medical data
to find the most relevant policy and determine applicable clinical requirements.

Version: 1.2.0 - Fixed datetime serialization in all handlers
"""

import json
from string import Template
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from common.aws_clients import get_bedrock_service
from common.model_config import resolve as resolve_model_role
from common.models import ExtractedData, SelectedPolicy, ClinicalCriterion
from common.exceptions import PolicySelectionError, AWSServiceError
from common.decorators import handle_processing_errors, monitor_performance, retry_with_backoff
from common.logging_config import get_logger, get_audit_logger
from common.utils import calculate_confidence_score, safe_json_loads
from knowledge.policy_retrieval import PolicyRetrievalService, SearchQuery, PolicySearchResult

logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class PolicySelectionCriteria:
    """Criteria for policy selection scoring"""
    payor_match_weight: float = 0.35
    code_overlap_weight: float = 0.25
    clinical_area_weight: float = 0.15
    service_type_weight: float = 0.10
    recency_weight: float = 0.10
    completeness_weight: float = 0.05


class IntelligentPolicySelector:
    """AI-powered policy selection with multi-factor scoring"""
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
        self.policy_retrieval = PolicyRetrievalService()
        # Used only by _extract_clinical_summary; other calls use invoke_role
        self.model_id = resolve_model_role("policy_selection").model_id
        self.selection_criteria = PolicySelectionCriteria()
    
    @handle_processing_errors
    @monitor_performance
    def select_best_policy(self, extracted_data: ExtractedData, 
                          max_candidates: int = 5) -> SelectedPolicy:
        """Select the most relevant policy based on extracted medical data"""
        
        logger.info("Starting policy selection",
                   document_id=extracted_data.document_id,
                   payor=extracted_data.payor_name,
                   cpt_codes=len(extracted_data.cpt_codes),
                   icd_codes=len(extracted_data.icd_codes))
        
        # Step 1: Retrieve candidate policies
        candidate_policies = self._retrieve_candidate_policies(extracted_data, max_candidates)
        
        if not candidate_policies:
            raise PolicySelectionError(
                f"No policies found for payor: {extracted_data.payor_name}",
                payor=extracted_data.payor_name
            )
        
        # Step 2: Score and rank candidates
        scored_policies = self._score_policy_candidates(candidate_policies, extracted_data)
        
        # Step 3: Select best policy using AI
        best_policy = self._ai_select_best_policy(scored_policies, extracted_data)
        
        # Step 4: Extract applicable clinical criteria
        clinical_criteria = self._extract_clinical_criteria(best_policy, extracted_data)
        
        # Step 5: Create final selected policy object
        # Extract S3 URI from Bedrock metadata (x-amz-bedrock-kb-source-uri)
        s3_uri = best_policy.metadata.get('x-amz-bedrock-kb-source-uri', best_policy.metadata.get('s3_uri', ''))
        
        # Try to construct source URL from metadata
        source_url = best_policy.metadata.get('source_url', '')
        if not source_url and 'source' in best_policy.metadata:
            # If we have a source filename, we can't construct the full URL without additional info
            # Just use the source filename as a reference
            source_url = best_policy.metadata.get('source', '')
        
        selected_policy = SelectedPolicy(
            policy_id=best_policy.policy_id,
            payor=best_policy.payor,
            policy_title=best_policy.policy_title,
            effective_date=best_policy.metadata.get('effective_date', ''),
            relevance_score=best_policy.relevance_score,
            applicable_criteria=clinical_criteria,
            policy_version=best_policy.metadata.get('version', '1.0'),
            policy_metadata={
                'source_url': source_url,
                's3_uri': s3_uri,
                'policy_title': best_policy.policy_title,
                'effective_date': best_policy.metadata.get('effective_date', ''),
                'payor': best_policy.payor,
                'policy_number': best_policy.metadata.get('policy-number', ''),
                'service_category': best_policy.metadata.get('service-category', '')
            }
        )
        
        # Log successful selection
        audit_logger.log_policy_access(
            policy_id=selected_policy.policy_id,
            payor=selected_policy.payor,
            access_type="select"
        )
        
        logger.info("Policy selection completed",
                   document_id=extracted_data.document_id,
                   selected_policy_id=selected_policy.policy_id,
                   relevance_score=selected_policy.relevance_score,
                   criteria_count=len(selected_policy.applicable_criteria))
        
        return selected_policy
    
    def _retrieve_candidate_policies(self, extracted_data: ExtractedData, 
                                   max_candidates: int) -> List[PolicySearchResult]:
        """Retrieve candidate policies using the knowledge base with metadata filtering
        
        NOTE: This now aggregates KB chunks by policy to avoid treating chunks as separate policies.
        """
        
        # Convert ExtractedData to dictionary format for two-tier filtering
        extraction_result = {
            'payor_name': extracted_data.payor_name,
            'cpt_codes': extracted_data.cpt_codes,
            'icd_codes': extracted_data.icd_codes,
            'service_type': extracted_data.service_type,
            'patient_info': {
                'age_range': extracted_data.patient_info.age_range,
                'gender': extracted_data.patient_info.gender,
                'primary_diagnosis': extracted_data.patient_info.primary_diagnosis
            },
            'structured_fields': {}
        }
        
        # Use two-tier metadata filtering for precise policy retrieval
        # This returns KB chunks, which we need to aggregate by policy
        # Note: Bedrock KB has a hard limit of 30 results max
        kb_chunks = self.policy_retrieval.search_policies_with_two_tier_filtering(
            extraction_result, 
            min(30, max_candidates * 2)  # Get more chunks but respect Bedrock's 30 limit
        )
        
        # FIX 1: Aggregate chunks by policy
        aggregated_policies = self._aggregate_chunks_by_policy(kb_chunks, extracted_data)
        
        logger.info("Aggregated KB chunks into policies",
                   total_chunks=len(kb_chunks),
                   unique_policies=len(aggregated_policies))
        
        # Return top policies (not chunks)
        return aggregated_policies[:max_candidates]
    
    def _aggregate_chunks_by_policy(self, kb_chunks: List[PolicySearchResult], 
                                   extracted_data: ExtractedData) -> List[PolicySearchResult]:
        """Aggregate KB chunks by policy to treat each policy as a single candidate
        
        This fixes the issue where 10 chunks from 2 policies were treated as 10 separate policies.
        """
        
        policy_groups = {}
        
        for chunk in kb_chunks:
            # Create unique policy key from policy number and payor
            policy_number = chunk.metadata.get('policy-number', 'unknown')
            payor = chunk.payor
            policy_key = f"{payor}_{policy_number}"
            
            if policy_key not in policy_groups:
                policy_groups[policy_key] = {
                    'policy_id': policy_number,
                    'policy_title': chunk.policy_title,
                    'payor': payor,
                    'chunks': [],
                    'metadata': chunk.metadata.copy(),
                    'all_chunk_scores': []
                }
            
            # Add chunk to this policy group
            policy_groups[policy_key]['chunks'].append(chunk)
            policy_groups[policy_key]['all_chunk_scores'].append(chunk.relevance_score)
        
        # Convert policy groups to aggregated PolicySearchResult objects
        aggregated_policies = []
        
        for policy_key, group in policy_groups.items():
            chunks = group['chunks']
            
            # Calculate aggregated relevance score
            # Use max score as primary indicator (best chunk match)
            max_score = max(group['all_chunk_scores'])
            avg_score = sum(group['all_chunk_scores']) / len(group['all_chunk_scores'])
            
            # Weighted score: 70% max, 30% average
            # This rewards policies with at least one highly relevant chunk
            # while also considering overall relevance across all chunks
            aggregated_score = (0.7 * max_score) + (0.3 * avg_score)
            
            # Combine content from all chunks (for AI selection context)
            combined_content = '\n\n---CHUNK SEPARATOR---\n\n'.join(
                chunk.matched_content for chunk in chunks
            )
            
            # Create aggregated policy result
            aggregated_policy = PolicySearchResult(
                policy_id=group['policy_id'],
                payor=group['payor'],
                policy_title=group['policy_title'],
                relevance_score=aggregated_score,
                confidence_score=max(chunk.confidence_score for chunk in chunks),
                matched_content=combined_content[:2000],  # Limit for AI prompt
                metadata={
                    **group['metadata'],
                    'chunk_count': len(chunks),
                    'max_relevance_score': max_score,
                    'avg_relevance_score': avg_score,
                    'all_chunks': chunks  # Store all chunks for later retrieval
                },
                retrieved_at=chunks[0].retrieved_at  # Use retrieved_at from first chunk
            )
            
            aggregated_policies.append(aggregated_policy)
            
            logger.info("Aggregated policy",
                       policy_id=group['policy_id'],
                       policy_title=group['policy_title'],
                       chunk_count=len(chunks),
                       max_score=max_score,
                       avg_score=avg_score,
                       aggregated_score=aggregated_score)
        
        # Sort by aggregated relevance score (descending)
        aggregated_policies.sort(key=lambda p: p.relevance_score, reverse=True)
        
        # Filter for exact payor match
        exact_payor_matches = [
            policy for policy in aggregated_policies
            if policy.payor.lower() == extracted_data.payor_name.lower()
        ]
        
        # Return exact matches if available, otherwise all aggregated policies
        return exact_payor_matches if exact_payor_matches else aggregated_policies
    
    def _determine_clinical_area(self, extracted_data: ExtractedData) -> Optional[str]:
        """Determine clinical area from extracted data"""
        
        # Use primary diagnosis to infer clinical area
        diagnosis = extracted_data.patient_info.primary_diagnosis.lower()
        
        clinical_area_keywords = {
            'cardiology': ['cardiac', 'heart', 'cardiovascular', 'coronary', 'angina', 'myocardial'],
            'orthopedics': ['bone', 'joint', 'fracture', 'arthritis', 'spine', 'musculoskeletal'],
            'radiology': ['imaging', 'scan', 'x-ray', 'mri', 'ct', 'ultrasound'],
            'oncology': ['cancer', 'tumor', 'malignant', 'neoplasm', 'carcinoma'],
            'neurology': ['neurological', 'brain', 'nervous', 'seizure', 'stroke', 'epilepsy']
        }
        
        for area, keywords in clinical_area_keywords.items():
            if any(keyword in diagnosis for keyword in keywords):
                return area
        
        # Check CPT codes for clinical area hints
        if extracted_data.cpt_codes:
            first_cpt = extracted_data.cpt_codes[0]
            # Basic CPT code range mapping
            if first_cpt.startswith(('93', '92')):
                return 'cardiology'
            elif first_cpt.startswith(('27', '29')):
                return 'orthopedics'
            elif first_cpt.startswith(('70', '71', '72', '73', '74', '75', '76')):
                return 'radiology'
        
        return None
    
    def _score_policy_candidates(self, candidates: List[PolicySearchResult], 
                               extracted_data: ExtractedData) -> List[Tuple[PolicySearchResult, float]]:
        """Score policy candidates using multi-factor analysis"""
        
        scored_policies = []
        
        for policy in candidates:
            score_factors = self._calculate_policy_score_factors(policy, extracted_data)
            
            # Calculate weighted score
            weights = {
                'payor_match': self.selection_criteria.payor_match_weight,
                'code_overlap': self.selection_criteria.code_overlap_weight,
                'clinical_area': self.selection_criteria.clinical_area_weight,
                'service_type': self.selection_criteria.service_type_weight,
                'recency': self.selection_criteria.recency_weight,
                'completeness': self.selection_criteria.completeness_weight
            }
            
            overall_score = calculate_confidence_score(score_factors, weights)
            scored_policies.append((policy, overall_score))
        
        # Sort by score (descending)
        scored_policies.sort(key=lambda x: x[1], reverse=True)
        
        return scored_policies
    
    def _calculate_policy_score_factors(self, policy: PolicySearchResult, 
                                      extracted_data: ExtractedData) -> Dict[str, float]:
        """Calculate individual scoring factors for a policy"""
        
        metadata = policy.metadata
        
        # Payor match score
        payor_match = 1.0 if policy.payor.lower() == extracted_data.payor_name.lower() else 0.0
        
        # Code overlap scores
        policy_cpts = set(metadata.get('cpt_codes', []))
        policy_icds = set(metadata.get('icd_codes', []))
        extracted_cpts = set(extracted_data.cpt_codes)
        extracted_icds = set(extracted_data.icd_codes)
        
        cpt_overlap = len(policy_cpts & extracted_cpts)
        icd_overlap = len(policy_icds & extracted_icds)
        
        total_codes = len(extracted_cpts | extracted_icds)
        code_overlap_score = (cpt_overlap + icd_overlap) / max(1, total_codes)
        
        # Clinical area match
        policy_clinical_area = metadata.get('clinical_area', '').lower()
        extracted_clinical_area = self._determine_clinical_area(extracted_data)
        clinical_area_score = 1.0 if (extracted_clinical_area and 
                                    policy_clinical_area == extracted_clinical_area.lower()) else 0.5
        
        # Service type match
        policy_services = metadata.get('service_categories', [])
        service_match = any(
            extracted_data.service_type.lower() in service.lower() 
            for service in policy_services
        )
        service_type_score = 1.0 if service_match else 0.3
        
        # Policy recency score
        recency_score = self._calculate_recency_score(metadata.get('effective_date'))
        
        # Metadata completeness score
        completeness_score = self._calculate_completeness_score(metadata)
        
        return {
            'payor_match': payor_match,
            'code_overlap': code_overlap_score,
            'clinical_area': clinical_area_score,
            'service_type': service_type_score,
            'recency': recency_score,
            'completeness': completeness_score
        }
    
    def _calculate_recency_score(self, effective_date_str: Optional[str]) -> float:
        """Calculate recency score based on policy effective date"""
        
        if not effective_date_str:
            return 0.5  # Unknown date gets medium score
        
        try:
            from datetime import date
            effective_date = date.fromisoformat(effective_date_str)
            days_old = (date.today() - effective_date).days
            
            # Policies are fresh for 1 year, then decay
            if days_old <= 365:
                return 1.0
            elif days_old <= 730:  # 2 years
                return 0.8
            elif days_old <= 1095:  # 3 years
                return 0.6
            else:
                return 0.4
        
        except (ValueError, TypeError):
            return 0.5
    
    def _calculate_completeness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate metadata completeness score"""
        
        important_fields = [
            'policy_id', 'payor', 'policy_title', 'policy_type',
            'clinical_area', 'effective_date', 'cpt_codes', 'icd_codes'
        ]
        
        present_fields = sum(1 for field in important_fields if metadata.get(field))
        return present_fields / len(important_fields)
    
    @retry_with_backoff(max_attempts=3)
    def _ai_select_best_policy(self, scored_policies: List[Tuple[PolicySearchResult, float]], 
                             extracted_data: ExtractedData) -> PolicySearchResult:
        """Use AI to make final policy selection decision"""
        
        if len(scored_policies) == 1:
            return scored_policies[0][0]  # Only one candidate
        
        # Prepare policy comparison data for AI
        policy_comparison = self._prepare_policy_comparison(scored_policies, extracted_data)
        
        # Create AI prompt for policy selection
        prompt = self._create_policy_selection_prompt(policy_comparison, extracted_data)
        
        try:
            # Get AI recommendation
            response = self.bedrock_service.invoke_role(
                "policy_selection",
                prompt=prompt,
                max_tokens=1000
            )
            
            # Parse AI response
            ai_decision = self._parse_ai_selection_response(response, scored_policies)
            
            return ai_decision
            
        except Exception as e:
            logger.warning("AI selection failed", error=str(e))
            # Return error instead of fallback
            raise PolicySelectionError(f"AI policy selection failed: {str(e)}")
    
    def _prepare_policy_comparison(self, scored_policies: List[Tuple[PolicySearchResult, float]], 
                                 extracted_data: ExtractedData) -> Dict[str, Any]:
        """Prepare policy comparison data for AI analysis"""
        
        # Extract additional context from structured fields
        structured_fields = extracted_data.structured_fields or {}
        service_details = structured_fields.get('service_details', {})
        diagnosis_details = structured_fields.get('diagnosis_details', {})
        
        # Get CPT and ICD descriptions
        cpt_descriptions = service_details.get('all_cpt_descriptions', [])
        diagnosis_descriptions = diagnosis_details.get('diagnosis_descriptions', [])
        
        # Extract clinical summary from raw text if available
        clinical_summary = self._extract_clinical_summary(extracted_data.raw_text)
        
        comparison_data = {
            'extracted_data_summary': {
                'payor': extracted_data.payor_name,
                'cpt_codes': extracted_data.cpt_codes,
                'cpt_descriptions': cpt_descriptions,  # NEW: Procedure descriptions
                'icd_codes': extracted_data.icd_codes,
                'icd_descriptions': diagnosis_descriptions,  # NEW: Diagnosis descriptions
                'service_type': extracted_data.service_type,
                'primary_diagnosis': extracted_data.patient_info.primary_diagnosis,
                'patient_age_range': extracted_data.patient_info.age_range,
                'patient_gender': extracted_data.patient_info.gender,
                'clinical_summary': clinical_summary  # NEW: Clinical context
            },
            'policy_candidates': []
        }
        
        for i, (policy, score) in enumerate(scored_policies[:3]):  # Top 3 candidates
            policy_data = {
                'rank': i + 1,
                'policy_id': policy.policy_id,
                'payor': policy.payor,
                'title': policy.policy_title,
                'relevance_score': policy.relevance_score,
                'calculated_score': score,
                'matched_content_preview': policy.matched_content[:200],
                'metadata': {
                    'clinical_area': policy.metadata.get('clinical_area'),
                    'policy_type': policy.metadata.get('policy_type'),
                    'cpt_codes': policy.metadata.get('cpt_codes', [])[:5],  # First 5
                    'icd_codes': policy.metadata.get('icd_codes', [])[:5],  # First 5
                    'effective_date': policy.metadata.get('effective_date'),
                    'service_categories': policy.metadata.get('service_categories', [])
                }
            }
            comparison_data['policy_candidates'].append(policy_data)
        
        return comparison_data
    
    @retry_with_backoff(max_attempts=2)
    def _extract_clinical_summary(self, raw_text: str) -> str:
        """Extract brief clinical summary from raw medical records using AI"""
        if not raw_text or len(raw_text) < 100:
            return "No clinical summary available"
        
        # Extract the decoded document content if present
        decoded_start = raw_text.find('DECODED DOCUMENT CONTENTS:')
        if decoded_start > 0:
            # Get content after the decoded marker
            content_to_summarize = raw_text[decoded_start:decoded_start+3000]  # First 3000 chars
        else:
            # Use first 3000 chars of raw text
            content_to_summarize = raw_text[:3000]
        
        # Create AI prompt for clinical summary extraction
        prompt = f"""Extract a concise clinical summary from these medical records. Focus ONLY on the most relevant information for prior authorization:

1. Chief complaint and primary symptoms
2. Failed conservative treatments (if any)
3. Key diagnostic findings (imaging, labs, physical exam)

Medical Records:
{content_to_summarize}

IMPORTANT:
- Keep the summary under 200 words
- Use bullet points or pipe-separated format
- Include only factual clinical information
- Omit patient identifiers, dates, and provider names

Return only the clinical summary, no additional text or explanations."""
        
        try:
            response = self.bedrock_service.invoke_model(
                model_id=self.model_id,
                prompt=prompt,
                max_tokens=400,
                temperature=0.3  # Lower temperature for more factual extraction
            )
            
            # Clean up the response
            summary = response.strip()
            
            # Limit to 500 characters for the prompt
            if len(summary) > 500:
                summary = summary[:497] + "..."
            
            logger.info("AI clinical summary extracted",
                       summary_length=len(summary),
                       source_length=len(content_to_summarize))
            
            return summary if summary else "Clinical summary extraction failed"
            
        except Exception as e:
            logger.warning("AI clinical summary extraction failed, using fallback",
                          error=str(e))
            
            # Fallback to simple extraction
            return self._extract_clinical_summary_fallback(raw_text)
    
    def _extract_clinical_summary_fallback(self, raw_text: str) -> str:
        """Fallback method for clinical summary extraction using simple string matching"""
        summary_parts = []
        
        # Extract chief complaint
        if 'CHIEF COMPLAINT:' in raw_text:
            start = raw_text.find('CHIEF COMPLAINT:')
            end = raw_text.find('\n\n', start)
            if end > start:
                summary_parts.append(raw_text[start:end].strip())
        
        # Extract conservative treatment
        if 'CONSERVATIVE TREATMENT' in raw_text:
            start = raw_text.find('CONSERVATIVE TREATMENT')
            end = raw_text.find('\n\n', start)
            if end > start:
                summary_parts.append(raw_text[start:end].strip())
        
        # Extract imaging findings
        if 'IMAGING:' in raw_text or 'MRI' in raw_text:
            start = raw_text.find('IMAGING:') if 'IMAGING:' in raw_text else raw_text.find('MRI')
            end = raw_text.find('\n\n', start)
            if end > start:
                summary_parts.append(raw_text[start:end].strip())
        
        if summary_parts:
            return ' | '.join(summary_parts)[:500]
        
        # Last resort: return first 300 chars of decoded content
        decoded_start = raw_text.find('DECODED DOCUMENT CONTENTS:')
        if decoded_start > 0:
            return raw_text[decoded_start:decoded_start+300].strip()
        
        return "Clinical summary not available"
    
    def _create_policy_selection_prompt(self, comparison_data: Dict[str, Any], 
                                      extracted_data: ExtractedData) -> str:
        """Create AI prompt for policy selection"""
        
        # Format CPT and ICD codes + descriptions for better readability
        cpt_codes = comparison_data['extracted_data_summary'].get('cpt_codes', [])
        cpt_desc = comparison_data['extracted_data_summary'].get('cpt_descriptions', [])
        icd_codes = comparison_data['extracted_data_summary'].get('icd_codes', [])
        icd_desc = comparison_data['extracted_data_summary'].get('icd_descriptions', [])
        clinical_summary = comparison_data['extracted_data_summary'].get('clinical_summary', '')
        
        # Format CPT: show codes + descriptions
        if cpt_desc:
            cpt_display = '\n  '.join(cpt_desc)
        elif cpt_codes:
            cpt_display = 'CPT ' + ', '.join(cpt_codes)
        else:
            cpt_display = 'Not specified'
        
        # Format ICD: show codes + descriptions together
        if icd_codes and icd_desc:
            # Combine codes with descriptions
            icd_parts = []
            for i, code in enumerate(icd_codes):
                if i < len(icd_desc):
                    icd_parts.append(f"ICD {code} - {icd_desc[i]}")
                else:
                    icd_parts.append(f"ICD {code}")
            icd_display = '\n  '.join(icd_parts)
        elif icd_desc:
            icd_display = '\n  '.join(icd_desc)
        elif icd_codes:
            icd_display = 'ICD ' + ', '.join(icd_codes)
        else:
            icd_display = 'Not specified'
        
        # Extract variables for prompt template (avoids f-string which triggers B608)
        summary = comparison_data['extracted_data_summary']
        prompt_payor = summary['payor']
        prompt_age = summary['patient_age_range']
        prompt_gender = summary['patient_gender']
        prompt_service_type = summary['service_type']
        prompt_candidates = json.dumps(comparison_data['policy_candidates'], indent=2)

        _POLICY_SELECTION_TEMPLATE = Template(
            """
You are a medical policy expert tasked with selecting the most appropriate prior authorization policy for a specific medical case.

MEDICAL CASE SUMMARY:
- Payor: $payor
- Patient: $age year old $gender

REQUESTED PROCEDURE:
  $cpt_display

DIAGNOSIS/CONDITION:
  $icd_display

CLINICAL CONTEXT:
  $clinical_summary

SERVICE TYPE: $service_type

POLICY CANDIDATES (ranked by knowledge base relevance):
$candidates

SELECTION CRITERIA (in order of importance):
1. **Knowledge Base Relevance Score** - The relevance_score from the knowledge base search is the PRIMARY indicator of policy match. Higher scores indicate better semantic matching to the requested procedure.
2. **Policy Title Match** - The policy title should directly relate to the requested procedure (e.g., "Spinal Surgery" for spinal fusion procedures).
3. **CPT/HCPCS/RxNorm Code Coverage** - The policy should cover the requested procedure codes. Note: Policy metadata may contain CODE RANGES (e.g., "22532-22634" or "M48.00-M48.03"). A discrete code matches if it falls within any listed range or matches exactly.
4. **Exact Payor Match** - The policy must be from the correct insurance payor.
5. **ICD Code Alignment** - The policy should cover the diagnosis codes. Note: Policy metadata may contain CODE RANGES (e.g., "M48.00-M48.03"). A discrete code like "M48.06" matches if it falls within any listed range.
6. **Clinical Area Alignment** - The policy should match the clinical specialty.

CODE RANGE MATCHING RULES:
- Policy metadata contains code ranges in various formats:
  * CPT ranges: "22532-22634" (procedures)
  * HCPCS ranges: "E0100-E0199" (durable medical equipment), "J0100-J0299" (drugs)
  * ICD ranges: "M48.00-M48.03" (diagnoses)
  * RxNorm ranges: "197361-197500" (medications)
- A discrete code matches if it falls within the range: CPT 22558 matches range "22532-22634"
- A discrete code matches if it equals any individual code: ICD M51.26 matches if "M51.26" is listed
- When evaluating code coverage, check if the requested codes fall within ANY of the policy's code ranges

CRITICAL GUIDANCE:
- The knowledge base relevance_score is calculated using semantic search and is the MOST RELIABLE indicator of policy relevance
- A policy with a HIGHER relevance_score is almost always the better choice
- Prioritize policies whose titles directly describe the PRIMARY procedure being requested, not ancillary or supporting services
- Distinguish between primary procedure policies (e.g., surgical procedures, diagnostic tests, treatments) and ancillary service policies (e.g., monitoring, anesthesia, facility services)
- When multiple policies match, select the one that addresses the MAIN medical necessity criteria for the requested procedure
- When comparing code coverage, remember that ranges like "22532-22634" include all codes between those numbers

TASK:
Select the most appropriate policy for evaluating medical necessity of the requested procedure. Consider:
- Which policy has the highest relevance_score from the knowledge base?
- Which policy title most directly describes the PRIMARY procedure being requested (not ancillary services)?
- Which policy is most likely to contain the medical necessity criteria for the MAIN procedure?
- Does the policy cover the requested CPT/HCPCS/RxNorm codes (including ranges)?
- Does the policy cover the diagnosis ICD codes (including ranges)?

Respond with a JSON object containing:
{
    "selected_policy_id": "policy_id_of_best_match",
    "selection_reasoning": "detailed explanation emphasizing relevance_score and policy title match",
    "confidence_level": 0.85,
    "key_matching_factors": ["factor1", "factor2", "factor3"]
}

Return only the JSON object, no additional text.
"""
        )
        return _POLICY_SELECTION_TEMPLATE.substitute(
            payor=prompt_payor,
            age=prompt_age,
            gender=prompt_gender,
            cpt_display=cpt_display,
            icd_display=icd_display,
            clinical_summary=clinical_summary,
            service_type=prompt_service_type,
            candidates=prompt_candidates
        )
    
    def _parse_ai_selection_response(self, response: str, 
                                   scored_policies: List[Tuple[PolicySearchResult, float]]) -> PolicySearchResult:
        """Parse AI selection response and return selected policy"""
        
        try:
            # Try to parse JSON response
            ai_decision = safe_json_loads(response, {})
            
            if not ai_decision:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    ai_decision = safe_json_loads(json_match.group(), {})
            
            selected_policy_id = ai_decision.get('selected_policy_id')
            
            if selected_policy_id:
                # Find the selected policy
                for policy, score in scored_policies:
                    if policy.policy_id == selected_policy_id:
                        logger.info("AI policy selection successful",
                                   selected_policy_id=selected_policy_id,
                                   confidence=ai_decision.get('confidence_level', 0.0),
                                   reasoning=ai_decision.get('selection_reasoning', ''))
                        return policy
            
            # Return error instead of fallback
            logger.warning("AI selection parsing failed")
            raise PolicySelectionError("AI policy selection parsing failed")
            
        except Exception as e:
            logger.warning("Failed to parse AI selection response", error=str(e))
            raise PolicySelectionError(f"Failed to parse AI selection response: {str(e)}")
    
    def _extract_clinical_criteria(self, policy: PolicySearchResult, 
                                 extracted_data: ExtractedData) -> List[ClinicalCriterion]:
        """Extract applicable clinical criteria from selected policy using enhanced AI extraction"""
        
        # Use the enhanced ClinicalCriteriaExtractor with patient context - NO FALLBACK
        from selector.criteria_extractor import ClinicalCriteriaExtractor
        
        criteria_extractor = ClinicalCriteriaExtractor()
        clinical_criteria = criteria_extractor.extract_clinical_criteria(policy, extracted_data)
        
        logger.info("Enhanced clinical criteria extraction completed",
                   policy_id=policy.policy_id,
                   criteria_count=len(clinical_criteria))
        
        return clinical_criteria

    
    @handle_processing_errors
    def get_policy_selection_explanation(self, selected_policy: SelectedPolicy, 
                                       extracted_data: ExtractedData) -> Dict[str, Any]:
        """Generate explanation for policy selection decision"""
        
        explanation = {
            'selection_summary': {
                'policy_id': selected_policy.policy_id,
                'policy_title': selected_policy.policy_title,
                'payor': selected_policy.payor,
                'relevance_score': selected_policy.relevance_score,
                'criteria_count': len(selected_policy.applicable_criteria)
            },
            'matching_factors': [],
            'coverage_analysis': {},
            'confidence_assessment': {}
        }
        
        # Analyze matching factors
        if selected_policy.payor.lower() == extracted_data.payor_name.lower():
            explanation['matching_factors'].append('Exact payor match')
        
        # Check code coverage
        policy_cpts = [criterion.description for criterion in selected_policy.applicable_criteria 
                      if any(code in criterion.description for code in extracted_data.cpt_codes)]
        if policy_cpts:
            explanation['matching_factors'].append(f'CPT code coverage: {len(policy_cpts)} criteria')
        
        # Coverage analysis
        explanation['coverage_analysis'] = {
            'cpt_codes_covered': len([code for code in extracted_data.cpt_codes 
                                    if any(code in str(criterion.description) 
                                          for criterion in selected_policy.applicable_criteria)]),
            'total_cpt_codes': len(extracted_data.cpt_codes),
            'service_type_match': extracted_data.service_type in selected_policy.policy_title.lower(),
            'clinical_criteria_count': len(selected_policy.applicable_criteria)
        }
        
        # Confidence assessment
        explanation['confidence_assessment'] = {
            'overall_confidence': selected_policy.relevance_score,
            'payor_confidence': 1.0 if selected_policy.payor.lower() == extracted_data.payor_name.lower() else 0.5,
            'coverage_confidence': min(1.0, explanation['coverage_analysis']['cpt_codes_covered'] / 
                                     max(1, explanation['coverage_analysis']['total_cpt_codes'])),
            'policy_quality': min(1.0, len(selected_policy.applicable_criteria) / 5.0)  # Normalize to 5 criteria
        }
        
        return explanation


class PolicySelectionValidator:
    """Validates policy selection results and provides quality assurance"""
    
    @staticmethod
    def validate_policy_selection(selected_policy: SelectedPolicy, 
                                extracted_data: ExtractedData) -> Tuple[bool, List[str]]:
        """Validate policy selection and return validation results"""
        
        validation_errors = []
        
        # Check payor match
        if selected_policy.payor.lower() != extracted_data.payor_name.lower():
            validation_errors.append(f"Payor mismatch: selected {selected_policy.payor}, expected {extracted_data.payor_name}")
        
        # Check if policy has applicable criteria
        if not selected_policy.applicable_criteria:
            validation_errors.append("Selected policy has no applicable clinical criteria")
        
        # Check relevance score threshold
        if selected_policy.relevance_score < 0.3:
            validation_errors.append(f"Low relevance score: {selected_policy.relevance_score}")
        
        # Check policy recency (warn if older than 3 years)
        try:
            from datetime import date
            effective_date = date.fromisoformat(selected_policy.effective_date)
            days_old = (date.today() - effective_date).days
            if days_old > 1095:  # 3 years
                validation_errors.append(f"Policy is {days_old} days old, may be outdated")
        except (ValueError, TypeError):
            validation_errors.append("Policy effective date is invalid or missing")
        
        # Check for minimum criteria coverage
        if len(selected_policy.applicable_criteria) < 2:
            validation_errors.append("Policy has insufficient clinical criteria (minimum 2 expected)")
        
        # NEW: Validate policy title matches procedure type
        policy_title_validation = PolicySelectionValidator._validate_policy_title_match(
            selected_policy.policy_title, 
            extracted_data
        )
        if not policy_title_validation['is_valid']:
            validation_errors.append(policy_title_validation['error'])
        
        is_valid = len(validation_errors) == 0
        
        return is_valid, validation_errors
    
    @staticmethod
    def _validate_policy_title_match(policy_title: str, extracted_data: ExtractedData) -> Dict[str, Any]:
        """Validate that policy title semantically matches the requested procedure
        
        This prevents selecting ancillary service policies (e.g., monitoring) 
        when the primary procedure policy (e.g., surgery) should be selected.
        """
        
        policy_title_lower = policy_title.lower()
        
        # Get procedure descriptions from structured fields
        structured_fields = extracted_data.structured_fields or {}
        service_details = structured_fields.get('service_details', {})
        cpt_descriptions = service_details.get('all_cpt_descriptions', [])
        
        # Extract key procedure terms from CPT descriptions
        procedure_keywords = set()
        for desc in cpt_descriptions:
            desc_lower = desc.lower()
            # Extract major procedure types
            if 'surgery' in desc_lower or 'surgical' in desc_lower:
                procedure_keywords.add('surgery')
            if 'fusion' in desc_lower or 'arthrodesis' in desc_lower:
                procedure_keywords.add('fusion')
            if 'spinal' in desc_lower or 'spine' in desc_lower:
                procedure_keywords.add('spinal')
            if 'cardiac' in desc_lower or 'heart' in desc_lower:
                procedure_keywords.add('cardiac')
            if 'imaging' in desc_lower or 'scan' in desc_lower:
                procedure_keywords.add('imaging')
        
        # Check if policy title contains procedure keywords
        title_matches_procedure = any(keyword in policy_title_lower for keyword in procedure_keywords)
        
        # Identify ancillary service policies (should not be primary)
        ancillary_keywords = ['monitoring', 'anesthesia', 'consultation', 'evaluation']
        is_ancillary_policy = any(keyword in policy_title_lower for keyword in ancillary_keywords)
        
        # Validation logic
        if is_ancillary_policy and not title_matches_procedure:
            return {
                'is_valid': False,
                'error': f"Policy '{policy_title}' appears to be an ancillary service policy, not the primary procedure policy. Expected policy covering: {', '.join(procedure_keywords)}"
            }
        
        if procedure_keywords and not title_matches_procedure:
            return {
                'is_valid': False,
                'error': f"Policy title '{policy_title}' does not match requested procedure type. Expected keywords: {', '.join(procedure_keywords)}"
            }
        
        return {
            'is_valid': True,
            'error': None
        }
    
    @staticmethod
    def assess_selection_quality(selected_policy: SelectedPolicy, 
                               extracted_data: ExtractedData) -> Dict[str, Any]:
        """Assess the quality of policy selection"""
        
        quality_metrics = {
            'payor_accuracy': 1.0 if selected_policy.payor.lower() == extracted_data.payor_name.lower() else 0.0,
            'relevance_score': selected_policy.relevance_score,
            'criteria_richness': min(1.0, len(selected_policy.applicable_criteria) / 5.0),
            'policy_recency': PolicySelectionValidator._calculate_recency_score(selected_policy.effective_date),
            'overall_quality': 0.0
        }
        
        # Calculate overall quality score
        weights = {'payor_accuracy': 0.4, 'relevance_score': 0.3, 'criteria_richness': 0.2, 'policy_recency': 0.1}
        quality_metrics['overall_quality'] = calculate_confidence_score(quality_metrics, weights)
        
        return quality_metrics
    
    @staticmethod
    def _calculate_recency_score(effective_date_str: str) -> float:
        """Calculate recency score for policy"""
        try:
            from datetime import date
            effective_date = date.fromisoformat(effective_date_str)
            days_old = (date.today() - effective_date).days
            
            if days_old <= 365:
                return 1.0
            elif days_old <= 730:
                return 0.8
            elif days_old <= 1095:
                return 0.6
            else:
                return 0.4
        except (ValueError, TypeError):
            return 0.5
        """Extract applicable clinical criteria from selected policy"""
        
        # For now, create basic criteria from the policy content
        # In a full implementation, this would parse the actual policy document
        
        criteria = []
        
        # Create criteria based on CPT codes
        for cpt_code in extracted_data.cpt_codes:
            criterion = ClinicalCriterion(
                criterion_id=f"cpt_{cpt_code}",
                description=f"Medical necessity criteria for CPT {cpt_code}",
                required_evidence=[
                    "Clinical documentation supporting medical necessity",
                    "Relevant diagnostic test results",
                    "Treatment history and previous interventions"
                ],
                clinical_indicators=[
                    f"Appropriate diagnosis codes (ICD-10)",
                    f"Clinical symptoms requiring {cpt_code} procedure",
                    "Failed conservative treatment (if applicable)"
                ],
                priority="critical"
            )
            criteria.append(criterion)
        
        # Create criteria based on diagnosis codes
        for icd_code in extracted_data.icd_codes:
            criterion = ClinicalCriterion(
                criterion_id=f"icd_{icd_code}",
                description=f"Documentation requirements for diagnosis {icd_code}",
                required_evidence=[
                    "Diagnostic confirmation",
                    "Clinical assessment notes",
                    "Supporting laboratory or imaging results"
                ],
                clinical_indicators=[
                    f"Documented symptoms consistent with {icd_code}",
                    "Appropriate diagnostic workup",
                    "Clinical examination findings"
                ],
                priority="standard"
            )
            criteria.append(criterion)
        
        # Add general medical necessity criterion
        general_criterion = ClinicalCriterion(
            criterion_id="general_medical_necessity",
            description="General medical necessity requirements",
            required_evidence=[
                "Complete medical history",
                "Physical examination findings",
                "Treatment plan documentation"
            ],
            clinical_indicators=[
                "Medical necessity clearly documented",
                "Appropriate provider qualifications",
                "Service provided in appropriate setting"
            ],
            priority="critical"
        )
        criteria.append(general_criterion)
        


