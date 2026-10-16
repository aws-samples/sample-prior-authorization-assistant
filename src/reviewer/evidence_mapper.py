"""
Clinical evidence mapping system for the Medical Record Reviewer module.

This module maps medical record content to clinical criteria and identifies
supporting evidence for medical necessity requirements.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from common.aws_clients import get_bedrock_service
from common.models import ClinicalCriterion, ClinicalEvidence, EvidenceType
from common.exceptions import AssessmentError, AWSServiceError
from common.decorators import handle_processing_errors, monitor_performance, retry_with_backoff
from common.logging_config import get_logger
from common.utils import safe_json_loads, clean_text, calculate_confidence_score

logger = get_logger(__name__)


class EvidenceMatchType(Enum):
    """Types of evidence matches"""
    EXACT_MATCH = "exact_match"
    PARTIAL_MATCH = "partial_match"
    CONTEXTUAL_MATCH = "contextual_match"
    INFERRED_MATCH = "inferred_match"
    NO_MATCH = "no_match"


@dataclass
class EvidenceMatch:
    """Represents a match between medical record content and clinical criteria"""
    criterion_id: str
    evidence_text: str
    match_type: EvidenceMatchType
    confidence_score: float
    source_section: str
    page_reference: Optional[int] = None
    clinical_reasoning: str = ""
    supporting_quotes: List[str] = None


@dataclass
class MedicalRecordSection:
    """Represents a section of medical records"""
    section_type: str  # "history", "examination", "assessment", "plan", "labs", "imaging"
    content: str
    page_number: Optional[int] = None
    date: Optional[str] = None
    author: Optional[str] = None


class ClinicalEvidenceMapper:
    """Maps medical record content to clinical criteria requirements"""
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
    
    @handle_processing_errors
    @monitor_performance
    def map_evidence_to_criteria(self, medical_records: str, 
                               clinical_criteria: List[ClinicalCriterion],
                               document_id: str) -> Dict[str, List[EvidenceMatch]]:
        """Map medical record evidence to clinical criteria"""
        
        logger.info("Starting evidence mapping",
                   document_id=document_id,
                   criteria_count=len(clinical_criteria),
                   record_length=len(medical_records))
        
        # Pass complete medical records directly to AI without section parsing
        # This ensures no content is lost and AI sees all available information
        
        # Map evidence for each criterion
        evidence_mapping = {}
        
        for criterion in clinical_criteria:
            criterion_evidence = self._map_criterion_evidence_direct(
                criterion, medical_records, document_id
            )
            evidence_mapping[criterion.criterion_id] = criterion_evidence
        
        logger.info("Evidence mapping completed",
                   document_id=document_id,
                   mapped_criteria=len(evidence_mapping),
                   total_matches=sum(len(matches) for matches in evidence_mapping.values()))
        
        return evidence_mapping
    
    def _parse_medical_records(self, medical_records: str) -> List[MedicalRecordSection]:
        """Parse medical records into structured sections"""
        
        sections = []
        
        # Common medical record section patterns
        section_patterns = {
            'chief_complaint': r'(?i)chief\s+complaint[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'history': r'(?i)(?:history\s+of\s+present\s+illness|hpi)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'past_medical_history': r'(?i)(?:past\s+medical\s+history|pmh)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'medications': r'(?i)(?:medications?|current\s+medications)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'allergies': r'(?i)allergies[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'social_history': r'(?i)social\s+history[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'family_history': r'(?i)family\s+history[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'physical_exam': r'(?i)(?:physical\s+exam(?:ination)?|pe)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'vital_signs': r'(?i)vital\s+signs[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'assessment': r'(?i)(?:assessment|impression)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'plan': r'(?i)(?:plan|treatment\s+plan)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'labs': r'(?i)(?:laboratory|lab\s+results?|labs)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'imaging': r'(?i)(?:imaging|radiology|x-ray|ct|mri|ultrasound)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)',
            'procedures': r'(?i)(?:procedures?|interventions?)[:\s]*(.+?)(?=\n\s*[A-Z][^:]*:|$)'
        }
        
        # Extract sections using patterns
        for section_type, pattern in section_patterns.items():
            matches = re.finditer(pattern, medical_records, re.MULTILINE | re.DOTALL)
            for match in matches:
                content = clean_text(match.group(1))
                if content and len(content) > 10:  # Minimum meaningful content
                    sections.append(MedicalRecordSection(
                        section_type=section_type,
                        content=content
                    ))
        
        # If no structured sections found, treat entire content as one section
        if not sections:
            sections.append(MedicalRecordSection(
                section_type="general",
                content=clean_text(medical_records)
            ))
        
        return sections
    
    @retry_with_backoff(max_attempts=3)
    def _map_criterion_evidence_direct(self, criterion: ClinicalCriterion, 
                                      medical_records: str,
                                      document_id: str) -> List[EvidenceMatch]:
        """Map evidence for a specific clinical criterion using complete medical records"""
        
        # Use AI to identify relevant evidence from complete medical records
        evidence_matches = self._ai_identify_evidence_direct(criterion, medical_records)
        
        # Enhance matches with additional analysis
        enhanced_matches = self._enhance_evidence_matches(evidence_matches, criterion)
        
        # Validate and score matches
        validated_matches = self._validate_evidence_matches(enhanced_matches, criterion)
        
        return validated_matches
    
    def _ai_identify_evidence_direct(self, criterion: ClinicalCriterion, 
                                    medical_records: str) -> List[EvidenceMatch]:
        """Use AI to identify evidence supporting a clinical criterion from complete medical records"""
        
        # Create AI prompt for evidence identification with complete medical records
        prompt = self._create_evidence_mapping_prompt_direct(criterion, medical_records)
        
        try:
            response = self.bedrock_service.invoke_role(
                "evidence_mapping",
                prompt=prompt,
                max_tokens=3000
            )
            
            # Parse AI response
            evidence_data = safe_json_loads(response, {})
            
            if not evidence_data:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    evidence_data = safe_json_loads(json_match.group(), {})
            
            # Convert to EvidenceMatch objects
            evidence_matches = self._parse_ai_evidence_response(evidence_data, criterion.criterion_id)
            
            return evidence_matches
            
        except Exception as e:
            logger.error("AI evidence identification failed",
                        criterion_id=criterion.criterion_id,
                        error=str(e))
            return []
    
    def _prepare_record_content(self, record_sections: List[MedicalRecordSection]) -> str:
        """Prepare medical record content for AI analysis"""
        
        content_parts = []
        
        for section in record_sections:
            section_header = f"\n=== {section.section_type.upper().replace('_', ' ')} ==="
            content_parts.append(f"{section_header}\n{section.content}")
        
        return "\n".join(content_parts)
    
    def _create_evidence_mapping_prompt_direct(self, criterion: ClinicalCriterion, 
                                              medical_records: str) -> str:
        """Create AI prompt for evidence mapping using complete medical records"""
        
        return f"""
You are a clinical documentation specialist tasked with identifying evidence in medical records that supports or contradicts specific medical necessity criteria.

CLINICAL CRITERION TO EVALUATE:
Criterion ID: {criterion.criterion_id}
Description: {criterion.description}
Priority: {criterion.priority}

Required Evidence: {', '.join(criterion.required_evidence) if criterion.required_evidence else 'Not specified'}
Clinical Indicators: {', '.join(criterion.clinical_indicators) if criterion.clinical_indicators else 'Not specified'}
Exclusion Criteria: {', '.join(criterion.exclusion_criteria) if criterion.exclusion_criteria else 'None specified'}

COMPLETE MEDICAL RECORD CONTENT:
{medical_records}

TASK:
Carefully analyze the medical record content to identify evidence that:
1. Directly supports the clinical criterion
2. Provides partial support for the criterion
3. Contradicts or excludes the criterion
4. Is contextually relevant but not definitive

For each piece of evidence found, determine:
- The specific text that serves as evidence
- How well it matches the criterion requirements
- The confidence level of the match
- The clinical reasoning for the assessment

RESPONSE FORMAT:
Return a JSON object with this structure:
{{
    "evidence_matches": [
        {{
            "evidence_text": "Exact quote from medical record",
            "match_type": "exact_match|partial_match|contextual_match|inferred_match|no_match",
            "confidence_score": 0.85,
            "source_section": "section_name_where_found",
            "clinical_reasoning": "Explanation of why this evidence supports/contradicts the criterion",
            "supporting_quotes": ["Additional relevant quotes if any"],
            "relevance_explanation": "How this evidence relates to the specific criterion"
        }}
    ],
    "overall_assessment": "Brief summary of evidence quality for this criterion",
    "missing_evidence": ["List of required evidence not found in the records"],
    "contradictory_evidence": ["Any evidence that contradicts the criterion"]
}}

IMPORTANT GUIDELINES:
- Only include evidence that is clearly documented in the medical records
- Be precise with quotes - use exact text from the records
- Consider both positive evidence (supports criterion) and negative evidence (contradicts)
- Assess confidence based on clarity and specificity of documentation
- Look for evidence in all sections of the medical record
- Consider temporal relationships (when tests were done, symptoms started, etc.)
- Be conservative with confidence scores - only use high scores for clear, unambiguous evidence

Return only the JSON object, no additional text.
"""
    
    def _parse_ai_evidence_response(self, evidence_data: Dict[str, Any], 
                                  criterion_id: str) -> List[EvidenceMatch]:
        """Parse AI response into EvidenceMatch objects"""
        
        evidence_matches = []
        
        matches = evidence_data.get('evidence_matches', [])
        
        for match_data in matches:
            try:
                # Validate match type
                match_type_str = match_data.get('match_type', 'no_match')
                try:
                    match_type = EvidenceMatchType(match_type_str)
                except ValueError:
                    match_type = EvidenceMatchType.NO_MATCH
                
                # Validate confidence score
                confidence = match_data.get('confidence_score', 0.0)
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    confidence = 0.0
                
                evidence_match = EvidenceMatch(
                    criterion_id=criterion_id,
                    evidence_text=match_data.get('evidence_text', ''),
                    match_type=match_type,
                    confidence_score=confidence,
                    source_section=match_data.get('source_section', 'unknown'),
                    clinical_reasoning=match_data.get('clinical_reasoning', ''),
                    supporting_quotes=match_data.get('supporting_quotes', [])
                )
                
                # Only include matches with meaningful evidence
                if evidence_match.evidence_text and len(evidence_match.evidence_text) > 5:
                    evidence_matches.append(evidence_match)
                
            except Exception as e:
                logger.warning("Failed to parse evidence match",
                             criterion_id=criterion_id,
                             error=str(e))
                continue
        
        return evidence_matches
    
    def _enhance_evidence_matches(self, evidence_matches: List[EvidenceMatch], 
                                criterion: ClinicalCriterion) -> List[EvidenceMatch]:
        """Enhance evidence matches with additional analysis"""
        
        enhanced_matches = []
        
        for match in evidence_matches:
            # Enhance confidence score based on evidence quality
            enhanced_confidence = self._calculate_enhanced_confidence(match, criterion)
            
            # Add contextual information
            enhanced_reasoning = self._enhance_clinical_reasoning(match, criterion)
            
            # Create enhanced match
            enhanced_match = EvidenceMatch(
                criterion_id=match.criterion_id,
                evidence_text=match.evidence_text,
                match_type=match.match_type,
                confidence_score=enhanced_confidence,
                source_section=match.source_section,
                page_reference=match.page_reference,
                clinical_reasoning=enhanced_reasoning,
                supporting_quotes=match.supporting_quotes
            )
            
            enhanced_matches.append(enhanced_match)
        
        return enhanced_matches
    
    def _calculate_enhanced_confidence(self, match: EvidenceMatch, 
                                     criterion: ClinicalCriterion) -> float:
        """Calculate enhanced confidence score for evidence match"""
        
        confidence_factors = {
            'base_confidence': match.confidence_score
        }
        
        # Match type factor
        match_type_scores = {
            EvidenceMatchType.EXACT_MATCH: 1.0,
            EvidenceMatchType.PARTIAL_MATCH: 0.8,
            EvidenceMatchType.CONTEXTUAL_MATCH: 0.6,
            EvidenceMatchType.INFERRED_MATCH: 0.4,
            EvidenceMatchType.NO_MATCH: 0.0
        }
        confidence_factors['match_type'] = match_type_scores.get(match.match_type, 0.0)
        
        # Evidence specificity factor
        evidence_length = len(match.evidence_text)
        specificity_score = min(1.0, evidence_length / 100.0)  # Normalize to 100 chars
        confidence_factors['specificity'] = specificity_score
        
        # Clinical reasoning quality factor
        reasoning_quality = min(1.0, len(match.clinical_reasoning) / 50.0)  # Normalize to 50 chars
        confidence_factors['reasoning_quality'] = reasoning_quality
        
        # Priority weighting
        priority_weights = {'critical': 1.0, 'standard': 0.9, 'optional': 0.8}
        priority_weight = priority_weights.get(criterion.priority, 0.9)
        
        # Calculate weighted confidence
        weights = {
            'base_confidence': 0.4,
            'match_type': 0.3,
            'specificity': 0.2,
            'reasoning_quality': 0.1
        }
        
        enhanced_confidence = calculate_confidence_score(confidence_factors, weights)
        
        # Apply priority weighting
        return enhanced_confidence * priority_weight
    
    def _enhance_clinical_reasoning(self, match: EvidenceMatch, 
                                  criterion: ClinicalCriterion) -> str:
        """Enhance clinical reasoning with additional context"""
        
        reasoning_parts = [match.clinical_reasoning] if match.clinical_reasoning else []
        
        # Add criterion context
        if criterion.priority == 'critical':
            reasoning_parts.append("This is a critical criterion for medical necessity.")
        
        # Add match type context
        if match.match_type == EvidenceMatchType.EXACT_MATCH:
            reasoning_parts.append("Evidence directly matches the requirement.")
        elif match.match_type == EvidenceMatchType.PARTIAL_MATCH:
            reasoning_parts.append("Evidence partially supports the requirement.")
        elif match.match_type == EvidenceMatchType.CONTEXTUAL_MATCH:
            reasoning_parts.append("Evidence provides contextual support.")
        
        # Add source section context
        section_context = {
            'history': "Found in patient history documentation.",
            'physical_exam': "Documented during physical examination.",
            'assessment': "Noted in clinical assessment.",
            'labs': "Supported by laboratory results.",
            'imaging': "Confirmed by imaging studies."
        }
        
        if match.source_section in section_context:
            reasoning_parts.append(section_context[match.source_section])
        
        return " ".join(reasoning_parts)
    
    def _validate_evidence_matches(self, evidence_matches: List[EvidenceMatch], 
                                 criterion: ClinicalCriterion) -> List[EvidenceMatch]:
        """Validate and filter evidence matches"""
        
        validated_matches = []
        
        for match in evidence_matches:
            # Skip matches with very low confidence
            if match.confidence_score < 0.1:
                continue
            
            # Skip matches with empty evidence text
            if not match.evidence_text or len(match.evidence_text.strip()) < 5:
                continue
            
            # Skip matches without clinical reasoning for critical criteria
            if criterion.priority == 'critical' and not match.clinical_reasoning:
                continue
            
            validated_matches.append(match)
        
        # Sort by confidence score (descending)
        validated_matches.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Limit to top matches to avoid noise
        return validated_matches[:5]  # Top 5 matches per criterion


class EvidenceQualityAssessor:
    """Assesses the quality and completeness of evidence mapping"""
    
    @staticmethod
    def assess_evidence_quality(evidence_mapping: Dict[str, List[EvidenceMatch]], 
                              clinical_criteria: List[ClinicalCriterion]) -> Dict[str, Any]:
        """Assess overall quality of evidence mapping"""
        
        assessment = {
            'total_criteria': len(clinical_criteria),
            'criteria_with_evidence': 0,
            'high_confidence_matches': 0,
            'medium_confidence_matches': 0,
            'low_confidence_matches': 0,
            'criteria_without_evidence': [],
            'strongest_evidence': [],
            'weakest_evidence': [],
            'overall_quality_score': 0.0
        }
        
        all_matches = []
        
        for criterion_id, matches in evidence_mapping.items():
            if matches:
                assessment['criteria_with_evidence'] += 1
                all_matches.extend(matches)
                
                # Find best match for this criterion
                best_match = max(matches, key=lambda x: x.confidence_score)
                if best_match.confidence_score >= 0.8:
                    assessment['strongest_evidence'].append({
                        'criterion_id': criterion_id,
                        'confidence': best_match.confidence_score,
                        'evidence': best_match.evidence_text[:100]
                    })
                elif best_match.confidence_score < 0.4:
                    assessment['weakest_evidence'].append({
                        'criterion_id': criterion_id,
                        'confidence': best_match.confidence_score,
                        'evidence': best_match.evidence_text[:100]
                    })
            else:
                assessment['criteria_without_evidence'].append(criterion_id)
        
        # Categorize matches by confidence
        for match in all_matches:
            if match.confidence_score >= 0.7:
                assessment['high_confidence_matches'] += 1
            elif match.confidence_score >= 0.4:
                assessment['medium_confidence_matches'] += 1
            else:
                assessment['low_confidence_matches'] += 1
        
        # Calculate overall quality score
        if assessment['total_criteria'] > 0:
            coverage_score = assessment['criteria_with_evidence'] / assessment['total_criteria']
            
            if all_matches:
                avg_confidence = sum(m.confidence_score for m in all_matches) / len(all_matches)
            else:
                avg_confidence = 0.0
            
            assessment['overall_quality_score'] = (coverage_score * 0.6) + (avg_confidence * 0.4)
        
        return assessment
    
    @staticmethod
    def identify_evidence_gaps(evidence_mapping: Dict[str, List[EvidenceMatch]], 
                             clinical_criteria: List[ClinicalCriterion]) -> List[Dict[str, Any]]:
        """Identify gaps in evidence coverage"""
        
        gaps = []
        
        for criterion in clinical_criteria:
            criterion_matches = evidence_mapping.get(criterion.criterion_id, [])
            
            # Check for missing evidence
            if not criterion_matches:
                gaps.append({
                    'type': 'no_evidence',
                    'criterion_id': criterion.criterion_id,
                    'description': criterion.description,
                    'priority': criterion.priority,
                    'required_evidence': criterion.required_evidence,
                    'recommendation': 'No supporting evidence found in medical records'
                })
            else:
                # Check for low confidence evidence
                best_confidence = max(m.confidence_score for m in criterion_matches)
                if best_confidence < 0.5:
                    gaps.append({
                        'type': 'low_confidence',
                        'criterion_id': criterion.criterion_id,
                        'description': criterion.description,
                        'priority': criterion.priority,
                        'best_confidence': best_confidence,
                        'recommendation': 'Evidence found but with low confidence - additional documentation may be needed'
                    })
                
                # Check for missing required evidence types
                if criterion.required_evidence:
                    found_evidence_types = set()
                    for match in criterion_matches:
                        found_evidence_types.add(match.source_section)
                    
                    # This is a simplified check - in practice, would need more sophisticated mapping
                    if len(found_evidence_types) < len(criterion.required_evidence):
                        gaps.append({
                            'type': 'incomplete_evidence',
                            'criterion_id': criterion.criterion_id,
                            'description': criterion.description,
                            'priority': criterion.priority,
                            'found_types': list(found_evidence_types),
                            'required_evidence': criterion.required_evidence,
                            'recommendation': 'Some required evidence types may be missing'
                        })
        
        return gaps

