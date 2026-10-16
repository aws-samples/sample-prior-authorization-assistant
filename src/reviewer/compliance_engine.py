"""
Compliance assessment engine for the Medical Record Reviewer module.

This module implements criterion-by-criterion assessment logic with clinical
reasoning generation and support for complex medical necessity criteria evaluation.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from common.aws_clients import get_bedrock_service
from common.models import ClinicalCriterion, CriterionAssessment, ComplianceStatus
from common.exceptions import AssessmentError, AWSServiceError
from common.decorators import handle_processing_errors, monitor_performance, retry_with_backoff
from common.logging_config import get_logger
from common.utils import safe_json_loads, calculate_confidence_score
from .evidence_mapper import EvidenceMatch, EvidenceMatchType

logger = get_logger(__name__)


class AssessmentDecision(Enum):
    """Assessment decision types"""
    MET = "met"
    NOT_MET = "not_met"
    PARTIALLY_MET = "partially_met"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"


@dataclass
class AssessmentContext:
    """Context information for criterion assessment"""
    patient_age_range: str
    patient_gender: str
    primary_diagnosis: str
    comorbidities: List[str]
    payor: str
    service_type: str
    icd_codes: List[str] = field(default_factory=list)
    cpt_codes: List[str] = field(default_factory=list)
    urgency_level: str = "routine"


@dataclass
class CriterionEvaluation:
    """Detailed evaluation of a single criterion"""
    criterion_id: str
    decision: AssessmentDecision
    confidence_score: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    clinical_reasoning: str
    evidence_quality_score: float
    recommendation: Optional[str] = None


class ComplianceAssessmentEngine:
    """Engine for assessing compliance with medical necessity criteria"""
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
    
    @handle_processing_errors
    @monitor_performance
    def assess_criteria_compliance(self, clinical_criteria: List[ClinicalCriterion],
                                 evidence_mapping: Dict[str, List[EvidenceMatch]],
                                 assessment_context: AssessmentContext) -> List[CriterionAssessment]:
        """Assess compliance for all clinical criteria"""
        
        logger.info("Starting compliance assessment",
                   criteria_count=len(clinical_criteria),
                   payor=assessment_context.payor)
        
        assessments = []
        
        for criterion in clinical_criteria:
            try:
                # Get evidence for this criterion
                criterion_evidence = evidence_mapping.get(criterion.criterion_id, [])
                
                # Perform detailed evaluation
                evaluation = self._evaluate_single_criterion(
                    criterion, criterion_evidence, assessment_context
                )
                
                # Convert to CriterionAssessment
                assessment = self._create_criterion_assessment(evaluation)
                assessments.append(assessment)
                
            except Exception as e:
                logger.error("Failed to assess criterion",
                           criterion_id=criterion.criterion_id,
                           error=str(e))
                
                # Create failed assessment
                failed_assessment = CriterionAssessment(
                    criterion_id=criterion.criterion_id,
                    status=ComplianceStatus.NOT_MET,
                    supporting_evidence=[],
                    confidence_score=0.0,
                    clinical_rationale=f"Assessment failed: {str(e)}",
                    gaps_identified=["Assessment error occurred"],
                    recommendations=["Manual review required due to assessment error"]
                )
                assessments.append(failed_assessment)
        
        logger.info("Compliance assessment completed",
                   total_assessments=len(assessments),
                   met_count=len([a for a in assessments if a.status == ComplianceStatus.MET]),
                   not_met_count=len([a for a in assessments if a.status == ComplianceStatus.NOT_MET]))
        
        return assessments
    
    @retry_with_backoff(max_attempts=10, base_delay=3.0, max_delay=180.0)
    def _evaluate_single_criterion(self, criterion: ClinicalCriterion,
                                 evidence_matches: List[EvidenceMatch],
                                 context: AssessmentContext) -> CriterionEvaluation:
        """Evaluate a single clinical criterion using AI"""
        
        # Prepare evaluation prompt
        prompt = self._create_assessment_prompt(criterion, evidence_matches, context)
        
        try:
            response = self.bedrock_service.invoke_role(
                "compliance_assessment",
                prompt=prompt,
                max_tokens=2000
            )
            
            # Parse AI response
            evaluation_data = safe_json_loads(response, {})
            
            if not evaluation_data:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    evaluation_data = safe_json_loads(json_match.group(), {})
            
            # Convert to CriterionEvaluation
            evaluation = self._parse_evaluation_response(
                evaluation_data, criterion.criterion_id, evidence_matches
            )
            
            return evaluation
            
        except Exception as e:
            logger.warning("AI criterion evaluation failed",
                        criterion_id=criterion.criterion_id,
                        error=str(e))
            
            # Return error instead of fallback mock data
            raise AssessmentError(f"AI assessment failed for criterion {criterion.criterion_id}: {str(e)}")
    
    def _create_assessment_prompt(self, criterion: ClinicalCriterion,
                                evidence_matches: List[EvidenceMatch],
                                context: AssessmentContext) -> str:
        """Create AI prompt for criterion assessment"""
        
        # Prepare evidence summary
        evidence_summary = self._prepare_evidence_summary(evidence_matches)
        
        return f"""
You are a clinical reviewer conducting a medical necessity assessment for prior authorization. Your task is to evaluate whether a specific clinical criterion is met based on the available medical record evidence.

PATIENT CONTEXT:
- Age Range: {context.patient_age_range}
- Gender: {context.patient_gender}
- Primary Diagnosis: {context.primary_diagnosis}
- Diagnosis Codes (ICD): {', '.join(context.icd_codes) if context.icd_codes else 'None documented'}
- Procedure Codes (CPT): {', '.join(context.cpt_codes) if context.cpt_codes else 'None documented'}
- Comorbidities: {', '.join(context.comorbidities) if context.comorbidities else 'None documented'}
- Payor: {context.payor}
- Service Type: {context.service_type}
- Urgency: {context.urgency_level}

CLINICAL CRITERION TO EVALUATE:
Criterion ID: {criterion.criterion_id}
Description: {criterion.description}
Priority Level: {criterion.priority}

Required Evidence: {', '.join(criterion.required_evidence) if criterion.required_evidence else 'Not specified'}
Clinical Indicators: {', '.join(criterion.clinical_indicators) if criterion.clinical_indicators else 'Not specified'}
Exclusion Criteria: {', '.join(criterion.exclusion_criteria) if criterion.exclusion_criteria else 'None specified'}

AVAILABLE EVIDENCE FROM MEDICAL RECORDS:
{evidence_summary}

ASSESSMENT TASK:
Based on the medical record evidence and the clinical criterion requirements, determine:

1. COMPLIANCE STATUS: Does the evidence support that this criterion is met?
   - "met": Evidence clearly supports the criterion
   - "not_met": Evidence clearly contradicts or fails to support the criterion
   - "partially_met": Evidence provides partial support but is incomplete
   - "insufficient_evidence": Not enough evidence to make a determination
   - "contradictory_evidence": Evidence contains conflicting information

2. CONFIDENCE LEVEL: How confident are you in this assessment? (0.0 to 1.0)

3. CLINICAL REASONING: Provide detailed medical reasoning for your decision

4. EVIDENCE ANALYSIS: Identify which pieces of evidence support or contradict the criterion

RESPONSE FORMAT:
Return a JSON object with this exact structure:
{{
    "compliance_status": "met|not_met|partially_met|insufficient_evidence|contradictory_evidence",
    "confidence_score": 0.85,
    "clinical_reasoning": "Detailed explanation of the clinical assessment and decision-making process",
    "supporting_evidence": [
        "Specific evidence that supports the criterion",
        "Another piece of supporting evidence"
    ],
    "contradicting_evidence": [
        "Evidence that contradicts or fails to support the criterion"
    ],
    "evidence_quality_assessment": {{
        "completeness": 0.8,
        "clarity": 0.9,
        "relevance": 0.85,
        "timeliness": 0.7
    }},
    "gaps_identified": [
        "Specific documentation gaps that affect the assessment",
        "Missing evidence that would strengthen the case"
    ],
    "recommendations": [
        "Specific actions to address identified gaps",
        "Additional documentation that should be obtained"
    ],
    "risk_factors": [
        "Clinical risk factors that should be considered",
        "Potential complications or contraindications"
    ]
}}

ASSESSMENT GUIDELINES:
- Base your assessment strictly on documented evidence in the medical records
- Consider the patient's specific clinical context (age, gender, diagnosis, comorbidities)
- Apply appropriate clinical judgment while following the criterion requirements
- Be conservative with confidence scores - only use high scores for clear, unambiguous evidence
- Consider the priority level of the criterion (critical criteria require stronger evidence)
- Look for both positive evidence (supports criterion) and negative evidence (contradicts criterion)
- Consider temporal relationships and clinical progression
- Account for standard medical practice and clinical guidelines
- If evidence is ambiguous, err on the side of requesting additional documentation

CRITICAL CONSIDERATIONS:
- For "critical" priority criteria, require strong, clear evidence
- For exclusion criteria, any contradicting evidence should result in "not_met"
- Consider patient safety implications in your assessment
- Ensure recommendations are specific and actionable

Return only the JSON object, no additional text or explanation.
"""
    
    def _prepare_evidence_summary(self, evidence_matches: List[EvidenceMatch]) -> str:
        """Prepare summary of evidence for AI assessment"""
        
        if not evidence_matches:
            return "No relevant evidence found in the medical records."
        
        evidence_parts = []
        
        for i, match in enumerate(evidence_matches, 1):
            evidence_part = f"""
Evidence {i}:
- Source: {match.source_section}
- Match Type: {match.match_type.value}
- Confidence: {match.confidence_score:.2f}
- Content: "{match.evidence_text}"
- Clinical Reasoning: {match.clinical_reasoning}
"""
            evidence_parts.append(evidence_part)
        
        return "\n".join(evidence_parts)
    
    def _parse_evaluation_response(self, evaluation_data: Dict[str, Any],
                                 criterion_id: str,
                                 evidence_matches: List[EvidenceMatch]) -> CriterionEvaluation:
        """Parse AI evaluation response into CriterionEvaluation object"""
        
        # Parse compliance status
        status_str = evaluation_data.get('compliance_status', 'insufficient_evidence')
        try:
            decision = AssessmentDecision(status_str)
        except ValueError:
            decision = AssessmentDecision.INSUFFICIENT_EVIDENCE
        
        # Parse confidence score
        confidence = evaluation_data.get('confidence_score', 0.0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            confidence = 0.0
        
        # Parse evidence quality
        evidence_quality_data = evaluation_data.get('evidence_quality_assessment', {})
        evidence_quality_score = self._calculate_evidence_quality_score(evidence_quality_data)
        
        # Extract supporting and contradicting evidence
        supporting_evidence = evaluation_data.get('supporting_evidence', [])
        contradicting_evidence = evaluation_data.get('contradicting_evidence', [])
        
        # Combine all evidence for supporting_evidence field with deduplication
        all_supporting = []
        seen_evidence = set()  # Track unique evidence to avoid duplicates
        
        # Add AI-provided supporting evidence
        if supporting_evidence:
            for evidence in supporting_evidence:
                # Normalize evidence for comparison (lowercase, strip whitespace)
                normalized = evidence.lower().strip()[:100]  # Use first 100 chars for comparison
                if normalized not in seen_evidence:
                    all_supporting.append(evidence)
                    seen_evidence.add(normalized)
        
        # Add high-confidence evidence matches (only if not already included)
        for match in evidence_matches:
            if match.confidence_score >= 0.7 and match.match_type in [
                EvidenceMatchType.EXACT_MATCH, EvidenceMatchType.PARTIAL_MATCH
            ]:
                truncated_evidence = match.evidence_text[:200]  # Truncate for brevity
                normalized = truncated_evidence.lower().strip()[:100]
                if normalized not in seen_evidence:
                    all_supporting.append(truncated_evidence)
                    seen_evidence.add(normalized)
        
        return CriterionEvaluation(
            criterion_id=criterion_id,
            decision=decision,
            confidence_score=confidence,
            supporting_evidence=all_supporting[:5],  # Limit to top 5
            contradicting_evidence=contradicting_evidence[:3],  # Limit to top 3
            clinical_reasoning=evaluation_data.get('clinical_reasoning', ''),
            evidence_quality_score=evidence_quality_score,
            recommendation=self._generate_recommendation(evaluation_data, decision)
        )
    
    def _calculate_evidence_quality_score(self, quality_data: Dict[str, Any]) -> float:
        """Calculate overall evidence quality score"""
        
        quality_factors = {
            'completeness': quality_data.get('completeness', 0.5),
            'clarity': quality_data.get('clarity', 0.5),
            'relevance': quality_data.get('relevance', 0.5),
            'timeliness': quality_data.get('timeliness', 0.5)
        }
        
        # Validate scores
        for key, value in quality_factors.items():
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                quality_factors[key] = 0.5
        
        return calculate_confidence_score(quality_factors)
    
    def _generate_recommendation(self, evaluation_data: Dict[str, Any],
                               decision: AssessmentDecision) -> Optional[str]:
        """Generate recommendation based on evaluation"""
        
        recommendations = evaluation_data.get('recommendations', [])
        
        if recommendations:
            return "; ".join(recommendations[:3])  # Top 3 recommendations
        
        # Generate default recommendations based on decision
        if decision == AssessmentDecision.NOT_MET:
            return "Obtain additional documentation to support medical necessity"
        elif decision == AssessmentDecision.PARTIALLY_MET:
            return "Provide supplemental documentation to complete the evidence"
        elif decision == AssessmentDecision.INSUFFICIENT_EVIDENCE:
            return "Submit comprehensive medical records supporting the clinical indication"
        
        return None
    

    
    def _create_criterion_assessment(self, evaluation: CriterionEvaluation) -> CriterionAssessment:
        """Convert CriterionEvaluation to CriterionAssessment"""
        
        # Map AssessmentDecision to ComplianceStatus
        status_mapping = {
            AssessmentDecision.MET: ComplianceStatus.MET,
            AssessmentDecision.NOT_MET: ComplianceStatus.NOT_MET,
            AssessmentDecision.PARTIALLY_MET: ComplianceStatus.PARTIALLY_MET,
            AssessmentDecision.INSUFFICIENT_EVIDENCE: ComplianceStatus.NOT_MET,
            AssessmentDecision.CONTRADICTORY_EVIDENCE: ComplianceStatus.NOT_MET
        }
        
        status = status_mapping.get(evaluation.decision, ComplianceStatus.NOT_MET)
        
        # Generate gaps and recommendations
        gaps_identified = []
        recommendations = []
        
        if evaluation.decision in [AssessmentDecision.NOT_MET, AssessmentDecision.INSUFFICIENT_EVIDENCE]:
            gaps_identified.append("Insufficient evidence to support medical necessity")
            recommendations.append("Provide additional clinical documentation")
        
        if evaluation.decision == AssessmentDecision.PARTIALLY_MET:
            gaps_identified.append("Incomplete evidence for full criterion compliance")
            recommendations.append("Submit supplemental documentation")
        
        if evaluation.decision == AssessmentDecision.CONTRADICTORY_EVIDENCE:
            gaps_identified.append("Conflicting evidence in medical records")
            recommendations.append("Clarify contradictory clinical findings")
        
        # Add specific recommendation if available
        if evaluation.recommendation:
            recommendations.append(evaluation.recommendation)
        
        return CriterionAssessment(
            criterion_id=evaluation.criterion_id,
            status=status,
            supporting_evidence=evaluation.supporting_evidence,
            confidence_score=evaluation.confidence_score,
            clinical_rationale=evaluation.clinical_reasoning,
            gaps_identified=gaps_identified,
            recommendations=recommendations[:3]  # Limit to 3 recommendations
        )


class ComplianceAggregator:
    """Aggregates individual criterion assessments into overall compliance"""
    
    @staticmethod
    def calculate_overall_compliance(assessments: List[CriterionAssessment],
                                   clinical_criteria: List[ClinicalCriterion]) -> Dict[str, Any]:
        """Calculate overall compliance status and metrics"""
        
        if not assessments:
            return {
                'overall_status': 'non_compliant',
                'compliance_percentage': 0.0,
                'met_criteria': 0,
                'total_criteria': len(clinical_criteria),
                'critical_criteria_met': 0,
                'critical_criteria_total': 0,
                'average_confidence': 0.0
            }
        
        # Categorize criteria by priority
        criteria_by_priority = {}
        for criterion in clinical_criteria:
            priority = criterion.priority
            if priority not in criteria_by_priority:
                criteria_by_priority[priority] = []
            criteria_by_priority[priority].append(criterion.criterion_id)
        
        # Count met criteria
        met_criteria = len([a for a in assessments if a.status == ComplianceStatus.MET])
        total_criteria = len(assessments)
        
        # Count critical criteria
        critical_criteria_ids = criteria_by_priority.get('critical', [])
        critical_met = len([
            a for a in assessments 
            if a.criterion_id in critical_criteria_ids and a.status == ComplianceStatus.MET
        ])
        critical_total = len(critical_criteria_ids)
        
        # Calculate compliance percentage
        compliance_percentage = (met_criteria / total_criteria) if total_criteria > 0 else 0.0
        
        # Calculate average confidence
        avg_confidence = sum(a.confidence_score for a in assessments) / len(assessments)
        
        # Determine overall status
        overall_status = ComplianceAggregator._determine_overall_status(
            compliance_percentage, critical_met, critical_total, avg_confidence
        )
        
        return {
            'overall_status': overall_status,
            'compliance_percentage': compliance_percentage,
            'met_criteria': met_criteria,
            'total_criteria': total_criteria,
            'critical_criteria_met': critical_met,
            'critical_criteria_total': critical_total,
            'average_confidence': avg_confidence,
            'status_breakdown': ComplianceAggregator._get_status_breakdown(assessments),
            'priority_breakdown': ComplianceAggregator._get_priority_breakdown(
                assessments, criteria_by_priority
            )
        }
    
    @staticmethod
    def _determine_overall_status(compliance_percentage: float, critical_met: int,
                                critical_total: int, avg_confidence: float) -> str:
        """Determine overall compliance status"""
        
        # Critical criteria must be met
        if critical_total > 0 and critical_met < critical_total:
            return 'non_compliant'
        
        # High compliance threshold
        if compliance_percentage >= 0.8 and avg_confidence >= 0.7:
            return 'compliant'
        
        # Partial compliance
        if compliance_percentage >= 0.5:
            return 'partially_compliant'
        
        # Low compliance
        return 'non_compliant'
    
    @staticmethod
    def _get_status_breakdown(assessments: List[CriterionAssessment]) -> Dict[str, int]:
        """Get breakdown of assessments by status"""
        
        breakdown = {
            'met': 0,
            'not_met': 0,
            'partially_met': 0
        }
        
        for assessment in assessments:
            if assessment.status == ComplianceStatus.MET:
                breakdown['met'] += 1
            elif assessment.status == ComplianceStatus.NOT_MET:
                breakdown['not_met'] += 1
            elif assessment.status == ComplianceStatus.PARTIALLY_MET:
                breakdown['partially_met'] += 1
        
        return breakdown
    
    @staticmethod
    def _get_priority_breakdown(assessments: List[CriterionAssessment],
                              criteria_by_priority: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """Get breakdown of assessments by priority level"""
        
        priority_breakdown = {}
        
        for priority, criterion_ids in criteria_by_priority.items():
            priority_assessments = [
                a for a in assessments if a.criterion_id in criterion_ids
            ]
            
            priority_breakdown[priority] = {
                'total': len(priority_assessments),
                'met': len([a for a in priority_assessments if a.status == ComplianceStatus.MET]),
                'not_met': len([a for a in priority_assessments if a.status == ComplianceStatus.NOT_MET]),
                'partially_met': len([a for a in priority_assessments if a.status == ComplianceStatus.PARTIALLY_MET])
            }
        
        return priority_breakdown

