"""
Gap analysis and recommendation system for the Medical Record Reviewer module.

This module identifies missing clinical documentation, generates specific
recommendations for addressing documentation gaps, and prioritizes actions
for improving prior authorization approval likelihood.
"""

import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from common.aws_clients import get_bedrock_service
from common.models import ClinicalCriterion, CriterionAssessment, GapAnalysis, ComplianceStatus
from common.exceptions import AssessmentError, AWSServiceError
from common.decorators import handle_processing_errors, monitor_performance, retry_with_backoff
from common.logging_config import get_logger
from common.utils import safe_json_loads, calculate_confidence_score
from .evidence_mapper import EvidenceMatch, EvidenceMatchType
from .compliance_engine import AssessmentContext

logger = get_logger(__name__)


class GapSeverity(Enum):
    """Severity levels for documentation gaps"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapCategory(Enum):
    """Categories of documentation gaps"""
    MISSING_EVIDENCE = "missing_evidence"
    INSUFFICIENT_DETAIL = "insufficient_detail"
    CONTRADICTORY_INFORMATION = "contradictory_information"
    OUTDATED_INFORMATION = "outdated_information"
    INCOMPLETE_ASSESSMENT = "incomplete_assessment"
    MISSING_CLINICAL_INDICATORS = "missing_clinical_indicators"
    INADEQUATE_JUSTIFICATION = "inadequate_justification"


@dataclass
class DocumentationGap:
    """Represents a specific documentation gap"""
    gap_id: str
    category: GapCategory
    severity: GapSeverity
    criterion_id: str
    description: str
    missing_evidence: List[str]
    impact_on_approval: str
    specific_recommendations: List[str]
    priority_score: float
    estimated_effort: str  # "low", "medium", "high"
    clinical_rationale: str


@dataclass
class RecommendationAction:
    """Specific action recommendation"""
    action_id: str
    action_type: str  # "obtain_documentation", "clarify_information", "additional_testing"
    description: str
    priority: str  # "immediate", "high", "medium", "low"
    estimated_timeline: str
    responsible_party: str  # "provider", "patient", "specialist"
    expected_impact: float  # 0.0 to 1.0
    implementation_notes: str


class GapAnalysisEngine:
    """Engine for analyzing documentation gaps and generating recommendations"""
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
    
    @handle_processing_errors
    @monitor_performance
    def analyze_documentation_gaps(self, assessments: List[CriterionAssessment],
                                 clinical_criteria: List[ClinicalCriterion],
                                 evidence_mapping: Dict[str, List[EvidenceMatch]],
                                 context: AssessmentContext) -> GapAnalysis:
        """Perform comprehensive gap analysis and generate recommendations"""
        
        logger.info("Starting gap analysis",
                   assessments_count=len(assessments),
                   criteria_count=len(clinical_criteria),
                   payor=context.payor)
        
        # Identify documentation gaps
        documentation_gaps = self._identify_documentation_gaps(
            assessments, clinical_criteria, evidence_mapping, context
        )
        
        # Generate specific recommendations
        recommendations = self._generate_recommendations(
            documentation_gaps, assessments, context
        )
        
        # Prioritize actions
        priority_actions = self._prioritize_actions(recommendations, documentation_gaps)
        
        # Create gap analysis summary (removed approval_likelihood calculation)
        gap_analysis = GapAnalysis(
            missing_evidence=self._extract_missing_evidence(documentation_gaps),
            critical_gaps=self._extract_critical_gaps(documentation_gaps),
            recommendations=[rec.description for rec in recommendations[:10]],  # Top 10
            priority_actions=[action.description for action in priority_actions[:5]],  # Top 5
            estimated_approval_likelihood=0.0  # Deprecated - no longer calculated
        )
        
        logger.info("Gap analysis completed",
                   gaps_identified=len(documentation_gaps),
                   recommendations_generated=len(recommendations))
        
        return gap_analysis
    
    def _identify_documentation_gaps(self, assessments: List[CriterionAssessment],
                                   clinical_criteria: List[ClinicalCriterion],
                                   evidence_mapping: Dict[str, List[EvidenceMatch]],
                                   context: AssessmentContext) -> List[DocumentationGap]:
        """Identify specific documentation gaps"""
        
        gaps = []
        
        for assessment in assessments:
            # Find corresponding criterion
            criterion = next(
                (c for c in clinical_criteria if c.criterion_id == assessment.criterion_id),
                None
            )
            
            if not criterion:
                continue
            
            # Analyze gaps for this criterion
            criterion_gaps = self._analyze_criterion_gaps(
                assessment, criterion, evidence_mapping.get(assessment.criterion_id, []), context
            )
            
            gaps.extend(criterion_gaps)
        
        return gaps
    
    def _analyze_criterion_gaps(self, assessment: CriterionAssessment,
                              criterion: ClinicalCriterion,
                              evidence_matches: List[EvidenceMatch],
                              context: AssessmentContext) -> List[DocumentationGap]:
        """Analyze gaps for a specific criterion"""
        
        gaps = []
        
        # Gap for unmet criteria
        if assessment.status in [ComplianceStatus.NOT_MET, ComplianceStatus.PARTIALLY_MET]:
            gap = self._create_unmet_criterion_gap(assessment, criterion, evidence_matches, context)
            if gap:
                gaps.append(gap)
        
        # Gap for insufficient evidence
        if not evidence_matches or all(m.confidence_score < 0.5 for m in evidence_matches):
            gap = self._create_insufficient_evidence_gap(assessment, criterion, context)
            if gap:
                gaps.append(gap)
        
        # Gap for missing required evidence types
        if criterion.required_evidence:
            missing_evidence_gap = self._create_missing_evidence_type_gap(
                assessment, criterion, evidence_matches, context
            )
            if missing_evidence_gap:
                gaps.append(missing_evidence_gap)
        
        # Gap for contradictory evidence
        contradictory_gap = self._check_contradictory_evidence(
            assessment, criterion, evidence_matches, context
        )
        if contradictory_gap:
            gaps.append(contradictory_gap)
        
        return gaps
    
    def _create_unmet_criterion_gap(self, assessment: CriterionAssessment,
                                  criterion: ClinicalCriterion,
                                  evidence_matches: List[EvidenceMatch],
                                  context: AssessmentContext) -> Optional[DocumentationGap]:
        """Create gap for unmet criterion"""
        
        # Determine severity based on criterion priority
        severity_mapping = {
            'critical': GapSeverity.CRITICAL,
            'standard': GapSeverity.HIGH,
            'optional': GapSeverity.MEDIUM
        }
        severity = severity_mapping.get(criterion.priority, GapSeverity.MEDIUM)
        
        # Generate specific recommendations
        recommendations = []
        if criterion.required_evidence:
            recommendations.extend([
                f"Provide documentation for: {evidence}"
                for evidence in criterion.required_evidence[:3]
            ])
        
        if criterion.clinical_indicators:
            recommendations.append(
                f"Document clinical findings: {', '.join(criterion.clinical_indicators[:2])}"
            )
        
        if not recommendations:
            recommendations.append("Provide comprehensive clinical documentation supporting this criterion")
        
        # Calculate priority score
        priority_score = self._calculate_gap_priority_score(
            severity, assessment.confidence_score, criterion.priority
        )
        
        return DocumentationGap(
            gap_id=f"unmet_{assessment.criterion_id}",
            category=GapCategory.MISSING_EVIDENCE,
            severity=severity,
            criterion_id=assessment.criterion_id,
            description=f"Criterion not met: {criterion.description[:100]}...",
            missing_evidence=criterion.required_evidence[:5],
            impact_on_approval="High - may result in denial if not addressed",
            specific_recommendations=recommendations,
            priority_score=priority_score,
            estimated_effort="medium" if severity == GapSeverity.CRITICAL else "low",
            clinical_rationale=assessment.clinical_rationale[:200]
        )
    
    def _create_insufficient_evidence_gap(self, assessment: CriterionAssessment,
                                        criterion: ClinicalCriterion,
                                        context: AssessmentContext) -> Optional[DocumentationGap]:
        """Create gap for insufficient evidence"""
        
        return DocumentationGap(
            gap_id=f"insufficient_{assessment.criterion_id}",
            category=GapCategory.INSUFFICIENT_DETAIL,
            severity=GapSeverity.HIGH,
            criterion_id=assessment.criterion_id,
            description=f"Insufficient evidence for: {criterion.description[:100]}...",
            missing_evidence=["Detailed clinical documentation", "Supporting test results"],
            impact_on_approval="Medium - additional documentation needed",
            specific_recommendations=[
                "Provide more detailed clinical documentation",
                "Include relevant diagnostic test results",
                "Add physician assessment and clinical reasoning"
            ],
            priority_score=0.7,
            estimated_effort="medium",
            clinical_rationale="Limited evidence available to support medical necessity"
        )
    
    def _create_missing_evidence_type_gap(self, assessment: CriterionAssessment,
                                        criterion: ClinicalCriterion,
                                        evidence_matches: List[EvidenceMatch],
                                        context: AssessmentContext) -> Optional[DocumentationGap]:
        """Create gap for missing evidence types"""
        
        # Identify evidence types found
        found_sections = set(match.source_section for match in evidence_matches)
        
        # Expected evidence types based on criterion
        expected_sections = self._get_expected_evidence_sections(criterion)
        
        # Find missing sections
        missing_sections = expected_sections - found_sections
        
        if not missing_sections:
            return None
        
        missing_evidence = [
            f"Documentation from {section.replace('_', ' ')}"
            for section in missing_sections
        ]
        
        return DocumentationGap(
            gap_id=f"missing_types_{assessment.criterion_id}",
            category=GapCategory.MISSING_EVIDENCE,
            severity=GapSeverity.MEDIUM,
            criterion_id=assessment.criterion_id,
            description=f"Missing evidence types for: {criterion.description[:100]}...",
            missing_evidence=missing_evidence,
            impact_on_approval="Medium - specific documentation types needed",
            specific_recommendations=[
                f"Provide {section.replace('_', ' ')} documentation"
                for section in list(missing_sections)[:3]
            ],
            priority_score=0.6,
            estimated_effort="low",
            clinical_rationale="Some evidence types are missing from the medical record"
        )
    
    def _check_contradictory_evidence(self, assessment: CriterionAssessment,
                                    criterion: ClinicalCriterion,
                                    evidence_matches: List[EvidenceMatch],
                                    context: AssessmentContext) -> Optional[DocumentationGap]:
        """Check for contradictory evidence"""
        
        # Look for conflicting evidence in matches
        high_confidence_matches = [m for m in evidence_matches if m.confidence_score >= 0.7]
        low_confidence_matches = [m for m in evidence_matches if m.confidence_score < 0.3]
        
        # If we have both high and low confidence matches, there might be contradiction
        if high_confidence_matches and low_confidence_matches:
            return DocumentationGap(
                gap_id=f"contradictory_{assessment.criterion_id}",
                category=GapCategory.CONTRADICTORY_INFORMATION,
                severity=GapSeverity.HIGH,
                criterion_id=assessment.criterion_id,
                description=f"Contradictory evidence for: {criterion.description[:100]}...",
                missing_evidence=["Clarification of conflicting information"],
                impact_on_approval="High - conflicting information may cause denial",
                specific_recommendations=[
                    "Clarify contradictory clinical findings",
                    "Provide additional context for conflicting information",
                    "Submit updated clinical assessment"
                ],
                priority_score=0.8,
                estimated_effort="medium",
                clinical_rationale="Medical record contains conflicting information"
            )
        
        return None
    
    def _get_expected_evidence_sections(self, criterion: ClinicalCriterion) -> Set[str]:
        """Get expected evidence sections based on criterion type"""
        
        expected_sections = set()
        
        # Basic sections always expected
        expected_sections.update(['history', 'physical_exam', 'assessment'])
        
        # Add specific sections based on criterion content
        criterion_text = f"{criterion.description} {' '.join(criterion.required_evidence)}".lower()
        
        if any(term in criterion_text for term in ['lab', 'blood', 'test']):
            expected_sections.add('labs')
        
        if any(term in criterion_text for term in ['imaging', 'x-ray', 'mri', 'ct', 'scan']):
            expected_sections.add('imaging')
        
        if any(term in criterion_text for term in ['medication', 'drug', 'treatment']):
            expected_sections.add('medications')
        
        if any(term in criterion_text for term in ['procedure', 'surgery', 'intervention']):
            expected_sections.add('procedures')
        
        return expected_sections
    
    def _calculate_gap_priority_score(self, severity: GapSeverity,
                                    confidence_score: float,
                                    criterion_priority: str) -> float:
        """Calculate priority score for a gap"""
        
        # Base score from severity
        severity_scores = {
            GapSeverity.CRITICAL: 1.0,
            GapSeverity.HIGH: 0.8,
            GapSeverity.MEDIUM: 0.6,
            GapSeverity.LOW: 0.4
        }
        base_score = severity_scores.get(severity, 0.5)
        
        # Adjust for confidence (lower confidence = higher priority to fix)
        confidence_adjustment = 1.0 - confidence_score
        
        # Adjust for criterion priority
        priority_multipliers = {'critical': 1.2, 'standard': 1.0, 'optional': 0.8}
        priority_multiplier = priority_multipliers.get(criterion_priority, 1.0)
        
        # Calculate final score
        priority_score = (base_score * 0.6) + (confidence_adjustment * 0.4)
        priority_score *= priority_multiplier
        
        return min(1.0, priority_score)
    
    @retry_with_backoff(max_attempts=2)
    def _generate_recommendations(self, gaps: List[DocumentationGap],
                                assessments: List[CriterionAssessment],
                                context: AssessmentContext) -> List[RecommendationAction]:
        """Generate specific recommendation actions using AI"""
        
        if not gaps:
            return []
        
        # Use AI to generate comprehensive recommendations
        ai_recommendations = self._ai_generate_recommendations(gaps, assessments, context)
        
        # Add standard recommendations
        standard_recommendations = self._generate_standard_recommendations(gaps, context)
        
        # Combine and deduplicate
        all_recommendations = ai_recommendations + standard_recommendations
        
        # Remove duplicates and prioritize
        unique_recommendations = self._deduplicate_recommendations(all_recommendations)
        
        return unique_recommendations[:15]  # Limit to top 15 recommendations
    
    def _ai_generate_recommendations(self, gaps: List[DocumentationGap],
                                   assessments: List[CriterionAssessment],
                                   context: AssessmentContext) -> List[RecommendationAction]:
        """Use AI to generate specific recommendations"""
        
        # Prepare gap summary for AI
        gap_summary = self._prepare_gap_summary(gaps)
        
        prompt = f"""
You are a healthcare documentation specialist helping providers improve their prior authorization submissions. Based on the identified documentation gaps, generate specific, actionable recommendations.

PATIENT CONTEXT:
- Payor: {context.payor}
- Service Type: {context.service_type}
- Primary Diagnosis: {context.primary_diagnosis}
- Patient: {context.patient_age_range} year old {context.patient_gender}

IDENTIFIED DOCUMENTATION GAPS:
{gap_summary}

TASK:
Generate specific, actionable recommendations to address these gaps and improve approval likelihood. For each recommendation:

1. Be specific about what documentation is needed
2. Indicate who should provide it (provider, specialist, patient)
3. Estimate the timeline for obtaining the documentation
4. Assess the expected impact on approval likelihood

RESPONSE FORMAT:
Return a JSON object with this structure:
{{
    "recommendations": [
        {{
            "action_type": "obtain_documentation|clarify_information|additional_testing|provider_assessment",
            "description": "Specific action to take",
            "priority": "immediate|high|medium|low",
            "estimated_timeline": "same_day|1-3_days|1_week|2-4_weeks",
            "responsible_party": "provider|specialist|patient|lab|imaging",
            "expected_impact": 0.85,
            "implementation_notes": "Specific guidance on how to implement this recommendation",
            "documentation_type": "clinical_notes|test_results|imaging|specialist_consult|patient_history"
        }}
    ]
}}

GUIDELINES:
- Focus on the most impactful recommendations first
- Be specific about what documentation is needed
- Consider the payor's typical requirements
- Prioritize recommendations that address critical gaps
- Provide practical implementation guidance
- Consider the clinical context and patient condition

Return only the JSON object, no additional text.
"""
        
        try:
            response = self.bedrock_service.invoke_role(
                "gap_analysis",
                prompt=prompt,
                max_tokens=2000
            )
            
            recommendations_data = safe_json_loads(response, {})
            
            if not recommendations_data:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    recommendations_data = safe_json_loads(json_match.group(), {})
            
            return self._parse_ai_recommendations(recommendations_data)
            
        except Exception as e:
            logger.warning("AI recommendation generation failed", error=str(e))
            return []
    
    def _prepare_gap_summary(self, gaps: List[DocumentationGap]) -> str:
        """Prepare summary of gaps for AI analysis"""
        
        gap_summaries = []
        
        for gap in gaps[:10]:  # Limit to top 10 gaps
            summary = f"""
Gap: {gap.description}
Severity: {gap.severity.value}
Category: {gap.category.value}
Missing Evidence: {', '.join(gap.missing_evidence[:3])}
Impact: {gap.impact_on_approval}
"""
            gap_summaries.append(summary)
        
        return "\n".join(gap_summaries)
    
    def _parse_ai_recommendations(self, recommendations_data: Dict[str, Any]) -> List[RecommendationAction]:
        """Parse AI-generated recommendations"""
        
        recommendations = []
        
        for i, rec_data in enumerate(recommendations_data.get('recommendations', [])):
            try:
                recommendation = RecommendationAction(
                    action_id=f"ai_rec_{i+1}",
                    action_type=rec_data.get('action_type', 'obtain_documentation'),
                    description=rec_data.get('description', ''),
                    priority=rec_data.get('priority', 'medium'),
                    estimated_timeline=rec_data.get('estimated_timeline', '1_week'),
                    responsible_party=rec_data.get('responsible_party', 'provider'),
                    expected_impact=rec_data.get('expected_impact', 0.5),
                    implementation_notes=rec_data.get('implementation_notes', '')
                )
                
                if recommendation.description:  # Only add if description exists
                    recommendations.append(recommendation)
                    
            except Exception as e:
                logger.warning("Failed to parse AI recommendation", index=i, error=str(e))
                continue
        
        return recommendations
    
    def _generate_standard_recommendations(self, gaps: List[DocumentationGap],
                                         context: AssessmentContext) -> List[RecommendationAction]:
        """Generate standard recommendations based on gap patterns"""
        
        recommendations = []
        
        # Group gaps by category
        gaps_by_category = {}
        for gap in gaps:
            category = gap.category
            if category not in gaps_by_category:
                gaps_by_category[category] = []
            gaps_by_category[category].append(gap)
        
        # Generate category-specific recommendations
        for category, category_gaps in gaps_by_category.items():
            if category == GapCategory.MISSING_EVIDENCE:
                recommendations.extend(self._generate_missing_evidence_recommendations(category_gaps))
            elif category == GapCategory.INSUFFICIENT_DETAIL:
                recommendations.extend(self._generate_insufficient_detail_recommendations(category_gaps))
            elif category == GapCategory.CONTRADICTORY_INFORMATION:
                recommendations.extend(self._generate_contradictory_info_recommendations(category_gaps))
        
        return recommendations
    
    def _generate_missing_evidence_recommendations(self, gaps: List[DocumentationGap]) -> List[RecommendationAction]:
        """Generate recommendations for missing evidence gaps"""
        
        recommendations = []
        
        # High-priority missing evidence
        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        
        if critical_gaps:
            recommendations.append(RecommendationAction(
                action_id="missing_critical_evidence",
                action_type="obtain_documentation",
                description="Obtain critical missing clinical documentation to support medical necessity",
                priority="immediate",
                estimated_timeline="1-3_days",
                responsible_party="provider",
                expected_impact=0.8,
                implementation_notes="Focus on the most critical missing evidence first"
            ))
        
        return recommendations
    
    def _generate_insufficient_detail_recommendations(self, gaps: List[DocumentationGap]) -> List[RecommendationAction]:
        """Generate recommendations for insufficient detail gaps"""
        
        return [
            RecommendationAction(
                action_id="enhance_documentation",
                action_type="provider_assessment",
                description="Provide more detailed clinical documentation with specific findings and assessments",
                priority="high",
                estimated_timeline="1_week",
                responsible_party="provider",
                expected_impact=0.6,
                implementation_notes="Include specific clinical findings, measurements, and clinical reasoning"
            )
        ]
    
    def _generate_contradictory_info_recommendations(self, gaps: List[DocumentationGap]) -> List[RecommendationAction]:
        """Generate recommendations for contradictory information gaps"""
        
        return [
            RecommendationAction(
                action_id="clarify_contradictions",
                action_type="clarify_information",
                description="Clarify contradictory clinical information with updated assessment",
                priority="high",
                estimated_timeline="1-3_days",
                responsible_party="provider",
                expected_impact=0.7,
                implementation_notes="Provide clarification note explaining any apparent contradictions"
            )
        ]
    
    def _deduplicate_recommendations(self, recommendations: List[RecommendationAction]) -> List[RecommendationAction]:
        """Remove duplicate recommendations and prioritize"""
        
        # Simple deduplication based on description similarity
        unique_recommendations = []
        seen_descriptions = set()
        
        for rec in recommendations:
            # Normalize description for comparison
            normalized_desc = rec.description.lower().strip()
            
            if normalized_desc not in seen_descriptions:
                seen_descriptions.add(normalized_desc)
                unique_recommendations.append(rec)
        
        # Sort by expected impact (descending)
        unique_recommendations.sort(key=lambda x: x.expected_impact, reverse=True)
        
        return unique_recommendations
    
    def _prioritize_actions(self, recommendations: List[RecommendationAction],
                          gaps: List[DocumentationGap]) -> List[RecommendationAction]:
        """Prioritize recommendation actions"""
        
        # Calculate priority scores
        for rec in recommendations:
            priority_score = self._calculate_recommendation_priority(rec, gaps)
            rec.expected_impact = max(rec.expected_impact, priority_score)
        
        # Sort by priority
        priority_order = {'immediate': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        recommendations.sort(key=lambda x: (
            priority_order.get(x.priority, 0),
            x.expected_impact
        ), reverse=True)
        
        return recommendations
    
    def _calculate_recommendation_priority(self, recommendation: RecommendationAction,
                                        gaps: List[DocumentationGap]) -> float:
        """Calculate priority score for a recommendation"""
        
        # Base score from expected impact
        base_score = recommendation.expected_impact
        
        # Boost for immediate priority
        if recommendation.priority == 'immediate':
            base_score *= 1.2
        elif recommendation.priority == 'high':
            base_score *= 1.1
        
        # Boost for critical gaps
        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        if critical_gaps:
            base_score *= 1.15
        
        return min(1.0, base_score)
    
    def _estimate_approval_likelihood(self, assessments: List[CriterionAssessment],
                                    gaps: List[DocumentationGap],
                                    context: AssessmentContext) -> float:
        """Estimate approval likelihood based on current state"""
        
        if not assessments:
            return 0.0
        
        # Base likelihood from assessment compliance
        met_assessments = [a for a in assessments if a.status == ComplianceStatus.MET]
        base_likelihood = len(met_assessments) / len(assessments)
        
        # Adjust for gap severity
        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        high_gaps = [g for g in gaps if g.severity == GapSeverity.HIGH]
        
        # Penalty for critical gaps
        critical_penalty = len(critical_gaps) * 0.2
        high_penalty = len(high_gaps) * 0.1
        
        # Adjust likelihood
        adjusted_likelihood = base_likelihood - critical_penalty - high_penalty
        
        # Consider average confidence
        avg_confidence = sum(a.confidence_score for a in assessments) / len(assessments)
        
        # Final likelihood calculation
        final_likelihood = (adjusted_likelihood * 0.7) + (avg_confidence * 0.3)
        
        return max(0.0, min(1.0, final_likelihood))
    
    def _extract_missing_evidence(self, gaps: List[DocumentationGap]) -> List[str]:
        """Extract list of missing evidence from gaps"""
        
        missing_evidence = set()
        
        for gap in gaps:
            missing_evidence.update(gap.missing_evidence)
        
        return list(missing_evidence)[:10]  # Limit to top 10
    
    def _extract_critical_gaps(self, gaps: List[DocumentationGap]) -> List[str]:
        """Extract list of critical gaps"""
        
        critical_gaps = [
            gap.description for gap in gaps 
            if gap.severity == GapSeverity.CRITICAL
        ]
        
        return critical_gaps[:5]  # Limit to top 5

