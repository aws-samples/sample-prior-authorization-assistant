"""
Policy retrieval and search functionality for Bedrock Knowledge Base.

This module provides semantic search, policy ranking, and retrieval
capabilities for finding relevant payor policies based on medical data.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
import math

from common.aws_clients import get_bedrock_service, get_s3_service
from common.exceptions import AWSServiceError, PolicySelectionError
from common.decorators import handle_processing_errors, monitor_performance, retry_with_backoff, cache_result
from common.logging_config import get_logger, get_audit_logger
from common.utils import calculate_confidence_score, safe_json_loads
from .policy_metadata import PolicyMetadata, PolicyMetadataManager, ClinicalArea
from .query_constructor import build_search_query
from .metadata_filter import apply_two_tier_filtering

logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class PolicySearchResult:
    """Result from policy search operation"""
    policy_id: str
    payor: str
    policy_title: str
    relevance_score: float
    confidence_score: float
    matched_content: str
    metadata: Dict[str, Any]
    retrieved_at: datetime


@dataclass
class SearchQuery:
    """Structured search query for policy retrieval"""
    payor: str
    cpt_codes: List[str]
    icd_codes: List[str]
    service_type: str
    hcpcs_codes: List[str] = None  # HCPCS codes for DME, home health, etc.
    rxnorm_codes: List[str] = None  # RxNorm codes for medications
    clinical_area: Optional[str] = None
    urgency: str = "routine"
    patient_age_range: Optional[str] = None
    patient_gender: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.hcpcs_codes is None:
            self.hcpcs_codes = []
        if self.rxnorm_codes is None:
            self.rxnorm_codes = []


class PolicyRetrievalService:
    """Service for retrieving and ranking policies from Knowledge Base"""
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
        self.s3_service = get_s3_service()
        self.metadata_manager = PolicyMetadataManager()
        
        # Configuration
        self.knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID')
        self.policies_bucket = os.getenv('POLICIES_BUCKET')
        
        if not self.knowledge_base_id:
            logger.warning("KNOWLEDGE_BASE_ID not set, using fallback search")
    
    @handle_processing_errors
    @monitor_performance
    def search_policies(self, query: SearchQuery, max_results: int = 15) -> List[PolicySearchResult]:
        """Search for relevant policies based on medical data"""
        
        logger.info("Searching policies",
                   payor=query.payor,
                   cpt_codes=query.cpt_codes,
                   icd_codes=query.icd_codes,
                   service_type=query.service_type)
        
        # Create search query text
        search_text = self._create_search_query_text(query)
        
        # Perform semantic search
        if self.knowledge_base_id:
            search_results = self._semantic_search(search_text, max_results * 2)
        else:
            search_results = self._fallback_search(query, max_results * 2)
        
        # Rank and filter results
        ranked_results = self._rank_search_results(search_results, query)
        
        # Limit to requested number of results
        final_results = ranked_results[:max_results]
        
        # Log search results
        audit_logger.log_policy_access(
            policy_id=f"search_{len(final_results)}_policies",
            payor=query.payor,
            access_type="search"
        )
        
        logger.info("Policy search completed",
                   query_payor=query.payor,
                   results_count=len(final_results),
                   top_score=final_results[0].relevance_score if final_results else 0.0)
        
        return final_results
    
    @handle_processing_errors
    @monitor_performance
    def search_policies_with_extraction(self, extraction_result: Dict[str, Any], 
                                       max_results: int = 15) -> List[PolicySearchResult]:
        """
        Search for relevant policies using enhanced query from extraction result.
        
        This method uses the new query constructor to build a rich search query
        with code descriptions, clinical context, and patient information.
        
        Args:
            extraction_result: Complete extraction result with all fields
            max_results: Maximum number of results to return
            
        Returns:
            List of ranked policy search results
        """
        payor_name = extraction_result.get('payor_name', 'Unknown')
        
        logger.info("Searching policies with enhanced query",
                   payor=payor_name,
                   has_structured_fields=bool(extraction_result.get('structured_fields')),
                   has_patient_info=bool(extraction_result.get('patient_info')))
        
        # Create enhanced search query
        search_text = self.create_enhanced_search_query(extraction_result)
        
        logger.debug("Enhanced search query created",
                    query_length=len(search_text),
                    payor=payor_name)
        
        # Perform semantic search
        if self.knowledge_base_id:
            search_results = self._semantic_search(search_text, max_results * 2)
        else:
            # For fallback, create a basic SearchQuery from extraction result
            query = self._extraction_to_search_query(extraction_result)
            search_results = self._fallback_search(query, max_results * 2)
        
        # Rank and filter results using extraction result context
        ranked_results = self._rank_search_results_with_extraction(
            search_results, extraction_result
        )
        
        # Limit to requested number of results
        final_results = ranked_results[:max_results]
        
        # Log search results
        audit_logger.log_policy_access(
            policy_id=f"search_{len(final_results)}_policies",
            payor=payor_name,
            access_type="enhanced_search"
        )
        
        logger.info("Enhanced policy search completed",
                   query_payor=payor_name,
                   results_count=len(final_results),
                   top_score=final_results[0].relevance_score if final_results else 0.0)
        
        return final_results
    
    @handle_processing_errors
    @monitor_performance
    def search_policies_with_two_tier_filtering(self, extraction_result: Dict[str, Any],
                                                max_results: int = 15) -> List[PolicySearchResult]:
        """
        Search for relevant policies using two-tier metadata filtering.
        
        This method applies metadata filtering before semantic search:
        - Tier 1: Payor + Procedure Codes + ICD Codes
        - Tier 2: Payor + Procedure Codes only (fallback if Tier 1 < max_results)
        
        Args:
            extraction_result: Complete extraction result with all fields
            max_results: Maximum number of results to return
            
        Returns:
            List of ranked policy search results
        """
        payor_name = extraction_result.get('payor_name', 'Unknown')
        cpt_codes = extraction_result.get('cpt_codes', [])
        icd_codes = extraction_result.get('icd_codes', [])
        
        # Collect all procedure codes (CPT, HCPCS, J-codes, RxNorm)
        procedure_codes = cpt_codes.copy()
        
        # Add HCPCS codes if present in structured fields
        structured_fields = extraction_result.get('structured_fields', {})
        service_details = structured_fields.get('service_details', {})
        
        if 'hcpcs_codes' in service_details:
            procedure_codes.extend(service_details['hcpcs_codes'])
        if 'rxnorm_codes' in service_details:
            procedure_codes.extend(service_details['rxnorm_codes'])
        
        logger.info("Searching policies with two-tier filtering",
                   payor=payor_name,
                   procedure_codes_count=len(procedure_codes),
                   icd_codes_count=len(icd_codes))
        
        # Create enhanced search query
        search_text = self.create_enhanced_search_query(extraction_result)
        
        # Create a wrapper function for Bedrock retrieve
        def bedrock_retrieve_fn(query_text: str, filter: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
            return self._semantic_search(query_text, max_results, filter=filter)
        
        # Apply two-tier filtering
        if self.knowledge_base_id:
            search_results = apply_two_tier_filtering(
                bedrock_retrieve_fn=bedrock_retrieve_fn,
                query_text=search_text,
                payor=payor_name,
                procedure_codes=procedure_codes,
                icd_codes=icd_codes,
                max_results=max_results * 2  # Get more for ranking
            )
        else:
            # Fallback to non-filtered search
            query = self._extraction_to_search_query(extraction_result)
            search_results = self._fallback_search(query, max_results * 2)
        
        # Rank and filter results
        ranked_results = self._rank_search_results_with_extraction(
            search_results, extraction_result
        )
        
        # Limit to requested number of results
        final_results = ranked_results[:max_results]
        
        # Log search results
        audit_logger.log_policy_access(
            policy_id=f"search_{len(final_results)}_policies",
            payor=payor_name,
            access_type="two_tier_search"
        )
        
        logger.info("Two-tier policy search completed",
                   query_payor=payor_name,
                   results_count=len(final_results),
                   top_score=final_results[0].relevance_score if final_results else 0.0,
                   tier1_count=sum(1 for r in search_results if r.get('tier') == 1),
                   tier2_count=sum(1 for r in search_results if r.get('tier') == 2))
        
        return final_results
    
    def _extraction_to_search_query(self, extraction_result: Dict[str, Any]) -> SearchQuery:
        """Convert extraction result to SearchQuery for fallback search"""
        patient_info = extraction_result.get('patient_info', {})
        
        return SearchQuery(
            payor=extraction_result.get('payor_name', ''),
            cpt_codes=extraction_result.get('cpt_codes', []),
            icd_codes=extraction_result.get('icd_codes', []),
            service_type=extraction_result.get('service_type', 'general'),
            patient_age_range=patient_info.get('age_range'),
            patient_gender=patient_info.get('gender')
        )
    
    def _rank_search_results_with_extraction(self, search_results: List[Dict[str, Any]], 
                                            extraction_result: Dict[str, Any]) -> List[PolicySearchResult]:
        """
        Rank search results using extraction result context.
        
        This is a wrapper that converts extraction_result to SearchQuery format
        for compatibility with existing ranking logic.
        """
        # Convert extraction result to SearchQuery for ranking
        query = self._extraction_to_search_query(extraction_result)
        
        # Use existing ranking logic
        return self._rank_search_results(search_results, query)
    
    def _create_search_query_text(self, query: SearchQuery) -> str:
        """Create optimized search query text for semantic search"""
        
        query_parts = []
        
        # Add payor information
        query_parts.append(f"Insurance payor: {query.payor}")
        
        # Add medical codes
        if query.cpt_codes:
            query_parts.append(f"CPT codes: {', '.join(query.cpt_codes)}")
        
        if query.hcpcs_codes:
            query_parts.append(f"HCPCS codes: {', '.join(query.hcpcs_codes)}")
        
        if query.rxnorm_codes:
            query_parts.append(f"RxNorm medication codes: {', '.join(query.rxnorm_codes)}")
        
        if query.icd_codes:
            query_parts.append(f"ICD diagnosis codes: {', '.join(query.icd_codes)}")
        
        # Add service information
        query_parts.append(f"Medical service type: {query.service_type}")
        
        if query.clinical_area:
            query_parts.append(f"Clinical specialty: {query.clinical_area}")
        
        # Add patient context
        if query.patient_age_range:
            query_parts.append(f"Patient age range: {query.patient_age_range}")
        
        if query.patient_gender:
            query_parts.append(f"Patient gender: {query.patient_gender}")
        
        # Add urgency
        query_parts.append(f"Urgency level: {query.urgency}")
        
        # Add context for better matching
        query_parts.append("Prior authorization medical necessity criteria coverage policy")
        
        return ". ".join(query_parts)
    
    def create_enhanced_search_query(self, extraction_result: Dict[str, Any]) -> str:
        """
        Create enhanced search query from extraction result.
        
        Uses the new query constructor to build a rich, clinical-context-aware
        search query with code descriptions and patient information.
        
        Args:
            extraction_result: Complete extraction result with all fields
            
        Returns:
            Enhanced natural language search query
        """
        return build_search_query(extraction_result)
    
    @retry_with_backoff(max_attempts=3)
    def _semantic_search(self, query_text: str, max_results: int, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using Bedrock Knowledge Base with optional metadata filter"""
        
        try:
            # Build retrieve parameters
            retrieve_params = {
                'knowledge_base_id': self.knowledge_base_id,
                'query': query_text,
                'max_results': max_results
            }
            
            # Add filter if provided
            if filter:
                retrieve_params['filter'] = filter
            
            response = self.bedrock_service.retrieve_from_knowledge_base(**retrieve_params)
            
            search_results = []
            
            for result in response.get('retrievalResults', []):
                content = result.get('content', {}).get('text', '')
                metadata = result.get('metadata', {})
                score = result.get('score', 0.0)
                
                # Sanitize metadata to ensure proper types for serialization
                sanitized_metadata = self._sanitize_metadata(metadata)
                
                search_results.append({
                    'content': content,
                    'metadata': sanitized_metadata,
                    'score': score,
                    'source': 'knowledge_base'
                })
            
            logger.info("Semantic search completed",
                       results_count=len(search_results),
                       query_length=len(query_text),
                       filtered=filter is not None)
            
            return search_results
            
        except Exception as e:
            logger.warning("Semantic search failed", error=str(e))
            raise AWSServiceError(f"Semantic search failed: {str(e)}")
    
    def _fallback_search(self, query: SearchQuery, max_results: int) -> List[Dict[str, Any]]:
        """Fallback search using metadata when Knowledge Base is not available"""
        
        logger.info("Using fallback metadata search")
        
        # Search metadata cache
        metadata_results = self.metadata_manager.search_metadata(
            query=f"{query.payor} {query.service_type}",
            payor=query.payor
        )
        
        search_results = []
        
        for metadata in metadata_results[:max_results]:
            # Calculate basic relevance score
            relevance_score = self._calculate_metadata_relevance(metadata, query)
            
            search_results.append({
                'content': metadata.summary,
                'metadata': metadata.to_dict(),
                'score': relevance_score,
                'source': 'metadata_fallback'
            })
        
        return search_results
    
    def _calculate_metadata_relevance(self, metadata: PolicyMetadata, query: SearchQuery) -> float:
        """Calculate relevance score based on metadata matching"""
        
        score_factors = {}
        
        # Payor match (highest weight)
        if metadata.payor.lower() == query.payor.lower():
            score_factors['payor_match'] = 1.0
        else:
            score_factors['payor_match'] = 0.0
        
        # CPT code matches
        cpt_matches = len(set(metadata.cpt_codes) & set(query.cpt_codes))
        total_cpts = len(set(metadata.cpt_codes) | set(query.cpt_codes))
        score_factors['cpt_match'] = cpt_matches / total_cpts if total_cpts > 0 else 0.0
        
        # ICD code matches
        icd_matches = len(set(metadata.icd_codes) & set(query.icd_codes))
        total_icds = len(set(metadata.icd_codes) | set(query.icd_codes))
        score_factors['icd_match'] = icd_matches / total_icds if total_icds > 0 else 0.0
        
        # Clinical area match
        if query.clinical_area and metadata.clinical_area.value == query.clinical_area.lower():
            score_factors['clinical_area_match'] = 1.0
        else:
            score_factors['clinical_area_match'] = 0.5
        
        # Service category match
        service_match = any(
            query.service_type.lower() in category.lower() 
            for category in metadata.service_categories
        )
        score_factors['service_match'] = 1.0 if service_match else 0.3
        
        # Policy recency (newer policies get higher scores)
        days_old = (date.today() - metadata.effective_date).days
        recency_score = max(0.0, 1.0 - (days_old / 365.0))  # Decay over 1 year
        score_factors['recency'] = recency_score
        
        # Calculate weighted score
        weights = {
            'payor_match': 0.4,
            'cpt_match': 0.2,
            'icd_match': 0.2,
            'clinical_area_match': 0.1,
            'service_match': 0.05,
            'recency': 0.05
        }
        
        return calculate_confidence_score(score_factors, weights)
    
    def _rank_search_results(self, search_results: List[Dict[str, Any]], 
                           query: SearchQuery) -> List[PolicySearchResult]:
        """Rank and convert search results to PolicySearchResult objects"""
        
        ranked_results = []
        
        for result in search_results:
            # Extract metadata
            metadata = result.get('metadata', {})
            
            # Calculate enhanced relevance score
            relevance_score = self._calculate_enhanced_relevance_score(result, query)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(result, query)
            
            # Create PolicySearchResult
            # Check both 'title' and 'policy_title' keys for backwards compatibility
            policy_title = metadata.get('title') or metadata.get('policy_title', 'Unknown Policy')
            
            policy_result = PolicySearchResult(
                policy_id=metadata.get('policy_id', 'unknown'),
                payor=metadata.get('payor', query.payor),
                policy_title=policy_title,
                relevance_score=relevance_score,
                confidence_score=confidence_score,
                matched_content=result.get('content', '')[:500],  # First 500 chars
                metadata=metadata,
                retrieved_at=datetime.utcnow()
            )
            
            ranked_results.append(policy_result)
        
        # Sort by relevance score (descending)
        ranked_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return ranked_results
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure proper types for JSON serialization"""
        sanitized = {}
        
        for key, value in metadata.items():
            if key == 'x-amz-bedrock-kb-document-page-number':
                # Convert float page numbers to integers
                try:
                    sanitized[key] = int(float(value)) if value is not None else None
                except (ValueError, TypeError):
                    sanitized[key] = value
            elif isinstance(value, float):
                # Handle other float values that might cause serialization issues
                if value.is_integer():
                    sanitized[key] = int(value)
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _calculate_enhanced_relevance_score(self, result: Dict[str, Any], 
                                          query: SearchQuery) -> float:
        """Calculate enhanced relevance score combining semantic and metadata factors"""
        
        base_score = result.get('score', 0.0)
        metadata = result.get('metadata', {})
        content = result.get('content', '').lower()
        
        score_factors = {
            'semantic_score': base_score
        }
        
        # Payor matching
        payor_match = metadata.get('payor', '').lower() == query.payor.lower()
        score_factors['payor_match'] = 1.0 if payor_match else 0.0
        
        # Medical code matching
        metadata_cpts = metadata.get('cpt_codes', [])
        metadata_icds = metadata.get('icd_codes', [])
        
        cpt_overlap = len(set(metadata_cpts) & set(query.cpt_codes))
        icd_overlap = len(set(metadata_icds) & set(query.icd_codes))
        
        score_factors['cpt_relevance'] = min(1.0, cpt_overlap / max(1, len(query.cpt_codes)))
        score_factors['icd_relevance'] = min(1.0, icd_overlap / max(1, len(query.icd_codes)))
        
        # Content relevance
        service_in_content = query.service_type.lower() in content
        score_factors['content_relevance'] = 1.0 if service_in_content else 0.5
        
        # Clinical area matching
        clinical_area_match = (
            query.clinical_area and 
            metadata.get('clinical_area', '').lower() == query.clinical_area.lower()
        )
        score_factors['clinical_match'] = 1.0 if clinical_area_match else 0.7
        
        # Urgency consideration
        urgency_mentioned = query.urgency.lower() in content
        score_factors['urgency_relevance'] = 1.0 if urgency_mentioned else 0.8
        
        # Calculate weighted final score
        # Note: payor_match and cpt_relevance removed as policies are already filtered by these
        weights = {
            'semantic_score': 0.50,      # Bedrock KB semantic search score (UP from 0.30)
            'payor_match': 0.0,          # REMOVED - already filtered in search
            'cpt_relevance': 0.0,        # REMOVED - already filtered in search
            'icd_relevance': 0.35,       # ICD code overlap (UP from 0.15) - critical for condition-specific policies
            'content_relevance': 0.10,   # Service type in content (same)
            'clinical_match': 0.03,      # Clinical area match (same)
            'urgency_relevance': 0.02    # Urgency mentioned (same)
        }
        
        return calculate_confidence_score(score_factors, weights)
    
    def _calculate_confidence_score(self, result: Dict[str, Any], 
                                  query: SearchQuery) -> float:
        """Calculate confidence score for the search result"""
        
        metadata = result.get('metadata', {})
        
        confidence_factors = {
            'metadata_completeness': self._assess_metadata_completeness(metadata),
            'query_coverage': self._assess_query_coverage(result, query),
            'policy_recency': self._assess_policy_recency(metadata),
            'content_quality': self._assess_content_quality(result.get('content', ''))
        }
        
        return calculate_confidence_score(confidence_factors)
    
    def _assess_metadata_completeness(self, metadata: Dict[str, Any]) -> float:
        """Assess completeness of policy metadata"""
        
        required_fields = [
            'policy_id', 'payor', 'policy_title', 'policy_type',
            'clinical_area', 'effective_date'
        ]
        
        present_fields = sum(1 for field in required_fields if metadata.get(field))
        return present_fields / len(required_fields)
    
    def _assess_query_coverage(self, result: Dict[str, Any], query: SearchQuery) -> float:
        """Assess how well the result covers the query requirements"""
        
        metadata = result.get('metadata', {})
        content = result.get('content', '').lower()
        
        coverage_score = 0.0
        total_aspects = 0
        
        # Check payor coverage
        if metadata.get('payor'):
            total_aspects += 1
            if metadata.get('payor', '').lower() == query.payor.lower():
                coverage_score += 1.0
        
        # Check medical codes coverage
        if query.cpt_codes:
            total_aspects += 1
            metadata_cpts = metadata.get('cpt_codes', [])
            if any(code in metadata_cpts for code in query.cpt_codes):
                coverage_score += 1.0
        
        if query.icd_codes:
            total_aspects += 1
            metadata_icds = metadata.get('icd_codes', [])
            if any(code in metadata_icds for code in query.icd_codes):
                coverage_score += 1.0
        
        # Check service type coverage
        total_aspects += 1
        if query.service_type.lower() in content:
            coverage_score += 1.0
        
        return coverage_score / total_aspects if total_aspects > 0 else 0.0
    
    def _assess_policy_recency(self, metadata: Dict[str, Any]) -> float:
        """Assess policy recency (newer policies are more reliable)"""
        
        effective_date_str = metadata.get('effective_date')
        if not effective_date_str:
            return 0.5  # Unknown date gets medium score
        
        try:
            effective_date = date.fromisoformat(effective_date_str)
            days_old = (date.today() - effective_date).days
            
            # Policies are considered fresh for 2 years, then decay
            if days_old <= 730:  # 2 years
                return 1.0
            else:
                # Linear decay over next 3 years
                decay_factor = max(0.0, 1.0 - ((days_old - 730) / 1095))
                return 0.5 + (0.5 * decay_factor)
        
        except (ValueError, TypeError):
            return 0.5
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess quality of policy content"""
        
        if not content:
            return 0.0
        
        quality_factors = {
            'length': min(1.0, len(content) / 1000),  # Normalize to 1000 chars
            'structure': 1.0 if any(marker in content.lower() for marker in 
                                  ['criteria', 'requirements', 'conditions']) else 0.5,
            'medical_terms': 1.0 if any(term in content.lower() for term in 
                                      ['medical', 'clinical', 'diagnosis', 'treatment']) else 0.5
        }
        
        return calculate_confidence_score(quality_factors)
    
    @handle_processing_errors
    @cache_result(ttl_seconds=300)  # Cache for 5 minutes
    def get_policy_by_id(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific policy by ID"""
        
        try:
            # Try to get from metadata first
            if policy_id in self.metadata_manager.metadata_cache:
                metadata = self.metadata_manager.metadata_cache[policy_id]
                
                # Get full policy content from S3
                policy_content = self._get_policy_content_from_s3(metadata.document_path)
                
                return {
                    'policy_id': policy_id,
                    'metadata': metadata.to_dict(),
                    'content': policy_content,
                    'retrieved_at': datetime.utcnow().isoformat()
                }
            
            # Fallback: search in S3 metadata
            return self._get_policy_from_s3_metadata(policy_id)
            
        except Exception as e:
            logger.error("Failed to retrieve policy by ID",
                        policy_id=policy_id, error=str(e))
            return None
    
    def _get_policy_content_from_s3(self, document_path: str) -> str:
        """Get policy content from S3 document"""
        
        try:
            # For now, return a placeholder since we'd need to extract text again
            # In a full implementation, we might store extracted text separately
            return f"Policy content from {document_path}"
            
        except Exception as e:
            logger.error("Failed to get policy content from S3",
                        document_path=document_path, error=str(e))
            return ""
    
    def _get_policy_from_s3_metadata(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get policy information from S3 metadata files"""
        
        try:
            metadata_key = f"policies/metadata/{policy_id}_index.json"
            
            response = self.s3_service.s3_client.get_object(
                Bucket=self.policies_bucket,
                Key=metadata_key
            )
            
            metadata_content = response['Body'].read().decode('utf-8')
            metadata = safe_json_loads(metadata_content, {})
            
            return metadata
            
        except Exception as e:
            logger.warning("Policy metadata not found in S3",
                          policy_id=policy_id, error=str(e))
            return None
    
    @handle_processing_errors
    def get_policies_by_payor(self, payor: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all policies for a specific payor"""
        
        # Search using payor name
        query = SearchQuery(
            payor=payor,
            cpt_codes=[],
            icd_codes=[],
            service_type="general"
        )
        
        results = self.search_policies(query, max_results=limit)
        
        return [
            {
                'policy_id': result.policy_id,
                'policy_title': result.policy_title,
                'relevance_score': result.relevance_score,
                'metadata': result.metadata
            }
            for result in results
        ]
    
    @handle_processing_errors
    def get_policies_by_clinical_area(self, clinical_area: str, 
                                    limit: int = 20) -> List[Dict[str, Any]]:
        """Get policies for a specific clinical area"""
        
        # Use metadata manager for clinical area search
        try:
            clinical_area_enum = ClinicalArea(clinical_area.lower())
            metadata_results = self.metadata_manager.get_metadata_by_clinical_area(clinical_area_enum)
            
            return [
                {
                    'policy_id': metadata.policy_id,
                    'policy_title': metadata.policy_title,
                    'payor': metadata.payor,
                    'clinical_area': metadata.clinical_area.value,
                    'metadata': metadata.to_dict()
                }
                for metadata in metadata_results[:limit]
            ]
            
        except ValueError:
            logger.warning("Invalid clinical area", clinical_area=clinical_area)
            return []

