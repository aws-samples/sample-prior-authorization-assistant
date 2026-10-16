"""
Policy document ingestion and indexing for Bedrock Knowledge Base.

This module handles the ingestion of policy documents, chunking,
index metadata, and Knowledge Base sync.
"""

import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

from common.aws_clients import get_bedrock_service, get_s3_service, get_textract_service
from common.exceptions import AWSServiceError, DocumentParsingException
from common.decorators import handle_processing_errors, monitor_performance, retry_with_backoff
from common.logging_config import get_logger, get_audit_logger
from common.utils import (
    generate_document_id, 
    clean_text, 
    chunk_list,
    safe_json_dumps,
    sanitize_filename
)
from .policy_metadata import PolicyMetadataManager, PolicyType, ClinicalArea

logger = get_logger(__name__)
audit_logger = get_audit_logger()

# Embedding model of the Knowledge Base vector index
# (see infrastructure/stacks/consolidated_stack.py, PolicyKnowledgeBase)
KB_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"


class PolicyDocumentProcessor:
    """Processes policy documents for ingestion into Knowledge Base"""
    
    def __init__(self):
        self.textract_service = get_textract_service()
        self.s3_service = get_s3_service()
        self.policies_bucket = os.getenv('POLICIES_BUCKET')
        
        if not self.policies_bucket:
            raise ValueError("POLICIES_BUCKET environment variable is required")
    
    @handle_processing_errors
    @monitor_performance
    def process_policy_document(self, file_content: bytes, filename: str,
                              payor: str, policy_type: str = None) -> Dict[str, Any]:
        """Process a policy document and extract structured information"""
        
        logger.info("Processing policy document", 
                   filename=filename, payor=payor)
        
        # Generate policy ID and upload to S3
        policy_id = self._generate_policy_id(payor, filename)
        document_path = self._upload_policy_document(
            policy_id, file_content, filename, payor
        )
        
        # Extract text content
        extracted_text = self._extract_text_from_policy(document_path)
        
        # Parse policy structure and content
        policy_data = self._parse_policy_content(
            extracted_text, payor, policy_type, filename
        )
        
        # Add document metadata
        policy_data.update({
            'policy_id': policy_id,
            'document_path': document_path,
            'file_size_bytes': len(file_content),
            'processed_at': datetime.utcnow().isoformat()
        })
        
        logger.info("Policy document processed successfully",
                   policy_id=policy_id,
                   payor=payor,
                   text_length=len(extracted_text))
        
        return policy_data
    
    def _generate_policy_id(self, payor: str, filename: str) -> str:
        """Generate unique policy ID"""
        
        # Create hash from payor and filename for consistency
        content = f"{payor.lower()}_{filename}_{datetime.utcnow().date()}"
        hash_object = hashlib.md5(content.encode(), usedforsecurity=False)
        hash_hex = hash_object.hexdigest()[:8]
        
        return f"policy_{payor.lower().replace(' ', '_')}_{hash_hex}"
    
    def _upload_policy_document(self, policy_id: str, file_content: bytes,
                              filename: str, payor: str) -> str:
        """Upload policy document to S3 with proper organization"""
        
        # Sanitize filename and create S3 key
        safe_filename = sanitize_filename(filename)
        payor_folder = payor.lower().replace(' ', '_').replace('-', '_')
        s3_key = f"policies/payors/{payor_folder}/{policy_id}/{safe_filename}"
        
        # Upload to S3
        self.s3_service.upload_document(
            bucket=self.policies_bucket,
            key=s3_key,
            content=file_content,
            content_type=self._get_content_type(filename)
        )
        
        return s3_key
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type from filename"""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        content_types = {
            'pdf': 'application/pdf',
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt': 'text/plain',
            'html': 'text/html'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    @retry_with_backoff(max_attempts=3)
    def _extract_text_from_policy(self, document_path: str) -> str:
        """Extract text from policy document using Textract"""
        
        try:
            # Extract text using Textract
            raw_text = self.textract_service.extract_text_from_document(
                bucket=self.policies_bucket,
                key=document_path
            )
            
            # Clean and normalize text
            cleaned_text = clean_text(raw_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.warning("Failed to extract text from policy document",
                        document_path=document_path, error=str(e))
            raise DocumentParsingException(f"Text extraction failed: {str(e)}")
    
    def _parse_policy_content(self, text: str, payor: str, 
                            policy_type: str = None, filename: str = "") -> Dict[str, Any]:
        """Parse policy content to extract structured information"""
        
        policy_data = {
            'payor': payor,
            'title': self._extract_policy_title(text, filename),
            'policy_type': policy_type or self._determine_policy_type(text),
            'clinical_area': self._determine_clinical_area(text),
            'content': text,
            'effective_date': self._extract_effective_date(text),
            'expiration_date': self._extract_expiration_date(text),
            'version': self._extract_version(text),
            'cpt_codes': self._extract_cpt_codes(text),
            'icd_codes': self._extract_icd_codes(text),
            'service_categories': self._extract_service_categories(text),
            'medical_necessity_criteria': self._extract_medical_necessity_criteria(text),
            'prior_auth_requirements': self._extract_prior_auth_requirements(text),
            'coverage_limitations': self._extract_coverage_limitations(text)
        }
        
        return policy_data
    
    def _extract_policy_title(self, text: str, filename: str) -> str:
        """Extract policy title from document"""
        
        # Look for title patterns in the first few lines
        lines = text.split('\n')[:10]
        
        # Common title patterns
        title_patterns = [
            r'(?i)^(.+?)\s*policy',
            r'(?i)^(.+?)\s*medical\s+necessity',
            r'(?i)^(.+?)\s*coverage\s+determination',
            r'(?i)^(.+?)\s*prior\s+authorization',
            r'(?i)title:\s*(.+)',
            r'(?i)policy\s+title:\s*(.+)',
            r'(?i)subject:\s*(.+)'
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Skip very short lines
                for pattern in title_patterns:
                    match = re.search(pattern, line)
                    if match:
                        title = match.group(1).strip()
                        if len(title) > 5:  # Ensure meaningful title
                            return title
        
        # Fallback to filename-based title
        if filename:
            base_name = filename.split('.')[0]
            return base_name.replace('_', ' ').replace('-', ' ').title()
        
        return "Policy Document"
    
    def _determine_policy_type(self, text: str) -> str:
        """Determine policy type from content"""
        
        text_lower = text.lower()
        
        if 'medical necessity' in text_lower:
            return PolicyType.MEDICAL_NECESSITY.value
        elif 'coverage determination' in text_lower:
            return PolicyType.COVERAGE_DETERMINATION.value
        elif 'prior authorization' in text_lower or 'preauthorization' in text_lower:
            return PolicyType.PRIOR_AUTHORIZATION.value
        elif 'clinical guideline' in text_lower:
            return PolicyType.CLINICAL_GUIDELINE.value
        elif 'formulary' in text_lower:
            return PolicyType.FORMULARY.value
        else:
            return PolicyType.MEDICAL_NECESSITY.value
    
    def _determine_clinical_area(self, text: str) -> str:
        """Determine clinical area from content"""
        
        text_lower = text.lower()
        
        # Clinical area keywords mapping
        area_keywords = {
            ClinicalArea.CARDIOLOGY.value: [
                'cardiac', 'heart', 'cardiovascular', 'coronary', 'echocardiogram', 
                'catheterization', 'angiography', 'stent', 'bypass'
            ],
            ClinicalArea.ORTHOPEDICS.value: [
                'orthopedic', 'bone', 'joint', 'fracture', 'arthroscopy', 
                'spine', 'knee', 'hip', 'shoulder'
            ],
            ClinicalArea.RADIOLOGY.value: [
                'imaging', 'mri', 'ct scan', 'x-ray', 'ultrasound', 
                'mammography', 'pet scan', 'nuclear medicine'
            ],
            ClinicalArea.ONCOLOGY.value: [
                'cancer', 'oncology', 'chemotherapy', 'radiation', 
                'tumor', 'malignant', 'neoplasm'
            ],
            ClinicalArea.NEUROLOGY.value: [
                'neurological', 'brain', 'nervous system', 'seizure', 
                'stroke', 'epilepsy', 'migraine'
            ]
        }
        
        # Count keyword matches for each area
        area_scores = {}
        for area, keywords in area_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                area_scores[area] = score
        
        # Return area with highest score
        if area_scores:
            return max(area_scores, key=area_scores.get)
        
        return ClinicalArea.GENERAL_MEDICINE.value
    
    def _extract_effective_date(self, text: str) -> Optional[str]:
        """Extract effective date from policy text"""
        
        # Date patterns
        date_patterns = [
            r'(?i)effective\s+date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?i)effective\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?i)date\s+of\s+issue:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?i)issued:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_expiration_date(self, text: str) -> Optional[str]:
        """Extract expiration date from policy text"""
        
        date_patterns = [
            r'(?i)expir(?:es|ation)\s+date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?i)expir(?:es|ation)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?i)valid\s+until:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?i)end\s+date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_version(self, text: str) -> str:
        """Extract version from policy text"""
        
        version_patterns = [
            r'(?i)version:\s*([0-9]+(?:\.[0-9]+)*)',
            r'(?i)v\.?\s*([0-9]+(?:\.[0-9]+)*)',
            r'(?i)revision:\s*([0-9]+(?:\.[0-9]+)*)',
            r'(?i)rev\.?\s*([0-9]+(?:\.[0-9]+)*)'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "1.0"  # Default version
    
    def _extract_cpt_codes(self, text: str) -> List[str]:
        """Extract CPT codes from policy text"""
        
        cpt_pattern = re.compile(r'\b(\d{5}(?:-[A-Z0-9]{2})?)\b')
        matches = cpt_pattern.findall(text)
        
        # Validate and deduplicate
        valid_codes = set()
        for match in matches:
            # Basic validation - 5 digits
            if len(match.split('-')[0]) == 5 and match.split('-')[0].isdigit():
                valid_codes.add(match)
        
        return list(valid_codes)
    
    def _extract_icd_codes(self, text: str) -> List[str]:
        """Extract ICD codes from policy text"""
        
        icd_pattern = re.compile(r'\b([A-Z]\d{2}(?:\.[A-Z0-9]{1,4})?)\b')
        matches = icd_pattern.findall(text)
        
        # Validate and deduplicate
        valid_codes = set()
        for match in matches:
            # Basic ICD-10 validation
            if re.match(r'^[A-Z]\d{2}', match):
                valid_codes.add(match)
        
        return list(valid_codes)
    
    def _extract_service_categories(self, text: str) -> List[str]:
        """Extract service categories from policy text"""
        
        text_lower = text.lower()
        categories = set()
        
        category_keywords = {
            'diagnostic': ['diagnostic', 'diagnosis', 'test', 'screening', 'examination'],
            'therapeutic': ['therapeutic', 'treatment', 'therapy', 'intervention'],
            'surgical': ['surgical', 'surgery', 'procedure', 'operation'],
            'imaging': ['imaging', 'radiology', 'scan', 'x-ray', 'mri', 'ct'],
            'laboratory': ['laboratory', 'lab', 'blood test', 'pathology'],
            'preventive': ['preventive', 'prevention', 'screening', 'wellness'],
            'emergency': ['emergency', 'urgent', 'emergent', 'acute'],
            'inpatient': ['inpatient', 'hospital', 'admission', 'stay'],
            'outpatient': ['outpatient', 'ambulatory', 'clinic', 'office']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.add(category)
        
        return list(categories)
    
    def _extract_medical_necessity_criteria(self, text: str) -> List[str]:
        """Extract medical necessity criteria from policy text"""
        
        criteria = []
        
        # Look for numbered or bulleted criteria
        criteria_patterns = [
            r'(?i)(?:medical\s+necessity\s+)?criteria?[:\s]*\n((?:(?:\d+\.|\*|\-|\•).+\n?)+)',
            r'(?i)requirements?[:\s]*\n((?:(?:\d+\.|\*|\-|\•).+\n?)+)',
            r'(?i)conditions?[:\s]*\n((?:(?:\d+\.|\*|\-|\•).+\n?)+)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                # Split into individual criteria
                lines = match.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 10:  # Skip very short lines
                        # Clean up formatting
                        cleaned = re.sub(r'^[\d\.\*\-\•\s]+', '', line).strip()
                        if cleaned:
                            criteria.append(cleaned)
        
        return criteria[:10]  # Limit to first 10 criteria
    
    def _extract_prior_auth_requirements(self, text: str) -> Dict[str, Any]:
        """Extract prior authorization requirements"""
        
        requirements = {
            'required': 'prior authorization' in text.lower() or 'preauthorization' in text.lower(),
            'urgency_exceptions': [],
            'documentation_required': []
        }
        
        # Look for urgency exceptions
        if 'emergency' in text.lower() or 'urgent' in text.lower():
            requirements['urgency_exceptions'].append('emergency')
        
        # Look for documentation requirements
        doc_keywords = [
            'medical records', 'physician notes', 'test results', 
            'imaging studies', 'laboratory results', 'clinical documentation'
        ]
        
        for keyword in doc_keywords:
            if keyword in text.lower():
                requirements['documentation_required'].append(keyword)
        
        return requirements
    
    def _extract_coverage_limitations(self, text: str) -> Dict[str, Any]:
        """Extract coverage limitations and exclusions"""
        
        limitations = {
            'age_limits': None,
            'frequency_limits': None,
            'exclusions': [],
            'conditions': []
        }
        
        # Look for age limitations
        age_pattern = r'(?i)age[s]?\s+(\d+)(?:\s*(?:to|-)\s*(\d+))?'
        age_match = re.search(age_pattern, text)
        if age_match:
            if age_match.group(2):
                limitations['age_limits'] = f"{age_match.group(1)}-{age_match.group(2)}"
            else:
                limitations['age_limits'] = f"{age_match.group(1)}+"
        
        # Look for frequency limitations
        freq_patterns = [
            r'(?i)once\s+per\s+(\w+)',
            r'(?i)(\d+)\s+times?\s+per\s+(\w+)',
            r'(?i)maximum\s+of\s+(\d+)'
        ]
        
        for pattern in freq_patterns:
            match = re.search(pattern, text)
            if match:
                limitations['frequency_limits'] = match.group(0)
                break
        
        # Look for exclusions
        exclusion_keywords = [
            'not covered', 'excluded', 'not eligible', 'contraindicated',
            'not medically necessary', 'experimental', 'investigational'
        ]
        
        for keyword in exclusion_keywords:
            if keyword in text.lower():
                limitations['exclusions'].append(keyword)
        
        return limitations


class PolicyIndexingService:
    """Chunks policy documents and records KB index metadata"""
    
    def __init__(self):
        self.bedrock_service = get_bedrock_service()
        self.s3_service = get_s3_service()
        self.metadata_manager = PolicyMetadataManager()
        self.policies_bucket = os.getenv('POLICIES_BUCKET')
        self.embedding_model = KB_EMBEDDING_MODEL_ID
    
    @handle_processing_errors
    @monitor_performance
    def index_policy_document(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Index a policy document with semantic embeddings"""
        
        logger.info("Indexing policy document", 
                   policy_id=policy_data.get('policy_id'))
        
        # Create metadata
        metadata = self.metadata_manager.create_policy_metadata(policy_data)
        
        # Chunk the document content
        chunks = self._chunk_policy_content(policy_data['content'])
        
        # Embeddings are generated by the Knowledge Base ingestion job

        # Save metadata and index information
        index_info = {
            'policy_id': policy_data['policy_id'],
            'metadata': metadata.to_dict(),
            'chunks_count': len(chunks),
            'indexed_at': datetime.utcnow().isoformat(),
            'embedding_model': self.embedding_model
        }
        
        # Save index information to S3
        self._save_index_metadata(policy_data['policy_id'], index_info)
        
        logger.info("Policy document indexed successfully",
                   policy_id=policy_data['policy_id'],
                   chunks_count=len(chunks))
        
        # Log to audit trail
        audit_logger.log_policy_access(
            policy_id=policy_data['policy_id'],
            payor=metadata.payor,
            access_type="index"
        )
        
        return index_info
    
    def _chunk_policy_content(self, content: str, chunk_size: int = 1000, 
                            overlap: int = 200) -> List[str]:
        """Chunk policy content for index metadata"""
        
        # Split into sentences first
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _save_index_metadata(self, policy_id: str, index_info: Dict[str, Any]) -> None:
        """Save indexing metadata to S3"""
        
        try:
            metadata_key = f"policies/metadata/{policy_id}_index.json"
            
            self.s3_service.s3_client.put_object(
                Bucket=self.policies_bucket,
                Key=metadata_key,
                Body=safe_json_dumps(index_info),
                ContentType='application/json'
            )
            
            logger.info("Index metadata saved",
                       policy_id=policy_id,
                       metadata_key=metadata_key)
            
        except Exception as e:
            logger.warning("Failed to save index metadata",
                        policy_id=policy_id,
                        error=str(e))
            raise AWSServiceError(f"Failed to save index metadata: {str(e)}")
    
    @handle_processing_errors
    def sync_with_knowledge_base(self, knowledge_base_id: str, 
                               data_source_id: str) -> Dict[str, Any]:
        """Sync indexed documents with Bedrock Knowledge Base"""
        
        try:
            # Start ingestion job
            response = self.bedrock_service.bedrock_agent_client.start_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                description=f"Policy ingestion job - {datetime.utcnow().isoformat()}"
            )
            
            ingestion_job_id = response['ingestionJob']['ingestionJobId']
            
            logger.info("Knowledge Base sync started",
                       knowledge_base_id=knowledge_base_id,
                       ingestion_job_id=ingestion_job_id)
            
            return {
                'ingestion_job_id': ingestion_job_id,
                'status': 'started',
                'started_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning("Failed to sync with Knowledge Base",
                        knowledge_base_id=knowledge_base_id,
                        error=str(e))
            raise AWSServiceError(f"Knowledge Base sync failed: {str(e)}")
    
    def get_ingestion_job_status(self, knowledge_base_id: str, 
                               data_source_id: str, job_id: str) -> Dict[str, Any]:
        """Get status of Knowledge Base ingestion job"""
        
        try:
            response = self.bedrock_service.bedrock_agent_client.get_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                ingestionJobId=job_id
            )
            
            job_info = response['ingestionJob']
            
            return {
                'job_id': job_id,
                'status': job_info['status'],
                'started_at': job_info.get('startedAt'),
                'completed_at': job_info.get('completedAt'),
                'failure_reasons': job_info.get('failureReasons', [])
            }
            
        except Exception as e:
            logger.warning("Failed to get ingestion job status",
                        job_id=job_id, error=str(e))
            raise AWSServiceError(f"Failed to get job status: {str(e)}")

