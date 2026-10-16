"""
AWS service clients and utilities for the Prior Authorization Application.

This module provides centralized AWS service client management with
proper error handling, retry logic, and configuration.
"""

import boto3
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional, Dict, Any, List
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config
import structlog

from .exceptions import AWSServiceError, BedrockThrottlingException
from .logging_config import get_logger
from .model_config import resolve as resolve_model_role

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelUsage:
    """Token usage and latency for one model invocation"""
    model_id: str
    role: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: Optional[int]
    stop_reason: Optional[str]


_usage_observers: List[Callable[[ModelUsage], None]] = []


def register_usage_observer(observer: Callable[[ModelUsage], None]) -> None:
    """Register a callable that receives a ModelUsage after each invocation"""
    _usage_observers.append(observer)


def unregister_usage_observer(observer: Callable[[ModelUsage], None]) -> None:
    """Remove a previously registered usage observer"""
    if observer in _usage_observers:
        _usage_observers.remove(observer)


def _notify_usage_observers(usage: ModelUsage) -> None:
    for observer in list(_usage_observers):
        try:
            observer(usage)
        except Exception as e:
            # Usage reporting must never fail a model invocation
            logger.warning("Usage observer failed", error=str(e))


def _converse(client, model_id: str, prompt: str, max_tokens: int,
              temperature: Optional[float] = None,
              top_p: Optional[float] = None,
              role: Optional[str] = None) -> str:
    """Call the Converse API with any bedrock-runtime client; return the text"""
    try:
        inference_config: Dict[str, Any] = {"maxTokens": max_tokens}
        if temperature is not None:
            inference_config["temperature"] = temperature
        if top_p is not None:
            inference_config["topP"] = top_p

        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=inference_config
        )

        usage = response.get('usage', {})
        _notify_usage_observers(ModelUsage(
            model_id=model_id,
            role=role,
            input_tokens=usage.get('inputTokens'),
            output_tokens=usage.get('outputTokens'),
            latency_ms=response.get('metrics', {}).get('latencyMs'),
            stop_reason=response.get('stopReason')
        ))

        # Reasoning models return reasoningContent blocks before the text.
        # As with the previous content[0]['text'] parsing, an empty text
        # block yields "" and a response without any text block is an error.
        content = response['output']['message']['content']
        text_blocks = [block['text'] for block in content if 'text' in block]
        if not text_blocks:
            raise AWSServiceError(
                "Bedrock model returned no text blocks "
                f"(stopReason: {response.get('stopReason')})"
            )
        return "".join(text_blocks)

    except AWSServiceError:
        raise
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ThrottlingException':
            raise BedrockThrottlingException(f"Bedrock throttling: {str(e)}")
        raise AWSServiceError(f"Bedrock model invocation failed: {str(e)}")
    except Exception as e:
        raise AWSServiceError(f"Unexpected error in Bedrock invocation: {str(e)}")


def converse_role(client, role: str, prompt: str, max_tokens: int) -> str:
    """Invoke the model configured for a pipeline role with the given client"""
    config = resolve_model_role(role)
    return _converse(client, config.model_id, prompt, max_tokens,
                     temperature=config.temperature, top_p=config.top_p,
                     role=role)


class AWSClientManager:
    """Centralized AWS client management with configuration and error handling"""
    
    def __init__(self, region: Optional[str] = None):
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        self._clients: Dict[str, Any] = {}
        
        # Common client configuration
        self.config = Config(
            region_name=self.region,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            },
            max_pool_connections=50
        )
    
    def get_bedrock_client(self):
        """Get Bedrock runtime client with optimized configuration"""
        if 'bedrock' not in self._clients:
            self._clients['bedrock'] = boto3.client(
                'bedrock-runtime',
                config=self.config
            )
        return self._clients['bedrock']
    
    def get_bedrock_agent_client(self):
        """Get Bedrock agent runtime client for knowledge base operations"""
        if 'bedrock_agent' not in self._clients:
            self._clients['bedrock_agent'] = boto3.client(
                'bedrock-agent-runtime',
                config=self.config
            )
        return self._clients['bedrock_agent']
    
    def get_textract_client(self):
        """Get Textract client for document processing"""
        if 'textract' not in self._clients:
            self._clients['textract'] = boto3.client(
                'textract',
                config=self.config
            )
        return self._clients['textract']
    
    def get_s3_client(self):
        """Get S3 client for document storage"""
        if 's3' not in self._clients:
            self._clients['s3'] = boto3.client(
                's3',
                config=self.config
            )
        return self._clients['s3']
    
    def get_dynamodb_client(self):
        """Get DynamoDB client for state management"""
        if 'dynamodb' not in self._clients:
            self._clients['dynamodb'] = boto3.client(
                'dynamodb',
                config=self.config
            )
        return self._clients['dynamodb']
    
    def get_dynamodb_resource(self):
        """Get DynamoDB resource for higher-level operations"""
        if 'dynamodb_resource' not in self._clients:
            self._clients['dynamodb_resource'] = boto3.resource(
                'dynamodb',
                config=self.config
            )
        return self._clients['dynamodb_resource']
    
    def get_stepfunctions_client(self):
        """Get Step Functions client for workflow management"""
        if 'stepfunctions' not in self._clients:
            self._clients['stepfunctions'] = boto3.client(
                'stepfunctions',
                config=self.config
            )
        return self._clients['stepfunctions']


class BedrockService:
    """Service wrapper for Amazon Bedrock operations"""
    
    def __init__(self, client_manager: AWSClientManager):
        self.client_manager = client_manager
        self.bedrock_client = client_manager.get_bedrock_client()
        self.bedrock_agent_client = client_manager.get_bedrock_agent_client()
        self.logger = get_logger(self.__class__.__name__)
    
    def invoke_model(self, model_id: str, prompt: str, max_tokens: int = 4000) -> str:
        """Invoke a Bedrock model with the original fixed sampling settings"""
        # Low temperature for consistent medical analysis
        return _converse(self.bedrock_client, model_id, prompt, max_tokens,
                         temperature=0.1, top_p=0.9)

    def invoke_role(self, role: str, prompt: str, max_tokens: int) -> str:
        """Invoke the model configured for a pipeline role"""
        return converse_role(self.bedrock_client, role, prompt, max_tokens)

    def retrieve_from_knowledge_base(self, knowledge_base_id: str, query: str, 
                                   max_results: int = 10, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve information from Bedrock Knowledge Base with optional metadata filtering"""
        try:
            # Build retrieval configuration
            retrieval_config = {
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
            
            # Add metadata filter if provided
            if filter:
                retrieval_config['vectorSearchConfiguration']['filter'] = filter
            
            response = self.bedrock_agent_client.retrieve(
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={
                    'text': query
                },
                retrievalConfiguration=retrieval_config
            )
            return response
            
        except ClientError as e:
            raise AWSServiceError(f"Knowledge base retrieval failed: {str(e)}")


class TextractService:
    """Service wrapper for Amazon Textract operations"""
    
    def __init__(self, client_manager: AWSClientManager):
        self.client_manager = client_manager
        self.textract_client = client_manager.get_textract_client()
        self.logger = get_logger(self.__class__.__name__)
    
    def extract_text_from_document(self, bucket: str, key: str) -> str:
        """Extract text from document stored in S3"""
        
        # Strategy: Try async first (handles both single and multi-page), 
        # then fall back to sync if async fails
        
        # First attempt: Async Textract (handles multi-page documents)
        try:
            self.logger.info(f"Attempting async Textract extraction for: {key}")
            return self._extract_text_async(bucket, key)
            
        except Exception as async_error:
            self.logger.warning(f"Async Textract failed: {str(async_error)}")
            
            # Second attempt: Sync Textract (for single-page documents)
            try:
                self.logger.info(f"Falling back to sync Textract extraction for: {key}")
                response = self.textract_client.detect_document_text(
                    Document={
                        'S3Object': {
                            'Bucket': bucket,
                            'Name': key
                        }
                    }
                )
                
                # Extract text from response
                text_blocks = []
                for block in response.get('Blocks', []):
                    if block['BlockType'] == 'LINE':
                        text_blocks.append(block['Text'])
                
                extracted_text = '\n'.join(text_blocks)
                self.logger.info(f"Sync Textract succeeded, extracted {len(text_blocks)} lines")
                return extracted_text
                
            except ClientError as sync_error:
                self.logger.error(f"Both async and sync Textract failed. Async: {str(async_error)}, Sync: {str(sync_error)}")
                raise AWSServiceError(
                    f"Document text extraction failed. Async error: {str(async_error)}. Sync error: {str(sync_error)}"
                )
    
    def _extract_text_async(self, bucket: str, key: str) -> str:
        """Extract text from multi-page document using async Textract"""
        import time
        
        try:
            # Start async text detection job
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket,
                        'Name': key
                    }
                }
            )
            
            job_id = response['JobId']
            self.logger.info(f"Started async Textract job: {job_id}")
            
            # Poll for job completion (with timeout)
            max_wait_time = 300  # 5 minutes max
            poll_interval = 5    # Check every 5 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                job_response = self.textract_client.get_document_text_detection(JobId=job_id)
                job_status = job_response['JobStatus']
                
                if job_status == 'SUCCEEDED':
                    self.logger.info(f"Async Textract job completed: {job_id}")
                    return self._extract_text_from_async_response(job_id)
                elif job_status == 'FAILED':
                    error_msg = job_response.get('StatusMessage', 'Unknown error')
                    raise AWSServiceError(f"Async Textract job failed: {error_msg}")
                elif job_status in ['IN_PROGRESS']:
                    self.logger.info(f"Textract job in progress: {job_id}, waiting...")
                    time.sleep(poll_interval)  # nosemgrep: arbitrary-sleep
                    elapsed_time += poll_interval
                else:
                    raise AWSServiceError(f"Unexpected job status: {job_status}")
            
            # Timeout reached
            raise AWSServiceError(f"Textract job timed out after {max_wait_time} seconds")
            
        except ClientError as e:
            raise AWSServiceError(f"Async Textract processing failed: {str(e)}")
    
    def _extract_text_from_async_response(self, job_id: str) -> str:
        """Extract text from async Textract job results"""
        text_blocks = []
        next_token = None
        
        try:
            while True:
                # Get results (with pagination)
                if next_token:
                    response = self.textract_client.get_document_text_detection(
                        JobId=job_id,
                        NextToken=next_token
                    )
                else:
                    response = self.textract_client.get_document_text_detection(JobId=job_id)
                
                # Extract text from blocks
                for block in response.get('Blocks', []):
                    if block['BlockType'] == 'LINE':
                        text_blocks.append(block['Text'])
                
                # Check for more pages
                next_token = response.get('NextToken')
                if not next_token:
                    break
            
            extracted_text = '\n'.join(text_blocks)
            self.logger.info(f"Extracted {len(text_blocks)} text lines from async job {job_id}")
            
            return extracted_text
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to retrieve async Textract results: {str(e)}")
    
    def analyze_document(self, bucket: str, key: str, 
                        feature_types: list = None) -> Dict[str, Any]:
        """Analyze document with advanced features"""
        if feature_types is None:
            feature_types = ['TABLES', 'FORMS']
        
        try:
            response = self.textract_client.analyze_document(
                Document={
                    'S3Object': {
                        'Bucket': bucket,
                        'Name': key
                    }
                },
                FeatureTypes=feature_types
            )
            return response
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'UnsupportedDocumentException':
                self.logger.warning(f"Document format not supported for analysis: {key}")
                # Return empty analysis for unsupported formats
                return {'Blocks': []}
            raise AWSServiceError(f"Textract document analysis failed: {str(e)}")
    
    def extract_text_from_multipage_document(self, bucket: str, key: str, 
                                           sns_topic_arn: str, sns_role_arn: str) -> str:
        """Extract text from multi-page document using async Textract (for future use)"""
        # This would implement the async workflow for multi-page documents
        # For now, we'll use the synchronous method and handle errors gracefully
        return self.extract_text_from_document(bucket, key)


class DynamoDBService:
    """Service wrapper for DynamoDB operations"""
    
    def __init__(self, client_manager: AWSClientManager):
        self.client_manager = client_manager
        self.dynamodb = client_manager.get_dynamodb_resource()
        self.logger = get_logger(self.__class__.__name__)
        
        # Get table names from environment
        self.extraction_table_name = os.getenv('EXTRACTION_TABLE', 'prior-auth-extraction-dev')
        self.results_table_name = os.getenv('RESULTS_TABLE', 'prior-auth-results-dev')
        self.extraction_table = self.dynamodb.Table(self.extraction_table_name)
        self.results_table = self.dynamodb.Table(self.results_table_name)
    
    def _convert_for_dynamodb(self, obj):
        """Convert Python types to DynamoDB-compatible types"""
        from datetime import datetime
        
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_for_dynamodb(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_dynamodb(item) for item in obj]
        return obj

    def save_processing_state(self, state: 'ProcessingState') -> None:
        """Save processing state to DynamoDB"""
        try:
            item = {
                'document_id': state.document_id,
                'step_name': state.current_step,
                'status': state.status.value,
                'progress_percentage': state.progress_percentage,
                'error_details': state.error_details,
                'step_results': self._convert_for_dynamodb(state.step_results),
                'created_at': state.created_at.isoformat(),
                'updated_at': state.updated_at.isoformat(),
                'timestamp': state.updated_at.isoformat()  # For GSI
            }
            
            self.extraction_table.put_item(Item=item)
            self.logger.info("Processing state saved", document_id=state.document_id)
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to save processing state: {str(e)}")
    
    def put_item(self, table_name: str, item: Dict[str, Any]) -> None:
        """Generic put_item method for DynamoDB with float-to-Decimal conversion"""
        try:
            # Convert floats to Decimals for DynamoDB compatibility
            converted_item = self._convert_floats_to_decimal(item)
            
            # Get the table
            table = self.dynamodb.Table(table_name)
            table.put_item(Item=converted_item)
            
            self.logger.info("Item saved to DynamoDB", table=table_name)
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to save item to {table_name}: {str(e)}")
    
    def update_item(self, table_name: str, key: Dict[str, Any], 
                   update_expression: str, expression_values: Dict[str, Any]) -> None:
        """Generic update_item method for DynamoDB with float-to-Decimal conversion"""
        try:
            # Convert floats to Decimals
            converted_values = self._convert_floats_to_decimal(expression_values)
            
            # Get the table
            table = self.dynamodb.Table(table_name)
            table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=converted_values
            )
            
            self.logger.info("Item updated in DynamoDB", table=table_name)
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to update item in {table_name}: {str(e)}")
    
    def delete_item(self, table_name: str, key: Dict[str, Any]) -> None:
        """Generic delete_item method for DynamoDB"""
        try:
            # Get the table
            table = self.dynamodb.Table(table_name)
            table.delete_item(Key=key)
            
            self.logger.info("Item deleted from DynamoDB", table=table_name)
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to delete item from {table_name}: {str(e)}")

    def get_processing_state(self, document_id: str, step_name: str) -> Optional[Dict]:
        """Get processing state from DynamoDB"""
        try:
            response = self.extraction_table.get_item(
                Key={
                    'document_id': document_id,
                    'step_name': step_name
                }
            )
            return response.get('Item')
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to get processing state: {str(e)}")
    
    def query_processing_states(self, document_id: str) -> List[Dict]:
        """Query all processing states for a document"""
        try:
            response = self.extraction_table.query(
                KeyConditionExpression='document_id = :doc_id',
                ExpressionAttributeValues={':doc_id': document_id}
            )
            return response.get('Items', [])
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to query processing states: {str(e)}")


class S3Service:
    """Service wrapper for S3 operations"""
    
    def __init__(self, client_manager: AWSClientManager):
        self.client_manager = client_manager
        self.s3_client = client_manager.get_s3_client()
        self.logger = get_logger(self.__class__.__name__)
    
    def upload_document(self, bucket: str, key: str, content: bytes, 
                       content_type: str = 'application/octet-stream') -> str:
        """Upload document to S3 with encryption"""
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
                ServerSideEncryption='AES256'
            )
            
            self.logger.info("Document uploaded to S3", bucket=bucket, key=key)
            return f"s3://{bucket}/{key}"
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to upload document to S3: {str(e)}")
    
    def download_document(self, bucket: str, key: str) -> bytes:
        """Download document from S3"""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to download document from S3: {str(e)}")
    
    def generate_presigned_url(self, bucket: str, key: str, 
                              expiration: int = 3600) -> str:
        """Generate presigned URL for document access"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
            
        except ClientError as e:
            raise AWSServiceError(f"Failed to generate presigned URL: {str(e)}")


# Global client manager instance
_client_manager: Optional[AWSClientManager] = None


def get_aws_client_manager() -> AWSClientManager:
    """Get global AWS client manager instance"""
    global _client_manager
    if _client_manager is None:
        _client_manager = AWSClientManager()
    return _client_manager


def get_bedrock_service() -> BedrockService:
    """Get Bedrock service instance"""
    return BedrockService(get_aws_client_manager())


def get_textract_service() -> TextractService:
    """Get Textract service instance"""
    return TextractService(get_aws_client_manager())


def get_dynamodb_service() -> DynamoDBService:
    """Get DynamoDB service instance"""
    return DynamoDBService(get_aws_client_manager())


def get_s3_service() -> S3Service:
    """Get S3 service instance"""
    return S3Service(get_aws_client_manager())
