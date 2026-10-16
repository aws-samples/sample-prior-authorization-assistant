"""Unit tests for PolicyIndexingService.

The Bedrock Knowledge Base embeds policy content itself during its ingestion
job, so indexing must not call an embedding model directly.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from knowledge import policy_ingestion

KB_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"


@pytest.fixture
def indexing_service(monkeypatch):
    monkeypatch.setenv("POLICIES_BUCKET", "test-policies-bucket")

    bedrock_service = MagicMock()
    s3_service = MagicMock()
    metadata = MagicMock()
    metadata.to_dict.return_value = {"policy_id": "policy-123", "payor": "Aetna"}
    metadata.payor = "Aetna"
    metadata.policy_title = "Physical Therapy"

    with (
        patch.object(
            policy_ingestion, "get_bedrock_service", return_value=bedrock_service
        ),
        patch.object(policy_ingestion, "get_s3_service", return_value=s3_service),
        patch.object(policy_ingestion, "audit_logger"),
        patch.object(policy_ingestion, "PolicyMetadataManager") as manager_cls,
    ):
        manager_cls.return_value.create_policy_metadata.return_value = metadata
        service = policy_ingestion.PolicyIndexingService()
        yield service, bedrock_service, s3_service


def _multi_chunk_policy():
    sentence = (
        "Physical therapy requires documented functional limitation and a plan of care"
    )
    return {"policy_id": "policy-123", "content": ". ".join([sentence] * 40) + "."}


def test_indexing_makes_no_embedding_calls(indexing_service):
    service, bedrock_service, _ = indexing_service

    policy = _multi_chunk_policy()

    result = service.index_policy_document(policy)

    assert bedrock_service.mock_calls == []
    expected_chunks = len(service._chunk_policy_content(policy["content"]))
    assert expected_chunks > 1
    assert result["chunks_count"] == expected_chunks


def test_indexing_succeeds_without_embedding_model_access(indexing_service):
    service, bedrock_service, _ = indexing_service
    bedrock_service.bedrock_client.invoke_model.side_effect = Exception(
        "AccessDeniedException"
    )

    policy = _multi_chunk_policy()

    result = service.index_policy_document(policy)

    expected_chunks = len(service._chunk_policy_content(policy["content"]))
    assert result["chunks_count"] == expected_chunks


def test_index_metadata_reports_kb_embedding_model(indexing_service):
    service, _, s3_service = indexing_service

    result = service.index_policy_document(_multi_chunk_policy())

    s3_service.s3_client.put_object.assert_called_once()
    put_kwargs = s3_service.s3_client.put_object.call_args.kwargs
    assert put_kwargs["Key"] == "policies/metadata/policy-123_index.json"
    persisted = json.loads(put_kwargs["Body"])

    for index_info in (persisted, result):
        assert index_info["embedding_model"] == KB_EMBEDDING_MODEL
        assert {
            "policy_id",
            "metadata",
            "chunks_count",
            "indexed_at",
        } <= index_info.keys()
