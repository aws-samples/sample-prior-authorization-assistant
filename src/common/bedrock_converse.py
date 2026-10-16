"""
Bedrock Converse invocation for pipeline roles.

Kept free of logging configuration at import time, so modules that import it
(for example metadata_extractor) keep their own logging setup.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import structlog
from botocore.exceptions import ClientError

from .exceptions import AWSServiceError, BedrockThrottlingException
from .model_config import resolve as resolve_model_role

# structlog.get_logger does not configure logging
logger = structlog.get_logger(__name__)


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


def converse(
    client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    role: Optional[str] = None,
) -> str:
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
            inferenceConfig=inference_config,
        )

        usage = response.get("usage", {})
        _notify_usage_observers(
            ModelUsage(
                model_id=model_id,
                role=role,
                input_tokens=usage.get("inputTokens"),
                output_tokens=usage.get("outputTokens"),
                latency_ms=response.get("metrics", {}).get("latencyMs"),
                stop_reason=response.get("stopReason"),
            )
        )

        # Same as the previous content[0]['text'] parsing, except that
        # reasoning blocks (from reasoning models) are skipped first.
        content = response["output"]["message"]["content"]
        text_blocks = [block["text"] for block in content if "text" in block]
        if not text_blocks:
            raise AWSServiceError(
                "Bedrock model returned no text blocks "
                f"(stopReason: {response.get('stopReason')})"
            )
        return text_blocks[0]

    except AWSServiceError:
        raise
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ThrottlingException":
            raise BedrockThrottlingException(f"Bedrock throttling: {str(e)}")
        raise AWSServiceError(f"Bedrock model invocation failed: {str(e)}")
    except Exception as e:
        raise AWSServiceError(f"Unexpected error in Bedrock invocation: {str(e)}")


def converse_role(client, role: str, prompt: str, max_tokens: int) -> str:
    """Invoke the model configured for a pipeline role with the given client"""
    config = resolve_model_role(role)
    return converse(
        client,
        config.model_id,
        prompt,
        max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        role=role,
    )
