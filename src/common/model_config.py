"""
Bedrock generation model registry.

Maps each pipeline role to the model ID and sampling settings it uses.
Defaults match the values each call site used before the registry existed.

Overrides (environment variables):
- PA_MODEL_ID_<ROLE>: model ID for one role (takes precedence)
- PA_MODEL_ID: model ID for every role
- PA_TOP_P_<ROLE>: top_p for an overridden role (the default top_p is
  dropped when a role's model is overridden, since some models reject
  temperature and top_p together)
"""

import os
from dataclasses import dataclass, replace
from typing import Dict, Optional

NOVA_PRO = "amazon.nova-pro-v1:0"
CLAUDE_SONNET_4_6 = "us.anthropic.claude-sonnet-4-6"


@dataclass(frozen=True)
class ModelRole:
    """Model and sampling settings for one pipeline role"""

    model_id: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None


_NOVA_DEFAULT = ModelRole(NOVA_PRO, temperature=0.1, top_p=0.9)
_SONNET_DEFAULT = ModelRole(CLAUDE_SONNET_4_6, temperature=0.0)

DEFAULTS: Dict[str, ModelRole] = {
    "medical_extraction": _NOVA_DEFAULT,
    "policy_selection": _NOVA_DEFAULT,
    "policy_criteria_extraction": _NOVA_DEFAULT,
    "compliance_assessment": _NOVA_DEFAULT,
    "evidence_mapping": _NOVA_DEFAULT,
    "gap_analysis": _NOVA_DEFAULT,
    "clinical_criteria_extraction": _SONNET_DEFAULT,
    "policy_metadata_extraction": _SONNET_DEFAULT,
    "kb_generation": ModelRole(CLAUDE_SONNET_4_6),
}


def resolve(role: str) -> ModelRole:
    """Resolve the model and sampling settings for a pipeline role"""
    if role not in DEFAULTS:
        raise KeyError(f"Unknown model role: {role}")

    default = DEFAULTS[role]
    suffix = role.upper()
    model_id = os.getenv(f"PA_MODEL_ID_{suffix}") or os.getenv("PA_MODEL_ID")

    if not model_id or model_id == default.model_id:
        return default

    top_p = os.getenv(f"PA_TOP_P_{suffix}")
    return replace(default, model_id=model_id, top_p=float(top_p) if top_p else None)
