"""
Bedrock generation model registry.

Maps each pipeline role to the model ID and sampling settings it uses.
Defaults match the values each call site used before the registry existed.

Overrides (environment variables):
- PA_MODEL_ID_<ROLE>: model ID for one role (takes precedence)
- PA_MODEL_ID: model ID for every role
- PA_TEMPERATURE_<ROLE>, PA_TOP_P_<ROLE>: sampling settings for an
  overridden role

Valid sampling settings depend on the model (some reject temperature and
top_p together, some reject any non-default value), so an overridden role
does not inherit the default role's settings.
"""

import os
from dataclasses import dataclass
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

    return ModelRole(
        model_id,
        temperature=_optional_float(f"PA_TEMPERATURE_{suffix}"),
        top_p=_optional_float(f"PA_TOP_P_{suffix}"),
    )


def _optional_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    return float(value) if value else None
