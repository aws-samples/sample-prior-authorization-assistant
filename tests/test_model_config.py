"""Unit tests for the Bedrock model registry."""

import os

import pytest

from common import model_config
from common.model_config import ModelRole, resolve

NOVA_PRO = "amazon.nova-pro-v1:0"
SONNET_4_6 = "us.anthropic.claude-sonnet-4-6"

NOVA_ROLES = [
    "medical_extraction",
    "policy_selection",
    "policy_criteria_extraction",
    "compliance_assessment",
    "evidence_mapping",
    "gap_analysis",
]
SONNET_SAMPLED_ROLES = ["clinical_criteria_extraction", "policy_metadata_extraction"]
ALL_ROLES = NOVA_ROLES + SONNET_SAMPLED_ROLES + ["kb_generation"]


@pytest.fixture(autouse=True)
def clear_overrides(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("PA_MODEL_ID", "PA_TOP_P_")):
            monkeypatch.delenv(name)


def test_registry_covers_exactly_the_pipeline_roles():
    assert set(model_config.DEFAULTS) == set(ALL_ROLES)


@pytest.mark.parametrize("role", NOVA_ROLES)
def test_nova_pro_role_defaults(role):
    assert resolve(role) == ModelRole(NOVA_PRO, temperature=0.1, top_p=0.9)


@pytest.mark.parametrize("role", SONNET_SAMPLED_ROLES)
def test_sonnet_role_defaults(role):
    assert resolve(role) == ModelRole(SONNET_4_6, temperature=0.0)


def test_kb_generation_default_has_no_sampling_settings():
    assert resolve("kb_generation") == ModelRole(SONNET_4_6)


def test_unknown_role_is_rejected():
    with pytest.raises(KeyError, match="not_a_role"):
        resolve("not_a_role")


def test_role_override_wins_over_global(monkeypatch):
    monkeypatch.setenv("PA_MODEL_ID", "model-a")
    monkeypatch.setenv("PA_MODEL_ID_GAP_ANALYSIS", "model-b")

    assert resolve("gap_analysis").model_id == "model-b"
    for role in ALL_ROLES:
        if role != "gap_analysis":
            assert resolve(role).model_id == "model-a"


def test_empty_override_is_ignored(monkeypatch):
    monkeypatch.setenv("PA_MODEL_ID", "")
    monkeypatch.setenv("PA_MODEL_ID_GAP_ANALYSIS", "")

    assert resolve("gap_analysis") == model_config.DEFAULTS["gap_analysis"]


def test_override_drops_default_top_p_and_keeps_temperature(monkeypatch):
    monkeypatch.setenv("PA_MODEL_ID_GAP_ANALYSIS", "model-b")

    assert resolve("gap_analysis") == ModelRole("model-b", temperature=0.1)


def test_override_with_explicit_top_p(monkeypatch):
    monkeypatch.setenv("PA_MODEL_ID_GAP_ANALYSIS", "model-b")
    monkeypatch.setenv("PA_TOP_P_GAP_ANALYSIS", "0.8")

    assert resolve("gap_analysis") == ModelRole("model-b", temperature=0.1, top_p=0.8)


def test_override_equal_to_default_keeps_full_defaults(monkeypatch):
    monkeypatch.setenv("PA_MODEL_ID_GAP_ANALYSIS", NOVA_PRO)

    assert resolve("gap_analysis") == ModelRole(NOVA_PRO, temperature=0.1, top_p=0.9)
