"""TTS voice-fallback policy validation, resolution, and factory wiring."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.schemas.llm_fallback import TTSFallbackPolicy
from api.schemas.model_connections import ServiceSelection
from api.services.configuration.registry import REGISTRY, ServiceType
from api.services.workflow.agent_runtime import AgentRuntime


def test_only_error_condition_is_accepted():
    rule = TTSFallbackPolicy[ServiceSelection].model_validate(
        {
            "rules": [
                {
                    "condition": {"type": "error"},
                    "target": {
                        "provider_connection_uuid": "11111111-1111-4111-8111-111111111111"
                    },
                }
            ]
        }
    )
    assert rule.rules[0].condition.type == "error"


_VALID_TARGET = {"provider_connection_uuid": "11111111-1111-4111-8111-111111111111"}


@pytest.mark.parametrize(
    "policy,loc",
    [
        (
            {
                "rules": [
                    {
                        "condition": {"type": "no_output", "after_ms": 900},
                        "target": _VALID_TARGET,
                    }
                ]
            },
            ("rules", 0, "condition", "type"),
        ),
        (
            {
                "rules": [
                    {"condition": {"type": "error"}, "target": _VALID_TARGET},
                    {"condition": {"type": "error"}, "target": _VALID_TARGET},
                ]
            },
            ("rules",),
        ),
        ({"version": 2, "rules": []}, ("version",)),
        (
            {
                "rules": [
                    {
                        "condition": {"type": "error"},
                        "extra": True,
                        "target": _VALID_TARGET,
                    }
                ]
            },
            ("rules", 0, "extra"),
        ),
    ],
)
def test_schema_rejects_invalid_policies(policy, loc):
    with pytest.raises(ValidationError) as exc:
        TTSFallbackPolicy[ServiceSelection].model_validate(policy)
    assert exc.value.errors()[0]["loc"] == loc


def tts_configuration(*, error_target=None):
    payload = {
        "llm": {"provider": "google", "api_key": "x", "model": "gemini-3.5-flash"},
        "stt": {"provider": "deepgram", "api_key": "x", "model": "nova-3"},
        "tts": {
            "provider": "openai",
            "api_key": "primary",
            "model": "tts-1",
            "voice": "alloy",
        },
    }
    if error_target:
        payload["tts_fallback"] = {
            "version": 1,
            "rules": [{"condition": {"type": "error"}, "target": error_target}],
        }
    return EffectiveAIModelConfiguration.model_validate(payload)


def test_effective_configuration_rejects_dograh_primary():
    config = tts_configuration(
        error_target={
            "provider": "elevenlabs",
            "api_key": "b",
            "model": "m",
            "voice": "v",
        }
    )
    payload = config.model_dump(exclude_computed_fields=True)
    payload["tts"] = {"provider": "dograh", "api_key": "primary"}
    with pytest.raises(ValidationError, match="non-Dograh primary TTS"):
        EffectiveAIModelConfiguration.model_validate(payload)


def test_effective_configuration_rejects_dograh_target():
    with pytest.raises(ValidationError, match="cannot be a fallback target"):
        tts_configuration(error_target={"provider": "dograh", "api_key": "managed"})


def test_no_rules_returns_plain_tts():
    from api.services.pipecat.service_factory import create_tts_service

    with patch(
        "api.services.pipecat.service_factory.create_single_tts_service"
    ) as create:
        primary = object()
        create.return_value = primary
        config = tts_configuration()
        assert create_tts_service(config, None) is primary
        create.assert_called_once()


def test_rules_build_switcher():
    from pipecat.tests.mock_tts_service import MockTTSService

    from api.services.pipecat.fallback_tts import FallbackTTSService
    from api.services.pipecat.service_factory import create_tts_service

    backup = {"provider": "elevenlabs", "api_key": "b", "model": "m", "voice": "v"}
    with patch(
        "api.services.pipecat.service_factory.create_single_tts_service"
    ) as create:
        primary = MockTTSService(name="primary")
        backup_service = MockTTSService(name="backup")
        create.side_effect = [primary, backup_service]
        config = tts_configuration(error_target=backup)
        result = create_tts_service(config, None)
    assert isinstance(result, FallbackTTSService)
    assert create.call_count == 2
    assert create.call_args_list[1].args[0].tts == config.tts_fallback.rules[0].target
    assert create.call_args_list[1].kwargs["register_campaign_errors"] is False


def test_broken_backup_returns_primary():
    from api.services.pipecat.service_factory import create_tts_service

    backup = {"provider": "elevenlabs", "api_key": "b", "model": "m", "voice": "v"}
    with patch(
        "api.services.pipecat.service_factory.create_single_tts_service"
    ) as create:
        primary = object()
        create.side_effect = [primary, RuntimeError("bad")]
        result = create_tts_service(tts_configuration(error_target=backup), None)
    assert result is primary


def test_runtime_metadata_omits_credentials_and_records_rules():
    from api.services.pipecat.service_factory import get_tts_runtime_configuration

    config = tts_configuration(
        error_target={
            "provider": "elevenlabs",
            "api_key": "secret",
            "model": "eleven_flash_v2_5",
            "voice": "voice-id",
        }
    )
    metadata = get_tts_runtime_configuration(config)
    assert metadata["tts_fallback"]["rules"][0] == {
        "condition": {"type": "error"},
        "target": {
            "provider": "elevenlabs",
            "model": "eleven_flash_v2_5",
            "voice": "voice-id",
        },
    }
    assert "secret" not in str(metadata)
    assert "api_key" not in metadata
    assert "tts_voice" not in metadata


def test_registry_has_no_provider_specific_tts_fallback_fields():
    for cls in REGISTRY[ServiceType.TTS].values():
        assert not any(field.startswith("fallback_") for field in cls.model_fields)


def test_agent_runtime_visit_description_records_tts_fallback():
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.tts = SimpleNamespace(fallback_metrics={"switched": 1})
    runtime.llm = SimpleNamespace()
    runtime.visit_id = "v1"
    runtime.workflow_id = 1
    runtime.workflow_name = "wf"
    runtime.definition_id = "def"
    runtime.runtime_configuration = {}
    runtime.entered_at = None
    runtime.exited_at = None
    runtime.exit_reason = None
    runtime.error = None
    runtime.current_node = None
    description = runtime.describe()
    assert description["tts_fallback_metrics"] == {"switched": 1}


@pytest.mark.asyncio
async def test_tts_fallback_counters_sum_retired_and_active_visits():
    from api.services.workflow.pipecat_engine import PipecatEngine

    engine = PipecatEngine.__new__(PipecatEngine)
    engine._gathered_context = {"existing": "value"}
    engine._agent_visits = [{"tts_fallback_metrics": {"switched": 2}}]
    agent = AgentRuntime(
        visit_id="active",
        workflow_id=1,
        definition_id=1,
        workflow_name="test",
        workflow=SimpleNamespace(),
        llm=SimpleNamespace(),
        inference_llm=None,
        variable_extraction_llm=None,
        tts=SimpleNamespace(fallback_metrics={"switched": 3}),
    )
    engine._active_agent = agent
    first = await engine.get_gathered_context()
    assert first["tts_fallback_metrics"] == {"switched": 5, "backup_unavailable": 0}
    assert await engine.get_gathered_context() == first

    engine._gathered_context["tts_fallback_metrics"] = {"switched": 4}
    assert (await engine.get_gathered_context())["tts_fallback_metrics"] == {
        "switched": 9,
        "backup_unavailable": 0,
    }
    engine._gathered_context["tts_fallback_metrics"] = {"switched": "corrupt"}
    assert (await engine.get_gathered_context())["tts_fallback_metrics"] == {
        "switched": 5,
        "backup_unavailable": 0,
    }


def test_masking_hides_tts_fallback_secrets():
    from api.services.configuration.masking import mask_user_config

    config = tts_configuration(
        error_target={
            "provider": "elevenlabs",
            "api_key": "secret-backup",
            "model": "eleven_flash_v2_5",
            "voice": "voice-id",
        }
    )
    masked = mask_user_config(config)
    dumped = str(masked)
    assert "secret-backup" not in dumped
    assert "***" in dumped


def test_tts_fallback_realtime_validator_allows_policy_on_config():
    payload = tts_configuration(
        error_target={
            "provider": "elevenlabs",
            "api_key": "b",
            "model": "m",
            "voice": "v",
        }
    ).model_dump(exclude_computed_fields=True)
    payload["is_realtime"] = True
    payload["tts"] = None
    config = EffectiveAIModelConfiguration.model_validate(payload)
    assert config.tts_fallback is not None


def test_tts_fallback_pipeline_without_primary_tts_is_rejected():
    payload = tts_configuration(
        error_target={
            "provider": "elevenlabs",
            "api_key": "b",
            "model": "m",
            "voice": "v",
        }
    ).model_dump(exclude_computed_fields=True)
    payload["tts"] = None
    with pytest.raises(ValidationError, match="non-Dograh primary TTS"):
        EffectiveAIModelConfiguration.model_validate(payload)
