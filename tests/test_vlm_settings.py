import json

import pytest

from view2grasp.vlm_settings import VLMSettings


ENVIRONMENT_KEYS = (
    "VLM_PROVIDER", "VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY", "ZHIPU_API_KEY",
    "VLM_TIMEOUT_SECONDS", "VLM_MAX_TOKENS", "VLM_THINKING", "VLM_REASONING_EFFORT",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_compatible_defaults_preserve_existing_local_service():
    settings = VLMSettings.from_env()
    assert settings.provider == "openai-compatible"
    assert settings.base_url == "http://127.0.0.1:8318/v1"
    assert settings.model == "gt-6-as-a"
    assert settings.api_key == "local"
    assert settings.timeout_seconds == 120
    assert settings.reasoning_effort is None
    assert settings.request_options() == {}


def test_glm_defaults_use_official_vision_endpoint_and_key_alias(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    monkeypatch.setenv("ZHIPU_API_KEY", "glm-test-credential")
    settings = VLMSettings.from_env()
    assert settings.provider == "glm"
    assert settings.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert settings.model == "glm-5.3-flashx"
    assert settings.api_key == "glm-test-credential"
    assert settings.reasoning_effort == "low"
    assert settings.request_options() == {
        "max_tokens": 4096, "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }
    assert "response_format" not in settings.request_options()


def test_explicit_overrides_take_precedence_and_normalize_base_url(monkeypatch):
    values = {
        "VLM_PROVIDER": " GLM ",
        "VLM_BASE_URL": " https://example.test/custom/v1/// ",
        "VLM_MODEL": " alternate-vision-model ",
        "VLM_API_KEY": "preferred-credential",
        "ZHIPU_API_KEY": "fallback-credential",
        "VLM_TIMEOUT_SECONDS": "7.5",
        "VLM_MAX_TOKENS": "512",
        "VLM_THINKING": "enabled",
        "VLM_REASONING_EFFORT": "high",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = VLMSettings.from_env()
    assert settings.provider == "glm"
    assert settings.base_url == "https://example.test/custom/v1"
    assert settings.model == "alternate-vision-model"
    assert settings.api_key == "preferred-credential"
    assert settings.timeout_seconds == 7.5
    assert settings.reasoning_effort == "high"
    assert settings.request_options() == {
        "max_tokens": 512, "thinking": {"type": "enabled"}, "reasoning_effort": "high",
    }


def test_empty_reasoning_effort_override_omits_glm_default(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    monkeypatch.setenv("ZHIPU_API_KEY", "glm-test-credential")
    monkeypatch.setenv("VLM_REASONING_EFFORT", "")
    settings = VLMSettings.from_env()
    assert settings.reasoning_effort is None
    assert "reasoning_effort" not in settings.request_options()
    assert settings.request_options() == {"max_tokens": 4096, "thinking": {"type": "enabled"}}


def test_glm_requires_a_key_for_live_calls_but_not_replay(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    with pytest.raises(ValueError, match="ZHIPU_API_KEY or VLM_API_KEY"):
        VLMSettings.from_env()
    settings = VLMSettings.from_env(require_api_key=False)
    assert settings.api_key == ""
    assert settings.provider == "glm"


def test_compatible_provider_does_not_adopt_a_glm_key(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "unrelated-glm-key")
    assert VLMSettings.from_env().api_key == "local"


@pytest.mark.parametrize("name,value", [
    ("VLM_TIMEOUT_SECONDS", "0"),
    ("VLM_TIMEOUT_SECONDS", "-1"),
    ("VLM_TIMEOUT_SECONDS", "nan"),
    ("VLM_TIMEOUT_SECONDS", "inf"),
    ("VLM_TIMEOUT_SECONDS", "invalid"),
    ("VLM_MAX_TOKENS", "0"),
    ("VLM_MAX_TOKENS", "-1"),
    ("VLM_MAX_TOKENS", "1.5"),
    ("VLM_MAX_TOKENS", "nan"),
])
def test_invalid_numeric_overrides_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="VLM_TIMEOUT_SECONDS.*VLM_MAX_TOKENS"):
        VLMSettings.from_env()


@pytest.mark.parametrize("url", [
    "file:///tmp/service",
    "https://user:password@example.test/v1",
    "https://example.test/v1?api_key=private",
    "https://example.test/v1#private",
    "https://example.test/v1/chat/completions",
])
def test_base_url_rejects_credentials_and_non_base_endpoints(monkeypatch, url):
    monkeypatch.setenv("VLM_BASE_URL", url)
    with pytest.raises(ValueError, match="VLM_BASE_URL"):
        VLMSettings.from_env()


@pytest.mark.parametrize("name,value", [
    ("VLM_PROVIDER", "unsupported"),
    ("VLM_MODEL", " "),
    ("VLM_THINKING", "true"),
    ("VLM_REASONING_EFFORT", "medium"),
])
def test_invalid_provider_model_or_reasoning_option_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        VLMSettings.from_env()


def test_public_metadata_and_repr_do_not_expose_credentials(monkeypatch):
    secret = "private-test-credential"
    monkeypatch.setenv("VLM_API_KEY", secret)
    settings = VLMSettings.from_env()
    metadata = settings.public_metadata()
    assert "api_key" not in metadata
    assert secret not in json.dumps(metadata)
    assert secret not in repr(settings)
    assert "api_key=" not in repr(settings)
    assert settings.redact("Credential: " + secret) == "Credential: [REDACTED]"
    assert "local" in VLMSettings("openai-compatible", "http://localhost/v1", "test", "local").redact("local")
