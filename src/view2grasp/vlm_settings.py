"""Provider settings shared by the robot client and connectivity checks."""
from dataclasses import dataclass, field
import math
import os
from urllib.parse import urlsplit


@dataclass(frozen=True)
class VLMSettings:
    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 120.
    max_tokens: int | None = None
    thinking: str | None = None
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls, *, require_api_key=True):
        provider = os.getenv('VLM_PROVIDER', 'openai-compatible').strip().lower()
        if provider not in ('openai-compatible', 'glm'):
            raise ValueError('VLM_PROVIDER must be openai-compatible or glm')
        glm = provider == 'glm'
        base_url = os.getenv('VLM_BASE_URL',
            'https://open.bigmodel.cn/api/paas/v4' if glm else 'http://127.0.0.1:8318/v1').strip().rstrip('/')
        url = urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('VLM_BASE_URL must be an HTTP(S) base URL without credentials, query or fragment')
        if url.path.endswith('/chat/completions'):
            raise ValueError('VLM_BASE_URL must omit /chat/completions')
        model = os.getenv('VLM_MODEL', 'glm-5.3-flashx' if glm else 'gt-6-as-a').strip()
        if not model:
            raise ValueError('VLM_MODEL must not be empty')
        api_key = os.getenv('VLM_API_KEY') or (os.getenv('ZHIPU_API_KEY', '') if glm else 'local')
        if require_api_key and not api_key.strip():
            raise ValueError('Set ZHIPU_API_KEY or VLM_API_KEY for the GLM API')
        try:
            timeout = float(os.getenv('VLM_TIMEOUT_SECONDS', '120'))
            raw_tokens = os.getenv('VLM_MAX_TOKENS', '4096' if glm else '')
            max_tokens = int(raw_tokens) if raw_tokens else None
        except ValueError:
            raise ValueError('VLM_TIMEOUT_SECONDS must be a positive number; VLM_MAX_TOKENS must be a positive integer') from None
        if not math.isfinite(timeout) or timeout <= 0 or (max_tokens is not None and max_tokens <= 0):
            raise ValueError('VLM_TIMEOUT_SECONDS and VLM_MAX_TOKENS must be finite and positive')
        thinking = os.getenv('VLM_THINKING', 'enabled' if glm else '').strip() or None
        if thinking not in (None, 'enabled', 'disabled'):
            raise ValueError('VLM_THINKING must be enabled or disabled')
        effort = os.getenv('VLM_REASONING_EFFORT', 'low' if glm else '').strip() or None
        if effort not in (None, 'low', 'high', 'max'):
            raise ValueError('VLM_REASONING_EFFORT must be low, high or max')
        return cls(provider, base_url, model, api_key, timeout, max_tokens, thinking, effort)

    def request_options(self):
        options = {}
        if self.max_tokens is not None:
            options['max_tokens'] = self.max_tokens
        if self.thinking is not None:
            options['thinking'] = {'type': self.thinking}
        if self.reasoning_effort is not None:
            options['reasoning_effort'] = self.reasoning_effort
        # GLM vision does not support the text-only response_format field.
        return options

    def public_metadata(self):
        return {'provider': self.provider, 'base_url': self.base_url, 'model': self.model,
                'timeout_seconds': self.timeout_seconds, 'request_options': self.request_options()}

    def redact(self, text):
        # Local compatible servers often use the literal placeholder "local".
        # Only redact configured credentials, never that public default.
        return text.replace(self.api_key, '[REDACTED]') if self.api_key and self.api_key != 'local' else text
