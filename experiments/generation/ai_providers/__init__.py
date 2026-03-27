"""AI provider registry for code generation.

This module provides a unified interface for multiple AI providers through
OpenAI-compatible APIs and native SDKs.

## Supported Providers

### OpenAI (Native)
```bash
export OPENAI_API_KEY="sk-..."
ironcore generate rmsnorm --provider openai --model gpt-4o
```

### Anthropic Claude (Native SDK)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
ironcore generate rmsnorm --provider anthropic --model claude-opus-4-20250514
```

### Zhipu GLM (OpenAI-Compatible)
```bash
export OPENAI_API_KEY="your-glm-api-key"
ironcore generate rmsnorm --provider openai --model glm-4.7 --base-url "https://api.z.ai/api/paas/v4/"
```

### Moonshot Kimi (OpenAI-Compatible)
```bash
export OPENAI_API_KEY="your-kimi-api-key"
ironcore generate rmsnorm --provider openai --model moonshot-v1-8k --base-url "https://api.moonshot.cn/v1"
```

### Local vLLM Server (OpenAI-Compatible)
```bash
export OPENAI_API_KEY="dummy"  # vLLM may not require a key
ironcore generate rmsnorm --provider openai --model llama-3-70b --base-url "http://localhost:8000/v1/"
```

### Ollama (OpenAI-Compatible)
```bash
export OPENAI_API_KEY="dummy"
ironcore generate rmsnorm --provider openai --model llama3 --base-url "http://localhost:11434/v1/"
```

## Provider Aliases

The CLI supports convenient aliases for common providers:
- `--provider glm` → Uses OpenAI provider with GLM base URL
- `--provider kimi` → Uses OpenAI provider with Kimi base URL
- `--provider vllm` → Uses OpenAI provider with localhost vLLM

## Programmatic Usage

```python
from experiments.generation.ai_providers import get_provider, list_providers

# List available providers
print(list_providers())  # ["anthropic", "openai"]

# Get provider instance
provider = get_provider("openai",
                       api_key="your-key",
                       model="glm-4.7",
                       base_url="https://api.z.ai/api/paas/v4/")

# Generate code
result = provider.generate_code(prompt)
print(result.code)
```
"""

import os
from experiments.generation.ai_providers.base import BaseProvider, GenerationResult
from experiments.generation.ai_providers.anthropic import AnthropicProvider
from experiments.generation.ai_providers.openai import OpenAIProvider

__all__ = [
    "BaseProvider",
    "GenerationResult",
    "AnthropicProvider",
    "OpenAIProvider",
    "get_provider",
    "list_providers",
    "register_provider",
    "PROVIDER_PRESETS",
]

# Core providers that use native SDKs
_PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}

# Provider presets for common AI services using OpenAI-compatible APIs
# Format: provider_name -> (base_url, api_key_env, default_model)
PROVIDER_PRESETS: dict[str, tuple[str, str, str]] = {
    # Zhipu GLM
    "glm": ("https://api.z.ai/api/coding/paas/v4", "OPENAI_API_KEY", "glm-5"),
    "glm-5": ("https://api.z.ai/api/coding/paas/v4", "OPENAI_API_KEY", "glm-5"),
    "glm5": ("https://api.z.ai/api/coding/paas/v4", "OPENAI_API_KEY", "glm-5"),
    "glm-4": ("https://api.z.ai/api/coding/paas/v4", "OPENAI_API_KEY", "glm-4"),
    "glm-4.7": ("https://api.z.ai/api/coding/paas/v4", "OPENAI_API_KEY", "glm-4.7"),
    "glm-4-air": ("https://api.z.ai/api/coding/paas/v4", "OPENAI_API_KEY", "glm-4-air"),
    "glm-4-flash": ("https://api.z.ai/api/coding/paas/v4", "OPENAI_API_KEY", "glm-4-flash"),

    # Moonshot Kimi
    "kimi": ("https://api.moonshot.cn/v1", "OPENAI_API_KEY", "moonshot-v1-8k"),
    "moonshot": ("https://api.moonshot.cn/v1", "OPENAI_API_KEY", "moonshot-v1-8k"),
    "moonshot-v1-8k": ("https://api.moonshot.cn/v1", "OPENAI_API_KEY", "moonshot-v1-8k"),
    "moonshot-v1-32k": ("https://api.moonshot.cn/v1", "OPENAI_API_KEY", "moonshot-v1-32k"),
    "moonshot-v1-128k": ("https://api.moonshot.cn/v1", "OPENAI_API_KEY", "moonshot-v1-128k"),

    # Local servers
    "vllm": ("http://localhost:8000/v1", "OPENAI_API_KEY", "dummy-model"),
    "ollama": ("http://localhost:11434/v1", "OPENAI_API_KEY", "llama3"),
}


def register_provider(name: str, provider_cls: type[BaseProvider]) -> None:
    """Register a new AI provider.

    Args:
        name: Provider name (e.g., "anthropic", "openai")
        provider_cls: Provider class that inherits from BaseProvider
    """
    _PROVIDER_REGISTRY[name] = provider_cls


def resolve_provider_alias(provider_name: str) -> tuple[str, str | None, str]:
    """Resolve provider alias to (actual_provider, base_url, default_model).

    Args:
        provider_name: Provider name or alias (e.g., "glm", "kimi", "openai")

    Returns:
        Tuple of (actual_provider_name, base_url_or_None, default_model)

    Examples:
        >>> resolve_provider_alias("glm")
        ('openai', 'https://api.z.ai/api/paas/v4/', 'glm-4.7')
        >>> resolve_provider_alias("anthropic")
        ('anthropic', None, '')
        >>> resolve_provider_alias("openai")
        ('openai', None, '')
    """
    if provider_name in _PROVIDER_REGISTRY:
        return provider_name, None, ""
    if provider_name in PROVIDER_PRESETS:
        base_url, api_key_env, default_model = PROVIDER_PRESETS[provider_name]
        return "openai", base_url, default_model
    raise ValueError(f"Unknown provider '{provider_name}'. "
                    f"Available: {', '.join(sorted(list(_PROVIDER_REGISTRY.keys()) + list(PROVIDER_PRESETS.keys())))}")


def get_provider(name: str, **kwargs) -> BaseProvider:
    """Get a provider instance by name.

    Args:
        name: Provider name (anthropic, openai) or alias (glm, kimi, vllm, etc.)
        **kwargs: Additional arguments passed to provider constructor
                  (api_key, model, base_url, etc.)

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is unknown

    Examples:
        # Direct OpenAI
        provider = get_provider("openai", api_key="sk-...", model="gpt-4o")

        # GLM via alias (auto-configures base_url)
        provider = get_provider("glm", api_key="your-key")

        # Kimi via alias
        provider = get_provider("kimi", api_key="your-key", model="moonshot-v1-32k")

        # Local vLLM
        provider = get_provider("vllm", api_key="dummy", base_url="http://localhost:8000/v1")
    """
    provider_name, base_url, default_model = resolve_provider_alias(name)

    if provider_name not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")

    provider_cls = _PROVIDER_REGISTRY[provider_name]

    # Set base_url from preset if not explicitly provided
    if base_url and "base_url" not in kwargs:
        kwargs["base_url"] = base_url

    # Set default model if not explicitly provided
    if default_model and "model" not in kwargs:
        kwargs["model"] = default_model

    return provider_cls(**kwargs)


def list_providers() -> list[str]:
    """List all available provider names and aliases.

    Returns:
        Sorted list of provider names and aliases
    """
    return sorted(list(_PROVIDER_REGISTRY.keys()) + list(PROVIDER_PRESETS.keys()))


def list_provider_presets() -> dict[str, dict[str, str]]:
    """Get detailed information about provider presets.

    Returns:
        Dictionary mapping preset names to their configuration
    """
    return {
        name: {
            "base_url": url,
            "api_key_env": api_key_env,
            "default_model": model,
        }
        for name, (url, api_key_env, model) in PROVIDER_PRESETS.items()
    }
