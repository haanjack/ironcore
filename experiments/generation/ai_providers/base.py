"""Base AI provider interface and result dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result from AI code generation.

    Attributes:
        code: Generated kernel code
        model: Model identifier used for generation
        tokens_used: Total tokens consumed (input + output)
        finish_reason: Why generation stopped (e.g., "stop", "length", "error")
        raw_response: Raw API response for debugging
    """
    code: str
    model: str
    tokens_used: int
    finish_reason: str
    raw_response: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "raw_response": self.raw_response,
        }


class BaseProvider(ABC):
    """Abstract base class for AI code generation providers.

    Implementations:
        - AnthropicProvider: Claude API
        - OpenAIProvider: OpenAI API
        - GLMProvider: Zhipu GLM API
    """

    def __init__(self, api_key: str, model: str):
        """Initialize provider.

        Args:
            api_key: API key for authentication
            model: Model identifier (e.g., "claude-opus-4-20250514")
        """
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available.

        Returns:
            True if provider package is installed and API key is valid, False otherwise
        """
        ...

    @abstractmethod
    def generate_code(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate kernel code from prompt.

        Args:
            prompt: Generation prompt with kernel specification
            **kwargs: Additional provider-specific parameters

        Returns:
            GenerationResult with generated code and metadata
        """
        ...

    def _validate_api_key(self) -> bool:
        """Validate that API key is non-empty.

        Returns:
            True if API key is set, False otherwise
        """
        return bool(self.api_key and self.api_key.strip())
