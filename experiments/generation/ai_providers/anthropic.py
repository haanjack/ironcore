"""Anthropic Claude API provider for code generation."""

import sys
from experiments.generation.ai_providers.base import BaseProvider, GenerationResult


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API implementation.

    Supports models:
        - claude-opus-4-20250514 (most capable)
        - claude-sonnet-4-20250514 (balanced)
        - claude-haiku-4-20250514 (fastest)
    """

    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_TEMPERATURE = 0.0

    def is_available(self) -> bool:
        """Check if Anthropic package is installed and API key is valid."""
        if not self._validate_api_key():
            return False

        try:
            import anthropic
        except ImportError:
            return False

        # Optional: validate API key with a lightweight call
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            # Just check if we can create a client - actual validation happens on first call
            return True
        except Exception:
            return False

    def generate_code(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate code using Claude API.

        Args:
            prompt: Code generation prompt
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            GenerationResult with generated code
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            return GenerationResult(
                code="",
                model=self.model,
                tokens_used=0,
                finish_reason="error",
                raw_response="anthropic package not installed. Install with: pip install anthropic",
            )

        # Merge kwargs with defaults
        max_tokens = kwargs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)

        try:
            client = Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract code from response
            code = self._extract_code(response)

            # Count tokens
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return GenerationResult(
                code=code,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=response.stop_reason,
                raw_response=str(response),
            )

        except Exception as e:
            return GenerationResult(
                code="",
                model=self.model,
                tokens_used=0,
                finish_reason="error",
                raw_response=str(e),
            )

    def _extract_code(self, response) -> str:
        """Extract code from Claude response.

        Handles responses that may include markdown code blocks.

        Args:
            response: Anthropic API response object

        Returns:
            Extracted code string
        """
        content = response.content[0].text

        # Check if response is wrapped in markdown code blocks
        lines = content.split("\n")

        # Find code block boundaries
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if line.strip().startswith("```") and start_idx == 0:
                start_idx = i + 1
                # Skip language identifier
                if lines[start_idx].strip() and not lines[start_idx].strip().startswith("```"):
                    start_idx += 1
            elif line.strip().startswith("```") and start_idx > 0:
                end_idx = i
                break

        # Extract code
        if start_idx > 0:
            code_lines = lines[start_idx:end_idx]
        else:
            code_lines = lines

        code = "\n".join(code_lines).strip()

        # Remove any trailing whitespace
        return code
