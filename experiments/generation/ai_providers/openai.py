"""OpenAI API provider for code generation."""

import os

from experiments.generation.ai_providers.base import BaseProvider, GenerationResult


class OpenAIProvider(BaseProvider):
    """OpenAI API implementation.

    Supports models:
        - gpt-4o (most capable)
        - gpt-4o-mini (faster, cheaper)
        - o1 (reasoning model)
        - GLM models via OpenAI-compatible API (e.g., https://api.z.ai/api/paas/v4/)
    """

    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_TEMPERATURE = 0.0

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or compatible API key)
            model: Model name to use
            base_url: Optional custom base URL (e.g., for GLM API)
        """
        super().__init__(api_key, model)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    def is_available(self) -> bool:
        """Check if OpenAI package is installed and API key is valid."""
        if not self._validate_api_key():
            return False

        try:
            import openai
        except ImportError:
            return False

        try:
            if self.base_url:
                client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                client = openai.OpenAI(api_key=self.api_key)
            return True
        except Exception:
            return False

    def generate_code(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate code using OpenAI API.

        Args:
            prompt: Code generation prompt
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            GenerationResult with generated code
        """
        try:
            import openai
        except ImportError:
            return GenerationResult(
                code="",
                model=self.model,
                tokens_used=0,
                finish_reason="error",
                raw_response="openai package not installed. Install with: pip install openai",
            )

        # Merge kwargs with defaults
        max_tokens = kwargs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)

        try:
            if self.base_url:
                client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert GPU kernel developer. Generate only the kernel code without explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract code from response
            code = self._extract_code(response)

            # Count tokens
            tokens_used = response.usage.total_tokens

            return GenerationResult(
                code=code,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=response.choices[0].finish_reason,
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
        """Extract code from OpenAI response.

        Handles responses that may include markdown code blocks.
        Also handles GLM's extended reasoning format where reasoning_content contains the actual content.

        Args:
            response: OpenAI API response object

        Returns:
            Extracted code string
        """
        message = response.choices[0].message

        # GLM uses extended reasoning format - check reasoning_content if content is empty
        content = message.content or ""
        reasoning_content = getattr(message, 'reasoning_content', None) or ""

        # Prefer reasoning_content if content is empty (GLM extended reasoning)
        if not content and reasoning_content:
            content = reasoning_content

        # Check if response is wrapped in markdown code blocks
        lines = content.split("\n")

        # Find code block boundaries
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("```") and start_idx == 0:
                start_idx = i + 1
                # Only skip next line if language identifier is NOT on the same line
                # Language identifier on same line looks like: ```python, ```python, etc.
                has_lang_on_same_line = len(stripped) > 3 and stripped[3:4].isalpha()
                if not has_lang_on_same_line:
                    # Language identifier is on next line, skip it
                    if start_idx < len(lines) and lines[start_idx].strip() and not lines[start_idx].strip().startswith("```"):
                        start_idx += 1
            elif stripped.startswith("```") and start_idx > 0:
                end_idx = i
                break

        # Extract code
        if start_idx > 0:
            code_lines = lines[start_idx:end_idx]
        else:
            # No code blocks found - check if response looks like pure code
            # (starts with import or similar)
            if any(line.strip().startswith(("import", "from", "def", "class", "#"))
                   for line in lines[:10]):
                code_lines = lines
            else:
                # Return empty if response doesn't look like code
                return ""

        code = "\n".join(code_lines).strip()

        return code
