"""Zhipu GLM API provider for code generation."""

from experiments.generation.ai_providers.base import BaseProvider, GenerationResult


class GLMProvider(BaseProvider):
    """Zhipu GLM API implementation.

    Supports models:
        - glm-4-plus (most capable)
        - glm-4 (balanced)
        - glm-4-flash (fastest)
    """

    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_TEMPERATURE = 0.0

    def is_available(self) -> bool:
        """Check if Zhipu package is installed and API key is valid."""
        if not self._validate_api_key():
            return False

        try:
            import zhipuai
        except ImportError:
            return False

        try:
            client = zhipuai.ZhipuAI(api_key=self.api_key)
            return True
        except Exception:
            return False

    def generate_code(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate code using GLM API.

        Args:
            prompt: Code generation prompt
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            GenerationResult with generated code
        """
        try:
            import zhipuai
        except ImportError:
            return GenerationResult(
                code="",
                model=self.model,
                tokens_used=0,
                finish_reason="error",
                raw_response="zhipuai package not installed. Install with: pip install zhipuai",
            )

        # Merge kwargs with defaults
        max_tokens = kwargs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)

        try:
            client = zhipuai.ZhipuAI(api_key=self.api_key)

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

            # Map finish reason
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "stop":
                finish_reason = "stop"
            else:
                finish_reason = str(finish_reason)

            return GenerationResult(
                code=code,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
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
        """Extract code from GLM response.

        Handles responses that may include markdown code blocks.

        Args:
            response: GLM API response object

        Returns:
            Extracted code string
        """
        content = response.choices[0].message.content

        # Check if response is wrapped in markdown code blocks
        lines = content.split("\n")

        # Find code block boundaries
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if line.strip().startswith("```") and start_idx == 0:
                start_idx = i + 1
                # Skip language identifier
                if start_idx < len(lines) and lines[start_idx].strip() and not lines[start_idx].strip().startswith("```"):
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

        return code
