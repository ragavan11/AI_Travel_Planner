"""
llm_client.py
--------------
A tiny provider-agnostic wrapper so the rest of the app can call
`llm.generate(system_prompt, user_prompt)` without caring whether the
underlying model is Google Gemini or an OpenAI GPT model. Switch
providers purely via the LLM_PROVIDER environment variable -- no code
changes needed.
"""

from . import config


class LLMClient:
    def __init__(self):
        self.provider = config.LLM_PROVIDER

        if self.provider == "openai":
            from openai import OpenAI

            if not config.OPENAI_API_KEY:
                raise ValueError(
                    "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set in .env"
                )
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.model = config.OPENAI_MODEL

        elif self.provider == "gemini":
            from google import genai

            if not config.GOOGLE_API_KEY:
                raise ValueError(
                    "LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is not set in .env"
                )
            self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
            self.model = config.GEMINI_MODEL

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{self.provider}'. Use 'openai' or 'gemini'."
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a system + user prompt to the configured LLM and return text."""
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            )
            return response.choices[0].message.content

        # provider == "gemini"
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=config.LLM_TEMPERATURE,
                max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            ),
        )
        return response.text
