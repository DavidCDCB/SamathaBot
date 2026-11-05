"""AI Backend Router — selects and manages AI backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from openocto.ai.base import AIBackend

if TYPE_CHECKING:
    from openocto.config import AIConfig
    from openocto.skills.base import SkillRegistry

logger = logging.getLogger(__name__)


class AIRouter:
    """Routes requests to the configured AI backend."""

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._backends: dict[str, AIBackend] = {}
        self._active_backend_name = config.default_backend

        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize all configured backends."""
        from openocto.ai.openai_compat import OpenAICompatBackend

        errors = []

        # OpenAI-compatible providers (any number)
        for name, provider_config in self._config.providers.items():
            if not provider_config.model:
                continue  # skip unconfigured providers
            if not provider_config.api_key and not provider_config.no_auth:
                continue  # skip providers without auth
            try:
                self._backends[name] = OpenAICompatBackend.from_provider_config(
                    name,
                    provider_config,
                    turn_timeout=self._config.turn_timeout,
                    max_tokens=self._config.max_tokens,
                )
            except Exception as e:
                errors.append(f"{name}: {e}")

        if errors:
            logger.warning("Some backends failed to initialize: %s", "; ".join(errors))

        if not self._backends:
            raise RuntimeError(
                "No AI backends available. Set at least one API key:\n"
                "  export OPENAI_API_KEY=sk-...           (for OpenAI)\n"
                "or configure a compatible provider under ai.providers in config.yaml"
            )

        if self._active_backend_name not in self._backends:
            fallback = next(iter(self._backends))
            logger.warning(
                "Requested backend %r not available, using %r",
                self._active_backend_name,
                fallback,
            )
            self._active_backend_name = fallback

    def get_backend(self, name: str | None = None) -> AIBackend:
        backend_name = name or self._active_backend_name
        if backend_name not in self._backends:
            raise ValueError(f"Backend {backend_name!r} not available")
        return self._backends[backend_name]

    @property
    def active_backend_name(self) -> str:
        return self._active_backend_name

    async def health_check(self) -> tuple[bool, str]:
        """Send a tiny test request to the active backend.

        Returns (ok, message).
        """
        backend = self.get_backend()
        try:
            response = await backend.send(
                [{"role": "user", "content": "Hi"}],
                "Reply with one word: OK",
            )
            if response and response.strip():
                return True, f"{self._active_backend_name} is working"
            return False, f"{self._active_backend_name} returned empty response"
        except Exception as e:
            return False, f"{self._active_backend_name}: {e}"

    async def send_streaming(
        self,
        user_text: str,
        history: list[dict],
        system_prompt: str,
        on_chunk: Callable[[str], Awaitable[None]],
        skills: SkillRegistry | None = None,
    ) -> str:
        """Send a user message and stream the response.

        Args:
            user_text: the current user message.
            history: previous messages (caller loads from DB).
            system_prompt: assembled system prompt (persona + memory context).
            on_chunk: callback for each streamed token.
            skills: optional skill registry for tool-use loop.
        """
        messages = [*history, {"role": "user", "content": user_text}]
        # Defensa: algunos llamadores guardan el mensaje actual en el historial
        # antes de pedir la respuesta. Si el último mensaje del historial es
        # exactamente el mismo, quitarlo para no enviarlo dos veces.
        if (
            len(messages) >= 2
            and messages[-2].get("role") == "user"
            and messages[-2].get("content") == user_text
        ):
            del messages[-2]
        backend = self.get_backend()
        return await backend.send_streaming(
            messages,
            system_prompt,
            on_chunk,
            skills=skills,
        )
