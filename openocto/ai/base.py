"""Abstract base for AI backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from openocto.skills.base import SkillRegistry


class AIBackend(ABC):
    """Abstract AI backend."""

    @abstractmethod
    async def send(self, messages: list[dict], system_prompt: str) -> str:
        """Send messages and return a complete response, sin herramientas.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts
            system_prompt: System prompt for this request

        Returns:
            Complete response text
        """

    @abstractmethod
    async def send_streaming(
        self,
        messages: list[dict],
        system_prompt: str,
        on_chunk: Callable[[str], Awaitable[None]],
        skills: SkillRegistry | None = None,
    ) -> str:
        """Send messages and stream the response.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            on_chunk: Async callback called for each text chunk
            skills: Optional skill registry; if set, the backend runs a
                streaming tool-use loop and dispatches tool calls to it

        Returns:
            Complete response text (concatenated chunks)
        """
