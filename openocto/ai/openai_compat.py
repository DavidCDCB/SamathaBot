"""OpenAI-compatible AI backend for any provider.

Works with OpenAI, OpenCode Go, Z.AI, Gemini, DeepSeek, Ollama, or any
provider implementing the OpenAI chat completions API format.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import openai

from openocto.ai.base import AIBackend

if TYPE_CHECKING:
    from openocto.config import ProviderConfig
    from openocto.skills.base import SkillRegistry

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 6
# Para voz no queremos que el modelo encadene búsquedas indefinidamente.
# Tras dos consultas de investigación se fuerza una respuesta sin tools.
_MAX_RESEARCH_CALLS = 2
_RESEARCH_TOOLS = {"web_search", "wikipedia"}
# Espera del cliente para peticiones no interactivas (resumen de memoria y
# comprobación de arranque). Los turnos de voz usan turn_timeout.
_CLIENT_TIMEOUT = 120.0
_TOOL_LOOP_FALLBACK = (
    "No encontré una respuesta concluyente con lo que pude consultar. "
    "¿Me lo preguntas de otra forma?"
)


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Deja solo campos del protocolo OpenAI.

    `HistoryStore` devuelve también `id` y `created_at`. Algunos adaptadores
    (GPT-5.6 Luna en Go) interpretan `id` como estado de otro protocolo y
    responden 400: "cannot preserve this message field across protocols".
    """
    limpios: list[dict] = []
    for mensaje in messages:
        item = {
            "role": mensaje.get("role", "user"),
            "content": mensaje.get("content", ""),
        }
        # Campos válidos cuando el historial incluye herramientas.
        for clave in ("name", "tool_calls", "tool_call_id"):
            if clave in mensaje:
                item[clave] = mensaje[clave]
        limpios.append(item)
    return limpios


class OpenAICompatBackend(AIBackend):
    """AI backend for any OpenAI-compatible API."""

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str = "",
        no_auth: bool = False,
        headers: dict[str, str] | None = None,
        reasoning_effort: str = "",
        extra_body: dict[str, Any] | None = None,
        turn_timeout: float = 30.0,
        max_tokens: int | None = None,
    ) -> None:
        if not api_key and not no_auth:
            raise ValueError(
                f"API key for provider {name!r} is not set. "
                f"Set it in config or export the env var."
            )
        self._client = openai.AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url or None,
            default_headers=headers or None,
            timeout=_CLIENT_TIMEOUT,
            max_retries=1,
        )
        self._model = model
        self._name = name
        self._reasoning_effort = reasoning_effort
        self._extra_body = extra_body or {}
        self._turn_timeout = turn_timeout
        self._max_tokens = max_tokens
        logger.info(
            "%s backend initialized: %s (base_url=%s, reasoning_effort=%s)",
            name,
            model,
            base_url,
            reasoning_effort or "-",
        )

    def _gen_kwargs(self) -> dict:
        """Opciones de generación según la config. Vacío = no tocar."""
        kw: dict = {}
        if self._reasoning_effort:
            kw["reasoning_effort"] = self._reasoning_effort
        if self._extra_body:
            kw["extra_body"] = dict(self._extra_body)
        return kw

    def _turn_kwargs(self) -> dict:
        """Opciones de un turno de voz: tiempo máximo de espera y tope de tokens."""
        kw = self._gen_kwargs()
        kw["timeout"] = self._turn_timeout
        if self._max_tokens:
            kw["max_tokens"] = self._max_tokens
        return kw

    @classmethod
    def from_provider_config(
        cls,
        name: str,
        config: ProviderConfig,
        turn_timeout: float = 30.0,
        max_tokens: int | None = None,
    ) -> OpenAICompatBackend:
        return cls(
            name=name,
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            no_auth=config.no_auth,
            headers=config.headers,
            reasoning_effort=config.reasoning_effort,
            extra_body=config.extra_body,
            turn_timeout=turn_timeout,
            max_tokens=max_tokens,
        )

    async def send(self, messages: list[dict], system_prompt: str) -> str:
        """Respuesta completa, sin herramientas (resumen de memoria y arranque)."""
        all_messages = [
            {"role": "system", "content": system_prompt},
            *_sanitize_messages(messages),
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,
            **self._gen_kwargs(),
        )
        text = response.choices[0].message.content or ""
        logger.debug("%s response (%d chars)", self._name, len(text))
        return text

    async def send_streaming(
        self,
        messages: list[dict],
        system_prompt: str,
        on_chunk: Callable[[str], Awaitable[None]],
        skills: SkillRegistry | None = None,
    ) -> str:
        all_messages = [
            {"role": "system", "content": system_prompt},
            *_sanitize_messages(messages),
        ]
        if skills is not None and len(skills) > 0:
            return await self._tool_loop(all_messages, skills, on_chunk)

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,
            stream=True,
            **self._turn_kwargs(),
        )
        parts: list[str] = []
        finish_reason: str | None = None
        async for chunk in stream:
            if (
                not chunk.choices
            ):  # algunos proveedores envían un chunk final sin choices
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta.content
            if delta:
                parts.append(delta)
                await on_chunk(delta)
        response = "".join(parts)
        logger.debug(
            "%s stream response (%d chars, finish=%s)",
            self._name,
            len(response),
            finish_reason,
        )
        if not response.strip():
            logger.warning(
                "%s devolvio contenido vacio (finish=%s); puede ser limite de razonamiento",
                self._name,
                finish_reason,
            )
        return response

    # ── Tool-use loop (en streaming) ───────────────────────────────────

    async def _tool_loop(
        self,
        messages: list[dict[str, Any]],
        skills: SkillRegistry,
        on_chunk: Callable[[str], Awaitable[None]],
    ) -> str:
        """Bucle de herramientas en streaming.

        Cada vuelta pide la respuesta en streaming: el texto va a on_chunk según
        llega. Si el modelo pide herramientas, sus fragmentos (tool_calls) se
        acumulan, se ejecutan y se vuelve a pedir con el resultado. Devuelve
        todo el texto que se dijo en el turno.
        """
        convo: list[dict[str, Any]] = list(messages)
        tools = skills.openai_tools()
        spoken: list[str] = []
        research_calls = 0

        for _ in range(_MAX_TOOL_ITERATIONS):
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=convo,
                tools=tools,
                stream=True,
                **self._turn_kwargs(),
            )
            round_text: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    round_text.append(delta.content)
                    spoken.append(delta.content)
                    await on_chunk(delta.content)
                for tc in delta.tool_calls or []:
                    idx = tc.index if tc.index is not None else len(calls)
                    slot = calls.setdefault(
                        idx, {"id": f"call_{idx}", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        slot["arguments"] += tc.function.arguments or ""

            if not calls:
                return "".join(spoken)

            ordered = [calls[i] for i in sorted(calls)]
            convo.append(
                {
                    "role": "assistant",
                    "content": "".join(round_text),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in ordered
                    ],
                }
            )
            for call in ordered:
                try:
                    args = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                logger.info("Tool call: %s(%s)", call["name"], args)
                if call["name"] in _RESEARCH_TOOLS:
                    research_calls += 1
                result = await skills.call(call["name"], args)
                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )

            if research_calls >= _MAX_RESEARCH_CALLS:
                logger.info(
                    "Research budget reached (%d); forcing final answer",
                    research_calls,
                )
                return await self._final_answer(convo, spoken, on_chunk)

        logger.info("Tool loop hit max iterations (%d)", _MAX_TOOL_ITERATIONS)
        spoken.append(_TOOL_LOOP_FALLBACK)
        await on_chunk(_TOOL_LOOP_FALLBACK)
        return "".join(spoken)

    async def _final_answer(
        self,
        convo: list[dict[str, Any]],
        spoken: list[str],
        on_chunk: Callable[[str], Awaitable[None]],
    ) -> str:
        """Cierra el turno sin herramientas tras agotar la investigación."""
        final_parts: list[str] = []
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=convo,
                stream=True,
                **self._turn_kwargs(),
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    final_parts.append(delta)
                    spoken.append(delta)
                    await on_chunk(delta)
        except Exception:
            logger.exception("Final answer without tools failed")
            spoken.append(_TOOL_LOOP_FALLBACK)
            await on_chunk(_TOOL_LOOP_FALLBACK)
            return "".join(spoken)

        if not "".join(final_parts).strip():
            spoken.append(_TOOL_LOOP_FALLBACK)
            await on_chunk(_TOOL_LOOP_FALLBACK)
        return "".join(spoken)
