"""OpenAI-compatible AI backend for any provider.

Works with OpenAI, OpenCode Go, Z.AI, Gemini, DeepSeek, Ollama, or any
provider implementing the OpenAI chat completions API format.
"""

from __future__ import annotations

import json
import logging
import re
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
# Longitud máxima del texto de respuesta que se registra en el log.
_LOG_RESPONSE_CHARS = 500


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


class _ThinkScrubber:
    """Quita los bloques de razonamiento (etiquetas think) del texto en streaming.

    MiniMax M3.1 suele mandar el razonamiento en ``delta.reasoning_content``
    (que este backend ignora), pero a veces filtra las etiquetas de pensamiento
    dentro del ``content`` normal. Si esas etiquetas llegan al TTS, Edge las
    lee en voz alta ("barra think") y el usuario oye basura.

    Es incremental: una etiqueta puede llegar partida entre deltas. El texto
    que podría ser el inicio de una etiqueta se retiene hasta el siguiente
    delta y se emite o descarta según se confirme.
    """

    _OPEN_RE = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
    _CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
    # Etiquetas literales (en minúsculas) para detectar sufijos incompletos.
    _TAGS = ("<think>", "<thinking>", "</think>", "</thinking>")
    _MAX_PARTIAL = max(len(t) for t in _TAGS) - 1

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, delta: str) -> str:
        """Añade un trozo del stream y devuelve el texto limpio listo para usar."""
        self._buf += delta
        return self._consume(final=False)

    def flush(self) -> str:
        """Cierra el filtro: emite lo retenido pendiente (o lo tira si era el
        inicio de una etiqueta)."""
        return self._consume(final=True)

    def _held_len(self, text: str) -> int:
        """Longitud del sufijo de `text` que aún podría empezar una etiqueta."""
        bajo = text.lower()
        for k in range(min(len(bajo), self._MAX_PARTIAL), 0, -1):
            if any(etiqueta.startswith(bajo[-k:]) for etiqueta in self._TAGS):
                return k
        return 0

    def _consume(self, final: bool) -> str:
        trozos: list[str] = []
        while True:
            if self._inside:
                m = self._CLOSE_RE.search(self._buf)
                if not m:
                    if final:
                        # Stream truncado dentro del razonamiento: se descarta.
                        self._buf = ""
                    else:
                        retenido = self._held_len(self._buf)
                        self._buf = (
                            self._buf[len(self._buf) - retenido :] if retenido else ""
                        )
                    break
                self._buf = self._buf[m.end() :]
                self._inside = False
                continue

            m_open = self._OPEN_RE.search(self._buf)
            m_close = self._CLOSE_RE.search(self._buf)
            primer = None
            for m in (m_open, m_close):
                if m is not None and (primer is None or m.start() < primer[0].start()):
                    primer = (m, m_open is m)
            if primer is None:
                if final:
                    # Lo que queda es solo el prefijo dudoso de una etiqueta
                    # (todo lo demás ya se emitió en su momento): no es habla.
                    self._buf = ""
                else:
                    retenido = self._held_len(self._buf)
                    corte = len(self._buf) - retenido
                    trozos.append(self._buf[:corte])
                    self._buf = self._buf[corte:]
                break
            m, es_apertura = primer
            trozos.append(self._buf[: m.start()])
            self._buf = self._buf[m.end() :]
            if es_apertura:
                self._inside = True
        return "".join(trozos)

    @classmethod
    def strip(cls, text: str) -> str:
        """Versión no incremental (para respuestas completas, sin streaming)."""
        scrub = cls()
        return scrub.feed(text) + scrub.flush()


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
        text = _ThinkScrubber.strip(response.choices[0].message.content or "")
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
        finish_reason: str | None = None
        if skills is not None and len(skills) > 0:
            response = await self._tool_loop(all_messages, skills, on_chunk)
        else:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=all_messages,
                stream=True,
                **self._turn_kwargs(),
            )
            scrub = _ThinkScrubber()
            parts: list[str] = []
            async for chunk in stream:
                if (
                    not chunk.choices
                ):  # algunos proveedores envían un chunk final sin choices
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta.content
                if delta:
                    limpio = scrub.feed(delta)
                    if limpio:
                        parts.append(limpio)
                        await on_chunk(limpio)
            final = scrub.flush()
            if final:
                parts.append(final)
                await on_chunk(final)
            response = "".join(parts)
            logger.debug(
                "%s stream response (%d chars, finish=%s)",
                self._name,
                len(response),
                finish_reason,
            )
        # Queda SIEMPRE registrado lo que respondió el modelo (hasta ahora solo
        # se logueaba la longitud, y el tool loop directamente nada).
        logger.info(
            "%s respuesta del turno (%d chars): %s",
            self._name,
            len(response),
            response[:_LOG_RESPONSE_CHARS]
            + ("…" if len(response) > _LOG_RESPONSE_CHARS else ""),
        )
        if not response.strip():
            logger.warning(
                "%s devolvio contenido vacio (finish=%s); puede ser limite de razonamiento",
                self._name,
                finish_reason if finish_reason is not None else "-",
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
        # Un unico filtro para todo el turno: una etiqueta puede quedar partida
        # entre dos deltas, incluso al pasar de una vuelta del loop a la siguiente.
        scrub = _ThinkScrubber()

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
                    limpio = scrub.feed(delta.content)
                    if limpio:
                        round_text.append(limpio)
                        spoken.append(limpio)
                        await on_chunk(limpio)
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
                final = scrub.flush()
                if final:
                    spoken.append(final)
                    await on_chunk(final)
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
                return await self._final_answer(convo, spoken, on_chunk, scrub)

        logger.info("Tool loop hit max iterations (%d)", _MAX_TOOL_ITERATIONS)
        final = scrub.flush()
        if final:
            spoken.append(final)
            await on_chunk(final)
        spoken.append(_TOOL_LOOP_FALLBACK)
        await on_chunk(_TOOL_LOOP_FALLBACK)
        return "".join(spoken)

    async def _final_answer(
        self,
        convo: list[dict[str, Any]],
        spoken: list[str],
        on_chunk: Callable[[str], Awaitable[None]],
        scrub: _ThinkScrubber,
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
                    limpio = scrub.feed(delta)
                    if limpio:
                        final_parts.append(limpio)
                        spoken.append(limpio)
                        await on_chunk(limpio)
            final = scrub.flush()
            if final:
                final_parts.append(final)
                spoken.append(final)
                await on_chunk(final)
        except Exception:
            logger.exception("Final answer without tools failed")
            spoken.append(_TOOL_LOOP_FALLBACK)
            await on_chunk(_TOOL_LOOP_FALLBACK)
            return "".join(spoken)

        if not "".join(final_parts).strip():
            spoken.append(_TOOL_LOOP_FALLBACK)
            await on_chunk(_TOOL_LOOP_FALLBACK)
        return "".join(spoken)
