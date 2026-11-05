"""Web search skill using Firecrawl v2 Search (read-only).

The model calls this skill when it needs current web information.  The
skill returns compact title/description/URL lines; the LLM is responsible
for summarizing them for voice.  No page content is scraped here: that
would add latency and credits, and the search highlights are enough for a
short spoken answer.

If Firecrawl runs out of credits (HTTP 402), the skill falls back to
DuckDuckGo through the unofficial ``ddgs`` scraper.  That fallback is
best-effort: DuckDuckGo can answer with a CAPTCHA or rate limit, in which
case the caller gets a clear error instead of a crash.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import requests
from pydantic import BaseModel, Field

from openocto.skills.base import Skill, SkillError, SkillUnavailable

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.firecrawl.dev/v2/search"
_DDG_BACKENDS = ("lite", "html")


def _ddg_text(
    query: str, max_results: int, region: str, timeout: float
) -> list[dict[str, Any]]:
    """Best-effort DuckDuckGo search using the unofficial ddgs scraper.

    Tries the lightweight HTML backend first; when DuckDuckGo returns a
    CAPTCHA or a rate limit, the next backend gets a chance.  Raises when
    all backends fail so the caller can return a clear message.
    """
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("ddgs no está instalado") from exc

    last: Exception | None = None
    with DDGS(timeout=int(max(5, timeout))) as ddgs:
        for backend in _DDG_BACKENDS:
            try:
                rows = list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                        backend=backend,
                        region=region,
                    )
                    or []
                )
            except Exception as exc:  # noqa: BLE001 - backend-specific errors
                last = exc
                continue
            if rows:
                return rows
    if last is not None:
        raise last
    return []


class _Params(BaseModel):
    query: str = Field(description="Consulta web en lenguaje natural.")
    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Número de resultados (1-10).",
    )


class WebSearchSkill(Skill):
    name = "web_search"
    description = (
        "Busca en la web información actual o que no esté en tu conocimiento: "
        "noticias, precios, eventos, personas o datos verificables. Devuelve "
        "título, resumen y URL. Formula una consulta corta y natural, "
        "preferiblemente en español; no uses comillas ni operadores avanzados. "
        "No la uses para definiciones enciclopédicas si Wikipedia basta. No la "
        "uses para abrir YouTube o Google; para eso usa open_search."
    )
    Parameters = _Params

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._key = (os.environ.get("FIRECRAWL_API_KEY") or "").strip()
        if not self._key:
            raise SkillUnavailable(
                "falta FIRECRAWL_API_KEY (guárdala con: Samantha.exe key firecrawl)"
            )
        self._timeout = float(self._config.get("timeout", 15))
        self._max_results = int(self._config.get("max_results", 5))
        self._lang = str(self._config.get("lang", "es"))
        self._country = str(self._config.get("country", "CO"))
        self._fallback = str(self._config.get("fallback", "duckduckgo")).lower()
        self._region = str(self._config.get("region", "co-es"))

    async def execute(self, query: str, limit: int = 5) -> str:
        query = (query or "").strip()
        if not query:
            raise SkillError("la consulta está vacía")
        n = max(1, min(10, int(limit or self._max_results), self._max_results))
        body = {
            "query": query,
            "limit": n,
            "sources": ["web"],
            "lang": self._lang,
            "country": self._country,
        }
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        try:
            resp = await asyncio.to_thread(
                requests.post,
                _ENDPOINT,
                headers=headers,
                json=body,
                timeout=self._timeout,
            )
        except requests.Timeout:
            return "Error: la búsqueda tardó demasiado. Intenta de nuevo."
        except requests.RequestException as exc:
            logger.warning("Firecrawl red: %s", type(exc).__name__)
            return "Error: no pude conectar con el buscador."

        if resp.status_code == 401:
            return (
                "Error: Firecrawl rechazó la API key (401). "
                "Revisa Samantha.exe key firecrawl."
            )
        if resp.status_code == 402:
            logger.warning("Firecrawl sin créditos (402); usando DuckDuckGo")
            return await self._duckduckgo(query, n)
        if resp.status_code == 429:
            return "Error: límite de Firecrawl alcanzado (429). Intenta en un minuto."
        if resp.status_code >= 400:
            logger.warning("Firecrawl HTTP %s", resp.status_code)
            return f"Error: el buscador respondió HTTP {resp.status_code}."

        try:
            payload = resp.json()
        except ValueError:
            return "Error: el buscador devolvió una respuesta ilegible."

        if not payload.get("success"):
            error = str(payload.get("error") or "").lower()
            if any(word in error for word in ("credit", "payment", "quota")):
                logger.warning("Firecrawl sin créditos (payload); usando DuckDuckGo")
                return await self._duckduckgo(query, n)
            return "Error: el buscador no pudo completar la búsqueda."

        data = payload.get("data") or {}
        web = data.get("web") if isinstance(data, dict) else data
        if not isinstance(web, list) or not web:
            return "No encontré resultados web para esa consulta."

        lines: list[str] = []
        for i, item in enumerate(web[:n], 1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Sin título").strip()[:120]
            url = str(item.get("url") or "").strip()
            desc = " ".join(str(item.get("description") or "").split())[:300]
            lines.append(f"{i}. {title}\n   {desc}\n   {url}")
        if not lines:
            return "No encontré resultados web para esa consulta."
        return "\n".join(lines)

    async def _duckduckgo(self, query: str, n: int) -> str:
        """Fallback when Firecrawl reports no credits."""
        if self._fallback not in ("duckduckgo", "ddgs"):
            return (
                "Error: Firecrawl se quedó sin créditos (402) y el respaldo "
                "DuckDuckGo está desactivado."
            )
        try:
            rows = await asyncio.to_thread(
                _ddg_text, query, n, self._region, self._timeout
            )
        except Exception as exc:  # noqa: BLE001 - fallback is best-effort
            logger.warning("DuckDuckGo fallback falló: %s", type(exc).__name__)
            return (
                "Error: Firecrawl se quedó sin créditos y DuckDuckGo no pudo "
                "responder. Intenta más tarde."
            )
        if not rows:
            return (
                "Error: Firecrawl se quedó sin créditos y DuckDuckGo no "
                "encontró resultados."
            )

        lines: list[str] = []
        for i, item in enumerate(rows[:n], 1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Sin título").strip()[:120]
            url = str(item.get("href") or item.get("url") or "").strip()
            desc = " ".join(str(item.get("body") or item.get("snippet") or "").split())[
                :300
            ]
            lines.append(f"{i}. {title}\n   {desc}\n   {url}")
        if not lines:
            return (
                "Error: Firecrawl se quedó sin créditos y DuckDuckGo no "
                "encontró resultados."
            )
        logger.warning("Búsqueda respondida por DuckDuckGo (respaldo)")
        return "\n".join(lines)
