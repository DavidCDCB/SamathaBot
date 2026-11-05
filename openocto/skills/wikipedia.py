"""Wikipedia lookup skill (read-only, no API key).

Two requests: the MediaWiki search API finds the best article title, then
the REST summary endpoint returns a short extract.  The extract is capped
so the voice model receives a compact tool result.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field

from openocto.skills.base import Skill, SkillError

logger = logging.getLogger(__name__)

_USER_AGENT = "Samantha/1.0 (personal voice assistant)"


class _Params(BaseModel):
    query: str = Field(description="Tema o título a buscar en Wikipedia.")
    lang: str = Field(default="es", description="Código de idioma: es o en.")


class WikipediaSkill(Skill):
    name = "wikipedia"
    description = (
        "Busca definiciones, conceptos, historia y antecedentes en Wikipedia. "
        "Úsala para preguntas 'qué es', 'quién fue' o 'historia de'. Devuelve "
        "un resumen y la URL del artículo."
    )
    Parameters = _Params

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._timeout = float(self._config.get("timeout", 10))

    async def execute(self, query: str, lang: str = "es") -> str:
        query = (query or "").strip()
        if not query:
            raise SkillError("la consulta está vacía")
        lang = lang if re.fullmatch(r"[a-z]{2,3}", lang or "") else "es"

        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": 3,
        }
        try:
            resp = await asyncio.to_thread(
                requests.get,
                search_url,
                params=params,
                headers={"User-Agent": _USER_AGENT},
                timeout=self._timeout,
            )
        except requests.Timeout:
            return "Error: Wikipedia tardó demasiado en responder."
        except requests.RequestException as exc:
            logger.warning("Wikipedia red: %s", type(exc).__name__)
            return "Error: no pude conectar con Wikipedia."
        if resp.status_code != 200:
            return f"Error: Wikipedia respondió HTTP {resp.status_code}."
        try:
            hits = resp.json().get("query", {}).get("search", [])
        except ValueError:
            return "Error: Wikipedia devolvió una respuesta ilegible."
        if not hits:
            return f"No encontré un artículo de Wikipedia para {query!r}."

        title = str(hits[0].get("title") or "").strip()
        if not title:
            return f"No encontré un artículo de Wikipedia para {query!r}."

        summary_url = (
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
            f"{quote(title, safe='')}"
        )
        try:
            resp = await asyncio.to_thread(
                requests.get,
                summary_url,
                headers={"User-Agent": _USER_AGENT},
                timeout=self._timeout,
            )
        except requests.Timeout:
            return "Error: Wikipedia tardó demasiado en responder."
        except requests.RequestException as exc:
            logger.warning("Wikipedia red: %s", type(exc).__name__)
            return "Error: no pude conectar con Wikipedia."
        if resp.status_code != 200:
            return f"Error: Wikipedia respondió HTTP {resp.status_code}."
        try:
            data = resp.json()
        except ValueError:
            return "Error: Wikipedia devolvió una respuesta ilegible."

        extract = " ".join(str(data.get("extract") or "").split())[:1500]
        page_url = (data.get("content_urls") or {}).get("desktop", {}).get(
            "page"
        ) or summary_url
        if not extract:
            return (
                f"Encontré {title!r}, pero no tiene un resumen disponible.\n{page_url}"
            )
        return f"{data.get('title') or title}\n{extract}\n{page_url}"
