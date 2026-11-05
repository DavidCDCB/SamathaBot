"""Open a Google or YouTube search in the default browser.

This is the only PC-control skill: it cannot launch arbitrary programs or
open arbitrary URLs.  The provider is restricted to Google and YouTube and
the query is URL-encoded, so a spoken query cannot inject a different URL.
"""

from __future__ import annotations

import asyncio
import logging
import webbrowser
from typing import Literal
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, field_validator

from openocto.skills.base import Skill, SkillError

logger = logging.getLogger(__name__)

_GOOGLE = "https://www.google.com/search?q={}"
_YOUTUBE = "https://www.youtube.com/results?search_query={}"


def build_search_url(provider: str, query: str) -> str:
    """Build a Google/YouTube URL, rejecting any other provider."""
    provider = (provider or "").strip().lower()
    query = (query or "").strip()
    if not query:
        raise ValueError("la búsqueda está vacía")
    if provider == "google":
        return _GOOGLE.format(quote_plus(query))
    if provider == "youtube":
        return _YOUTUBE.format(quote_plus(query))
    raise ValueError("proveedor no permitido; usa google o youtube")


class _Params(BaseModel):
    provider: Literal["google", "youtube"] = Field(
        description="Motor de búsqueda: google o youtube."
    )
    query: str = Field(description="Texto a buscar.")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> str:
        return str(value).strip().lower()


class OpenSearchSkill(Skill):
    name = "open_search"
    description = (
        "Abre una búsqueda en Google o YouTube en el navegador predeterminado. "
        "Úsala cuando la persona pida explícitamente abrir o ver una búsqueda, "
        "o cuando mencione YouTube; si lo que quiere es una respuesta general, "
        "usa web_search."
    )
    Parameters = _Params

    async def execute(self, provider: str, query: str) -> str:
        query = (query or "").strip()
        if not query:
            raise SkillError("la búsqueda está vacía")
        try:
            url = build_search_url(provider, query)
        except ValueError as exc:
            raise SkillError(str(exc)) from exc

        try:
            opened = await asyncio.to_thread(webbrowser.open, url, new=2)
        except Exception as exc:  # noqa: BLE001 - browser backends vary
            logger.warning("open_search falló: %s", type(exc).__name__)
            return "Error: no pude abrir el navegador."

        if opened is False:
            return "Error: no encontré un navegador predeterminado."
        label = "Google" if provider.strip().lower() == "google" else "YouTube"
        logger.info("Búsqueda abierta en %s: %s", label, url)
        return f"Abrí una búsqueda en {label} para: {query}"
