"""Tests for the Wikipedia skill (no real network)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from openocto.skills.wikipedia import WikipediaSkill


class _Resp:
    def __init__(self, status: int = 200, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class WikipediaSkillTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.skill = WikipediaSkill({"timeout": 1})

    async def test_search_then_summary(self) -> None:
        search = _Resp(200, {"query": {"search": [{"title": "Fotosíntesis"}]}})
        summary = _Resp(
            200,
            {
                "title": "Fotosíntesis",
                "extract": "La fotosíntesis es el proceso por el cual las plantas...",
                "content_urls": {
                    "desktop": {
                        "page": "https://es.wikipedia.org/wiki/Fotos%C3%ADntesis"
                    }
                },
            },
        )
        with patch(
            "openocto.skills.wikipedia.requests.get",
            side_effect=[search, summary],
        ) as get:
            result = await self.skill.execute("fotosíntesis")

        self.assertIn("Fotosíntesis", result)
        self.assertIn("plantas", result)
        self.assertIn("es.wikipedia.org", result)
        self.assertEqual(get.call_count, 2)

    async def test_no_results(self) -> None:
        search = _Resp(200, {"query": {"search": []}})
        with patch(
            "openocto.skills.wikipedia.requests.get",
            return_value=search,
        ):
            result = await self.skill.execute("algo inexistente")
        self.assertIn("No encontré", result)

    async def test_invalid_lang_falls_back_to_spanish(self) -> None:
        search = _Resp(200, {"query": {"search": []}})
        with patch(
            "openocto.skills.wikipedia.requests.get",
            return_value=search,
        ) as get:
            await self.skill.execute("algo", lang="../../evil")
        url = get.call_args.args[0]
        self.assertTrue(url.startswith("https://es.wikipedia.org/"))

    async def test_timeout_reports_clear_error(self) -> None:
        with patch(
            "openocto.skills.wikipedia.requests.get",
            side_effect=requests.Timeout,
        ):
            result = await self.skill.execute("algo")
        self.assertIn("tardó", result)


if __name__ == "__main__":
    unittest.main()
