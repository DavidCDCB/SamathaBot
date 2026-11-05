"""Tests for the Firecrawl web_search skill (no real network)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import requests

from openocto.skills.base import SkillUnavailable
from openocto.skills.web_search import WebSearchSkill


class _Resp:
    def __init__(self, status: int = 200, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class WebSearchSkillTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        os.environ["FIRECRAWL_API_KEY"] = "fc-test"
        self.skill = WebSearchSkill(
            {"timeout": 1, "max_results": 5, "fallback": "duckduckgo"}
        )

    def tearDown(self) -> None:
        os.environ.pop("FIRECRAWL_API_KEY", None)

    async def test_success_returns_title_description_and_url(self) -> None:
        payload = {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Resultado uno",
                        "url": "https://example.com/uno",
                        "description": "Descripción uno",
                    },
                    {
                        "title": "Resultado dos",
                        "url": "https://example.com/dos",
                        "description": "Descripción dos",
                    },
                ]
            },
        }
        with patch(
            "openocto.skills.web_search.requests.post",
            return_value=_Resp(200, payload),
        ) as post:
            result = await self.skill.execute("noticias de hoy", limit=2)

        self.assertIn("Resultado uno", result)
        self.assertIn("https://example.com/uno", result)
        self.assertIn("Descripción dos", result)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["query"], "noticias de hoy")
        self.assertEqual(body["sources"], ["web"])
        self.assertEqual(body["limit"], 2)

    async def test_no_results(self) -> None:
        with patch(
            "openocto.skills.web_search.requests.post",
            return_value=_Resp(200, {"success": True, "data": {"web": []}}),
        ):
            result = await self.skill.execute("algo inexistente")
        self.assertIn("No encontré", result)

    async def test_401_reports_invalid_key(self) -> None:
        with patch(
            "openocto.skills.web_search.requests.post",
            return_value=_Resp(401, {}),
        ):
            result = await self.skill.execute("algo")
        self.assertIn("401", result)
        self.assertIn("key", result)

    async def test_429_reports_rate_limit(self) -> None:
        with patch(
            "openocto.skills.web_search.requests.post",
            return_value=_Resp(429, {}),
        ):
            result = await self.skill.execute("algo")
        self.assertIn("429", result)
        self.assertIn("límite", result)

    async def test_timeout_reports_clear_error(self) -> None:
        with patch(
            "openocto.skills.web_search.requests.post",
            side_effect=requests.Timeout,
        ):
            result = await self.skill.execute("algo")
        self.assertIn("tardó", result)

    async def test_402_falls_back_to_duckduckgo(self) -> None:
        rows = [
            {
                "title": "Respaldo uno",
                "href": "https://example.org/uno",
                "body": "Descripción uno",
            },
            {
                "title": "Respaldo dos",
                "href": "https://example.org/dos",
                "body": "Descripción dos",
            },
        ]
        with (
            patch(
                "openocto.skills.web_search.requests.post",
                return_value=_Resp(402, {}),
            ),
            patch(
                "openocto.skills.web_search._ddg_text",
                return_value=rows,
            ) as ddg,
        ):
            result = await self.skill.execute("algo", limit=2)

        self.assertIn("Respaldo uno", result)
        self.assertIn("https://example.org/uno", result)
        self.assertIn("Descripción dos", result)
        ddg.assert_called_once()
        self.assertEqual(ddg.call_args.args[0], "algo")
        self.assertEqual(ddg.call_args.args[1], 2)

    async def test_402_fallback_disabled(self) -> None:
        skill = WebSearchSkill(
            {"timeout": 1, "max_results": 5, "fallback": "none"}
        )
        with (
            patch(
                "openocto.skills.web_search.requests.post",
                return_value=_Resp(402, {}),
            ),
            patch("openocto.skills.web_search._ddg_text") as ddg,
        ):
            result = await skill.execute("algo")
        self.assertIn("desactivado", result)
        ddg.assert_not_called()

    async def test_402_fallback_empty(self) -> None:
        with (
            patch(
                "openocto.skills.web_search.requests.post",
                return_value=_Resp(402, {}),
            ),
            patch("openocto.skills.web_search._ddg_text", return_value=[]),
        ):
            result = await self.skill.execute("algo")
        self.assertIn("DuckDuckGo", result)
        self.assertIn("no encontró", result)

    async def test_402_fallback_failure(self) -> None:
        with (
            patch(
                "openocto.skills.web_search.requests.post",
                return_value=_Resp(402, {}),
            ),
            patch(
                "openocto.skills.web_search._ddg_text",
                side_effect=RuntimeError("blocked"),
            ),
        ):
            result = await self.skill.execute("algo")
        self.assertIn("DuckDuckGo", result)
        self.assertIn("no pudo responder", result)

    async def test_429_does_not_fall_back(self) -> None:
        with (
            patch(
                "openocto.skills.web_search.requests.post",
                return_value=_Resp(429, {}),
            ),
            patch("openocto.skills.web_search._ddg_text") as ddg,
        ):
            result = await self.skill.execute("algo")
        self.assertIn("429", result)
        ddg.assert_not_called()

    def test_missing_key_is_unavailable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SkillUnavailable):
                WebSearchSkill({})


if __name__ == "__main__":
    unittest.main()
