"""Tests for the open_search skill (no real browser)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from openocto.skills.base import SkillError
from openocto.skills.open_search import OpenSearchSkill, build_search_url


class BuildSearchUrlTest(unittest.TestCase):
    def test_google_encodes_spaces_and_accents(self):
        self.assertEqual(
            build_search_url("google", "clima en Bogotá"),
            "https://www.google.com/search?q=clima+en+Bogot%C3%A1",
        )

    def test_youtube_encodes_spaces_and_accents(self):
        self.assertEqual(
            build_search_url("youtube", "música para estudiar"),
            "https://www.youtube.com/results?search_query=m%C3%BAsica+para+estudiar",
        )

    def test_query_cannot_inject_another_url(self):
        url = build_search_url("google", "a&b https://evil.com")
        self.assertTrue(url.startswith("https://www.google.com/search?q="))
        self.assertNotIn("https://evil.com", url)
        self.assertIn("%26", url)

    def test_invalid_provider(self):
        with self.assertRaises(ValueError):
            build_search_url("bing", "algo")

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            build_search_url("google", "   ")


class OpenSearchSkillTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.skill = OpenSearchSkill({})

    def test_provider_is_normalized(self):
        params = OpenSearchSkill.Parameters(provider="Google", query="algo")
        self.assertEqual(params.provider, "google")

    async def test_execute_opens_youtube_in_browser(self):
        with patch(
            "openocto.skills.open_search.webbrowser.open",
            return_value=True,
        ) as opened:
            result = await self.skill.execute("youtube", "música para estudiar")

        self.assertIn("YouTube", result)
        self.assertIn("música para estudiar", result)
        opened.assert_called_once()
        self.assertEqual(
            opened.call_args.args[0],
            "https://www.youtube.com/results?search_query=m%C3%BAsica+para+estudiar",
        )
        self.assertEqual(opened.call_args.kwargs["new"], 2)

    async def test_execute_opens_google_in_browser(self):
        with patch(
            "openocto.skills.open_search.webbrowser.open",
            return_value=True,
        ) as opened:
            result = await self.skill.execute("google", "clima en Bogotá")

        self.assertIn("Google", result)
        self.assertEqual(
            opened.call_args.args[0],
            "https://www.google.com/search?q=clima+en+Bogot%C3%A1",
        )

    async def test_browser_returns_false(self):
        with patch(
            "openocto.skills.open_search.webbrowser.open",
            return_value=False,
        ):
            result = await self.skill.execute("google", "algo")
        self.assertIn("navegador", result)

    async def test_browser_raises(self):
        with patch(
            "openocto.skills.open_search.webbrowser.open",
            side_effect=OSError("sin navegador"),
        ):
            result = await self.skill.execute("google", "algo")
        self.assertIn("navegador", result)

    async def test_empty_query_is_rejected(self):
        with self.assertRaises(SkillError):
            await self.skill.execute("google", "   ")

    async def test_invalid_provider_is_rejected(self):
        with self.assertRaises(SkillError):
            await self.skill.execute("bing", "algo")


if __name__ == "__main__":
    unittest.main()
