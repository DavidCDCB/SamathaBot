"""Tests for skill registration and per-skill config."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from openocto.config import SkillsConfig
from openocto.skills import build_default_registry


class SkillsRegistryTest(unittest.TestCase):
    def test_web_search_and_wikipedia_register_with_key(self) -> None:
        cfg = SkillsConfig(enabled=["notes", "web_search", "wikipedia", "open_search"])
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            registry = build_default_registry(cfg)
        self.assertIn("web_search", registry.names())
        self.assertIn("wikipedia", registry.names())
        self.assertIn("open_search", registry.names())
        self.assertIn("manage_notes_and_facts", registry.names())

    def test_open_search_registers_without_key(self) -> None:
        cfg = SkillsConfig(enabled=["open_search"])
        with patch.dict(os.environ, {}, clear=True):
            registry = build_default_registry(cfg)
        self.assertIn("open_search", registry.names())

    def test_web_search_is_dropped_without_key(self) -> None:
        cfg = SkillsConfig(enabled=["web_search"])
        with patch.dict(os.environ, {}, clear=True):
            registry = build_default_registry(cfg)
        self.assertNotIn("web_search", registry.names())

    def test_firecrawl_defaults(self) -> None:
        cfg = SkillsConfig()
        self.assertEqual(cfg.firecrawl.max_results, 5)
        self.assertEqual(cfg.firecrawl.timeout, 15)
        self.assertEqual(cfg.firecrawl.lang, "es")
        self.assertEqual(cfg.firecrawl.country, "CO")
        self.assertEqual(cfg.firecrawl.fallback, "duckduckgo")
        self.assertEqual(cfg.firecrawl.region, "co-es")


if __name__ == "__main__":
    unittest.main()
