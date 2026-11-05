"""Persona Manager — carga la personalidad (persona.yaml + system_prompt.md)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from openocto.config import PERSONAS_DIR

logger = logging.getLogger(__name__)


@dataclass
class Persona:
    name: str
    display_name: str
    description: str
    system_prompt: str
    memory_summary_sections: list[str] | None = None


class PersonaManager:
    """Loads persona configurations from personas/<name>/ and activates one."""

    def __init__(self, personas_dir: Path | None = None) -> None:
        self._personas_dir = personas_dir or PERSONAS_DIR
        self._personas: dict[str, Persona] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self._personas_dir.exists():
            logger.warning("Personas directory not found: %s", self._personas_dir)
            return

        for persona_dir in self._personas_dir.iterdir():
            if persona_dir.is_dir() and (persona_dir / "persona.yaml").exists():
                try:
                    persona = self._load_persona(persona_dir)
                    self._personas[persona.name] = persona
                    logger.debug("Loaded persona: %s", persona.name)
                except Exception as e:
                    logger.warning("Failed to load persona from %s: %s", persona_dir, e)

        logger.info(
            "Loaded %d persona(s): %s", len(self._personas), list(self._personas)
        )

    def _load_persona(self, persona_dir: Path) -> Persona:
        config_path = persona_dir / "persona.yaml"
        prompt_path = persona_dir / "system_prompt.md"

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        system_prompt = (
            prompt_path.read_text(encoding="utf-8").strip()
            if prompt_path.exists()
            else ""
        )
        memory = config.get("memory", {})

        return Persona(
            name=config["name"],
            display_name=config.get("display_name", config["name"]),
            description=config.get("description", ""),
            system_prompt=system_prompt,
            memory_summary_sections=memory.get("summary_sections"),
        )

    def activate(self, name: str) -> Persona:
        """Activate a persona by name."""
        if name not in self._personas:
            raise ValueError(
                f"Persona {name!r} not found. Available: {list(self._personas)}"
            )
        logger.info("Activated persona: %s", name)
        return self._personas[name]
