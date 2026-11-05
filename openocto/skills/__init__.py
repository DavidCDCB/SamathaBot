"""OpenOcto Skills — extensible LLM-callable capability system.

The dispatcher is the LLM itself: each skill exposes a JSON Schema for
its parameters, the AI backend translates the registry into OpenAI tool format,
and the model decides when to call them.

Public API:
    Skill, SkillRegistry, SkillError, SkillUnavailable, build_default_registry
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openocto.skills.base import Skill, SkillError, SkillRegistry, SkillUnavailable

if TYPE_CHECKING:
    from openocto.config import SkillsConfig


# Maps the config name -> import path -> class.  Lazy imports so a
# missing optional dep (e.g. vlc bin) doesn't break the whole module
# load — each skill handles its own availability check at __init__.
_SKILL_TABLE: dict[str, tuple[str, str]] = {
    "notes": ("openocto.skills.notes", "NotesSkill"),
}


def build_default_registry(config: "SkillsConfig") -> SkillRegistry:
    """Instantiate every enabled skill from config and return the registry.

    Skills that fail their availability check (missing binary, missing
    config) are silently dropped.
    """
    import importlib

    registry = SkillRegistry()
    enabled = list(config.enabled)
    # Stable, dependency-aware order.
    priority = {"notes": 0}
    enabled.sort(key=lambda n: priority.get(n, 99))

    per_skill_config = {"notes": {}}

    for name in enabled:
        entry = _SKILL_TABLE.get(name)
        if not entry:
            continue
        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            skill_cls = getattr(module, class_name)
        except Exception:
            continue
        registry.try_register(skill_cls, per_skill_config.get(name, {}))

    return registry


__all__ = [
    "Skill",
    "SkillRegistry",
    "SkillError",
    "SkillUnavailable",
    "build_default_registry",
]
