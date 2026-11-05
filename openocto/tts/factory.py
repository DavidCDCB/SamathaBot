"""Factory for creating TTS engines based on config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openocto.config import TTSConfig
    from openocto.tts.base import TTSEngine

logger = logging.getLogger(__name__)


def _create(
    engine_name: str, lang: str, model_name: str, config: TTSConfig
) -> TTSEngine:
    if engine_name == "piper":
        from openocto.tts.piper import PiperTTSEngine

        return PiperTTSEngine(model_name=model_name, config=config)

    if engine_name == "edge":
        from openocto.tts.edge import EdgeTTSEngine

        return EdgeTTSEngine(lang=lang, config=config)

    raise ValueError(
        f"Unknown TTS engine: {engine_name!r}. Supported engines: 'piper', 'edge'."
    )


def create_tts_engine(lang: str, model_name: str, config: TTSConfig) -> TTSEngine:
    """Create a TTS engine for the given language.

    The engine type is resolved from config.engines[lang], falling back to
    config.engine. For Edge, model_name is ignored (the voice comes from
    config.edge.voices) and it is kept as the Piper voice for the fallback.

    When the primary engine is online (edge) and fallback_engine is set, the
    returned engine tries the primary first and falls back transparently.
    """
    engine_name = config.engines.get(lang, config.engine)
    fallback_name = config.fallback_engine

    if engine_name == "edge" and fallback_name and fallback_name != "edge":
        try:
            primary = _create("edge", lang, model_name, config)
        except Exception as exc:
            logger.warning("Edge TTS no disponible (%s); uso %s", exc, fallback_name)
            return _create(fallback_name, lang, model_name, config)

        try:
            fallback = _create(fallback_name, lang, model_name, config)
        except Exception as exc:
            logger.warning(
                "Fallback %s no disponible (%s); queda solo Edge",
                fallback_name,
                exc,
            )
            return primary

        from openocto.tts.fallback import FallbackTTSEngine

        return FallbackTTSEngine(
            primary,
            fallback,
            primary_name="edge",
            fallback_name=fallback_name,
            cooldown=config.fallback_cooldown,
        )

    return _create(engine_name, lang, model_name, config)
