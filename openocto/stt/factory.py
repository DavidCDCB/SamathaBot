"""Build the configured STT engine, with the whisper.cpp fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openocto.stt.base import STTEngine
from openocto.stt.fallback import FallbackSTTEngine
from openocto.stt.whisper_cpp import WhisperCppEngine

if TYPE_CHECKING:
    from openocto.config import STTConfig

logger = logging.getLogger(__name__)


def _normalize(name: str | None) -> str:
    return (name or "").strip().lower().replace("-", "_")


def create_stt_engine(config: STTConfig) -> STTEngine:
    """Create the primary engine and wrap it with the configured fallback."""
    engine = _normalize(config.engine) or "whisper_cpp"
    fallback = _normalize(config.fallback_engine)

    if engine in ("whisper_cpp", "whisper", "cpp"):
        return WhisperCppEngine(config)

    if engine in ("faster_whisper", "faster"):
        try:
            from openocto.stt.faster_whisper import FasterWhisperEngine

            primary: STTEngine = FasterWhisperEngine(config)
        except Exception:
            logger.exception(
                "faster-whisper no disponible; uso whisper.cpp como principal"
            )
            return WhisperCppEngine(config)

        if fallback == "none":
            return primary
        if fallback not in ("", "whisper_cpp", "whisper", "cpp"):
            logger.warning(
                "Fallback STT desconocido %r; uso faster-whisper solo", fallback
            )
            return primary

        try:
            base = WhisperCppEngine(config)
        except Exception:
            logger.exception(
                "No pude cargar el fallback whisper.cpp; sigo con faster-whisper"
            )
            return primary
        return FallbackSTTEngine(primary, base, max_latency_ms=config.fallback_after_ms)

    raise ValueError(f"Motor STT desconocido: {engine!r}")
