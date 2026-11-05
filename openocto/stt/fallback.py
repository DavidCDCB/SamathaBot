"""Fallback wrapper for STT engines."""

from __future__ import annotations

import logging

import numpy as np

from openocto.stt.base import STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)


class FallbackSTTEngine(STTEngine):
    """Try the primary engine; use the fallback on error, empty text or latency."""

    def __init__(
        self,
        primary: STTEngine,
        fallback: STTEngine,
        max_latency_ms: int = 0,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._max_latency_ms = max(0, int(max_latency_ms))
        self._prefer_fallback = False

    def describe(self) -> str:
        def _d(e: STTEngine) -> str:
            fn = getattr(e, "describe", None)
            return fn() if callable(fn) else type(e).__name__

        if self._prefer_fallback:
            return f"{_d(self._fallback)} (el grande era lento, cambié por latencia)"
        return f"{_d(self._primary)} -> {_d(self._fallback)}"

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        if self._prefer_fallback:
            return self._fallback.transcribe(audio, sample_rate)

        try:
            result = self._primary.transcribe(audio, sample_rate)
            if result.text.strip():
                if self._max_latency_ms and result.duration_ms > self._max_latency_ms:
                    logger.warning(
                        "STT principal tardó %.0fms (> %dms); "
                        "los siguientes turnos usan el fallback",
                        result.duration_ms,
                        self._max_latency_ms,
                    )
                    self._prefer_fallback = True
                return result
            logger.warning("STT principal devolvió texto vacío; uso el fallback")
        except Exception:
            logger.exception("STT principal falló; uso el fallback")

        return self._fallback.transcribe(audio, sample_rate)
