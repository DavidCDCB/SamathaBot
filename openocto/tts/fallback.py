"""TTS wrapper: usa un motor primario y cae a otro si falla.

Pensado para Edge TTS (online) con Piper (local) de respaldo. Tras un fallo
del primario se evita reintentarlo durante `cooldown` segundos: sin red, no
queremos pagar el timeout en cada frase del turno.
"""

from __future__ import annotations

import logging
import time

from openocto.tts.base import AudioSegment, TTSEngine

logger = logging.getLogger(__name__)


class FallbackTTSEngine(TTSEngine):
    def __init__(
        self,
        primary: TTSEngine,
        fallback: TTSEngine,
        primary_name: str = "primary",
        fallback_name: str = "fallback",
        cooldown: float = 90.0,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_name = primary_name
        self._fallback_name = fallback_name
        self._cooldown = cooldown
        self._retry_at = 0.0
        self._degraded = False
        self._last_engine = primary_name

    def synthesize(self, text: str) -> AudioSegment:
        now = time.monotonic()
        if now >= self._retry_at:
            try:
                audio = self._primary.synthesize(text)
            except Exception as exc:
                self._retry_at = time.monotonic() + self._cooldown
                if not self._degraded:
                    logger.warning(
                        "%s no disponible (%s: %s); uso %s durante %.0fs",
                        self._primary_name,
                        type(exc).__name__,
                        exc or "(sin mensaje)",
                        self._fallback_name,
                        self._cooldown,
                    )
                    self._degraded = True
                else:
                    logger.debug(
                        "%s sigue fallando (%s); uso %s",
                        self._primary_name,
                        exc,
                        self._fallback_name,
                    )
            else:
                if self._degraded:
                    logger.info("%s recuperado", self._primary_name)
                    self._degraded = False
                self._last_engine = self._primary_name
                return audio
        else:
            logger.debug(
                "%s en espera; uso %s", self._primary_name, self._fallback_name
            )

        self._last_engine = self._fallback_name
        return self._fallback.synthesize(text)

    @property
    def sample_rate(self) -> int:
        if self._last_engine == self._fallback_name:
            return self._fallback.sample_rate
        return self._primary.sample_rate

    @property
    def last_engine(self) -> str:
        """Motor que sintetizo la ultima frase (util para diagnostico)."""
        return self._last_engine
