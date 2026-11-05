"""Text-to-Speech using Microsoft Edge neural voices (online).

No usa credenciales: habla con el endpoint publico que usa el "Read Aloud"
de Microsoft Edge. Requiere internet y envia el texto a Microsoft. Si falla,
`FallbackTTSEngine` continua con Piper (local) sin cortar el turno.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from openocto.tts.base import AudioSegment, TTSEngine

if TYPE_CHECKING:
    from openocto.config import TTSConfig

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "es-CO-SalomeNeural"


class EdgeTTSEngine(TTSEngine):
    """Sintesis neuronal de Edge; una peticion por frase."""

    def __init__(self, lang: str, config: TTSConfig | None = None) -> None:
        # Falla en el arranque (y el factory cae a Piper) si faltan las libs.
        import edge_tts
        import miniaudio

        self._edge_tts = edge_tts
        self._miniaudio = miniaudio

        self._rate = config.edge.rate if config else "+0%"
        self._pitch = config.edge.pitch if config else "+0Hz"
        self._volume = config.edge.volume if config else "+0%"
        self._timeout = config.edge.timeout if config else 10.0
        self._trim = config.edge.trim_silence if config else True
        # pausa entre frases; el resto del silencio de Edge se recorta
        self._gap = config.sentence_silence if config else 0.25

        voices = config.edge.voices if config else {}
        prefix = lang.split("_")[0].lower()
        self._voice = (
            voices.get(lang)
            or voices.get(prefix)
            or next(iter(voices.values()), _DEFAULT_VOICE)
        )
        logger.info("Edge TTS listo: voice=%s", self._voice)

    def synthesize(self, text: str) -> AudioSegment:
        if not text.strip():
            return AudioSegment(audio=np.zeros(0, dtype=np.int16), sample_rate=24000)

        t0 = time.perf_counter()
        try:
            mp3 = asyncio.run(self._collect_with_timeout(text))
        except TimeoutError as exc:
            # asyncio.TimeoutError llega con str vacio; damos contexto al log.
            raise TimeoutError(f"sin respuesta en {self._timeout:.0f}s") from exc
        if not mp3:
            raise RuntimeError("Edge TTS no devolvio audio")

        decoded = self._miniaudio.decode(
            mp3,
            output_format=self._miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=24000,
        )
        audio = np.array(decoded.samples, dtype=np.int16)
        raw_dur = len(audio) / 24000
        audio = self._trim_edges(audio)
        logger.debug(
            "Edge TTS %s: %d chars -> %.2fs audio (%.2fs tras recorte) en %.2fs",
            self._voice,
            len(text),
            raw_dur,
            len(audio) / 24000,
            time.perf_counter() - t0,
        )
        return AudioSegment(audio=audio, sample_rate=24000)

    def _trim_edges(self, audio: np.ndarray) -> np.ndarray:
        """Recorta silencio inicial/final y deja una pausa de `sentence_silence`.

        Edge devuelve ~0,7 s de cola por frase; al sintetizar frase a frase
        eso se suma y hace que la respuesta suene mucho mas lenta.
        """
        if not self._trim or audio.size == 0:
            return audio
        sr = 24000
        frame = max(1, int(sr * 0.01))  # tramas de 10 ms
        n = audio.size // frame
        if n == 0:
            return audio
        frames = audio[: n * frame].reshape(n, frame).astype(np.float32)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        active = np.flatnonzero(rms >= 180.0)
        if active.size == 0:
            return np.zeros(0, dtype=np.int16)

        pad = int(sr * 0.03)  # 30 ms de margen para no cortar consonantes
        start = max(0, int(active[0]) * frame - pad)
        end = min(audio.size, (int(active[-1]) + 1) * frame + pad)
        trimmed = audio[start:end]

        gap_n = int(sr * max(0.0, self._gap))
        if gap_n <= 0:
            return trimmed
        return np.concatenate([trimmed, np.zeros(gap_n, dtype=np.int16)])

    async def _collect_with_timeout(self, text: str) -> bytes:
        return await asyncio.wait_for(self._collect(text), timeout=self._timeout)

    async def _collect(self, text: str) -> bytes:
        communicate = self._edge_tts.Communicate(
            text,
            self._voice,
            rate=self._rate,
            volume=self._volume,
            pitch=self._pitch,
        )
        data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                data.extend(chunk["data"])
        return bytes(data)

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def voice(self) -> str:
        return self._voice
