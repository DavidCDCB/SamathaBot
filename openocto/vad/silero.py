"""Voice Activity Detection using Silero VAD (ONNX, sin torch)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from openocto.utils.model_downloader import get_silero_vad_model
from openocto.vad.base import VADEngine

if TYPE_CHECKING:
    from openocto.config import VADConfig

logger = logging.getLogger(__name__)


class SileroVAD(VADEngine):
    """VAD con Silero ONNX.

    Soporta v4 (entradas h, c) y v5 (entrada state). El modelo que descarga
    el proyecto es v4 (~1,7 MB), que funciona igual en x64 y ARM64.
    """

    _V5_WINDOW = 512  # v5 exige exactamente 512 muestras por inferencia
    _SPEECH_STREAK_RESET = 2  # chunks seguidos para reiniciar el silencio

    def __init__(self, config: VADConfig | None = None) -> None:
        self._threshold = config.threshold if config else 0.5
        self._silence_duration = config.silence_duration if config else 0.9
        self._rms_threshold = config.rms_speech_threshold if config else 300
        self._sample_rate = np.array(16000, dtype=np.int64)
        self._silence_start: float | None = None
        self._speech_streak = 0
        self.last_prob = 0.0
        self.last_rms = 0.0
        # Throttle del debug por chunk: con silencio continuo son ~12 lineas/s
        # y entierran el resto del log (respuesta del modelo, tool calls).
        self._last_rms_log = 0.0

        model_path = get_silero_vad_model()
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._version = self._detect_model_version()
        self._reset_state()
        logger.info("Silero VAD loaded (%s, %s)", self._version, model_path)

    def _detect_model_version(self) -> str:
        names = {i.name for i in self._session.get_inputs()}
        return "v5" if "state" in names else "v4"

    def _reset_state(self) -> None:
        if self._version == "v5":
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _infer_v5(self, audio_f32: np.ndarray) -> float:
        """Ejecuta v5 en ventanas de 512 muestras y devuelve la prob. maxima."""
        max_prob = 0.0
        for offset in range(0, len(audio_f32), self._V5_WINDOW):
            window = audio_f32[offset : offset + self._V5_WINDOW]
            if len(window) < self._V5_WINDOW:
                window = np.pad(window, (0, self._V5_WINDOW - len(window)))
            output, self._state = self._session.run(
                None,
                {
                    "input": window.reshape(1, -1),
                    "state": self._state,
                    "sr": self._sample_rate,
                },
            )
            max_prob = max(max_prob, float(output.flat[0]))
        return max_prob

    def _infer_v4(self, audio_f32: np.ndarray) -> float:
        output, self._h, self._c = self._session.run(
            None,
            {
                "input": audio_f32.reshape(1, -1),
                "sr": self._sample_rate,
                "h": self._h,
                "c": self._c,
            },
        )
        return float(output.flat[0])

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """True si el chunk contiene voz.

        El RMS solo se usa como puerta rapida de silencio puro; la decision
        final la toma Silero (un micro sensible puede tener ruido electrico
        por encima de cualquier RMS razonable y Silero lo clasifica bien).
        """
        chunk_i16 = audio_chunk.flatten()
        rms_raw = float(np.sqrt(np.mean(chunk_i16.astype(np.float32) ** 2)))
        self.last_rms = rms_raw

        if rms_raw < self._rms_threshold:
            self.last_prob = 0.0
            ahora = time.monotonic()
            if ahora - self._last_rms_log >= 1.0:
                self._last_rms_log = ahora
                logger.debug(
                    "VAD rms_raw=%.0f < thr=%d -> silencio (sin Silero)",
                    rms_raw,
                    self._rms_threshold,
                )
            return False

        audio_f32 = chunk_i16.astype(np.float32) / 32768.0
        prob = (
            self._infer_v5(audio_f32)
            if self._version == "v5"
            else self._infer_v4(audio_f32)
        )
        self.last_prob = prob
        result = prob > self._threshold
        logger.debug(
            "VAD prob=%.3f rms_raw=%.0f (thr=%d/%.2f) -> %s",
            prob,
            rms_raw,
            self._rms_threshold,
            self._threshold,
            result,
        )
        return result

    def should_stop_recording(
        self, audio_chunk: np.ndarray, speech: bool | None = None
    ) -> bool:
        """True cuando el silencio dura >= silence_duration.

        Histeresis: un pico aislado de "voz" no reinicia el contador; hacen
        falta 2 chunks seguidos. Evita que un golpe o ruido del USB alargue
        la grabacion indefinidamente.
        """
        if speech is None:
            speech = self.is_speech(audio_chunk)

        if speech:
            self._speech_streak += 1
            if self._speech_streak >= self._SPEECH_STREAK_RESET:
                self._silence_start = None
            return False

        self._speech_streak = 0
        if self._silence_start is None:
            self._silence_start = time.monotonic()

        return (time.monotonic() - self._silence_start) >= self._silence_duration

    def reset(self) -> None:
        self._silence_start = None
        self._speech_streak = 0
        self._reset_state()
