"""Audio capture from microphone."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from openocto.config import AudioConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1280  # 80ms chunks


def _resolve_device(device: int | str | None, kind: str) -> int | None:
    """Resolve a device name substring to its index.

    If ``device`` is already an int or None, return as-is.
    If it's a string, search for the first device whose name contains it
    (case-insensitive) and has at least one channel of the requested kind.
    Logs a warning and returns None (system default) if not found.
    """
    if device is None or isinstance(device, int):
        return device

    max_ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()
    needle = device.lower()
    for idx, info in enumerate(devices):
        if needle in info["name"].lower() and info[max_ch_key] > 0:
            logger.info(
                "Resolved %s device %r → #%d (%s)", kind, device, idx, info["name"]
            )
            return idx

    logger.warning("Audio %s device %r not found, using system default", kind, device)
    return None


class AudioCapture:
    """Captures audio from the default (or configured) microphone."""

    # Watchdog: con silencio el stream igual entrega ~12 chunks/s, así que
    # 2,5 s sin callbacks significa dispositivo muerto (audífonos desconectados,
    # Bluetooth caído, otro programa tomó el micro en exclusiva).
    _WATCH_INTERVAL = 0.5
    _REOPEN_GRACE = 2.5

    def __init__(
        self, config: AudioConfig | None = None, mic_gain: float | None = None
    ) -> None:
        self._sample_rate = config.sample_rate if config else SAMPLE_RATE
        self._blocksize = config.blocksize if config else BLOCKSIZE
        raw = config.input_device if config else None
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.lower() in ("", "default", "auto", "predeterminado"):
                raw = None
        self._input_device = _resolve_device(raw, kind="input")
        # None = seguir el predeterminado de Windows EN CADA APERTURA. Si el
        # usuario fija un nombre/índice se respeta, con vuelta al default si
        # desaparece.
        self._follow_default = self._input_device is None
        self._stream_device: int | None = None
        self._stream_wanted = False
        self._last_callback = 0.0
        self._callback_since_open = False
        self._last_open_try = 0.0
        self._stream_lock = threading.RLock()
        self._watchdog: threading.Thread | None = None
        self._mic_gain = mic_gain

        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._chunks_recibidos = 0  # cuenta chunks aunque no se esté grabando
        self._chunk_callback = None  # wake word, etc. (hilo de audio)

    def _apply_gain(self, audio_f32: np.ndarray) -> np.ndarray:
        """Ganancia fija opcional por chunk; None = passthrough.

        El AGC por chunk se elimino: amplificaba el ruido entre palabras y
        el CER con ruido pasaba de 3,2% a 10,6% (medido). La ganancia util
        ahora se aplica UNA vez por toma completa antes del STT
        (openocto.app.normalize_for_stt).
        """
        if self._mic_gain is not None:
            return np.clip(audio_f32 * self._mic_gain, -1.0, 1.0)
        return audio_f32

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: object
    ) -> None:
        try:
            self._last_callback = time.monotonic()
            self._callback_since_open = True
            if status:
                logger.warning("Audio capture status: %s", status)
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            # Apply mic gain before passing to VAD and recording buffer
            mono_f32 = mono.astype(np.float32) / 32768.0
            mono_f32 = self._apply_gain(mono_f32)
            mono = np.clip(mono_f32 * 32768.0, -32768, 32767).astype(np.int16)
            self._chunks_recibidos += 1
            if self._chunk_callback:
                self._chunk_callback(mono.copy())
            if self._recording:
                with self._lock:
                    self._buffer.append(mono.copy())
        except Exception as e:
            import sys

            print(f"Audio callback error: {e}", file=sys.stderr, flush=True)
            logger.error("Audio callback error: %s", e)

    def _current_default_input(self) -> int | None:
        """Índice del micrófono predeterminado de Windows AHORA mismo."""
        try:
            info = sd.query_devices(kind="input")
            idx = info.get("index")
            return int(idx) if idx is not None else None
        except Exception:
            return None

    def _open_stream(self) -> None:
        with self._stream_lock:
            device = self._input_device
            if self._follow_default:
                device = self._current_default_input()
            self._stream_device = device
            try:
                self._stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    blocksize=self._blocksize,
                    device=device,
                    callback=self._callback,
                )
            except sd.PortAudioError as e:
                if device is not None:
                    logger.warning(
                        "Failed to open device %s (%s), falling back to default device",
                        device,
                        e,
                    )
                    # El dispositivo fijado desapareció (audífonos fuera):
                    # desde ahora seguimos el predeterminado de Windows.
                    self._input_device = None
                    self._follow_default = True
                    self._stream_device = self._current_default_input()
                    self._stream = sd.InputStream(
                        samplerate=self._sample_rate,
                        channels=CHANNELS,
                        dtype=DTYPE,
                        blocksize=self._blocksize,
                        device=self._stream_device,
                        callback=self._callback,
                    )
                else:
                    raise
            self._last_callback = time.monotonic()
            self._callback_since_open = False
            self._stream.start()
        self._ensure_watchdog()

    # --- Seguimiento del micrófono predeterminado de Windows ---

    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog = threading.Thread(
            target=self._watch_loop, name="mic-watchdog", daemon=True
        )
        self._watchdog.start()

    def _watch_loop(self) -> None:
        while True:
            time.sleep(self._WATCH_INTERVAL)
            try:
                self._check_stream_health()
            except Exception:
                logger.debug("Watchdog del micrófono falló", exc_info=True)

    def _check_stream_health(self, now: float | None = None) -> str | None:
        """Reabre el micro si cambió el default o si el stream quedó mudo.

        Devuelve la acción tomada ("open"/"reopen"/None) para poder probarlo
        sin abrir audio real.
        """
        now = time.monotonic() if now is None else now
        if not self._stream_wanted:
            return None
        if self._stream is None:
            # Reintento tras un fallo al abrir: cada 3 s, no cada tick.
            if now - self._last_open_try < 3.0:
                return None
            self._last_open_try = now
            self._reopen_stream()
            return "open"
        if self._follow_default and self._stream_device is not None:
            actual = self._current_default_input()
            if actual is not None and actual != self._stream_device:
                logger.info(
                    "Micrófono predeterminado cambió (#%d → #%d); reabro el stream",
                    self._stream_device,
                    actual,
                )
                self._reopen_stream()
                return "reopen"
        if now - self._last_callback >= self._REOPEN_GRACE:
            logger.warning(
                "Sin audio del micrófono por %.1f s; reabro el dispositivo",
                now - self._last_callback,
            )
            self._reopen_stream()
            return "reopen"
        return None

    def _reopen_stream(self) -> None:
        with self._stream_lock:
            if not self._stream_wanted:
                return
            viejo = self._stream
            self._stream = None
            if viejo is not None:
                try:
                    viejo.stop()
                    viejo.close()
                except Exception:
                    logger.debug("No pude cerrar el stream viejo", exc_info=True)
            try:
                self._open_stream()
                logger.info(
                    "Micrófono reabierto (dispositivo #%s)", self._stream_device
                )
            except Exception as exc:
                logger.warning("No pude reabrir el micrófono: %s", exc)

    # --- PTT mode: open/close stream per recording ---

    def set_chunk_callback(self, callback) -> None:
        """Registra un callback por chunk (se ejecuta en el hilo de audio)."""
        self._chunk_callback = callback

    def start_stream(self) -> None:
        """Abre el micrófono sin buffer, para calentarlo antes de grabar."""
        self._stream_wanted = True
        if not self._stream:
            self._chunks_recibidos = 0
            self._open_stream()
            logger.debug("Audio stream opened (warm-up)")

    async def wait_ready(self, timeout: float = 2.0) -> bool:
        """Espera el primer callback real del dispositivo (arranque BT/MME).

        Exige un callback posterior a la apertura actual y de hace <=1,5 s:
        así un stream zombi (audífonos desconectados) no cuenta como listo.
        """
        import asyncio
        import time

        limite = time.monotonic() + timeout
        while True:
            ahora = time.monotonic()
            fresco = ahora - self._last_callback <= 1.5
            if self._chunks_recibidos > 0 and self._callback_since_open and fresco:
                return True
            if ahora >= limite:
                return False
            await asyncio.sleep(0.02)

    def reopen_now(self) -> None:
        """Reabre ya en el predeterminado actual (si el stream es deseado)."""
        if not self._stream_wanted:
            return
        logger.info("Reapertura forzada del micrófono")
        self._reopen_stream()

    def start_recording(self) -> None:
        """Empieza a guardar audio en un stream ya caliente."""
        with self._lock:
            self._buffer.clear()
        self._recording = True

    def start(self) -> None:
        """Open stream and start recording immediately (legacy/one-shot)."""
        with self._lock:
            self._buffer.clear()
        self._chunks_recibidos = 0
        self._recording = True
        self._stream_wanted = True
        self._open_stream()
        logger.debug("Audio capture started")

    def stop_recording(self) -> None:
        """Deja de guardar sin cerrar el stream (conversación en curso)."""
        self._recording = False

    def stop(self) -> None:
        """Stop recording and close stream (PTT mode)."""
        self._recording = False
        self._stream_wanted = False
        with self._stream_lock:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        logger.debug("Audio capture stopped")

    def get_recording(self) -> np.ndarray:
        """Return the recorded audio as a single concatenated array."""
        with self._lock:
            if not self._buffer:
                return np.zeros(0, dtype=np.int16)
            return np.concatenate(self._buffer, axis=0).flatten()

    def latest_chunk_index(self) -> int:
        """Index of the last buffered chunk (-1 if empty)."""
        with self._lock:
            return len(self._buffer) - 1

    def get_latest_chunk(self, after: int = -1) -> tuple[np.ndarray, int] | None:
        """Return the next unprocessed audio chunk for VAD streaming.

        Args:
            after: index of the last processed chunk (-1 = start from beginning).

        Returns:
            (chunk, index) or None if no new chunk is available.
        """
        with self._lock:
            next_idx = after + 1
            if next_idx < len(self._buffer):
                return self._buffer[next_idx].copy(), next_idx
            return None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
