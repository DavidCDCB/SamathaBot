"""Detector local de la frase de activación con Vosk.

No usa la nube: Vosk reconoce en streaming y solo busca la frase configurada
(por defecto "hola samantha"). El worker vive en su propio hilo para no
bloquear el callback de audio.
"""

from __future__ import annotations

import difflib
import json
import logging
import queue
import re
import threading
import time
import unicodedata
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_UMBRAL_PALABRA = 0.82


def _palabras(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9ñ]+", " ", t)
    return [w for w in t.split() if w]


class VoskWakeWord:
    """Reconoce la frase en el stream; no necesita entrenamiento propio."""

    def __init__(self, phrase: str, model_path) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        self.phrase = phrase
        self._objetivo = _palabras(phrase)
        if not self._objetivo:
            raise ValueError("La frase de activación está vacía")
        self._modelo = Model(str(model_path))
        self._rec = KaldiRecognizer(self._modelo, SAMPLE_RATE)

    def process_chunk(self, chunk: np.ndarray) -> bool:
        """True si la frase aparece en el texto parcial/final de este chunk."""
        pcm = np.ascontiguousarray(chunk, dtype=np.int16).tobytes()
        if self._rec.AcceptWaveform(pcm):
            texto = json.loads(self._rec.Result()).get("text", "")
            # nueva frase después de un silencio
            self._rec.Reset()
        else:
            texto = json.loads(self._rec.PartialResult()).get("partial", "")
        return self._contiene(texto)

    def _contiene(self, texto: str) -> bool:
        palabras = _palabras(texto)
        n = len(self._objetivo)
        if n == 0 or len(palabras) < n:
            return False
        for i in range(len(palabras) - n + 1):
            ventana = palabras[i : i + n]
            if all(
                difflib.SequenceMatcher(None, objetivo, palabra).ratio()
                >= _UMBRAL_PALABRA
                for objetivo, palabra in zip(self._objetivo, ventana)
            ):
                return True
        return False

    def reset(self) -> None:
        self._rec.Reset()


class WakeWordController:
    """Hilo que consume chunks y pausa el detector cuando el asistente habla."""

    def __init__(self, detector: VoskWakeWord, on_detect: Callable[[], None]) -> None:
        self._detector = detector
        self._on_detect = on_detect
        self._cola: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._pausado = threading.Event()
        self._detener = threading.Event()
        self._reiniciar = threading.Event()
        self._hilo = threading.Thread(
            target=self._corre, name="wakeword-vosk", daemon=True
        )

    def start(self) -> None:
        self._hilo.start()

    def feed(self, chunk: np.ndarray) -> None:
        """Llamado desde el hilo de audio; nunca bloquea."""
        if self._pausado.is_set() or self._detener.is_set():
            return
        try:
            self._cola.put_nowait(np.asarray(chunk, dtype=np.int16))
        except queue.Full:
            logger.debug("wakeword: cola llena, chunk descartado")

    def pause(self) -> None:
        if self._pausado.is_set():
            return
        self._pausado.set()
        self._vaciar()

    def resume(self) -> None:
        if not self._pausado.is_set():
            return
        self._vaciar()
        self._reiniciar.set()
        self._pausado.clear()

    def stop(self) -> None:
        self._detener.set()
        self._pausado.set()
        self._hilo.join(timeout=2.0)

    def _vaciar(self) -> None:
        while True:
            try:
                self._cola.get_nowait()
            except queue.Empty:
                return

    def _corre(self) -> None:
        while not self._detener.is_set():
            if self._pausado.is_set() or self._reiniciar.is_set():
                self._detector.reset()
                self._reiniciar.clear()
                time.sleep(0.05)
                continue
            try:
                chunk = self._cola.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._detector.process_chunk(chunk):
                    self._pausado.set()
                    self._vaciar()
                    self._detector.reset()
                    self._on_detect()
            except Exception:
                logger.exception("wakeword: fallo al procesar audio")
                time.sleep(0.1)
