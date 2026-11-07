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
# 0,84 por palabra: acepta "samanta"/"sananta" (0,86) y rechaza "santa"/"manta"
# (0,83), que son palabras comunes y dispararían en conversación normal.
_UMBRAL_PALABRA = 0.84


def _palabras(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9ñ]+", " ", t)
    return [w for w in t.split() if w]


def _fonetica(palabra: str) -> str:
    """Forma laxa para comparar: colapsa confusiones típicas del ASR es/en."""
    p = palabra.replace("th", "t").replace("ph", "f")
    p = p.replace("ck", "k").replace("qu", "k")
    p = p.replace("z", "s").replace("v", "b")
    p = re.sub(r"(.)\1+", r"\1", p)  # dobles letras
    return p.replace("h", "")  # h muda ("hola" ≈ "ola")


def _prepara_objetivos(variantes: list[str]) -> list[list[str]]:
    """Frases aceptadas, en forma fonética y sin duplicados."""
    objetivos: list[list[str]] = []
    for variante in variantes:
        palabras = [_fonetica(w) for w in _palabras(variante)]
        if palabras and palabras not in objetivos:
            objetivos.append(palabras)
    return objetivos


def _coincide(palabras: list[str], objetivos: list[list[str]]) -> str | None:
    """Ventana reconocida que coincide con alguna frase (texto) o None."""
    for objetivo in objetivos:
        n = len(objetivo)
        if n == 0 or len(palabras) < n:
            continue
        for i in range(len(palabras) - n + 1):
            ventana = palabras[i : i + n]
            if all(
                difflib.SequenceMatcher(None, o, p).ratio() >= _UMBRAL_PALABRA
                for o, p in zip(objetivo, ventana)
            ):
                return " ".join(ventana)
    return None


class VoskWakeWord:
    """Reconoce la frase en el stream; no necesita entrenamiento propio."""

    def __init__(
        self, phrase: str, model_path, aliases: list[str] | None = None
    ) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        self.phrase = phrase
        self._variantes = [phrase, *(aliases or [])]
        self._objetivos = _prepara_objetivos(self._variantes)
        if not self._objetivos:
            raise ValueError("La frase de activación está vacía")
        self._modelo = Model(str(model_path))
        self._rec = self._crea_recognizer(KaldiRecognizer)
        self.last_text = ""  # último texto oído (para el log)
        self.last_match = ""  # ventana que disparó

    def _crea_recognizer(self, KaldiRecognizer):
        """Reconocedor con gramática restringida a las variantes.

        Vosk busca solo esas frases (+[unk]), lo que con micro lejano sube
        mucho el recall y reduce falsos positivos. La gramática usa solo las
        variantes de 2+ palabras: meter el nombre suelto compite con la frase
        y empeora el reconocimiento (medido); el alias corto se acepta igual
        en el matcher cuando Vosk solo alcanza a oír el nombre.
        """
        frases = [v.strip() for v in self._variantes if len(_palabras(v)) >= 2]
        if not frases:
            frases = [v.strip() for v in self._variantes if v.strip()]
        gramatica = json.dumps([*frases, "[unk]"], ensure_ascii=False)
        try:
            return KaldiRecognizer(self._modelo, SAMPLE_RATE, gramatica)
        except Exception as exc:
            logger.warning(
                "Wake word: gramática no soportada (%s); uso reconocedor libre", exc
            )
            return KaldiRecognizer(self._modelo, SAMPLE_RATE)

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
        self.last_text = texto
        palabras = [_fonetica(w) for w in _palabras(texto)]
        match = _coincide(palabras, self._objetivos)
        if match is None:
            return False
        self.last_match = match
        logger.debug("Wake word: %r (oído %r)", match, texto)
        return True

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

    @property
    def last_text(self) -> str:
        """Último texto que oyó Vosk (para diagnóstico)."""
        return getattr(self._detector, "last_text", "")

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
