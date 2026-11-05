"""Speech-to-Text using faster-whisper (CTranslate2).

The big model (default ``large-v3-turbo``) is the primary engine.  If CUDA
is requested but the NVIDIA libraries are missing, the engine retries on
CPU with int8.  The app's previous whisper.cpp engine is used as a separate
fallback by ``openocto.stt.factory``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openocto.config import APP_DIR
from openocto.stt.base import STTEngine, TranscriptionResult
from openocto.stt.whisper_cpp import _HALLUCINATION_RE

if TYPE_CHECKING:
    from openocto.config import STTConfig

logger = logging.getLogger(__name__)

# Manejadores de os.add_dll_directory: mantienen vivas las carpetas de
# cublas/cudnn mientras dure el proceso.
_DLL_HANDLES: list = []


def _register_cuda_dll_dirs(fw) -> None:
    """Añade al PATH de DLLs las carpetas con cublas64_12/cudnn64_9.

    Fuentes, en orden: ``stt.faster_whisper.cuda_dll_dirs``, la carpeta
    ``cuda\\`` junto al ejecutable (portable construido con --gpu) y los
    paquetes pip ``nvidia-*-cu12`` instalados en el venv.
    """
    if os.name != "nt":
        return
    dirs: list[Path] = []
    for item in (fw.cuda_dll_dirs if fw else []) or []:
        p = Path(item)
        if not p.is_absolute():
            p = APP_DIR / p
        dirs.append(p)
    dirs.append(APP_DIR / "cuda")

    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
    except Exception:
        spec = None
    if spec and spec.submodule_search_locations:
        base = Path(next(iter(spec.submodule_search_locations)))
        for dll in base.rglob("*.dll"):
            dirs.append(dll.parent)

    seen: set[str] = set()
    path_parts = os.environ.get("PATH", "")
    extra: list[str] = []
    for d in dirs:
        key = str(d)
        if key in seen or not d.is_dir():
            continue
        seen.add(key)
        # ctranslate2 usa LoadLibrary("cublas64_12.dll") sin flags: solo
        # ve carpetas del PATH (y la suya propia). add_dll_directory no le
        # llega, pero no molesta a nadie mas, asi que hacemos ambos.
        try:
            _DLL_HANDLES.append(os.add_dll_directory(key))
        except OSError:
            logger.debug("CUDA: no se pudo registrar %s", key)
        if key.lower() not in (p.lower() for p in path_parts.split(os.pathsep)):
            extra.append(key)
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra) + os.pathsep + path_parts
        for key in extra:
            logger.info("CUDA: carpeta de DLLs registrada: %s", key)


class FasterWhisperEngine(STTEngine):
    """Primary STT engine: faster-whisper large-v3-turbo (GPU or CPU)."""

    def __init__(self, config: STTConfig | None = None) -> None:
        fw = config.faster_whisper if config else None
        self._model_name = (fw.model if fw else "large-v3-turbo") or "large-v3-turbo"
        self._device = (fw.device if fw else "auto") or "auto"
        self._compute_type = (fw.compute_type if fw else "default") or "default"
        self._beam_size = int(fw.beam_size if fw else 5)
        language = config.language if config else "auto"
        self._language = None if language == "auto" else language
        self._n_threads = int(config.n_threads if config else 4)

        hotwords = list(getattr(config, "hotwords", None) or []) if config else []
        self._initial_prompt = (
            "Nombres propios: " + ", ".join(hotwords) if hotwords else None
        )

        root = Path(fw.download_root if fw else "models/faster-whisper")
        if not root.is_absolute():
            root = APP_DIR / root
        root.mkdir(parents=True, exist_ok=True)
        self._download_root = str(root)

        # Sin Developer Mode, Hugging Face copia en vez de usar symlinks;
        # el aviso es normal y no afecta al modelo.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if self._device != "cpu":
            _register_cuda_dll_dirs(fw)
        from faster_whisper import WhisperModel

        self._whisper_model_cls = WhisperModel
        logger.info(
            "Loading faster-whisper %s (device=%s, compute=%s, download_root=%s)",
            self._model_name,
            self._device,
            self._compute_type,
            self._download_root,
        )
        try:
            self._model = self._crear_modelo(self._device, self._compute_type)
        except Exception as exc:
            if self._device == "cpu":
                raise
            logger.warning(
                "faster-whisper no pudo cargar %s (%s); uso CPU int8",
                self._device,
                exc,
            )
            self._cargar_cpu()
        else:
            # CTranslate2 puede cargar el modelo en CUDA aunque falte
            # cublas64_12.dll; el error aparece en el primer encode.
            actual = getattr(getattr(self._model, "model", None), "device", None)
            if actual:
                self._device = str(actual)
            if self._device != "cpu":
                self._calentar_gpu()

        logger.info(
            "faster-whisper loaded: %s (device=%s, compute=%s)",
            self._model_name,
            self._device,
            self._compute_type,
        )

    def describe(self) -> str:
        if self._device.startswith("cpu"):
            destino = f"CPU {self._compute_type}"
        elif self._device == "auto":
            destino = "auto"
        else:
            destino = f"GPU {self._compute_type}"
        return f"faster-whisper {self._model_name} ({destino})"

    def _crear_modelo(self, device: str, compute_type: str):
        return self._whisper_model_cls(
            self._model_name,
            device=device,
            compute_type=compute_type,
            download_root=self._download_root,
            cpu_threads=self._n_threads,
        )

    def _cargar_cpu(self) -> None:
        logger.warning("Cargando faster-whisper en CPU int8...")
        self._model = self._crear_modelo("cpu", "int8")
        self._device = "cpu"
        self._compute_type = "int8"

    def _calentar_gpu(self) -> None:
        """Fuerza un encode corto: detecta cuBLAS/cuDNN faltantes al inicio."""
        try:
            probe = np.zeros(16000, dtype=np.float32)
            segments, _ = self._model.transcribe(
                probe,
                language=self._language,
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                task="transcribe",
            )
            list(segments)  # el encode ocurre al iterar
        except Exception as exc:
            logger.warning("faster-whisper CUDA no operativo (%s); uso CPU int8", exc)
            self._cargar_cpu()

    def _transcribir(self, audio_float: np.ndarray) -> TranscriptionResult:
        start = time.perf_counter()
        segments, info = self._model.transcribe(
            audio_float,
            language=self._language,
            beam_size=self._beam_size,
            initial_prompt=self._initial_prompt,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
            temperature=0.0,
            task="transcribe",
        )
        # ``segments`` is a generator: transcription runs while iterating.
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration_ms = (time.perf_counter() - start) * 1000

        if text and _HALLUCINATION_RE.match(text):
            logger.debug("Filtered hallucination: %r", text)
            text = ""

        detected = self._language or getattr(info, "language", None) or "en"
        confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
        logger.debug(
            "Transcribed in %.0fms [%s]: %r",
            duration_ms,
            detected,
            text[:80],
        )
        return TranscriptionResult(
            text=text,
            language=detected,
            confidence=confidence,
            duration_ms=duration_ms,
        )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        if audio.size == 0:
            return TranscriptionResult(
                text="", language="en", confidence=0.0, duration_ms=0.0
            )

        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        try:
            return self._transcribir(audio_float)
        except Exception as exc:
            if self._device == "cpu":
                raise
            logger.warning(
                "faster-whisper falló en %s (%s); reintento CPU int8",
                self._device,
                exc,
            )
            self._cargar_cpu()
            return self._transcribir(audio_float)
