"""Model downloader for STT, TTS, and VAD models."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from openocto.config import MODELS_DIR

logger = logging.getLogger(__name__)

# Whisper GGML model download URLs
WHISPER_MODELS: dict[str, dict] = {
    "tiny": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "filename": "ggml-tiny.bin",
    },
    "base": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "filename": "ggml-base.bin",
    },
    "small": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "filename": "ggml-small.bin",
    },
    "medium": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "filename": "ggml-medium.bin",
    },
}

# Piper voice model download URLs
PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_MODELS: dict[str, dict] = {
    # --- English (US) — high quality ---
    "en_US-lessac-high": {
        "model_url": f"{PIPER_BASE_URL}/en/en_US/lessac/high/en_US-lessac-high.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/lessac/high/en_US-lessac-high.onnx.json",
        "description": "American English female, high quality (~109 MB)",
    },
    "en_US-ryan-high": {
        "model_url": f"{PIPER_BASE_URL}/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
        "description": "American English male, high quality (~115 MB)",
    },
    "en_US-ljspeech-high": {
        "model_url": f"{PIPER_BASE_URL}/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json",
        "description": "American English female (LJSpeech dataset), high quality (~109 MB)",
    },
    "en_US-libritts-high": {
        "model_url": f"{PIPER_BASE_URL}/en/en_US/libritts/high/en_US-libritts-high.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/libritts/high/en_US-libritts-high.onnx.json",
        "description": "American English multi-speaker (904 voices, LibriTTS dataset), high quality (~130 MB)",
    },
    # --- English (GB) — high quality ---
    "en_GB-cori-high": {
        "model_url": f"{PIPER_BASE_URL}/en/en_GB/cori/high/en_GB-cori-high.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_GB/cori/high/en_GB-cori-high.onnx.json",
        "description": "British English female, high quality (~109 MB)",
    },
    # --- English (US) — medium quality (smaller / faster) ---
    "en_US-amy-medium": {
        "model_url": f"{PIPER_BASE_URL}/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        "description": "American English female, medium quality (~60 MB)",
    },
    "en_US-ryan-medium": {
        "model_url": f"{PIPER_BASE_URL}/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
        "description": "American English male, medium quality (~60 MB)",
    },
    "en_US-hfc_female-medium": {
        "model_url": f"{PIPER_BASE_URL}/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json",
        "description": "American English female, medium quality (~60 MB)",
    },
    "en_US-hfc_male-medium": {
        "model_url": f"{PIPER_BASE_URL}/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx.json",
        "description": "American English male, medium quality (~60 MB)",
    },
    # --- English (GB) — medium quality ---
    "en_GB-jenny_dioco-medium": {
        "model_url": f"{PIPER_BASE_URL}/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json",
        "description": "British English female (Jenny), medium quality (~60 MB)",
    },
    "en_GB-alan-medium": {
        "model_url": f"{PIPER_BASE_URL}/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        "description": "British English male, medium quality (~60 MB)",
    },
    # --- Spanish (SPIKE PATCH 1: 9 voces) ---
    "es_MX-claude-high": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json",
        "description": "Mexican Spanish, high quality (~63 MB)",
    },
    "es_AR-daniela-high": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
        "description": "Argentine Spanish female, high quality (~114 MB)",
    },
    "es_MX-ald-medium": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json",
        "description": "Mexican Spanish male, medium quality (~63 MB)",
    },
    "es_ES-davefx-medium": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json",
        "description": "European Spanish male, medium quality (~63 MB)",
    },
    "es_ES-sharvard-medium": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json",
        "description": "European Spanish, medium quality (~75 MB)",
    },
    "es_ES-mls_10246-low": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx.json",
        "description": "European Spanish (MLS), low quality (~63 MB)",
    },
    "es_ES-mls_9972-low": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx.json",
        "description": "European Spanish (MLS), low quality (~63 MB)",
    },
    "es_MX-ald-x_low": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/x_low/es_MX-ald-x_low.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/x_low/es_MX-ald-x_low.onnx.json",
        "description": "Mexican Spanish male, x_low quality (~20 MB)",
    },
    "es_ES-carlfm-x_low": {
        "model_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx",
        "config_url": f"https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json",
        "description": "European Spanish male, x_low quality (~27 MB)",
    },
    # --- Russian ---
    "ru_RU-irina-medium": {
        "model_url": f"{PIPER_BASE_URL}/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json",
        "description": "Russian female, medium quality (~60 MB)",
    },
    "ru_RU-denis-medium": {
        "model_url": f"{PIPER_BASE_URL}/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json",
        "description": "Russian male, medium quality (~60 MB)",
    },
    "ru_RU-dmitri-medium": {
        "model_url": f"{PIPER_BASE_URL}/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json",
        "description": "Russian male, medium quality (~60 MB)",
    },
    "ru_RU-ruslan-medium": {
        "model_url": f"{PIPER_BASE_URL}/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx.json",
        "description": "Russian male, medium quality (~60 MB)",
    },
}


def _download_file(url: str, dest: Path, desc: str = "", max_retries: int = 5) -> None:
    """Download a file with progress bar, resume support, and retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    logger.info("Downloading %s to %s", desc or url, dest)

    for attempt in range(1, max_retries + 1):
        try:
            # Resume from partial download
            downloaded = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}

            response = requests.get(
                url, stream=True, timeout=(15, 120), headers=headers
            )
            response.raise_for_status()

            # Total size from Content-Range or Content-Length
            if response.status_code == 206:  # Partial content (resume)
                content_range = response.headers.get("Content-Range", "")
                total = int(content_range.split("/")[-1]) if "/" in content_range else 0
            else:
                total = int(response.headers.get("content-length", 0))
                downloaded = 0  # Server doesn't support range, start over

            mode = "ab" if downloaded > 0 else "wb"
            with (
                open(tmp, mode) as f,
                tqdm(
                    desc=desc or dest.name,
                    total=total,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=32768):
                    f.write(chunk)
                    bar.update(len(chunk))

            tmp.rename(dest)
            logger.info("Downloaded %s", dest)
            return

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            if attempt < max_retries:
                wait = attempt * 5
                logger.warning(
                    "Download interrupted (%s), retry %d/%d in %ds...",
                    type(e).__name__,
                    attempt,
                    max_retries,
                    wait,
                )
                import time

                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download {desc or url} after {max_retries} attempts. "
                    f"Check your network connection and try again."
                ) from e


def get_whisper_model(model_size: str) -> Path:
    """Get path to a whisper model, downloading if needed."""
    if model_size not in WHISPER_MODELS:
        raise ValueError(
            f"Unknown whisper model: {model_size}. Available: {list(WHISPER_MODELS)}"
        )

    info = WHISPER_MODELS[model_size]
    dest = MODELS_DIR / "whisper" / info["filename"]

    if not dest.exists():
        _download_file(info["url"], dest, desc=f"whisper-{model_size}")

    return dest


def get_piper_model(model_name: str) -> tuple[Path, Path]:
    """Get paths to (model.onnx, model.onnx.json), downloading if needed."""
    if model_name not in PIPER_MODELS:
        raise ValueError(
            f"Unknown piper model: {model_name}. Available: {list(PIPER_MODELS)}"
        )

    info = PIPER_MODELS[model_name]
    model_dir = MODELS_DIR / "piper"
    model_path = model_dir / f"{model_name}.onnx"
    config_path = model_dir / f"{model_name}.onnx.json"

    if not model_path.exists():
        _download_file(info["model_url"], model_path, desc=f"{model_name}.onnx")

    if not config_path.exists():
        _download_file(info["config_url"], config_path, desc=f"{model_name}.onnx.json")

    return model_path, config_path


# --- Silero VAD ---

SILERO_VAD = {
    "url": "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
    "filename": "silero_vad.onnx",
}


def get_silero_vad_model() -> Path:
    """Ruta al modelo Silero VAD ONNX, descargandolo si falta."""
    dest = MODELS_DIR / "vad" / SILERO_VAD["filename"]
    if not dest.exists():
        _download_file(SILERO_VAD["url"], dest, desc=SILERO_VAD["filename"])
    return dest


# --- Vosk (wake word local) ---

VOSK_MODELS = {
    "vosk-model-small-es-0.42": (
        "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
    ),
    # Modelo completo: ~1,4 GB, mucho mejor con micrófono lejano.
    "vosk-model-es-0.42": (
        "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip"
    ),
}


def get_vosk_model(model_name: str) -> Path:
    """Ruta a la carpeta del modelo Vosk, descargando y extrayendo si falta."""
    if model_name not in VOSK_MODELS:
        raise ValueError(
            f"Modelo Vosk desconocido: {model_name}. Disponibles: {list(VOSK_MODELS)}"
        )
    base = MODELS_DIR / "vosk"
    dest = base / model_name
    if dest.exists():
        return dest
    base.mkdir(parents=True, exist_ok=True)
    archivo = base / f"{model_name}.zip"
    if not archivo.exists():
        _download_file(VOSK_MODELS[model_name], archivo, desc=archivo.name)
    with zipfile.ZipFile(archivo) as z:
        z.extractall(base)
    return dest
