"""Configuration loader for OpenOcto.

Resolution order:
1. config/default.yaml (shipped with the package)
2. ~/.openocto/config.yaml (user overrides)
3. Environment variables (for API keys)
4. CLI flags
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from openocto.utils.hotkey import DEFAULT_HOTKEY, parse_hotkey


def _resolve_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} patterns with environment variable values."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(r"\$\{(\w+)\}", _replace, value)


def _resolve_env_recursive(data: dict | list | str) -> dict | list | str:
    """Recursively resolve ${ENV_VAR} patterns in config data."""
    if isinstance(data, dict):
        return {k: _resolve_env_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_recursive(item) for item in data]
    if isinstance(data, str):
        return _resolve_env_vars(data)
    return data


# --- Config models ---


class AudioConfig(BaseModel):
    input_device: int | str | None = None  # index or substring of device name
    output_device: int | str | None = None  # index or substring of device name
    sample_rate: int = 16000
    blocksize: int = 1280
    # ganancia fija o null = auto (antes vivia en vad.mic_gain)
    mic_gain: float | None = None
    # normaliza la toma completa antes del STT (ver app.normalize_for_stt)
    normalize: bool = True
    # "hold" = mantener el atajo mientras hablas (comportamiento clasico).
    # "vad"  = una pulsacion empieza a escuchar; se corta solo al callar.
    ptt_mode: str = "hold"
    # Tras responder, reabre el micro para el siguiente turno (conversacion).
    continuous: bool = False
    # Espera anti-eco antes de reabrir (segundos).
    continuous_tail_guard: float = 1.0
    # Tecla global del push-to-talk: combinación de ctrl/alt/shift/win + tecla (ver utils/hotkey.py).
    ptt_hotkey: str = DEFAULT_HOTKEY

    @field_validator("ptt_hotkey")
    @classmethod
    def _valida_ptt_hotkey(cls, valor: str) -> str:
        parse_hotkey(valor)  # lanza ValueError con el motivo
        return valor


class VADConfig(BaseModel):
    """Deteccion de voz/silencio para el modo ptt_mode: vad."""

    threshold: float = 0.5
    silence_duration: float = 0.9
    # RMS minimo (int16) para ni siquiera consultar a Silero: silencio puro.
    # 50 deja pasar voz de micro lejano; Silero decide si es voz o ruido.
    rms_speech_threshold: int = 50
    # Si no se detecta voz en este tiempo, se cancela la toma.
    no_speech_timeout: float = 8.0
    # Tope duro: aunque haya ruido continuo, se corta y se procesa.
    max_recording_duration: float = 30.0


class FasterWhisperConfig(BaseModel):
    model: str = "large-v3-turbo"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "default"  # default | float16 | int8_float16 | int8
    beam_size: int = 5
    download_root: str = "models/faster-whisper"
    # Carpetas extra con cublas64_12.dll / cudnn64_9.dll para activar CUDA.
    # Vacio = auto: busca cuda\ junto a la app y los paquetes pip
    # nvidia-cublas-cu12 / nvidia-cudnn-cu12 del venv.
    cuda_dll_dirs: list[str] = Field(default_factory=list)


class STTConfig(BaseModel):
    # primary y fallback: whisper_cpp | faster_whisper
    engine: str = "whisper_cpp"
    fallback_engine: str = "whisper_cpp"
    # Si el motor principal tarda más de esto, los siguientes turnos usan
    # el fallback. 0 desactiva el cambio por latencia.
    fallback_after_ms: int = 4000
    model_size: str = "small"
    language: str = "auto"
    n_threads: int = 4
    # P11: palabras que Whisper debe preferir (nombres propios, toponimos).
    # Se inyectan como initial_prompt del decode; corrige "Freddy"->"Fredy".
    hotwords: list[str] = Field(default_factory=list)
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)


class EdgeTTSConfig(BaseModel):
    """Microsoft Edge neural TTS (online, voces neuronales)."""

    voices: dict[str, str] = Field(
        default_factory=lambda: {
            "es": "es-CO-SalomeNeural",
            "en": "en-US-AriaNeural",
        }
    )
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    timeout: float = 10.0
    # Edge agrega ~0,7 s de silencio al final de cada frase; recortarlo y
    # dejar solo `tts.sentence_silence` acelera mucho las respuestas.
    trim_silence: bool = True


class TTSConfig(BaseModel):
    engine: str = "piper"
    engines: dict[str, str] = Field(
        default_factory=dict
    )  # per-lang overrides, e.g. {ru: silero}
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "en": "en_US-amy-medium",
            "ru": "ru_RU-irina-medium",
        }
    )
    # Motor neuronal online (engine: edge). Se resuelve la voz por idioma.
    edge: EdgeTTSConfig = Field(default_factory=EdgeTTSConfig)
    # Si el motor principal falla (sin red, endpoint caido), se usa este.
    # Cadena vacia = sin fallback.
    fallback_engine: str = "piper"
    # Segundos que se queda en el fallback antes de reintentar el principal.
    fallback_cooldown: float = 30.0
    # Volumen local del TTS. 1.0 = original; 1.6 ≈ +4 dB; limitador suave.
    gain: float = 1.0
    length_scale: float = 1.0
    sentence_silence: float = 0.3


class ProviderConfig(BaseModel):
    """Configuration for any OpenAI-compatible provider."""

    api_key: str = ""
    model: str = ""
    base_url: str = ""
    # If true, api_key is not required (e.g. local proxy)
    no_auth: bool = False
    # SPIKE PATCH 2a: OpenCode Go pide User-Agent propio y x-opencode-session
    headers: dict[str, str] = Field(default_factory=dict)
    # SPIKE PATCH 7: razonamiento opcional; solo algunos modelos de Go lo
    # aplican (ver spike_effort.py). Vacio = no se manda, igual que upstream.
    reasoning_effort: str = ""
    # SPIKE PATCH 7b: campos libres que se fusionan en el cuerpo de cada
    # peticion (para botones que el SDK aun no exponga). Cuidado: puede
    # pisar claves ya puestas por openocto.
    extra_body: dict[str, Any] = Field(default_factory=dict)


class AIConfig(BaseModel):
    default_backend: str = "go"
    # Cualquier proveedor compatible con la API de OpenAI (OpenCode Go, OpenAI, Ollama...)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    # Solo con la memoria desactivada; con memoria manda memory.recent_window.
    max_history: int = 20
    # Espera máxima (segundos) por la respuesta del LLM en un turno de voz.
    turn_timeout: float = 30.0
    # Tope de tokens por respuesta en un turno de voz (None = sin tope).
    max_tokens: int | None = 400


class MemoryConfig(BaseModel):
    enabled: bool = True
    # Turnos (usuario + Samantha) que se mandan como contexto; cada turno son 2 mensajes.
    recent_window: int = 80
    summarize_threshold: int = 40  # mensajes fuera de la ventana antes de resumir
    max_facts: int = 50  # max user facts
    max_notes: int = 20  # max active notes
    notes_ttl_days: int = 14  # auto-resolve notes older than N days


class FirecrawlConfig(BaseModel):
    timeout: float = 15.0
    max_results: int = 5
    lang: str = "es"
    country: str = "CO"
    # "duckduckgo" usa ddgs solo si Firecrawl responde 402 (sin créditos).
    fallback: str = "duckduckgo"
    region: str = "co-es"


class SkillsConfig(BaseModel):
    # Skills disponibles: ver openocto/skills/__init__.py (_SKILL_TABLE).
    enabled: list[str] = Field(default_factory=lambda: ["notes"])
    firecrawl: FirecrawlConfig = Field(default_factory=FirecrawlConfig)


class LoggingConfig(BaseModel):
    level: str = "DEBUG"  # nivel del ARCHIVO
    # nivel de la CONSOLA: WARNING por defecto; `assistant.py start --debug` lo sube a DEBUG
    console_level: str = "WARNING"
    file: str | None = None
    max_bytes: int = 5 * 1024 * 1024  # 5 MB per file
    backup_count: int = 3  # rotated files kept


class UIConfig(BaseModel):
    # Orbe de estado: ventana pequeña que aparece al activar la conversación.
    orb: bool = True


class WakeWordConfig(BaseModel):
    enabled: bool = False
    phrase: str = "Hola Samantha"
    model: str = "vosk-model-small-es-0.42"
    # Saludo de arranque: al iniciar, la frase se envía al modelo como si
    # la hubieras dicho; Samantha responde y (en modo continuo) se queda
    # escuchando, sin necesidad de decir la wake word.
    greet_on_start: bool = False
    greet_text: str = ""  # vacío = usar la frase de la wake word
    greet_delay: float = 1.5  # segundos tras el arranque


class AppConfig(BaseModel):
    persona: str = "samantha"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# --- Paths ---

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

# Modo portable: exe congelado (PyInstaller) o OCTO_PORTABLE=1. En ese modo
# todo lo escribible (config, datos, modelos, personas) vive junto al exe.
FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else PROJECT_ROOT
PORTABLE = FROZEN or os.environ.get("OCTO_PORTABLE") == "1"

if PORTABLE:
    USER_CONFIG_DIR = APP_DIR / "data"
    USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"
    MODELS_DIR = APP_DIR / "models"
    PERSONAS_DIR = APP_DIR / "personas"
else:
    USER_CONFIG_DIR = Path.home() / ".openocto"
    USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"
    MODELS_DIR = USER_CONFIG_DIR / "models"
    PERSONAS_DIR = PROJECT_ROOT / "personas"


# --- Loader ---


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration with resolution order:
    default.yaml -> ~/.openocto/config.yaml -> custom path -> env vars
    """
    data: dict = {}

    # 1. Default config
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            default_data = yaml.safe_load(f) or {}
            data = _deep_merge(data, default_data)

    # 2. User config
    if USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH, encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}
            data = _deep_merge(data, user_data)

    # 3. Custom config path (from CLI)
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                custom_data = yaml.safe_load(f) or {}
                data = _deep_merge(data, custom_data)

    # 4. Resolve environment variables in all string values
    data = _resolve_env_recursive(data)

    return AppConfig(**data)
