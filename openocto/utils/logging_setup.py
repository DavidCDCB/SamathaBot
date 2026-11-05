"""Logging: consola colorida y archivo.

Por defecto la consola solo muestra WARNING+ y el archivo guarda DEBUG
(`python assistant.py start --debug` sube la consola a DEBUG). Los loggers
ruidosos de terceros se fijan en WARNING siempre para no ensuciar la
conversación ni siquiera en modo debug.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from openocto.config import APP_DIR, LoggingConfig
from openocto.utils import console

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_LEVEL_COLORS = {
    logging.DEBUG: console.DIM,
    logging.INFO: console.BLUE,
    logging.WARNING: console.YELLOW,
    logging.ERROR: console.RED,
    logging.CRITICAL: console.BOLD + console.RED,
}


class ColorFormatter(logging.Formatter):
    """Nivel coloreado; logger atenuado. Los mensajes quedan sin color."""

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        color = _LEVEL_COLORS.get(record.levelno, "")
        if console.COLOR and color:
            record.levelname = color + level + console.RESET
            record.name = console.DIM + record.name + console.RESET
            try:
                return super().format(record)
            finally:
                record.levelname = level  # restaurar para otros handlers
                record.name = record.name.replace(console.DIM, "").replace(
                    console.RESET, ""
                )
        return super().format(record)


def _parse_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def setup_logging(config: LoggingConfig) -> None:
    """Configura consola + archivo rotativo con niveles independientes."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # los handlers deciden qué pasa

    for handler in list(root.handlers):
        root.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_parse_level(getattr(config, "console_level", "WARNING")))
    console_handler.setFormatter(ColorFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    if config.file:
        log_path = Path(config.file).expanduser()
        if not log_path.is_absolute():
            log_path = APP_DIR / log_path  # portable: junto al exe
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(_parse_level(config.level))
            file_handler.setFormatter(
                logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
            )
            root.addHandler(file_handler)
            root.info("Log de archivo: %s (nivel %s)", log_path, config.level)
        except OSError as e:
            root.warning("No se pudo abrir el log %s: %s", log_path, e)

    # Terceros ruidosos: siempre WARNING, incluso con --debug.
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "httpcore.connection",
        "httpcore2.connection",
        "httpcore.http11",
        "httpcore2.http11",
        "truststore",
        "openai",
        "openai._base_client",
        "aiohttp.access",
        "urllib3.connectionpool",
        "pywhispercpp.model",
        "piper.voice",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
