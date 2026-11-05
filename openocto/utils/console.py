"""Salida de consola: colores ANSI y helpers de la conversación.

- Colores solo si la salida es una terminal (o OCTO_COLOR=1) y no hay NO_COLOR.
- En Windows habilita VIRTUAL_TERMINAL_PROCESSING (consola clásica);
  Windows Terminal y PowerShell 7 ya lo traen.
"""

from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def _enable() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not (sys.stdout.isatty() or os.environ.get("OCTO_COLOR") == "1"):
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VT
        except Exception:
            return False
    return True


COLOR = _enable()


def c(text: str, *codes: str) -> str:
    if not COLOR or not codes:
        return text
    return "".join(codes) + text + RESET


def dim(text: str) -> str:
    return c(text, DIM)


def ok(text: str) -> str:
    return c(text, GREEN)


def warn(text: str) -> str:
    return c(text, YELLOW)


def err(text: str) -> str:
    return c(text, RED)


def info(text: str) -> str:
    return c(text, BLUE)


def header(text: str) -> str:
    return c(text, BOLD, CYAN)


def user(text: str) -> str:
    return c("Tú: ", BOLD, CYAN) + text


def ai_label(name: str) -> str:
    return c(f"{name}: ", BOLD, GREEN)


def clear_line() -> None:
    """Borra la línea actual (spinner/estado) antes de escribir contenido."""
    if COLOR:
        sys.stdout.write("\r\033[K")
    else:
        sys.stdout.write("\r" + " " * 72 + "\r")
    sys.stdout.flush()
