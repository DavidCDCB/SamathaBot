"""Combinaciones de teclas del push-to-talk: texto <-> piezas para RegisterHotKey y pynput.

Módulo puro (sin Windows ni pynput), para validarlo en config.py y en las pruebas.
Ejemplos válidos: "ctrl+shift+space", "ctrl+alt+enter", "win+alt+f12", "pause".
"""

from __future__ import annotations

from typing import NamedTuple

DEFAULT_HOTKEY = "ctrl+shift+space"

# Modificadores de RegisterHotKey (Win32)
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

_MODS = {"ctrl": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}
_VK_ESPECIALES = {"space": 0x20, "enter": 0x0D, "pause": 0x13, "scroll_lock": 0x91}
_ETIQUETAS = {
    "space": "Space",
    "enter": "Enter",
    "pause": "Pause",
    "scroll_lock": "Scroll Lock",
}


class Hotkey(NamedTuple):
    mods: frozenset[str]  # subconjunto de ctrl, alt, shift, win
    tecla: str  # space, enter, pause, scroll_lock, letra a-z, dígito 0-9 o f1-f12


def _es_tecla(nombre: str) -> bool:
    if nombre in _VK_ESPECIALES:
        return True
    if len(nombre) == 1 and nombre.isascii() and (nombre.isalpha() or nombre.isdigit()):
        return True
    return nombre[:1] == "f" and nombre[1:].isdigit() and 1 <= int(nombre[1:]) <= 12


def parse_hotkey(texto: str) -> Hotkey:
    """'Ctrl+Shift+Space' -> Hotkey(frozenset({'ctrl', 'shift'}), 'space').

    Lanza ValueError si la combinación no se entiende.
    """
    partes = [p.strip().lower() for p in texto.split("+")]
    if any(not p for p in partes):
        raise ValueError(f"combinación no válida: {texto!r}")
    *mods, tecla = partes
    if any(m not in _MODS for m in mods):
        raise ValueError(
            f"modificador no válido en {texto!r} (usa ctrl, alt, shift o win)"
        )
    if len(set(mods)) != len(mods):
        raise ValueError(f"modificador repetido en {texto!r}")
    if not _es_tecla(tecla):
        raise ValueError(
            f"tecla no válida en {texto!r} (letra, dígito, f1-f12, space, enter, pause o scroll_lock)"
        )
    return Hotkey(frozenset(mods), tecla)


def hotkey_label(hk: Hotkey) -> str:
    """Hotkey -> texto para mostrar: 'Ctrl+Shift+Space'."""
    nombres = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}
    mods = [nombres[m] for m in ("ctrl", "alt", "shift", "win") if m in hk.mods]
    tecla = _ETIQUETAS.get(hk.tecla, hk.tecla.upper())
    return "+".join([*mods, tecla])


def win_flags(hk: Hotkey) -> tuple[int, int]:
    """(modificadores, código virtual) para RegisterHotKey."""
    mods = 0
    for m in hk.mods:
        mods |= _MODS[m]
    if hk.tecla in _VK_ESPECIALES:
        vk = _VK_ESPECIALES[hk.tecla]
    elif hk.tecla[:1] == "f" and hk.tecla[1:].isdigit():
        vk = 0x70 + int(hk.tecla[1:]) - 1  # VK_F1 = 0x70
    else:
        vk = ord(hk.tecla.upper())  # letras 0x41-0x5A, dígitos 0x30-0x39
    return mods, vk
