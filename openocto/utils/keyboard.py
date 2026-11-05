"""Push-to-talk global (sin UI, sin wake word).

La combinación viene de audio.ptt_hotkey (por defecto Ctrl+Shift+Space).
Según audio.ptt_mode:
- "vad": RegisterHotKey de Windows. No instala ningún hook de teclado: el
  sistema solo avisa al pulsar, y en vad soltar la tecla no hace nada.
- "hold": pynput. Instala un hook global que ve todas las pulsaciones del
  sistema; hace falta para saber cuándo se suelta la tecla.
Si RegisterHotKey no puede registrar la combinación (otra aplicación la usa),
vad cae a pynput con un aviso.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import sys
import threading
from ctypes import wintypes
from typing import Callable

from pynput import keyboard

from openocto.utils.hotkey import (
    DEFAULT_HOTKEY,
    Hotkey,
    hotkey_label,
    parse_hotkey,
    win_flags,
)

logger = logging.getLogger(__name__)

# ── RegisterHotKey (Win32) ───────────────────────────────────────────────────
MOD_NOREPEAT = 0x4000  # Windows 7+: no repite el aviso mientras se mantiene pulsada
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
HOTKEY_ID = 0x0A11

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.RegisterHotKey.argtypes = [
        wintypes.HWND,
        ctypes.c_int,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.RegisterHotKey.restype = wintypes.BOOL
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UnregisterHotKey.restype = wintypes.BOOL
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.PeekMessageW.restype = wintypes.BOOL
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.GetMessageW.restype = ctypes.c_int  # -1 = error, 0 = WM_QUIT
    user32.PostThreadMessageW.argtypes = [
        wintypes.DWORD,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.PostThreadMessageW.restype = wintypes.BOOL
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD

# ── pynput (hook de teclado) ─────────────────────────────────────────────────
_CTRL = (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
_ALT = (
    keyboard.Key.alt,
    keyboard.Key.alt_l,
    keyboard.Key.alt_r,
    getattr(keyboard.Key, "alt_gr", None),
)
_SHIFT = (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r)
_WIN = (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)


def _key_name(key: object) -> str:
    """Nombre canónico de una tecla de pynput: ctrl, alt, shift, win, letra/dígito o nombre."""
    if key in _CTRL:
        return "ctrl"
    if key in _ALT:
        return "alt"
    if key in _SHIFT:
        return "shift"
    if key in _WIN:
        return "win"
    vk = getattr(key, "vk", None)  # con Ctrl pulsada, pynput puede no dar el carácter
    if vk is not None and (0x41 <= vk <= 0x5A or 0x30 <= vk <= 0x39):
        return chr(vk).lower()
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    return getattr(key, "name", "") or "?"


class ComboState:
    """Máquina pura y testeable de la combinación.

    feed(nombre, presionada) -> "start" | "stop" | None
    """

    def __init__(self, hotkey: Hotkey = parse_hotkey(DEFAULT_HOTKEY)) -> None:
        self._mods = hotkey.mods
        self._tecla = hotkey.tecla
        self._down: set[str] = set()
        self._active = False

    def feed(self, name: str, pressed: bool) -> str | None:
        if pressed:
            self._down.add(name)
            if name == self._tecla and self._mods <= self._down and not self._active:
                self._active = True
                return "start"
            return None
        self._down.discard(name)
        if self._active and (name == self._tecla or name in self._mods):
            self._active = False
            return "stop"
        return None


class PushToTalkListener:
    """Escucha global con pynput: on_press al pulsar la combinación y on_release al soltarla."""

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        hotkey: Hotkey = parse_hotkey(DEFAULT_HOTKEY),
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._state = ComboState(hotkey)
        self._label = hotkey_label(hotkey)
        self._listener: keyboard.Listener | None = None

    def _handle_press(self, key: object) -> None:
        if self._state.feed(_key_name(key), True) == "start":
            logger.debug("PTT combo pressed")
            self._on_press()

    def _handle_release(self, key: object) -> None:
        if self._state.feed(_key_name(key), False) == "stop":
            logger.debug("PTT combo released")
            self._on_release()

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._handle_press, on_release=self._handle_release
        )
        self._listener.start()
        logger.info("PTT listener started (combo: %s)", self._label)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None


class HotkeyPressListener:
    """Avisa al pulsar una combinación global con RegisterHotKey. Solo Windows.

    La combinación se registra en un hilo propio con su cola de mensajes: el
    sistema postea WM_HOTKEY a ese hilo y el callback corre allí.
    """

    def __init__(self, on_press: Callable[[], None], mods: int, vk: int) -> None:
        self._on_press = on_press
        self._mods = mods | MOD_NOREPEAT
        self._vk = vk
        self._thread: threading.Thread | None = None
        self._tid: int | None = None
        self._ready = threading.Event()
        self._registered = False

    def start(self) -> bool:
        """Registra la combinación. Devuelve False si Windows no la deja registrar."""
        self._thread = threading.Thread(
            target=self._run, name="ptt-hotkey", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=5)
        return self._registered

    def _run(self) -> None:
        msg = wintypes.MSG()
        self._tid = kernel32.GetCurrentThreadId()
        # Crea la cola de mensajes del hilo antes de registrar y de avisar a start()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        self._registered = bool(
            user32.RegisterHotKey(None, HOTKEY_ID, self._mods, self._vk)
        )
        self._ready.set()
        if not self._registered:
            return
        try:
            while True:
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret <= 0:  # 0 = WM_QUIT; -1 = error
                    break
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                    try:
                        self._on_press()
                    except Exception:
                        logger.exception("PTT: fallo en el callback de la tecla")
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID)

    def stop(self) -> None:
        if (
            self._thread is not None
            and self._thread.is_alive()
            and self._tid is not None
        ):
            user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
            self._thread.join(timeout=2)


class AsyncPushToTalkListener:
    """Puentea los callbacks al loop de asyncio.

    mode "vad": RegisterHotKey, sin hook. Si Windows no deja registrar la
    combinación, cae a pynput con un aviso. mode "hold": pynput.
    """

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        mode: str = "hold",
        combo: str = DEFAULT_HOTKEY,
    ) -> None:
        self._loop = asyncio.get_event_loop()
        hotkey = parse_hotkey(combo)
        self._label = hotkey_label(hotkey)
        self._sync = PushToTalkListener(
            lambda: self._loop.call_soon_threadsafe(on_press),
            lambda: self._loop.call_soon_threadsafe(on_release),
            hotkey=hotkey,
        )
        self._hotkey: HotkeyPressListener | None = None
        if mode == "vad" and sys.platform == "win32":
            mods, vk = win_flags(hotkey)
            self._hotkey = HotkeyPressListener(
                lambda: self._loop.call_soon_threadsafe(on_press), mods=mods, vk=vk
            )

    def start(self) -> None:
        if self._hotkey is not None:
            if self._hotkey.start():
                logger.info(
                    "PTT %s con RegisterHotKey (sin hook de teclado)", self._label
                )
                return
            logger.warning(
                "RegisterHotKey no pudo registrar %s; uso pynput (hook global de teclado)",
                self._label,
            )
            self._hotkey = None
        self._sync.start()

    def stop(self) -> None:
        if self._hotkey is not None:
            self._hotkey.stop()
        self._sync.stop()
