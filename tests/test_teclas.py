"""Pruebas de RegisterHotKey (solo Windows). No pulsa teclas reales."""

import sys
import threading
import unittest

from openocto.utils import keyboard as kb
from openocto.utils.hotkey import parse_hotkey, win_flags

# Combinación poco usada para no chocar con nada de tu equipo: Ctrl+Alt+Shift+F12
MODS_PRUEBA, VK_PRUEBA = win_flags(parse_hotkey("ctrl+alt+shift+f12"))


@unittest.skipUnless(sys.platform == "win32", "RegisterHotKey es de Windows")
class RegisterHotKeyTest(unittest.TestCase):
    def test_mensaje_wm_hotkey_dispara_el_callback(self):
        disparos = []
        recibido = threading.Event()

        def al_pulsar():
            disparos.append(1)
            recibido.set()

        h = kb.HotkeyPressListener(al_pulsar, mods=MODS_PRUEBA, vk=VK_PRUEBA)
        self.assertTrue(
            h.start(), "RegisterHotKey no pudo registrar la combinación de prueba"
        )
        try:
            # Simula la pulsación: el sistema postea WM_HOTKEY a la cola del hilo del listener
            kb.user32.PostThreadMessageW(h._tid, kb.WM_HOTKEY, kb.HOTKEY_ID, 0)
            self.assertTrue(recibido.wait(2), "el callback no se llamó")
        finally:
            h.stop()
        self.assertEqual(len(disparos), 1)

    def test_al_parar_libera_la_combinacion(self):
        h = kb.HotkeyPressListener(lambda: None, mods=MODS_PRUEBA, vk=VK_PRUEBA)
        self.assertTrue(h.start())
        h.stop()
        otro = kb.HotkeyPressListener(lambda: None, mods=MODS_PRUEBA, vk=VK_PRUEBA)
        try:
            self.assertTrue(otro.start(), "la combinación sigue registrada tras parar")
        finally:
            otro.stop()


if __name__ == "__main__":
    unittest.main()
