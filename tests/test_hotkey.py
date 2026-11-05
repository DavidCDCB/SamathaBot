"""Pruebas de la combinación del push-to-talk (texto -> piezas). Sin Windows."""

import unittest

from openocto.utils.hotkey import (
    DEFAULT_HOTKEY,
    Hotkey,
    hotkey_label,
    parse_hotkey,
    win_flags,
)


class HotkeyTest(unittest.TestCase):
    def test_por_defecto_es_ctrl_shift_space(self):
        self.assertEqual(DEFAULT_HOTKEY, "ctrl+shift+space")
        self.assertEqual(
            parse_hotkey(DEFAULT_HOTKEY), Hotkey(frozenset({"ctrl", "shift"}), "space")
        )

    def test_acepta_mayusculas_y_espacios(self):
        self.assertEqual(
            parse_hotkey(" Ctrl + Alt + Enter "),
            Hotkey(frozenset({"ctrl", "alt"}), "enter"),
        )

    def test_etiqueta_para_mostrar(self):
        self.assertEqual(
            hotkey_label(parse_hotkey("ctrl+shift+space")), "Ctrl+Shift+Space"
        )
        self.assertEqual(hotkey_label(parse_hotkey("win+alt+f12")), "Alt+Win+F12")

    def test_banderas_de_registerhotkey(self):
        self.assertEqual(
            win_flags(parse_hotkey("ctrl+shift+space")), (0x0002 | 0x0004, 0x20)
        )
        self.assertEqual(win_flags(parse_hotkey("ctrl+shift+f12"))[1], 0x7B)
        self.assertEqual(win_flags(parse_hotkey("ctrl+shift+a"))[1], 0x41)
        self.assertEqual(win_flags(parse_hotkey("ctrl+shift+7"))[1], 0x37)

    def test_rechaza_lo_que_no_entiende(self):
        for malo in [
            "ctrl+foo",
            "hyper+space",
            "ctrl+shift+",
            "",
            "ctrl+ctrl+space",
            "ctrl+shift+f13",
        ]:
            with self.subTest(combinacion=malo):
                with self.assertRaises(ValueError):
                    parse_hotkey(malo)

    def test_la_config_rechaza_una_combinacion_no_valida(self):
        from pydantic import ValidationError

        from openocto.config import AudioConfig

        with self.assertRaises(ValidationError):
            AudioConfig(ptt_hotkey="ctrl+foo")
        self.assertEqual(AudioConfig().ptt_hotkey, DEFAULT_HOTKEY)


if __name__ == "__main__":
    unittest.main()
