"""Pruebas de funciones puras de audio, texto y teclado (requieren el venv con dependencias)."""

import unittest

import numpy as np

from openocto.app import OpenOctoApp, normalize_for_stt
from openocto.stt.whisper_cpp import _HALLUCINATION_RE
from openocto.tts.edge import EdgeTTSEngine
from openocto.utils.keyboard import ComboState


class NormalizeTest(unittest.TestCase):
    def test_silencio_se_devuelve_igual(self):
        audio = np.zeros(1600, dtype=np.int16)
        self.assertIs(normalize_for_stt(audio), audio)

    def test_toma_floja_sube_hasta_el_objetivo(self):
        audio = np.full(1600, 328, dtype=np.int16)  # RMS ~0,010
        out = normalize_for_stt(audio)
        self.assertAlmostEqual(int(out[0]), 3277, delta=2)  # 0,10 * 32768

    def test_no_amplifica_mas_de_20x(self):
        audio = np.full(1600, 10, dtype=np.int16)
        out = normalize_for_stt(audio)
        self.assertGreater(int(out[0]), 10)
        self.assertLessEqual(abs(int(out[0])), 20 * 10 + 1)

    def test_objetivo_y_tope_configurables(self):
        audio = np.full(1600, 328, dtype=np.int16)  # RMS ~0,010
        out = normalize_for_stt(audio, target_rms=0.05, max_gain=4.0)
        self.assertEqual(int(out[0]), 1312)  # 328 * 4 (manda el tope)

    def test_objetivo_mas_bajo_sube_menos(self):
        audio = np.full(1600, 328, dtype=np.int16)
        out = normalize_for_stt(audio, target_rms=0.05)
        self.assertAlmostEqual(int(out[0]), 1638, delta=3)  # ~5x


class AudioConfigTest(unittest.TestCase):
    def test_defaults_de_normalizacion(self):
        from openocto.config import AudioConfig

        c = AudioConfig()
        self.assertTrue(c.normalize)
        self.assertAlmostEqual(c.normalize_target_rms, 0.10)
        self.assertAlmostEqual(c.normalize_max_gain, 20.0)

    def test_assistant_config_trae_los_niveles(self):
        from pathlib import Path

        from openocto.config import load_config

        raiz = Path(__file__).resolve().parents[1]
        cfg = load_config(str(raiz / "assistant-config.yaml"))
        self.assertAlmostEqual(cfg.audio.normalize_target_rms, 0.10)
        self.assertAlmostEqual(cfg.audio.normalize_max_gain, 20.0)


class SplitFraseTest(unittest.TestCase):
    def test_corta_en_punto_seguido_de_espacio(self):
        self.assertEqual(
            OpenOctoApp._split_first_sentence("Hola. ¿Qué tal?"), ("Hola.", "¿Qué tal?")
        )

    def test_no_corta_decimales_ni_miles(self):
        self.assertIsNone(OpenOctoApp._split_first_sentence("cuesta 1.500 pesos"))

    def test_corta_en_salto_de_linea_tras_20_caracteres(self):
        self.assertEqual(
            OpenOctoApp._split_first_sentence(
                "Esta frase no tiene punto final\nSiguiente"
            ),
            ("Esta frase no tiene punto final", "Siguiente"),
        )

    def test_fragmento_corto_sin_cierre_no_corta(self):
        self.assertIsNone(OpenOctoApp._split_first_sentence("Hola"))


class AlucinacionesTest(unittest.TestCase):
    def test_filtra_marcas_entre_corchetes(self):
        self.assertTrue(_HALLUCINATION_RE.match("[música]"))

    def test_filtra_thank_you_for_watching(self):
        self.assertTrue(_HALLUCINATION_RE.match("Thank you for watching."))

    def test_deja_pasar_frases_normales(self):
        self.assertIsNone(_HALLUCINATION_RE.match("Ponme una alarma a las siete"))


class ComboPTTTest(unittest.TestCase):
    def test_ctrl_shift_space_dispara_start_y_al_soltar_stop(self):
        s = ComboState()  # por defecto: Ctrl+Shift+Space
        self.assertIsNone(s.feed("ctrl", True))
        self.assertIsNone(s.feed("shift", True))
        self.assertEqual(s.feed("space", True), "start")
        self.assertEqual(s.feed("space", False), "stop")

    def test_no_repite_start_mientras_se_mantiene(self):
        s = ComboState()
        s.feed("ctrl", True)
        s.feed("shift", True)
        self.assertEqual(s.feed("space", True), "start")
        self.assertIsNone(s.feed("space", True))  # autorepetición

    def test_otra_combinacion_no_dispara(self):
        s = ComboState()
        s.feed("ctrl", True)
        s.feed("alt", True)
        self.assertIsNone(s.feed("space", True))  # Ctrl+Alt+Space ya no es la tecla


class RecorteEdgeTest(unittest.TestCase):
    def test_recorta_silencio_y_deja_pausa_entre_frases(self):
        e = EdgeTTSEngine.__new__(EdgeTTSEngine)  # sin cargar edge_tts
        e._trim = True
        e._gap = 0.25
        audio = np.concatenate(
            [
                np.zeros(12000, np.int16),
                np.full(12000, 1000, np.int16),
                np.zeros(12000, np.int16),
            ]
        )
        # tono de 0,5 s con 30 ms de margen a cada lado + pausa de 0,25 s
        self.assertEqual(len(e._trim_edges(audio)), 13440 + 6000)


if __name__ == "__main__":
    unittest.main()
