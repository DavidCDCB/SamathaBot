"""Pruebas del matcher de la frase de activación (sin cargar Vosk)."""

import unittest
from pathlib import Path

from openocto.wakeword.vosk_detector import (
    _coincide,
    _fonetica,
    _palabras,
    _prepara_objetivos,
)


class FoneticaTest(unittest.TestCase):
    def test_colapsa_confusiones_tipicas(self):
        self.assertEqual(_fonetica("samantha"), "samanta")
        self.assertEqual(_fonetica("zapato"), "sapato")
        self.assertEqual(_fonetica("vaca"), "baca")
        self.assertEqual(_fonetica("pharmacia"), "farmacia")

    def test_quita_h_muda(self):
        self.assertEqual(_fonetica("hola"), "ola")


class CoincidenciaTest(unittest.TestCase):
    def _detecta(self, texto, *variantes):
        variantes = variantes or ("hola samantha",)
        objetivos = _prepara_objetivos(list(variantes))
        return _coincide([_fonetica(w) for w in _palabras(texto)], objetivos)

    def test_frase_exacta(self):
        self.assertIsNotNone(self._detecta("hola samantha"))

    def test_variantes_del_nombre(self):
        self.assertIsNotNone(self._detecta("hola samanta"))
        self.assertIsNotNone(self._detecta("ola sananta"))

    def test_solo_nombre_con_alias(self):
        self.assertIsNotNone(self._detecta("samantha", "hola samantha", "samantha"))

    def test_solo_nombre_sin_alias_no_dispara(self):
        self.assertIsNone(self._detecta("samantha"))

    def test_palabras_comunes_no_disparan(self):
        # "santa"/"manta" se parecen a "samanta" (0,83) pero quedan bajo el
        # umbral 0,84: no deben despertar a Samantha.
        self.assertIsNone(self._detecta("hola santa"))
        self.assertIsNone(self._detecta("hola manta"))
        self.assertIsNone(self._detecta("hoy hace mucho calor"))

    def test_duplicados_foneticos_se_colapsan(self):
        objetivos = _prepara_objetivos(["hola samantha", "hola samanta"])
        self.assertEqual(len(objetivos), 1)


class ConfigTest(unittest.TestCase):
    def test_alias_en_assistant_config(self):
        from openocto.config import load_config

        raiz = Path(__file__).resolve().parents[1]
        cfg = load_config(str(raiz / "assistant-config.yaml"))
        self.assertIn("samantha", cfg.wakeword.aliases)
        self.assertIn("hola samanta", cfg.wakeword.aliases)


if __name__ == "__main__":
    unittest.main()
