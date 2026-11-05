"""Tests for the Spanish date helpers injected into the system prompt."""

from __future__ import annotations

import unittest
from datetime import datetime

from openocto.utils.dates import fecha_es, hora_es


class DatesTest(unittest.TestCase):
    def test_fecha_es_format(self):
        d = datetime(2026, 11, 5, 15, 7)  # jueves
        self.assertEqual(fecha_es(d), "jueves, 5 de noviembre de 2026")

    def test_fecha_es_sin_ceros_y_weekday(self):
        self.assertEqual(fecha_es(datetime(2026, 1, 1)), "jueves, 1 de enero de 2026")
        self.assertEqual(
            fecha_es(datetime(2024, 2, 29)), "jueves, 29 de febrero de 2024"
        )

    def test_hora_es_am_pm(self):
        self.assertEqual(hora_es(datetime(2026, 11, 5, 0, 5)), "12:05 a. m.")
        self.assertEqual(hora_es(datetime(2026, 11, 5, 12, 0)), "12:00 p. m.")
        self.assertEqual(hora_es(datetime(2026, 11, 5, 15, 7)), "3:07 p. m.")


class FechaBlockTest(unittest.TestCase):
    def test_bloque_contiene_fecha_hoy(self):
        from openocto.app import OpenOctoApp

        app = object.__new__(OpenOctoApp)
        bloque = app._fecha_block()
        self.assertIn("## Fecha y hora actuales", bloque)
        self.assertIn(fecha_es(), bloque)
        self.assertIn("no lo busques en la web", bloque)


if __name__ == "__main__":
    unittest.main()
