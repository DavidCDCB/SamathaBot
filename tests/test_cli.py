"""Pruebas del comando por defecto del CLI (doble clic en el .exe)."""

import unittest

from assistant import FIRECRAWL_SECRET, _argv_por_defecto


class ComandoPorDefectoTest(unittest.TestCase):
    def test_sin_argumentos_arranca_con_start(self):
        self.assertEqual(_argv_por_defecto([]), ["start"])

    def test_comandos_explicitos_no_cambian(self):
        for argv in (
            ["start"],
            ["start", "--debug"],
            ["key"],
            ["key", "firecrawl"],
            ["check"],
            ["report"],
            ["say", "hola"],
            ["--help"],
        ):
            with self.subTest(argv=argv):
                self.assertEqual(_argv_por_defecto(argv), argv)


class FirecrawlSecretTest(unittest.TestCase):
    def test_firecrawl_secret_path(self):
        self.assertEqual(FIRECRAWL_SECRET.name, "FIRECRAWL_API_KEY")
        self.assertEqual(FIRECRAWL_SECRET.parent.name, "secret")


if __name__ == "__main__":
    unittest.main()
