"""Pruebas del comando por defecto del CLI (doble clic en el .exe)."""

import unittest

from assistant import FIRECRAWL_SECRET, _argv_por_defecto, _quitar_flag_consola


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


class ConsolaTest(unittest.TestCase):
    def test_console_flag_se_quita(self):
        self.assertEqual(_quitar_flag_consola(["--console"]), [])
        self.assertEqual(_quitar_flag_consola(["start", "--console"]), ["start"])
        self.assertEqual(_quitar_flag_consola(["--terminal", "check"]), ["check"])
        self.assertEqual(_quitar_flag_consola(["say", "hola"]), ["say", "hola"])

    def test_solo_console_equivale_a_start(self):
        self.assertEqual(
            _argv_por_defecto(_quitar_flag_consola(["--console"])),
            ["start"],
        )


class FirecrawlSecretTest(unittest.TestCase):
    def test_firecrawl_secret_path(self):
        self.assertEqual(FIRECRAWL_SECRET.name, "FIRECRAWL_API_KEY")
        self.assertEqual(FIRECRAWL_SECRET.parent.name, "secret")


if __name__ == "__main__":
    unittest.main()
