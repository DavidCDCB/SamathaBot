"""Pruebas de las funciones puras de memoria (solo stdlib)."""

import unittest

from openocto.memory import MemoryManager


class ParseRespuestaTest(unittest.TestCase):
    def setUp(self):
        self.mm = MemoryManager(None, None)

    def test_separa_resumen_notas_y_hechos(self):
        texto = (
            "## SUMMARY\nResumen de prueba\n"
            "## NOTES\n- Llamar a mamá el viernes\nRESOLVED: pagar la factura\n"
            "## FACTS\n[personal] Se llama Ana\nSin categoría\n"
        )
        resumen, notas, resueltas, hechos = self.mm._parse_response(texto)
        self.assertEqual(resumen, "Resumen de prueba")
        self.assertEqual(notas, ["Llamar a mamá el viernes"])
        self.assertEqual(resueltas, ["pagar la factura"])
        self.assertEqual(
            hechos, [("personal", "Se llama Ana"), ("personal", "Sin categoría")]
        )

    def test_none_no_genera_nada(self):
        _, notas, resueltas, hechos = self.mm._parse_response(
            "## NOTES\nNone\n## FACTS\nNone\n"
        )
        self.assertEqual((notas, resueltas, hechos), ([], [], []))


class ValidarResumenTest(unittest.TestCase):
    def setUp(self):
        self.mm = MemoryManager(None, None)

    def test_resumen_corto_con_muchos_mensajes_no_vale(self):
        self.assertFalse(self.mm._validate_summary("Corto", 30))

    def test_resumen_corto_con_pocos_mensajes_vale(self):
        self.assertTrue(self.mm._validate_summary("Corto", 5))

    def test_resumen_vacio_no_vale(self):
        self.assertFalse(self.mm._validate_summary("   ", 5))


class HechosParecidosTest(unittest.TestCase):
    def test_ignora_mayusculas_y_punto_final(self):
        self.assertTrue(
            MemoryManager._facts_similar("Vive en Bogotá.", "vive en bogotá")
        )

    def test_hechos_distintos(self):
        self.assertFalse(
            MemoryManager._facts_similar("Le gusta el café", "Trabaja en Bogotá")
        )


if __name__ == "__main__":
    unittest.main()
