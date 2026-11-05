"""Pruebas del orbe de estado: funciones puras y controlador sin ventana (sin Tk)."""

import math
import queue
import unittest

from openocto.ui import orb
from openocto.ui.orb import (
    ESTILOS,
    SIZE,
    OrbController,
    bgra_premultiplicado,
    dibuja_fotograma,
    mezcla,
    nivel_normalizado,
    parametros,
    puntos_simbolo,
)

ESTADOS = (
    "listening",
    "hearing",
    "transcribing",
    "thinking",
    "speaking",
    "waiting",
    "error",
)


class OrbPuroTest(unittest.TestCase):
    def test_mezcla_entre_dos_colores(self):
        self.assertEqual(mezcla("#000000", "#ffffff", 0), "#000000")
        self.assertEqual(mezcla("#000000", "#ffffff", 1), "#ffffff")
        self.assertEqual(
            mezcla("#000000", "#ffffff", 0.5), "#808080"
        )  # round(127.5) = 128

    def test_nivel_normalizado_acotado_y_creciente(self):
        self.assertEqual(nivel_normalizado(0), 0.0)
        self.assertEqual(nivel_normalizado(10**6), 1.0)
        self.assertLess(nivel_normalizado(150), nivel_normalizado(1500))

    def test_todos_los_estados_tienen_estilo_y_parametros_validos(self):
        for estado in ESTADOS:
            self.assertIn(estado, ESTILOS)
            for t in (0.0, 0.37, 1.9):
                for nivel in (0.0, 0.5, 1.0):
                    p = parametros(estado, t, nivel)
                    self.assertGreaterEqual(p["radio"], 0.25)
                    self.assertLessEqual(p["radio"], 0.95)
                    self.assertGreaterEqual(p["brillo"], 0.0)
                    self.assertLessEqual(p["brillo"], 1.0)
                    self.assertEqual(p["color"], ESTILOS[estado].color)

    def test_escuchar_la_voz_crece_con_el_nivel(self):
        bajo = parametros("hearing", 0.0, 0.1)["radio"]
        alto = parametros("hearing", 0.0, 0.9)["radio"]
        self.assertGreater(alto, bajo)

    def test_cada_estado_tiene_un_color_distinto(self):
        colores = {ESTILOS[e].color for e in ESTADOS}
        self.assertEqual(len(colores), len(ESTADOS))

    def test_simbolo_infinito_centrado_y_proporcional(self):
        radio = 100.0
        pts = puntos_simbolo(0.0, 0.0, radio)
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        self.assertAlmostEqual(max(xs), radio * 0.62, places=6)
        self.assertAlmostEqual(min(xs), -radio * 0.62, places=6)
        # el máximo de sin·cos cae entre muestras: tolerancia de medio píxel
        self.assertAlmostEqual(max(ys), radio * 0.23, delta=0.05)
        self.assertAlmostEqual(min(ys), -radio * 0.23, delta=0.05)
        # simetría central: por cada punto existe su opuesto
        for x, y in pts[:20]:
            self.assertTrue(
                any(abs(-x - x2) < 1e-9 and abs(-y - y2) < 1e-9 for x2, y2 in pts)
            )
        # todo el trazo cae dentro del cuerpo del orbe
        self.assertTrue(all(math.hypot(x, y) < radio for x, y in pts))


class OrbControladorTest(unittest.TestCase):
    def test_desactivado_no_hace_nada_y_no_falla(self):
        orbe = OrbController(enabled=False)
        orbe.start()
        orbe.show()
        orbe.state("thinking")
        orbe.level(900)
        orbe.end(error=True)
        orbe.close()
        self.assertEqual(orbe._cola.qsize(), 0)

    def test_sin_ventana_las_ordenes_se_encolan_en_orden(self):
        orbe = OrbController(enabled=True)  # sin start(): no se abre ninguna ventana
        orbe.show()
        orbe.state("thinking")
        orbe.end()
        self.assertEqual(
            list(orbe._cola.queue),
            [("show", None), ("state", "thinking"), ("end", False)],
        )


class OrbRenderTest(unittest.TestCase):
    """Dibujo con Pillow (sin abrir ventanas)."""

    def test_fotograma_rgba_del_tamano_de_la_ventana(self):
        img = dibuja_fotograma("listening", 0.0, 0.0)
        self.assertEqual(img.mode, "RGBA")
        self.assertEqual(img.size, (SIZE, SIZE))

    def test_escala_da_el_tamano_fisico_de_la_ventana(self):
        img = dibuja_fotograma("listening", 0.0, 0.0, 1.5)
        self.assertEqual(img.size, (330, 330))

    def test_fuera_del_orbe_es_transparente_y_el_centro_opaco(self):
        img = dibuja_fotograma("speaking", 0.0, 0.0)
        self.assertEqual(img.getpixel((0, 0))[3], 0)
        self.assertEqual(img.getpixel((SIZE - 1, SIZE - 1))[3], 0)
        self.assertEqual(img.getpixel((SIZE // 2, SIZE // 2))[3], 255)

    def test_el_nivel_del_micro_hace_crecer_el_orbe(self):
        bajo = dibuja_fotograma("hearing", 0.0, 0.1).getchannel("A").histogram()[255]
        alto = dibuja_fotograma("hearing", 0.0, 0.9).getchannel("A").histogram()[255]
        self.assertGreater(alto, bajo)

    def test_sin_efectos_omite_corona_y_ondas(self):
        def alfa_total(img):
            return sum(v * n for v, n in enumerate(img.getchannel("A").histogram()))

        con = dibuja_fotograma("speaking", 0.0, 0.0)
        sin = dibuja_fotograma("speaking", 0.0, 0.0, efectos=False)
        self.assertEqual(sin.mode, "RGBA")
        self.assertEqual(sin.size, (SIZE, SIZE))
        self.assertEqual(sin.getpixel((0, 0))[3], 0)
        self.assertEqual(sin.getpixel((SIZE // 2, SIZE // 2))[3], 255)
        self.assertLess(alfa_total(sin), alfa_total(con))

    def test_el_centro_tiene_el_simbolo_crema(self):
        img = dibuja_fotograma("listening", 0.4, 0.0)
        rojo, verde, azul, alfa = img.getpixel((SIZE // 2, SIZE // 2))
        self.assertEqual(alfa, 255)
        self.assertGreater(rojo, 180)
        self.assertGreater(verde, 150)
        self.assertGreater(azul, 100)

    def test_sin_brillo_especular_en_el_interior(self):
        img = dibuja_fotograma("listening", 0.4, 0.0)
        p = parametros("listening", 0.4, 0.0)
        radio = (SIZE / 2 - 16) * p["radio"]
        x = int(round(SIZE / 2 - radio * 0.40))
        y = int(round(SIZE / 2 - radio * 0.44))
        rojo, verde, azul, _ = img.getpixel((x, y))
        # antes había una chispa blanca aquí; ahora es el color del cuerpo
        self.assertLess(max(rojo, verde, azul), 250)

    def test_bgra_premultiplicado_ordena_bytes_y_multiplica_el_color(self):
        from PIL import Image

        semi = Image.new("RGBA", (1, 1), (255, 0, 0, 128))
        self.assertEqual(bgra_premultiplicado(semi), bytes([0, 0, 128, 128]))
        vacio = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        self.assertEqual(bgra_premultiplicado(vacio), bytes([0, 0, 0, 0]))

    def test_bgra_tiene_cuatro_bytes_por_pixel(self):
        datos = bgra_premultiplicado(dibuja_fotograma("waiting", 0.0, 0.0))
        self.assertEqual(len(datos), SIZE * SIZE * 4)

    def test_sin_pillow_no_se_intenta_la_ventana_con_alfa(self):
        original = orb.Image
        orb.Image = None
        try:
            with self.assertRaises(OSError):
                orb._VentanaAlfa(queue.Queue(), lambda: 0.0)
        finally:
            orb.Image = original


if __name__ == "__main__":
    unittest.main()
