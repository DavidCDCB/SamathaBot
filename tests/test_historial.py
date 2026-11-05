"""Pruebas de HistoryStore con una base temporal (requieren el venv con dependencias)."""

import tempfile
import unittest
from pathlib import Path

from openocto.history import HistoryStore


class HistorialTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = HistoryStore(Path(self._tmp.name) / "test.db")
        self.uid = self.store.create_user("Prueba", is_default=True)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_borrar_hecho_lo_saca_de_los_activos(self):
        f1 = self.store.add_fact(self.uid, "Vive en Bogotá")
        self.store.add_fact(self.uid, "Le gusta el café")
        self.store.deactivate_fact(f1)
        activos = [f["fact"] for f in self.store.get_active_facts(self.uid)]
        self.assertEqual(activos, ["Le gusta el café"])

    def test_recent_messages_devuelve_los_ultimos_turnos_pedidos(self):
        for i in range(10):
            self.store.add_message(self.uid, "samantha", "user", f"pregunta {i}")
            self.store.add_message(self.uid, "samantha", "assistant", f"respuesta {i}")
        recientes = self.store.get_recent_messages(self.uid, "samantha", limit=3)
        self.assertEqual(len(recientes), 6)  # 3 turnos = 6 mensajes
        self.assertEqual(recientes[0]["content"], "pregunta 7")
        self.assertEqual(recientes[-1]["content"], "respuesta 9")


if __name__ == "__main__":
    unittest.main()
