"""Pruebas del seguimiento del micrófono predeterminado de Windows.

No abren audio real: el stream y sounddevice se sustituyen por dobles.
"""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from openocto.audio import capture as captura_mod
from openocto.audio.capture import AudioCapture


def _captura(**cambios):
    cap = object.__new__(AudioCapture)
    cap._follow_default = True
    cap._stream = object()
    cap._stream_device = 1
    cap._last_callback = 100.0
    cap._stream_wanted = True
    cap._last_open_try = 0.0
    cap._watchdog = None
    cap._current_default_input = lambda: 1
    acciones = []
    cap._reopen_stream = lambda: acciones.append("reopen")
    for clave, valor in cambios.items():
        setattr(cap, clave, valor)
    return cap, acciones


class WatchdogTest(unittest.TestCase):
    def test_reabre_si_cambia_el_predeterminado(self):
        cap, acciones = _captura()
        cap._current_default_input = lambda: 2
        self.assertEqual(cap._check_stream_health(now=100.5), "reopen")
        self.assertEqual(acciones, ["reopen"])

    def test_no_hace_nada_si_esta_sano(self):
        cap, acciones = _captura()
        self.assertIsNone(cap._check_stream_health(now=100.5))
        self.assertEqual(acciones, [])

    def test_reabre_si_dejan_de_llegar_chunks(self):
        cap, acciones = _captura()
        cap._last_callback = 90.0
        self.assertEqual(cap._check_stream_health(now=100.0), "reopen")
        self.assertEqual(acciones, ["reopen"])

    def test_no_actua_si_el_stream_no_es_deseado(self):
        cap, acciones = _captura()
        cap._stream_wanted = False
        cap._current_default_input = lambda: 2
        self.assertIsNone(cap._check_stream_health(now=200.0))
        self.assertEqual(acciones, [])

    def test_reintento_con_backoff_si_no_hay_stream(self):
        cap, acciones = _captura()
        cap._stream = None
        cap._last_open_try = 99.0
        self.assertIsNone(cap._check_stream_health(now=100.0))
        self.assertEqual(acciones, [])
        self.assertEqual(cap._check_stream_health(now=102.1), "open")
        self.assertEqual(acciones, ["reopen"])


class _StreamFalso:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        pass


class _SdFalso:
    class PortAudioError(Exception):
        pass

    def __init__(self, default=2, falla_dispositivo=None):
        self.default = default
        self.falla_dispositivo = falla_dispositivo
        self.instancias = []

    def query_devices(self, kind=None):
        return {"index": self.default, "name": "Fake mic"}

    def InputStream(self, **kwargs):
        if (
            self.falla_dispositivo is not None
            and kwargs.get("device") == self.falla_dispositivo
        ):
            raise self.PortAudioError("dispositivo ausente")
        instancia = _StreamFalso(**kwargs)
        self.instancias.append(instancia)
        return instancia


def _captura_para_abrir(**cambios):
    cap = object.__new__(AudioCapture)
    cap._sample_rate = 16000
    cap._blocksize = 1280
    cap._input_device = None
    cap._follow_default = True
    cap._stream_device = None
    cap._stream = None
    cap._last_callback = 0.0
    cap._stream_lock = threading.RLock()
    # Evita lanzar el hilo real del watchdog en la prueba.
    cap._watchdog = NS(is_alive=lambda: True)
    cap._callback = lambda *a: None
    for clave, valor in cambios.items():
        setattr(cap, clave, valor)
    return cap


class OpenStreamTest(unittest.TestCase):
    def test_abre_con_el_default_actual(self):
        cap = _captura_para_abrir()
        fake = _SdFalso(default=2)
        with mock.patch.object(captura_mod, "sd", fake):
            cap._open_stream()
        self.assertEqual(fake.instancias[0].kwargs["device"], 2)
        self.assertEqual(cap._stream_device, 2)
        self.assertTrue(fake.instancias[0].started)

    def test_dispositivo_fijado_ausente_cae_al_default(self):
        cap = _captura_para_abrir(
            _input_device=5, _follow_default=False, _stream_device=5
        )
        fake = _SdFalso(default=2, falla_dispositivo=5)
        with mock.patch.object(captura_mod, "sd", fake):
            cap._open_stream()
        self.assertEqual(fake.instancias[0].kwargs["device"], 2)
        self.assertTrue(cap._follow_default)
        self.assertIsNone(cap._input_device)


class WaitReadyTest(unittest.TestCase):
    def _cap(self, chunks, last, since_open):
        cap = object.__new__(AudioCapture)
        cap._chunks_recibidos = chunks
        cap._callback_since_open = since_open
        cap._last_callback = last
        return cap

    def test_listo_con_audio_fresco(self):
        cap = self._cap(3, time.monotonic(), True)
        self.assertTrue(asyncio.run(cap.wait_ready(timeout=0.2)))

    def test_stream_zombi_no_cuenta(self):
        cap = self._cap(50, time.monotonic() - 10.0, True)
        self.assertFalse(asyncio.run(cap.wait_ready(timeout=0.05)))

    def test_sin_callback_desde_la_apertura(self):
        cap = self._cap(0, time.monotonic(), False)
        self.assertFalse(asyncio.run(cap.wait_ready(timeout=0.05)))


class ReopenNowTest(unittest.TestCase):
    def test_reabre_solo_si_el_stream_es_deseado(self):
        cap, acciones = _captura()
        cap.reopen_now()
        self.assertEqual(acciones, ["reopen"])
        cap._stream_wanted = False
        cap.reopen_now()
        self.assertEqual(acciones, ["reopen"])


if __name__ == "__main__":
    unittest.main()
