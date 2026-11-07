"""Tests for the startup greeting (wakeword.greet_on_start)."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace as NS

from openocto.config import AppConfig
from openocto.state_machine import State, StateMachine


def _bare_app():
    """OpenOctoApp sin __init__ (evita micrófono, modelos y red)."""
    from openocto.app import OpenOctoApp

    app = object.__new__(OpenOctoApp)
    app._processing = False
    app._state_machine = StateMachine()
    app._wakeword = None
    app._name_asked = False
    app._conversation_active = False
    app._pending_text = None
    app._greet_task = None
    app._ptt_mode = "vad"
    app._orb_shown = []
    app._orb = NS(
        show=lambda: app._orb_shown.append(True),
        state=lambda e: None,
        end=lambda **k: None,
        level=lambda v: None,
    )
    cfg = AppConfig()
    cfg.audio.continuous = True
    cfg.wakeword.greet_on_start = True
    cfg.wakeword.greet_delay = 0.0
    app._config = cfg
    return app


class InjectTextTest(unittest.TestCase):
    def test_inject_text_feeds_pipeline_as_if_transcribed(self):
        app = _bare_app()
        seen: dict = {}

        async def fake_handle(audio):
            # consume _pending_text like _handle_audio does
            seen["audio_size"] = audio.size
            seen["pending"] = app._pending_text
            app._pending_text = None

        app._handle_audio = fake_handle
        asyncio.run(app._inject_text("Hola Samantha"))

        result = seen["pending"]
        self.assertEqual(result.text, "Hola Samantha")
        self.assertEqual(result.language, "es")  # stt.language "auto" -> es
        self.assertAlmostEqual(result.confidence, 1.0)
        self.assertEqual(seen["audio_size"], 0)
        self.assertTrue(app._conversation_active)  # continuous + vad
        self.assertEqual(app._orb_shown, [True])  # el saludo muestra el orbe
        self.assertEqual(app._state_machine.state, State.RECORDING)

    def test_inject_text_pauses_wakeword_and_resets_name(self):
        app = _bare_app()
        paused = []
        app._wakeword = NS(pause=lambda: paused.append(True), resume=lambda: None)
        app._name_asked = True

        async def fake_handle(audio):
            app._pending_text = None

        app._handle_audio = fake_handle
        asyncio.run(app._inject_text("Hola Samantha"))
        self.assertEqual(paused, [True])
        self.assertFalse(app._name_asked)

    def test_inject_text_noop_while_processing(self):
        app = _bare_app()
        app._processing = True
        called = []

        async def fake_handle(audio):
            called.append(1)

        app._handle_audio = fake_handle
        asyncio.run(app._inject_text("Hola Samantha"))
        self.assertEqual(called, [])
        self.assertEqual(app._state_machine.state, State.IDLE)


class AutoGreetTest(unittest.TestCase):
    def test_uses_greet_text_then_phrase(self):
        app = _bare_app()
        got = []

        async def fake_inject(text):
            got.append(text)

        app._inject_text = fake_inject
        asyncio.run(app._auto_greet())
        self.assertEqual(got, ["Hola Samantha"])  # default phrase

        got.clear()
        app._config.wakeword.greet_text = "buenos días samantha"
        asyncio.run(app._auto_greet())
        self.assertEqual(got, ["buenos días samantha"])

    def test_respects_delay(self):
        import time

        app = _bare_app()
        app._config.wakeword.greet_delay = 0.25
        got = []

        async def fake_inject(text):
            got.append(text)

        app._inject_text = fake_inject
        t0 = time.monotonic()
        asyncio.run(app._auto_greet())
        self.assertGreaterEqual(time.monotonic() - t0, 0.2)
        self.assertEqual(got, ["Hola Samantha"])


class OrbAtStartupTest(unittest.TestCase):
    def _app_con_registro(self):
        app = _bare_app()
        eventos = []

        def _registro(nombre):
            def _f(*args):
                eventos.append(nombre if not args else (nombre, args[0]))

            return _f

        app._orb = NS(
            start=_registro("orb.start"),
            state=_registro("orb.state"),
            show=_registro("orb.show"),
        )
        app._config.ui.orb = True
        return app, eventos

    def test_orb_gris_antes_de_cargar_modelos(self):
        app, eventos = self._app_con_registro()

        async def fake_resolve():
            return 1, "User"

        async def fake_health():
            return True, "ok"

        async def fake_ptt(header):
            eventos.append("run_ptt")

        app._resolve_user = fake_resolve
        app._nombre_conocido = lambda uid: True
        app._persona = NS(display_name="Samantha")
        app._init_components = lambda: eventos.append("init_components")
        app._ai_router = NS(active_backend_name="go", health_check=fake_health)
        app._run_ptt_mode = fake_ptt

        asyncio.run(app.run())

        self.assertEqual(
            eventos[:4],
            ["orb.start", ("orb.state", "waiting"), "orb.show", "init_components"],
        )
        self.assertEqual(eventos[-1], "run_ptt")

    def test_respeta_ui_orb_desactivado(self):
        app, eventos = self._app_con_registro()
        app._config.ui.orb = False
        app._show_orb_at_start()
        self.assertEqual(eventos, [])


class ConfigTest(unittest.TestCase):
    def test_defaults_off(self):
        cfg = AppConfig()
        self.assertFalse(cfg.wakeword.greet_on_start)
        self.assertEqual(cfg.wakeword.greet_delay, 1.5)

    def test_assistant_config_enables_greeting(self):
        from pathlib import Path

        from openocto.config import load_config

        path = Path(__file__).resolve().parents[1] / "assistant-config.yaml"
        cfg = load_config(str(path))
        self.assertTrue(cfg.wakeword.greet_on_start)


if __name__ == "__main__":
    unittest.main()
