"""El router no debe enviar el mensaje actual dos veces al backend."""

from __future__ import annotations

import asyncio
import unittest

from openocto.ai.router import AIRouter


class _Backend:
    def __init__(self) -> None:
        self.messages: list[dict] | None = None
        self.system_prompt: str | None = None

    async def send(self, messages, system_prompt):  # pragma: no cover - no se usa
        return "ok"

    async def send_streaming(self, messages, system_prompt, on_chunk, skills=None):
        self.messages = messages
        self.system_prompt = system_prompt
        return "respuesta"


def _router() -> tuple[AIRouter, _Backend]:
    backend = _Backend()
    router = object.__new__(AIRouter)
    router._backends = {"fake": backend}
    router._active_backend_name = "fake"
    return router, backend


async def _noop(_chunk: str) -> None:
    return None


class RouterMensajeActualTest(unittest.TestCase):
    def test_quita_el_mensaje_actual_duplicado_del_historial(self):
        router, backend = _router()
        history = [
            {"role": "user", "content": "ayer"},
            {"role": "assistant", "content": "hola"},
            {"role": "user", "content": "hoy"},
        ]
        asyncio.run(router.send_streaming("hoy", history, "sys", _noop))
        self.assertEqual(
            [(m["role"], m["content"]) for m in backend.messages],
            [("user", "ayer"), ("assistant", "hola"), ("user", "hoy")],
        )

    def test_si_el_ultimo_es_del_asistente_no_quita_nada(self):
        router, backend = _router()
        history = [
            {"role": "user", "content": "hoy"},
            {"role": "assistant", "content": "ok"},
        ]
        asyncio.run(router.send_streaming("hoy", history, "sys", _noop))
        self.assertEqual(
            [m["role"] for m in backend.messages],
            ["user", "assistant", "user"],
        )

    def test_un_texto_distinto_se_conserva(self):
        router, backend = _router()
        history = [{"role": "user", "content": "otra cosa"}]
        asyncio.run(router.send_streaming("hoy", history, "sys", _noop))
        self.assertEqual(len(backend.messages), 2)


if __name__ == "__main__":
    unittest.main()
