"""Pruebas del bucle de herramientas en streaming con un cliente falso (sin red)."""

import asyncio
import unittest
from types import SimpleNamespace as NS

from openocto.ai.openai_compat import OpenAICompatBackend


def _chunk(content=None, tool_calls=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls))])


def _llamada(index, id=None, name=None, arguments=""):
    return NS(index=index, id=id, function=NS(name=name, arguments=arguments))


class _Flujo:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class _Completions:
    def __init__(self, respuestas):
        self._respuestas = list(respuestas)
        self.llamadas = []

    async def create(self, **kwargs):
        self.llamadas.append(kwargs)
        return _Flujo(self._respuestas.pop(0))


class _Herramientas:
    def __init__(self):
        self.llamadas = []

    def __len__(self):
        return 1

    def openai_tools(self):
        return [{"type": "function", "function": {"name": "manage_notes_and_facts"}}]

    async def call(self, name, args):
        self.llamadas.append((name, args))
        return "Saved note #1"


def _backend(respuestas):
    b = OpenAICompatBackend.__new__(OpenAICompatBackend)  # sin crear cliente real
    completions = _Completions(respuestas)
    b._client = NS(chat=NS(completions=completions))
    b._model = "modelo-prueba"
    b._name = "prueba"
    b._reasoning_effort = ""
    b._extra_body = {}
    b._turn_timeout = 30.0
    b._max_tokens = 400
    return b, completions


async def _ignorar(_texto):
    return None


class StreamingTest(unittest.TestCase):
    def test_texto_sin_herramientas_se_emite_por_fragmentos(self):
        b, comp = _backend([[_chunk("Hola "), _chunk(None), _chunk("mundo")]])
        emitido = []

        async def on_chunk(texto):
            emitido.append(texto)

        msgs = [{"role": "user", "content": "hola"}]
        texto = asyncio.run(b.send_streaming(msgs, "sistema", on_chunk))
        self.assertEqual(texto, "Hola mundo")
        self.assertEqual(emitido, ["Hola ", "mundo"])
        self.assertEqual(comp.llamadas[0]["max_tokens"], 400)
        self.assertEqual(comp.llamadas[0]["timeout"], 30.0)

    def test_chunk_sin_choices_no_rompe(self):
        b, _ = _backend([[NS(choices=[]), _chunk("ok")]])
        msgs = [{"role": "user", "content": "hola"}]
        self.assertEqual(asyncio.run(b.send_streaming(msgs, "sistema", _ignorar)), "ok")

    def test_tool_call_fragmentado_se_ejecuta_y_luego_responde(self):
        primera = [
            _chunk(
                None,
                [
                    _llamada(
                        0,
                        id="c1",
                        name="manage_notes_and_facts",
                        arguments='{"action": ',
                    )
                ],
            ),
            _chunk(
                None, [_llamada(0, arguments='"add_note", "content": "llamar a mamá"}')]
            ),
        ]
        segunda = [_chunk("Listo, lo anoté."), _chunk(None)]
        b, comp = _backend([primera, segunda])
        herramientas = _Herramientas()
        emitido = []

        async def on_chunk(texto):
            emitido.append(texto)

        msgs = [{"role": "user", "content": "recuerda llamar a mamá"}]
        texto = asyncio.run(
            b.send_streaming(msgs, "sistema", on_chunk, skills=herramientas)
        )
        self.assertEqual(
            herramientas.llamadas,
            [
                (
                    "manage_notes_and_facts",
                    {"action": "add_note", "content": "llamar a mamá"},
                )
            ],
        )
        self.assertEqual(emitido, ["Listo, lo anoté."])
        self.assertEqual(texto, "Listo, lo anoté.")
        self.assertEqual(
            comp.llamadas[1]["messages"][-1],
            {"role": "tool", "tool_call_id": "c1", "content": "Saved note #1"},
        )


if __name__ == "__main__":
    unittest.main()
