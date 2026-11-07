"""Pruebas del bucle de herramientas en streaming con un cliente falso (sin red)."""

import asyncio
import unittest
from types import SimpleNamespace as NS

from openocto.ai.openai_compat import OpenAICompatBackend


def _chunk(content=None, tool_calls=None, finish_reason=None):
    return NS(
        choices=[
            NS(
                delta=NS(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


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


class _HerramientasWeb(_Herramientas):
    def __len__(self):
        return 2

    def openai_tools(self):
        return [{"type": "function", "function": {"name": "web_search"}}]

    async def call(self, name, args):
        self.llamadas.append((name, args))
        return f"resultado web {len(self.llamadas)}"


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

    def test_busqueda_se_corta_tras_dos_llamadas_y_fuerza_respuesta(self):
        primera = [
            _chunk(
                None,
                [
                    _llamada(
                        0,
                        id="w1",
                        name="web_search",
                        arguments='{"query": "Bleach final"}',
                    )
                ],
            )
        ]
        segunda = [
            _chunk(
                None,
                [
                    _llamada(
                        0,
                        id="w2",
                        name="web_search",
                        arguments='{"query": "Bleach manga ended"}',
                    )
                ],
            )
        ]
        tercera = [_chunk("El manga de Bleach terminó en 2016."), _chunk(None)]
        b, comp = _backend([primera, segunda, tercera])
        herramientas = _HerramientasWeb()
        emitido = []

        async def on_chunk(texto):
            emitido.append(texto)

        msgs = [{"role": "user", "content": "¿Cuándo termina Bleach?"}]
        texto = asyncio.run(
            b.send_streaming(msgs, "sistema", on_chunk, skills=herramientas)
        )

        self.assertEqual(len(herramientas.llamadas), 2)
        self.assertEqual(len(comp.llamadas), 3)
        self.assertIn("tools", comp.llamadas[0])
        self.assertNotIn("tools", comp.llamadas[2])
        self.assertEqual(texto, "El manga de Bleach terminó en 2016.")
        self.assertEqual(emitido, ["El manga de Bleach terminó en 2016."])
        self.assertTrue(
            any(m.get("role") == "tool" for m in comp.llamadas[2]["messages"])
        )


class ThinkScrubberTest(unittest.TestCase):
    """El texto audible nunca debe llevar etiquetas de razonamiento.

    Las etiquetas se construyen con chr(60)/chr(62) para que este archivo no
    contenga el bloque completo (algunos filtros de edición lo corrompen).
    """

    AP = chr(60) + "think" + chr(62)
    CI = chr(60) + "/" + "think" + chr(62)
    APl = chr(60) + "thinking" + chr(62)
    CIl = chr(60) + "/" + "thinking" + chr(62)

    def _run(self, trozos):
        from openocto.ai.openai_compat import _ThinkScrubber

        s = _ThinkScrubber()
        out = [s.feed(t) for t in trozos]
        out.append(s.flush())
        return "".join(out)

    def test_bloque_de_razonamiento_se_descarta(self):
        self.assertEqual(
            self._run([self.AP + "no leer" + self.CI + "Hola."]),
            "Hola.",
        )

    def test_etiqueta_partida_entre_deltas(self):
        # Cada etiqueta puede llegar dividida en trozos; el prefijo dudoso se
        # retiene y solo se emite si se confirma que NO era una etiqueta.
        self.assertEqual(
            self._run(
                [
                    "Va",
                    chr(60) + "thi",
                    "nk>no leas esto",
                    chr(60) + "/th",
                    "ink> bien",
                ]
            ),
            "Va bien",
        )

    def test_cierre_suelto_sin_apertura(self):
        # El gateway a veces deja la etiqueta de cierre huerfana en content.
        self.assertEqual(self._run(["Respuesta" + self.CI]), "Respuesta")

    def test_prefijo_parcial_al_final_no_se_emite(self):
        self.assertEqual(self._run(["Texto " + chr(60) + "thi"]), "Texto ")

    def test_texto_normal_con_angulos_se_conserva(self):
        self.assertEqual(self._run(["10 < 15 y ", "a < b"]), "10 < 15 y a < b")

    def test_variante_thinking_larga(self):
        self.assertEqual(
            self._run([self.APl + "pienso" + self.CIl + "Listo."]),
            "Listo.",
        )

    def test_streaming_no_emite_etiquetas(self):
        b, _ = _backend(
            [[_chunk(self.AP + "Razon" + "ando" + self.CI), _chunk("Hola")]]
        )
        emitido = []

        async def on_chunk(texto):
            emitido.append(texto)

        msgs = [{"role": "user", "content": "hola"}]
        texto = asyncio.run(b.send_streaming(msgs, "sistema", on_chunk))
        self.assertEqual(texto, "Hola")
        for trozo in emitido:
            self.assertNotIn(chr(60), trozo)

    def test_tool_loop_filtra_y_registra_respuesta(self):
        # Con skills cargadas (el camino real de cada turno) la respuesta debe
        # quedar registrada en el log, incluso con razonamiento inline.
        primera = [
            _chunk(
                None,
                [
                    _llamada(
                        0,
                        id="c1",
                        name="manage_notes_and_facts",
                        arguments='{"action": "list_notes"}',
                    )
                ],
            )
        ]
        segunda = [
            _chunk(self.AP + "pensamiento" + self.CI + "Tengo una nota."),
            _chunk(None),
        ]
        b, _ = _backend([primera, segunda])
        herramientas = _Herramientas()
        emitido = []

        async def on_chunk(texto):
            emitido.append(texto)

        msgs = [{"role": "user", "content": "lista mis notas"}]
        with self.assertLogs("openocto.ai.openai_compat", level="INFO") as logs:
            texto = asyncio.run(
                b.send_streaming(msgs, "sistema", on_chunk, skills=herramientas)
            )
        self.assertEqual(texto, "Tengo una nota.")
        self.assertEqual(emitido, ["Tengo una nota."])
        # La respuesta queda registrada en el log (era el bug reportado).
        self.assertTrue(
            any("Tengo una nota." in linea for linea in logs.output),
            f"respuesta no registrada en log: {logs.output}",
        )


if __name__ == "__main__":
    unittest.main()
