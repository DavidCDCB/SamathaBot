"""Extrae el nombre de una respuesta hablada, sin depender del LLM."""

from __future__ import annotations

import re

# "me llamo Fredy", "mi nombre es María José", "puedes llamarme Fer"...
_PATRONES = (
    r"\bme llamo\s+(.+)",
    r"\bmi nombre es\s+(.+)",
    r"\bpuedes llamarme\s+(.+)",
    r"\bll[áa]mame\s+(.+)",
)

_STOP = {
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "y",
    "pero",
    "porque",
    "que",
    "mi",
    "tu",
    "un",
    "una",
    "en",
    "con",
    "para",
    "por",
    "es",
    "soy",
}

_NO_NOMBRE = {
    "no",
    "si",
    "sí",
    "nada",
    "prefiero",
    "gracias",
    "hola",
    "buenas",
    "buenos",
    "días",
    "dias",
    "tardes",
    "noches",
    "claro",
    "ok",
    "vale",
    "feliz",
    "bien",
    "mal",
    "cansado",
    "cansada",
    "colombiano",
    "colombiana",
    "listo",
    "aja",
    "ajá",
    "ninguno",
    "ingeniero",
    "ingeniera",
    "profesor",
    "profesora",
    "estudiante",
    "médico",
    "medico",
    "abogado",
    "abogada",
    "programador",
    "programadora",
    "desarrollador",
    "desarrolladora",
    "empresario",
    "empresaria",
}

_PALABRA = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+"


def _limpia_nombre(bruto: str) -> str | None:
    palabras = re.findall(_PALABRA, bruto)
    elegidas: list[str] = []
    for palabra in palabras:
        clave = palabra.casefold()
        if clave in _NO_NOMBRE:
            return None if not elegidas else " ".join(elegidas)
        if clave in _STOP:
            break
        elegidas.append(palabra)
        if len(elegidas) >= 3:
            break
    if not elegidas:
        return None
    return " ".join(p.capitalize() for p in elegidas)


def extraer_nombre(texto: str, esperando: bool = False) -> str | None:
    """Devuelve un nombre si la frase lo declara o si responde a la pregunta.

    `esperando` se activa después de que Samantha preguntó el nombre; permite
    aceptar una respuesta corta como "Fredy" o "soy Fredy" sin confundir
    "soy de Bogotá" con un nombre.
    """
    texto = (texto or "").strip()
    if not texto:
        return None

    for patron in _PATRONES:
        match = re.search(patron, texto, re.IGNORECASE)
        if match:
            nombre = _limpia_nombre(match.group(1))
            if nombre:
                return nombre

    if esperando:
        # respuesta breve: "Fredy", "Soy Fredy", "María José"
        match = re.fullmatch(
            rf"\s*(?:soy\s+)?({_PALABRA}(?:\s+{_PALABRA}){{0,2}})\s*[.!?]*\s*",
            texto,
            re.IGNORECASE,
        )
        if match:
            return _limpia_nombre(match.group(1))
    return None
