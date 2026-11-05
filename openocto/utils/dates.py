"""Fecha/hora formateadas en español, sin depender de la locale de Windows."""

from __future__ import annotations

from datetime import datetime

DIAS = [
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
]
MESES = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]


def fecha_es(ahora: datetime | None = None) -> str:
    """'jueves, 5 de noviembre de 2026' (sin ceros, en español)."""
    d = ahora or datetime.now()
    return f"{DIAS[d.weekday()]}, {d.day} de {MESES[d.month - 1]} de {d.year}"


def hora_es(ahora: datetime | None = None) -> str:
    """'3:07 p. m.' en formato colombiano."""
    d = ahora or datetime.now()
    h = d.hour % 12 or 12
    sufijo = "a. m." if d.hour < 12 else "p. m."
    return f"{h}:{d.minute:02d} {sufijo}"
