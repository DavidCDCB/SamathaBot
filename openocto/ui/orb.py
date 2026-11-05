"""Orbe de estado: ventana pequeña, sin bordes, que muestra la conversación con color y animación.

Aparece al iniciar una toma y se oculta al terminar la conversación. El asistente
solo manda órdenes (mostrar, cambiar de estado, nivel del micro, terminar): la
ventana vive en su propio hilo.

En Windows, el orbe se dibuja con Pillow y se muestra con alfa por píxel (bordes
suaves y fondo transparente). Si Pillow o esa ventana fallan, usa Tkinter como
plan B. Si tampoco hay ventana, el controlador no hace nada y el asistente sigue igual.
"""

from __future__ import annotations

import ctypes
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

try:
    from PIL import Image, ImageDraw
except ImportError:  # sin Pillow, el orbe usa la ventana de Tk
    Image = ImageDraw = None

logger = logging.getLogger(__name__)

SIZE = 220  # lado de la ventana, en píxeles
FRAME_MS = 33  # ~30 fotogramas por segundo
KEY = "#010203"  # color que Windows deja transparente (solo en la ventana de Tk)
RMS_TOPE = 2500.0  # RMS int16 con el que el orbe reacciona al máximo
SUPERMUESTREO = 2  # el orbe se dibuja al doble y se reduce: bordes suaves
SIMBOLO = "#f3e2bd"  # crema del infinito interior


@dataclass(frozen=True)
class Estilo:
    color: str  # color base del orbe
    descripcion: str  # para leer el código: el orbe no muestra texto
    ritmo: float  # ciclos por segundo de la animación
    radio: float  # tamaño base (fracción del radio disponible)
    ganancia: float  # cuánto crece con el nivel del micro (0 = no reacciona)
    brillo: float  # intensidad base (0 a 1)
    acento: str = "#ffffff"  # luz secundaria: borde, ondas y partículas


ESTILOS: dict[str, Estilo] = {
    "listening": Estilo(
        "#22d3ee", "esperando tu voz", 0.35, 0.50, 0.00, 0.85, "#67e8f9"
    ),
    "hearing": Estilo(
        "#2dd4bf", "escuchando tu voz", 0.90, 0.38, 0.55, 1.00, "#99f6e4"
    ),
    "transcribing": Estilo(
        "#a78bfa", "transcribiendo", 0.80, 0.52, 0.00, 0.90, "#c4b5fd"
    ),
    "thinking": Estilo("#fbbf24", "pensando", 1.40, 0.50, 0.00, 0.95, "#fde68a"),
    "speaking": Estilo("#f472b6", "hablando", 1.00, 0.52, 0.00, 1.00, "#f9a8d4"),
    "waiting": Estilo(
        "#64748b", "conversación abierta", 0.20, 0.42, 0.00, 0.45, "#94a3b8"
    ),
    "error": Estilo("#f87171", "error", 2.00, 0.55, 0.00, 1.00, "#fca5a5"),
}


def nivel_normalizado(rms: float) -> float:
    """RMS int16 -> 0 a 1. Voz cercana ~1000-3000; voz de micro lejano ~150-300."""
    return max(0.0, min(1.0, rms / RMS_TOPE))


def mezcla(c1: str, c2: str, t: float) -> str:
    """Mezcla dos colores '#rrggbb' (t=0 da c1, t=1 da c2)."""
    t = max(0.0, min(1.0, t))
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def parametros(estado: str, t: float, nivel: float) -> dict:
    """Radio, brillo y ángulo del orbe en el instante t. Sin Tk: se puede probar."""
    est = ESTILOS[estado]
    fase = 2 * math.pi * est.ritmo * t
    # dos senos desfasados: respiración menos mecánica que una sola onda
    radio = (
        est.radio
        + est.ganancia * nivel
        + 0.032 * math.sin(fase)
        + 0.010 * math.sin(2.3 * fase + 0.7)
    )
    if estado == "speaking":
        radio += 0.045 * math.sin(3 * fase + 0.4)  # ondulación de la voz
    elif estado == "hearing":
        radio += 0.010 * math.sin(5 * fase)
    elif estado == "error":
        radio += 0.010 * math.sin(7 * fase)
    radio = max(0.25, min(0.95, radio))
    brillo = est.brillo * (0.82 + 0.18 * math.sin(fase + 0.3))
    return {
        "radio": radio,
        "brillo": max(0.0, min(1.0, brillo)),
        "angulo": (360.0 * est.ritmo * t) % 360.0,
        "color": est.color,
        "acento": est.acento,
        "fase": fase,
        "pulso": 0.5 + 0.5 * math.sin(fase),
        "energia": max(0.0, min(1.0, nivel)),
    }


def _rgba(color: str, alfa: int = 255) -> tuple[int, int, int, int]:
    """'#rrggbb' y alfa -> tupla RGBA para Pillow."""
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), alfa)


def _capa(lado: int) -> Image.Image:
    return Image.new("RGBA", (lado, lado), (0, 0, 0, 0))


def _puntos_orbita(
    cx: float,
    cy: float,
    radio: float,
    fase: float,
    deformacion: float,
    n: int = 64,
) -> list[tuple[float, float]]:
    """Polígono casi circular con una deformación orgánica muy leve."""
    pts: list[tuple[float, float]] = []
    for k in range(n):
        a = math.tau * k / n
        rr = radio * (
            1
            + deformacion * math.sin(3 * a + fase)
            + 0.004 * math.sin(5 * a - fase * 1.7)
        )
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def puntos_simbolo(
    cx: float, cy: float, radio: float, n: int = 180
) -> list[tuple[float, float]]:
    """Lemniscata (infinito) centrada, ancho 1.24r y alto 0.46r.

    Curva de Gerono: x = cos(t), y = sin(t)·cos(t). Pasa por el centro en
    t=π/2 y 3π/2, que es donde se cruzan los dos bucles.
    """
    ancho = radio * 0.62
    alto = radio * 0.46
    pts: list[tuple[float, float]] = []
    for k in range(n):
        a = math.tau * k / n
        pts.append((cx + ancho * math.cos(a), cy + alto * math.sin(a) * math.cos(a)))
    return pts


def _dibuja_cuerpo(
    img: Image.Image,
    c: float,
    r: float,
    p: dict,
    estado: str,
    lado: int,
    unidad: float,
) -> None:
    """Cuerpo plano con viñeta sutil, canto oscuro y borde de vidrio."""
    base = p["color"]
    acento = p["acento"]
    oscuro = mezcla(base, "#04060b", 0.62)
    borde = mezcla(base, acento, 0.22)
    fase = p["fase"]
    deformacion = 0.020 if estado in ("hearing", "speaking") else 0.010

    d = ImageDraw.Draw(img)
    pasos = 44
    for i in range(pasos, 0, -1):
        f = i / pasos
        rr = r * f
        # interior plano, sin brillo central: solo viñeta hacia el borde
        tt = (1.0 - f) ** 0.55
        col = mezcla(oscuro, base, min(1.0, 0.30 + 0.70 * tt))
        if f > 0.84:
            col = mezcla(col, borde, (f - 0.84) * 1.4)
        d.polygon(
            _puntos_orbita(c, c, rr, fase, deformacion),
            fill=_rgba(col),
        )
    # canto: sombra abajo-derecha, luz fina arriba-izquierda y borde de vidrio
    canto = _capa(lado)
    cd = ImageDraw.Draw(canto)
    caja = (c - r * 0.97, c - r * 0.97, c + r * 0.97, c + r * 0.97)
    cd.arc(
        caja,
        15,
        165,
        fill=_rgba(oscuro, 150),
        width=max(2, int(round(5 * unidad))),
    )
    cd.arc(
        caja,
        198,
        342,
        fill=_rgba(acento, 90),
        width=max(1, int(round(2 * unidad))),
    )
    cd.arc(
        caja,
        20,
        150,
        fill=_rgba(acento, 32),
        width=max(1, int(round(1.2 * unidad))),
    )
    cd.ellipse(
        caja,
        outline=_rgba(acento, 42),
        width=max(1, int(round(1.0 * unidad))),
    )
    img.alpha_composite(canto)


def _dibuja_simbolo(
    img: Image.Image, c: float, r: float, p: dict, lado: int, unidad: float
) -> None:
    """Infinito crema en el centro; sin reflejo ni brillo interior."""
    pts = puntos_simbolo(c, c, r)
    trazo = [*pts, pts[0]]
    capa = _capa(lado)
    d = ImageDraw.Draw(capa)
    ancho = max(1, int(round(1.7 * unidad)))
    # sombra muy fina: mantiene el crema legible sobre estados claros (amarillo)
    d.line(
        trazo,
        fill=_rgba(mezcla(p["color"], "#000000", 0.55), 80),
        width=ancho + max(1, int(round(0.9 * unidad))),
        joint="curve",
    )
    d.line(trazo, fill=_rgba(SIMBOLO, 235), width=ancho, joint="curve")
    img.alpha_composite(capa)


def _dibuja_efectos(
    img: Image.Image,
    c: float,
    r: float,
    p: dict,
    estado: str,
    t: float,
    lado: int,
    unidad: float,
) -> None:
    """Anillos, ondas y partículas propias de cada estado."""
    base = p["color"]
    acento = p["acento"]
    brillo = p["brillo"]
    ang = p["angulo"]
    fase = p["fase"]
    capa = _capa(lado)
    d = ImageDraw.Draw(capa)

    if estado in ("hearing", "speaking"):
        # corona de barras: reacciona a tu voz (hearing) o al ritmo del habla
        n = 26
        r0 = r * 1.08
        for i in range(n):
            a = math.tau * i / n - math.pi / 2
            if estado == "speaking":
                energia = 0.30 + 0.45 * abs(math.sin(fase * 1.7 + i * 0.35))
            else:
                energia = max(0.05, p["energia"])
            h = (
                1.0 + 7.5 * energia * (0.35 + 0.65 * abs(math.sin(fase * 2 + i * 0.9)))
            ) * unidad
            x0, y0 = c + r0 * math.cos(a), c + r0 * math.sin(a)
            x1, y1 = c + (r0 + h) * math.cos(a), c + (r0 + h) * math.sin(a)
            alfa = int(105 + 105 * min(1.0, energia))
            d.line(
                (x0, y0, x1, y1),
                fill=_rgba(acento, alfa),
                width=max(1, int(round(1.5 * unidad))),
            )

    if estado == "speaking":
        # ondas que se expanden con la voz
        for k in range(3):
            f = (t * 0.55 + k / 3) % 1.0
            rr = r * (0.98 + 0.30 * f)
            alfa = int(70 * (1 - f) * brillo)
            if alfa > 0:
                d.ellipse(
                    (c - rr, c - rr, c + rr, c + rr),
                    outline=_rgba(acento, alfa),
                    width=max(1, int(round(1.2 * unidad))),
                )

    elif estado == "listening":
        # dos motas lentas y un arco tenue: respiración, no actividad
        rr = r * 1.13
        for k in range(2):
            a = math.radians(ang * 0.7 + k * 180)
            x, y = c + rr * math.cos(a), c + rr * math.sin(a)
            pt = max(2.0, 2.6 * unidad)
            d.ellipse((x - pt, y - pt, x + pt, y + pt), fill=_rgba(acento, 150))
        arco = (c - rr * 1.06, c - rr * 1.06, c + rr * 1.06, c + rr * 1.06)
        d.arc(
            arco,
            start=ang,
            end=ang + 58,
            fill=_rgba(acento, 65),
            width=max(1, int(round(1.2 * unidad))),
        )

    elif estado == "transcribing":
        # dos arcos contrarrotatorios: lectura/procesamiento
        rr = r * 1.12
        caja = (c - rr, c - rr, c + rr, c + rr)
        d.arc(
            caja,
            start=ang,
            end=ang + 105,
            fill=_rgba(acento, 210),
            width=max(2, int(round(2.8 * unidad))),
        )
        d.arc(
            caja,
            start=-ang * 1.55 + 30,
            end=-ang * 1.55 + 205,
            fill=_rgba(base, 120),
            width=max(1, int(round(1.4 * unidad))),
        )
        for k in (0, 1):
            a = math.radians(ang + (105 if k == 0 else 0))
            x, y = c + rr * math.cos(a), c + rr * math.sin(a)
            pt = max(1.5, 2.0 * unidad)
            d.ellipse((x - pt, y - pt, x + pt, y + pt), fill=_rgba(acento, 230))

    elif estado == "thinking":
        # tres cometas orbitando: ideas que giran
        rr = r * 1.13
        for k in range(3):
            a0 = math.radians(ang * 1.35 + 120 * k)
            for cola in range(6, 0, -1):
                a = a0 - cola * 0.15
                rad = rr * (1 - 0.012 * cola)
                x, y = c + rad * math.cos(a), c + rad * math.sin(a)
                pt = max(1.2, (4.6 - 0.55 * cola) * unidad)
                d.ellipse(
                    (x - pt, y - pt, x + pt, y + pt),
                    fill=_rgba(acento, int(155 * (1 - cola / 7))),
                )
            x, y = c + rr * math.cos(a0), c + rr * math.sin(a0)
            pt = max(2.2, 4.8 * unidad)
            d.ellipse((x - pt, y - pt, x + pt, y + pt), fill=_rgba("#ffffff", 235))
        arco = (c - rr * 0.86, c - rr * 0.86, c + rr * 0.86, c + rr * 0.86)
        d.arc(
            arco,
            start=-ang * 0.8,
            end=-ang * 0.8 + 120,
            fill=_rgba(acento, 55),
            width=max(1, int(round(1.0 * unidad))),
        )

    elif estado == "waiting":
        rr = r * 1.10
        d.arc(
            (c - rr, c - rr, c + rr, c + rr),
            start=ang,
            end=ang + 35,
            fill=_rgba(acento, 70),
            width=max(1, int(round(1.0 * unidad))),
        )
        for k in range(2):
            a = math.radians(ang * 0.5 + 180 * k)
            x, y = c + rr * math.cos(a), c + rr * math.sin(a)
            pt = max(1.5, 2.0 * unidad)
            d.ellipse((x - pt, y - pt, x + pt, y + pt), fill=_rgba(acento, 80))

    elif estado == "error":
        # corona discontinua + pulso rojo: algo salió mal
        rr = r * 1.10
        for k in range(12):
            a0 = ang * 1.8 + k * 30
            d.arc(
                (c - rr, c - rr, c + rr, c + rr),
                start=a0,
                end=a0 + 18,
                fill=_rgba(acento, 185),
                width=max(2, int(round(2.6 * unidad))),
            )
        for k in range(2):
            f = (t * 0.9 + k / 2) % 1.0
            rad = r * (0.98 + 0.22 * f)
            alfa = int(85 * (1 - f))
            if alfa > 0:
                d.ellipse(
                    (c - rad, c - rad, c + rad, c + rad),
                    outline=_rgba("#ff3b30", alfa),
                    width=max(1, int(round(1.5 * unidad))),
                )

    img.alpha_composite(capa)


def dibuja_fotograma(
    estado: str,
    t: float,
    nivel: float,
    escala: float = 1.0,
    opacidad: float = 1.0,
    efectos: bool = True,
) -> Image.Image:
    """Orbe en RGBA de SIZE * escala píxeles por lado: fuera del orbe, transparente.

    Con ``efectos=False`` se omite la corona, las ondas y las partículas: queda
    el cuerpo plano con su símbolo y su halo, útil para iconos y avatares.
    """
    p = parametros(estado, t, nivel)
    tam = round(SIZE * escala)
    ss = SUPERMUESTREO
    unidad = ss * escala
    lado = tam * ss
    c = lado / 2
    r = (tam / 2 - 16 * escala) * p["radio"] * ss
    img = Image.new("RGBA", (lado, lado), (0, 0, 0, 0))

    # halo: capas muy suaves que se apagan hacia afuera
    halo = _capa(lado)
    hd = ImageDraw.Draw(halo)
    acento = p["acento"]
    for k in range(22, 0, -1):
        f = k / 22
        rr = r * (1.0 + 0.20 * f)
        alfa = int(118 * p["brillo"] * (1 - f) ** 2.0)
        if alfa > 0:
            hd.ellipse(
                (c - rr, c - rr, c + rr, c + rr),
                fill=_rgba(mezcla(p["color"], acento, 0.35), alfa),
            )
    rr = r * 1.20
    hd.ellipse(
        (c - rr, c - rr, c + rr, c + rr),
        outline=_rgba(acento, int(42 * p["brillo"])),
        width=max(1, int(round(1.0 * unidad))),
    )
    img.alpha_composite(halo)

    _dibuja_cuerpo(img, c, r, p, estado, lado, unidad)
    _dibuja_simbolo(img, c, r, p, lado, unidad)
    if efectos:
        _dibuja_efectos(img, c, r, p, estado, t, lado, unidad)

    out = img.reduce(ss)
    if opacidad < 0.999:
        out.putalpha(
            out.getchannel("A").point(lambda v: int(v * max(0.0, min(1.0, opacidad))))
        )
    return out


def bgra_premultiplicado(imagen: Image.Image) -> bytes:
    """Bytes BGRA con alfa premultiplicado: el formato que pide UpdateLayeredWindow."""
    r, g, b, a = imagen.convert("RGBa").split()
    return Image.merge("RGBA", (b, g, r, a)).tobytes()


# ── ventana de Windows con alfa por píxel (UpdateLayeredWindow) ──────────────

_WS_POPUP = 0x80000000
_WS_EX_TOPMOST = 0x00000008
_WS_EX_TOOLWINDOW = 0x00000080  # ni barra de tareas ni Alt+Tab
_WS_EX_LAYERED = 0x00080000  # necesario para el alfa por píxel
_WS_EX_NOACTIVATE = 0x08000000  # no roba el foco, tampoco al arrastrarla
_WM_TIMER = 0x0113
_WM_NCHITTEST = 0x0084
_HTCAPTION = 2  # "barra de título": permite arrastrarla desde cualquier punto
_ULW_ALPHA = 0x00000002
_AC_SRC_OVER = 0x00
_AC_SRC_ALPHA = 0x01
_SW_HIDE = 0
_SW_SHOWNOACTIVATE = 4
_SPI_GETWORKAREA = 0x0030
_ID_TEMPORIZADOR = 1
_DPI_POR_MONITOR_V2 = -4  # consciente del DPI: Windows no estira la ventana

# Tipos de Win32 para ctypes (en 64 bits, handles y punteros ocupan 8 bytes)
_HANDLE = ctypes.c_void_p
_UINT = ctypes.c_uint32
_DWORD = ctypes.c_uint32
_LONG = ctypes.c_int32
_WORD = ctypes.c_uint16
_BOOL = ctypes.c_int32
_WPARAM = ctypes.c_size_t
_LPARAM = ctypes.c_ssize_t
_LRESULT = ctypes.c_ssize_t
_WNDPROC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
    _LRESULT, _HANDLE, _UINT, _WPARAM, _LPARAM
)


class _Punto(ctypes.Structure):
    _fields_ = [("x", _LONG), ("y", _LONG)]


class _Tamano(ctypes.Structure):
    _fields_ = [("cx", _LONG), ("cy", _LONG)]


class _Rect(ctypes.Structure):
    _fields_ = [("left", _LONG), ("top", _LONG), ("right", _LONG), ("bottom", _LONG)]


class _Fusion(ctypes.Structure):  # BLENDFUNCTION
    _fields_ = [
        ("BlendOp", ctypes.c_ubyte),
        ("BlendFlags", ctypes.c_ubyte),
        ("SourceConstantAlpha", ctypes.c_ubyte),
        ("AlphaFormat", ctypes.c_ubyte),
    ]


class _CabeceraBMP(ctypes.Structure):  # BITMAPINFOHEADER
    _fields_ = [
        ("biSize", _DWORD),
        ("biWidth", _LONG),
        ("biHeight", _LONG),
        ("biPlanes", _WORD),
        ("biBitCount", _WORD),
        ("biCompression", _DWORD),
        ("biSizeImage", _DWORD),
        ("biXPelsPerMeter", _LONG),
        ("biYPelsPerMeter", _LONG),
        ("biClrUsed", _DWORD),
        ("biClrImportant", _DWORD),
    ]


class _InfoBMP(ctypes.Structure):  # BITMAPINFO sin paleta: a 32 bits no hace falta
    _fields_ = [("bmiHeader", _CabeceraBMP)]


class _Mensaje(ctypes.Structure):  # MSG
    _fields_ = [
        ("hwnd", _HANDLE),
        ("message", _UINT),
        ("wParam", _WPARAM),
        ("lParam", _LPARAM),
        ("time", _DWORD),
        ("pt", _Punto),
    ]


class _ClaseVentana(ctypes.Structure):  # WNDCLASSW
    _fields_ = [
        ("style", _UINT),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", _HANDLE),
        ("hIcon", _HANDLE),
        ("hCursor", _HANDLE),
        ("hbrBackground", _HANDLE),
        ("lpszMenuName", ctypes.c_wchar_p),
        ("lpszClassName", ctypes.c_wchar_p),
    ]


def _firma(funcion, restype, *argtypes):
    """Declara el tipo de retorno y de argumentos de una función de Win32."""
    funcion.restype = restype
    funcion.argtypes = list(argtypes)
    return funcion


class _Win32:
    """Funciones de user32, gdi32 y kernel32 con sus tipos declarados."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("la ventana con alfa solo existe en Windows")
        u = ctypes.WinDLL("user32")
        g = ctypes.WinDLL("gdi32")
        k = ctypes.WinDLL("kernel32")
        self.RegisterClassW = _firma(
            u.RegisterClassW, _WORD, ctypes.POINTER(_ClaseVentana)
        )
        self.UnregisterClassW = _firma(
            u.UnregisterClassW, _BOOL, ctypes.c_wchar_p, _HANDLE
        )
        self.CreateWindowExW = _firma(
            u.CreateWindowExW,
            _HANDLE,
            _DWORD,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            _DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            _HANDLE,
            _HANDLE,
            _HANDLE,
            _HANDLE,
        )
        self.DestroyWindow = _firma(u.DestroyWindow, _BOOL, _HANDLE)
        self.DefWindowProcW = _firma(
            u.DefWindowProcW, _LRESULT, _HANDLE, _UINT, _WPARAM, _LPARAM
        )
        self.ShowWindow = _firma(u.ShowWindow, _BOOL, _HANDLE, ctypes.c_int)
        self.SetTimer = _firma(
            u.SetTimer,
            ctypes.c_size_t,
            _HANDLE,
            ctypes.c_size_t,
            _UINT,
            ctypes.c_void_p,
        )
        self.KillTimer = _firma(u.KillTimer, _BOOL, _HANDLE, ctypes.c_size_t)
        self.GetMessageW = _firma(
            u.GetMessageW, _BOOL, ctypes.POINTER(_Mensaje), _HANDLE, _UINT, _UINT
        )
        self.TranslateMessage = _firma(
            u.TranslateMessage, _BOOL, ctypes.POINTER(_Mensaje)
        )
        self.DispatchMessageW = _firma(
            u.DispatchMessageW, _LRESULT, ctypes.POINTER(_Mensaje)
        )
        self.PostQuitMessage = _firma(u.PostQuitMessage, None, ctypes.c_int)
        self.SystemParametersInfoW = _firma(
            u.SystemParametersInfoW,
            _BOOL,
            _UINT,
            _UINT,
            ctypes.POINTER(_Rect),
            _UINT,
        )
        self.UpdateLayeredWindow = _firma(
            u.UpdateLayeredWindow,
            _BOOL,
            _HANDLE,
            _HANDLE,
            ctypes.POINTER(_Punto),
            ctypes.POINTER(_Tamano),
            _HANDLE,
            ctypes.POINTER(_Punto),
            _DWORD,
            ctypes.POINTER(_Fusion),
            _DWORD,
        )
        self.CreateCompatibleDC = _firma(g.CreateCompatibleDC, _HANDLE, _HANDLE)
        self.DeleteDC = _firma(g.DeleteDC, _BOOL, _HANDLE)
        self.CreateDIBSection = _firma(
            g.CreateDIBSection,
            _HANDLE,
            _HANDLE,
            ctypes.POINTER(_InfoBMP),
            _UINT,
            ctypes.POINTER(ctypes.c_void_p),
            _HANDLE,
            _DWORD,
        )
        self.SelectObject = _firma(g.SelectObject, _HANDLE, _HANDLE, _HANDLE)
        self.DeleteObject = _firma(g.DeleteObject, _BOOL, _HANDLE)
        self.GetModuleHandleW = _firma(k.GetModuleHandleW, _HANDLE, ctypes.c_wchar_p)
        self.SetThreadDpiAwarenessContext = _firma(
            u.SetThreadDpiAwarenessContext, _HANDLE, _HANDLE
        )
        self.GetDpiForSystem = _firma(u.GetDpiForSystem, _UINT)


class _VentanaAlfa:
    """El orbe en una ventana de Windows con alfa por píxel. Solo la toca su hilo.

    Su temporizador hace de bucle: vacía la cola de órdenes y, si el orbe está
    visible, dibuja el siguiente fotograma. Así no hace falta otro hilo.
    """

    def __init__(self, cola: queue.Queue, nivel: Callable[[], float]) -> None:
        if Image is None:
            raise OSError("falta Pillow")
        self._w = _Win32()
        self._cola = cola
        self._nivel = nivel
        self._estado = "listening"
        self._visible = False
        self._cerrar = False
        self._oculta_en: float | None = None
        self._suave = 0.0
        self._alpha = 0.0
        self._alpha_obj = 1.0
        self._estado_prev: str | None = None
        self._trans_t0 = 0.0
        self._trans_dur = 0.30
        self._fallo_dibujo = False
        self._t0 = time.monotonic()
        # solo este hilo es consciente del DPI; si falla, se restaura el anterior
        previo = self._w.SetThreadDpiAwarenessContext(
            ctypes.c_void_p(_DPI_POR_MONITOR_V2)
        )
        try:
            self._escala = self._w.GetDpiForSystem() / 96
            self._tam = round(SIZE * self._escala)
            self._crea_ventana()
        except Exception:
            self._w.SetThreadDpiAwarenessContext(previo)
            raise

    def _crea_ventana(self) -> None:
        # una clase por ventana: cada clase lleva su propio callback
        self._clase = f"OpenOctoOrbe{id(self):x}"
        # esta referencia debe vivir mientras exista la clase
        self._proc = _WNDPROC(self._procesa)
        self._inst = self._w.GetModuleHandleW(None)
        clase = _ClaseVentana()
        clase.lpfnWndProc = self._proc
        clase.hInstance = self._inst
        clase.lpszClassName = self._clase
        if not self._w.RegisterClassW(ctypes.byref(clase)):
            raise OSError("RegisterClassW falló")
        area = _Rect()
        self._w.SystemParametersInfoW(_SPI_GETWORKAREA, 0, ctypes.byref(area), 0)
        self._hwnd = self._w.CreateWindowExW(
            _WS_EX_TOPMOST | _WS_EX_TOOLWINDOW | _WS_EX_LAYERED | _WS_EX_NOACTIVATE,
            self._clase,
            "Samantha orbe",  # el título permite localizar la ventana
            _WS_POPUP,
            area.left + (area.right - area.left - self._tam) // 2,
            area.top + (area.bottom - area.top - self._tam) // 2,
            self._tam,
            self._tam,
            None,
            None,
            self._inst,
            None,
        )
        if not self._hwnd:
            raise OSError("CreateWindowExW falló")
        self._dc = self._w.CreateCompatibleDC(None)
        info = _InfoBMP()
        cabecera = info.bmiHeader
        cabecera.biSize = ctypes.sizeof(_CabeceraBMP)
        cabecera.biWidth = self._tam
        cabecera.biHeight = -self._tam  # de arriba abajo, como las filas de Pillow
        cabecera.biPlanes = 1
        cabecera.biBitCount = 32
        bits = ctypes.c_void_p()
        self._dib = self._w.CreateDIBSection(
            self._dc, ctypes.byref(info), 0, ctypes.byref(bits), None, 0
        )
        if not self._dib:
            raise OSError("CreateDIBSection falló")
        self._bits = bits.value
        self._viejo = self._w.SelectObject(self._dc, self._dib)
        self._w.SetTimer(self._hwnd, _ID_TEMPORIZADOR, FRAME_MS, None)

    def mainloop(self) -> None:
        mensaje = _Mensaje()
        while self._w.GetMessageW(ctypes.byref(mensaje), None, 0, 0) > 0:
            self._w.TranslateMessage(ctypes.byref(mensaje))
            self._w.DispatchMessageW(ctypes.byref(mensaje))

    def _procesa(self, hwnd, msg, wparam, lparam) -> int:
        if msg == _WM_TIMER and wparam == _ID_TEMPORIZADOR:
            self._tick()
            return 0
        if msg == _WM_NCHITTEST:
            return _HTCAPTION  # se arrastra desde cualquier punto
        return self._w.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _tick(self) -> None:
        while True:
            try:
                accion, valor = self._cola.get_nowait()
            except queue.Empty:
                break
            try:
                self._aplica(accion, valor)
            except Exception:
                logger.exception("orbe: orden no válida")
        if self._cerrar:
            self._destruye()
            return
        if self._oculta_en is not None and time.monotonic() >= self._oculta_en:
            self._oculta_en = None
            self._alpha_obj = 0.0
        if self._visible:
            try:
                self._pinta()
            except Exception:
                if not self._fallo_dibujo:  # un fallo por fotograma no llena el log
                    logger.exception("orbe: fallo al dibujar")
                self._fallo_dibujo = True
            if self._visible and self._alpha_obj <= 0.0 and self._alpha < 0.02:
                self._oculta()

    def _aplica(self, accion: str, valor) -> None:
        if accion == "show":
            self._oculta_en = None
            self._estado_prev = None
            self._estado = "listening"
            self._alpha_obj = 1.0
            self._muestra()
        elif accion == "state" and valor in ESTILOS:
            if valor != self._estado:
                self._estado_prev = self._estado
                self._trans_t0 = time.monotonic()
            self._estado = valor
            self._alpha_obj = 1.0
        elif accion == "end":
            self._oculta_en = None
            if valor:
                if self._estado != "error":
                    self._estado_prev = self._estado
                    self._trans_t0 = time.monotonic()
                self._estado = "error"
                self._oculta_en = time.monotonic() + 1.2
                self._alpha_obj = 1.0
            else:
                # fundido de salida en vez de desaparecer de golpe
                self._alpha_obj = 0.0
        elif accion == "close":
            self._cerrar = True

    def _muestra(self) -> None:
        if self._visible:
            return
        if self._alpha <= 0.01:
            self._alpha = 0.12
        self._alpha_obj = 1.0
        self._pinta()  # la ventana con alfa necesita contenido antes de aparecer
        self._visible = True
        # SW_SHOWNOACTIVATE: aparece sin robar el foco de lo que estás usando
        self._w.ShowWindow(self._hwnd, _SW_SHOWNOACTIVATE)

    def _oculta(self) -> None:
        self._visible = False
        self._alpha = 0.0
        self._alpha_obj = 0.0
        self._w.ShowWindow(self._hwnd, _SW_HIDE)

    def _pinta(self) -> None:
        # ataque rápido cuando hablas, caída lenta al callar: menos nervioso
        nivel = self._nivel()
        k = 0.55 if nivel > self._suave else 0.18
        self._suave += (nivel - self._suave) * k
        self._alpha += (self._alpha_obj - self._alpha) * 0.18
        self._alpha = max(0.0, min(1.0, self._alpha))
        if self._alpha <= 0.01 and self._alpha_obj <= 0.0:
            return
        ahora = time.monotonic()
        t = ahora - self._t0
        imagen = dibuja_fotograma(
            self._estado, t, self._suave, self._escala, self._alpha
        )
        # fundido cruzado de 0,30 s entre estados (sin salto de color/forma)
        if self._estado_prev is not None:
            p = (ahora - self._trans_t0) / self._trans_dur
            if p >= 1.0:
                self._estado_prev = None
            else:
                p = max(0.0, p)
                p = p * p * (3 - 2 * p)  # smoothstep
                anterior = dibuja_fotograma(
                    self._estado_prev, t, self._suave, self._escala, self._alpha
                )
                imagen = Image.blend(anterior, imagen, p)
        datos = bgra_premultiplicado(imagen)
        ctypes.memmove(self._bits, datos, len(datos))
        # sin DC de pantalla ni posición: la ventana conserva donde la dejó el arrastre
        ok = self._w.UpdateLayeredWindow(
            self._hwnd,
            None,
            None,
            ctypes.byref(_Tamano(self._tam, self._tam)),
            self._dc,
            ctypes.byref(_Punto(0, 0)),
            0,
            ctypes.byref(_Fusion(_AC_SRC_OVER, 0, 255, _AC_SRC_ALPHA)),
            _ULW_ALPHA,
        )
        if not ok:
            raise OSError("UpdateLayeredWindow falló")

    def _destruye(self) -> None:
        self._w.KillTimer(self._hwnd, _ID_TEMPORIZADOR)
        self._w.SelectObject(self._dc, self._viejo)
        self._w.DeleteObject(self._dib)
        self._w.DeleteDC(self._dc)
        self._w.DestroyWindow(self._hwnd)
        self._w.UnregisterClassW(self._clase, self._inst)
        self._w.PostQuitMessage(0)


class _Ventana:
    """La ventana de Tk (plan B). Solo la toca su propio hilo."""

    def __init__(self, tk, cola: queue.Queue, nivel: Callable[[], float]) -> None:
        self._tk = tk
        self._cola = cola
        self._nivel = nivel
        self._estado = "listening"
        self._visible = False
        self._cerrar = False
        self._suave = 0.0
        self._t0 = time.monotonic()
        self._temporizador = None
        self._arrastre = (0, 0)
        self._root = tk.Tk()
        self._root.title("Samantha orbe")  # el título permite localizar la ventana
        self._root.overrideredirect(True)  # sin barra ni bordes
        self._root.attributes("-topmost", True)
        self._root.configure(bg=KEY)
        try:
            self._root.wm_attributes("-transparentcolor", KEY)
        except tk.TclError:
            logger.debug("el sistema no admite color transparente", exc_info=True)
        self._root.withdraw()
        ancho = self._root.winfo_screenwidth()
        alto = self._root.winfo_screenheight()
        self._root.geometry(f"{SIZE}x{SIZE}+{(ancho - SIZE) // 2}+{(alto - SIZE) // 2}")
        self._lienzo = tk.Canvas(
            self._root, width=SIZE, height=SIZE, bg=KEY, highlightthickness=0, bd=0
        )
        self._lienzo.pack()
        self._lienzo.bind("<ButtonPress-1>", self._pulsa)
        self._lienzo.bind("<B1-Motion>", self._arrastra)
        self._root.after(FRAME_MS, self._tick)

    def mainloop(self) -> None:
        self._root.mainloop()
        try:
            self._root.destroy()
        except Exception:
            pass

    def _tick(self) -> None:
        try:
            while True:
                accion, valor = self._cola.get_nowait()
                self._aplica(accion, valor)
        except queue.Empty:
            pass
        except Exception:
            logger.exception("orbe: orden no válida")
        if self._cerrar:
            self._root.quit()
            return
        if self._visible:
            try:
                self._dibuja(time.monotonic() - self._t0)
            except Exception:
                logger.exception("orbe: fallo al dibujar")
        self._root.after(FRAME_MS, self._tick)

    def _aplica(self, accion: str, valor) -> None:
        if accion == "show":
            self._cancela_temporizador()
            self._estado = "listening"
            self._muestra()
        elif accion == "state" and valor in ESTILOS:
            self._estado = valor
        elif accion == "end":
            self._cancela_temporizador()
            if valor:
                self._estado = "error"
                self._temporizador = self._root.after(1200, self._fin_error)
            else:
                self._oculta()
        elif accion == "close":
            self._cerrar = True

    def _muestra(self) -> None:
        if self._visible:
            return
        self._visible = True
        if sys.platform == "win32":
            try:
                self._root.update_idletasks()
                hwnd = int(self._root.wm_frame(), 16)
                if hwnd:
                    user32 = ctypes.windll.user32
                    # SW_SHOWNOACTIVATE: aparece sin robar el foco de lo que estás usando
                    user32.ShowWindow(ctypes.c_void_p(hwnd), 4)
                    # HWND_TOPMOST sin activar: se queda encima del resto de ventanas
                    user32.SetWindowPos(
                        ctypes.c_void_p(hwnd), ctypes.c_void_p(-1), 0, 0, 0, 0, 0x0013
                    )
                    return
            except Exception:
                logger.debug("ShowWindow sin activar no disponible", exc_info=True)
        self._root.deiconify()

    def _oculta(self) -> None:
        self._visible = False
        self._root.withdraw()

    def _fin_error(self) -> None:
        self._temporizador = None
        self._oculta()

    def _cancela_temporizador(self) -> None:
        if self._temporizador is not None:
            self._root.after_cancel(self._temporizador)
            self._temporizador = None

    def _pulsa(self, evento) -> None:
        self._arrastre = (evento.x, evento.y)

    def _arrastra(self, evento) -> None:
        dx, dy = self._arrastre
        self._root.geometry(f"+{evento.x_root - dx}+{evento.y_root - dy}")

    def _dibuja(self, t: float) -> None:
        c = self._lienzo
        c.delete("all")
        self._suave = 0.7 * self._suave + 0.3 * self._nivel()  # movimiento suave
        p = parametros(self._estado, t, self._suave)
        m = SIZE / 2
        r = (m - 16) * p["radio"]
        base = p["color"]
        oscuro = mezcla(base, "#04060b", 0.62)
        borde = mezcla(base, p["acento"], 0.22)
        claro = mezcla(base, "#ffffff", 0.35)
        # cuerpo plano con viñeta al borde: sin brillo ni reflejo interior
        pasos = 16
        for i in range(pasos, 0, -1):
            f = i / pasos
            rr = r * f
            tt = (1.0 - f) ** 0.55
            col = mezcla(oscuro, base, min(1.0, 0.30 + 0.70 * tt))
            if f > 0.84:
                col = mezcla(col, borde, (f - 0.84) * 1.4)
            c.create_oval(
                m - rr,
                m - rr,
                m + rr,
                m + rr,
                outline="",
                fill=col,
            )
        # símbolo infinito crema (mismo trazo que la ventana con alfa)
        pts = puntos_simbolo(m, m, r, n=72)
        coords = [v for pt in pts for v in pt]
        coords += coords[:2]
        c.create_line(
            *coords,
            fill=mezcla(base, "#000000", 0.35),
            width=4,
            smooth=True,
        )
        c.create_line(*coords, fill=SIMBOLO, width=2, smooth=True)
        # anillo giratorio (transcribiendo) o tres puntos orbitando (pensando)
        if self._estado == "transcribing":
            rr = r * 1.18
            c.create_arc(
                m - rr,
                m - rr,
                m + rr,
                m + rr,
                start=p["angulo"],
                extent=110,
                style="arc",
                outline=claro,
                width=3,
            )
        elif self._estado == "thinking":
            rr = r * 1.15
            for k in range(3):
                a = math.radians(p["angulo"] + 120 * k)
                x, y = m + rr * math.cos(a), m + rr * math.sin(a)
                c.create_oval(x - 4, y - 4, x + 4, y + 4, outline="", fill=claro)


class OrbController:
    """Lado del asistente: manda órdenes al orbe sin bloquear y sin lanzar errores."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._cola: queue.Queue = queue.Queue()
        self._nivel = 0.0
        self._hilo: threading.Thread | None = None

    def start(self) -> None:
        """Arranca la ventana en su hilo. Empieza oculta hasta que llegue show()."""
        if not self._enabled or self._hilo is not None:
            return
        self._hilo = threading.Thread(target=self._corre, name="orbe", daemon=True)
        self._hilo.start()

    def _corre(self) -> None:
        ventana = None
        try:
            ventana = _VentanaAlfa(self._cola, lambda: self._nivel)
        except Exception as exc:
            # Falta de Pillow es una instalacion incompleta, no un bug del
            # orbe: un aviso corto y se usa Tk. El traceback queda en DEBUG.
            if "falta Pillow" in str(exc):
                logger.warning(
                    "Orbe: Pillow no instalado; uso la ventana Tk "
                    "(instala: pip install pillow)"
                )
            else:
                logger.warning("Orbe: sin ventana con alfa (%s); uso la de Tk", exc)
            logger.debug("Orbe: fallo de ventana alfa", exc_info=True)
        try:
            if ventana is None:
                import tkinter

                ventana = _Ventana(tkinter, self._cola, lambda: self._nivel)
            ventana.mainloop()
        except Exception:
            logger.warning(
                "Orbe de estado no disponible; sigo sin ventana", exc_info=True
            )
            self._enabled = False

    def _envia(self, accion: str, valor=None) -> None:
        if self._enabled:
            self._cola.put((accion, valor))

    def show(self) -> None:
        self._envia("show")

    def state(self, estado: str) -> None:
        self._envia("state", estado)

    def level(self, rms: float) -> None:
        if self._enabled:
            self._nivel = nivel_normalizado(rms)

    def end(self, error: bool = False) -> None:
        self._envia("end", error)

    def close(self) -> None:
        if self._hilo is None:
            return
        self._envia("close")
        self._hilo.join(timeout=2)
        self._hilo = None
