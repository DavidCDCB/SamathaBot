"""Genera el icono de Samantha a partir del orbe rosado de la app.

Uso:
    .venv\\Scripts\\python.exe tools\\make_orb_icon.py

Produce:
    assets\\samantha-orb.png   vista previa de 1024 x 1024
    assets\\samantha.ico       icono multi-tamano para PyInstaller/Windows
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openocto.ui.orb import SIZE, dibuja_fotograma, parametros

ESCALA = 5.0  # se dibuja a 1100 px y se recorta antes de reducir
RECORTE = 2.9  # lado = 2.9 * radio: el orbe ocupa ~69 % del icono
LADO = 1024
TAMANOS = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def _emblema() -> Image.Image:
    """Orbe "speaking" sin corona ni ondas, centrado y a buen tamaño."""
    base = dibuja_fotograma("speaking", 0.0, 0.0, escala=ESCALA, efectos=False)
    p = parametros("speaking", 0.0, 0.0)
    tam = round(SIZE * ESCALA)
    radio = (tam / 2 - 16 * ESCALA) * p["radio"]
    centro = base.width / 2
    lado = int(round(radio * RECORTE))
    caja = (
        int(centro - lado / 2),
        int(centro - lado / 2),
        int(centro + lado / 2),
        int(centro + lado / 2),
    )
    return base.crop(caja).resize((LADO, LADO), Image.LANCZOS)


def main() -> int:
    icono = _emblema()
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)

    png = assets / "samantha-orb.png"
    ico = assets / "samantha.ico"
    icono.save(png, format="PNG")
    icono.save(ico, format="ICO", sizes=[(s, s) for s in TAMANOS])

    print(f"PNG: {png} ({icono.width}x{icono.height})")
    print(f"ICO: {ico} ({', '.join(str(s) for s in TAMANOS)} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
