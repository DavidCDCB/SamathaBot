"""Genera la carpeta portable dist\\Samantha (Windows x64, sin Python instalado).

Uso:
    .venv\\Scripts\\python.exe build_portable.py [--with-key] [--no-models] [--distpath dist\\otra]

Produce:
    dist\\Samantha\\Samantha.exe        (consola; icono del orbe; hotkey global Ctrl+Shift+Space)
    dist\\Samantha\\_internal\\          (runtime + onnxruntime + piper/espeak + DLLs)
    dist\\Samantha\\assistant-config.yaml  (editable, en la raiz)
    dist\\Samantha\\personas\\samantha\\
    dist\\Samantha\\models\\             (whisper + voces + VAD + Vosk)
    dist\\Samantha\\LEEME.txt

La key NO se copia salvo --with-key; se crea con: Samantha.exe key
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv" / "Scripts" / "python.exe"
DIST = ROOT / "dist"
APP = DIST / "Samantha"
ICON = ROOT / "assets" / "samantha.ico"
SITEPKG = Path(
    subprocess.check_output(
        [
            str(VENV),
            "-c",
            "import pywhispercpp, pathlib; "
            "print(pathlib.Path(pywhispercpp.__file__).resolve().parent.parent)",
        ],
        text=True,
    ).strip()
)

HIDDEN = [
    "openocto.skills.notes",
    "openocto.skills.base",
    "openocto.tts.piper",
    "openocto.tts.edge",
    "openocto.tts.fallback",
    "openocto.stt.whisper_cpp",
    "openocto.vad.silero",
    "openocto.vad.base",
    "openocto.wakeword.vosk_detector",
    "openocto.ai.openai_compat",
    "openocto.ai.router",
    "openocto.persona.manager",
    "openocto.audio.capture",
    "openocto.audio.player",
    "openocto.utils.keyboard",
    "openocto.utils.console",
    "openocto.ui.orb",
    "openocto.utils.logging_setup",
    "openocto.utils.model_downloader",
    "openocto.history",
    "openocto.memory",
    "openocto.state_machine",
    "pywhispercpp",
    "pywhispercpp.model",
    "pynput",
    "pynput.keyboard",
    "sounddevice",
    "questionary",
    "openai",
    "onnxruntime",
    "piper",
    "yaml",
    "click",
    "numpy",
    "requests",
    "tqdm",
    "edge_tts",
    "miniaudio",
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def _rename_retry(src: Path, dst: Path, attempts: int = 8, delay: float = 0.5) -> None:
    """Rename con reintentos: Defender/indexer puede soltar el handle tarde."""
    last: OSError | None = None
    for _ in range(attempts):
        try:
            src.rename(dst)
            return
        except OSError as exc:
            last = exc
            time.sleep(delay)
    if last is not None:
        raise last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--with-key", action="store_true", help="copiar la key actual a secret\\"
    )
    ap.add_argument(
        "--no-models",
        action="store_true",
        help="no copiar modelos (se bajan al primer arranque)",
    )
    ap.add_argument(
        "--distpath",
        default=str(DIST),
        help="carpeta dist alternativa (util si dist\\Samantha esta abierta)",
    )
    args = ap.parse_args()

    global APP
    dist = Path(args.distpath).expanduser().resolve()
    final = dist / "Samantha"
    stage = dist / "_stage_Samantha"
    APP = stage / "Samantha"  # se construye aparte y solo se intercambia al final

    if not VENV.exists():
        sys.exit("Falta .venv: crea el entorno primero (ver README).")

    try:
        subprocess.run(
            [str(VENV), "-c", "import PyInstaller"], check=True, capture_output=True
        )
    except subprocess.CalledProcessError:
        print("Instalando PyInstaller en el venv...")
        run([VENV, "-m", "pip", "install", "-q", "pyinstaller"])

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    dlls = sorted(SITEPKG.glob("whisper-*.dll")) + sorted(SITEPKG.glob("ggml-*.dll"))
    if not dlls:
        print("AVISO: no encontre whisper-*.dll/ggml-*.dll en site-packages")

    if not ICON.exists():
        print("Generando icono del orbe...")
        run([VENV, ROOT / "tools" / "make_orb_icon.py"])
    if not ICON.exists():
        sys.exit(f"Falta el icono: {ICON}")

    cmd = [
        VENV,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--name",
        "Samantha",
        "--icon",
        str(ICON),
        "--distpath",
        str(stage),
        "--paths",
        str(ROOT),
        "--add-data",
        f"{ROOT / 'config' / 'default.yaml'}{os.pathsep}config",
        "--collect-all",
        "onnxruntime",
        "--collect-all",
        "piper",
        "--collect-all",
        "sounddevice",
        "--collect-all",
        "_sounddevice_data",
        "--collect-all",
        "pywhispercpp",
        "--collect-all",
        "edge_tts",
        "--collect-all",
        "miniaudio",
        "--collect-all",
        "PIL",
        "--collect-all",
        "vosk",
    ]
    for mod in HIDDEN:
        cmd += ["--hidden-import", mod]
    for dll in dlls:
        cmd += ["--add-binary", f"{dll}{os.pathsep}."]
    cmd.append(str(ROOT / "assistant.py"))
    run(cmd)

    # Assets editables junto al exe
    shutil.copytree(ROOT / "personas", APP / "personas", dirs_exist_ok=True)
    shutil.copy2(ROOT / "assistant-config.yaml", APP / "assistant-config.yaml")

    # Modelos: solo el whisper y las voces que usa la config
    copied: list[str] = []
    if not args.no_models:
        cfg = yaml.safe_load(
            (ROOT / "assistant-config.yaml").read_text(encoding="utf-8")
        )
        src = Path.home() / ".openocto" / "models"
        dst = APP / "models"
        size = str(cfg["stt"].get("model_size", "base"))
        w = src / "whisper" / f"ggml-{size}.bin"
        if w.exists():
            (dst / "whisper").mkdir(parents=True, exist_ok=True)
            shutil.copy2(w, dst / "whisper" / w.name)
            copied.append(w.name)
        else:
            print(f"AVISO: falta {w} (se descargara al primer arranque)")
        for voice in cfg["tts"].get("models", {}).values():
            for ext in (".onnx", ".onnx.json"):
                f = src / "piper" / f"{voice}{ext}"
                if f.exists():
                    (dst / "piper").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst / "piper" / f.name)
                    copied.append(f.name)
                else:
                    print(f"AVISO: falta {f}")

        vad = src / "vad" / "silero_vad.onnx"
        if vad.exists():
            (dst / "vad").mkdir(parents=True, exist_ok=True)
            shutil.copy2(vad, dst / "vad" / vad.name)
            copied.append(vad.name)
        else:
            print(f"AVISO: falta {vad} (se descargara al primer arranque)")

        wm = cfg.get("wakeword", {}).get("model", "")
        if cfg.get("wakeword", {}).get("enabled") and wm:
            vm = src / "vosk" / wm
            if vm.exists():
                (dst / "vosk").mkdir(parents=True, exist_ok=True)
                shutil.copytree(vm, dst / "vosk" / wm, dirs_exist_ok=True)
                copied.append(wm)
            else:
                print(f"AVISO: falta {vm} (se descargara al primer arranque)")

    if args.with_key:
        key = ROOT / "secret" / "OPENCODE_GO_API_KEY"
        if key.exists():
            (APP / "secret").mkdir(exist_ok=True)
            shutil.copy2(key, APP / "secret" / "OPENCODE_GO_API_KEY")
            print("key copiada a secret\\ (OJO: viaja con la carpeta)")
        else:
            print("AVISO: no hay secret\\OPENCODE_GO_API_KEY para copiar")

    leeme = """Samantha — asistente de voz portable (Windows x64)

1) Valida la API key:      Samantha.exe check
2) Si no hay key:          Samantha.exe key     (se guarda en secret\\)
3) Arranca:                Samantha.exe start   (di "Hola Samantha" o pulsa
                           Ctrl+Shift+Space; tras responder sigue escuchando)
   Con debug en consola:   Samantha.exe start --debug
4) Prueba la voz:          Samantha.exe say "Hola, soy Samantha"
5) Latencia por turno:     Samantha.exe report

Config:    assistant-config.yaml  (en esta carpeta; edítalo y reinicia)
Modelos:   models\\               (whisper, voces Piper, VAD Silero y Vosk)
Datos:     data\\history.db        (memoria)  ·  logs\\assistant.log
Personas:  personas\\samantha\\

Requiere internet para el LLM (OpenCode Go). El audio y la memoria son locales.
Aviso: secret\\ guarda tu API key; no compartas esta carpeta si la copiaste.
"""
    (APP / "LEEME.txt").write_text(leeme, encoding="utf-8")

    # Conservar memoria, logs y key del portable anterior. El config nuevo
    # manda, pero se deja el anterior como assistant-config.previous.yaml.
    if final.exists():
        for name in ("data", "logs", "secret"):
            src_dir = final / name
            if src_dir.exists():
                shutil.copytree(src_dir, APP / name, dirs_exist_ok=True)
        old_cfg = final / "assistant-config.yaml"
        if old_cfg.exists():
            shutil.copy2(old_cfg, APP / "assistant-config.previous.yaml")

    # Intercambio seguro: si la carpeta final esta abierta, el rename falla
    # ANTES de borrar nada. La build nueva queda intacta en stage.
    old = dist / f"Samantha.old-{int(time.time())}"
    if final.exists():
        try:
            _rename_retry(final, old)
        except OSError as exc:
            print(f"\nERROR: no pude reemplazar {final}: {exc}")
            print("Cierra Explorer/terminal que tengan esa carpeta abierta.")
            print(f"La build nueva quedo intacta en: {APP}")
            return 1
    try:
        _rename_retry(APP, final)
    except OSError:
        if old.exists() and not final.exists():
            _rename_retry(old, final)
        raise
    APP = final
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(stage, ignore_errors=True)

    total = sum(f.stat().st_size for f in APP.rglob("*") if f.is_file())
    print(f"\nPortable listo: {APP}")
    print(f"  tamano total: {total / 1048576:.1f} MB")
    print(f"  modelos copiados: {', '.join(copied) if copied else 'ninguno'}")
    print(
        "  key incluida: "
        + ("si (secret\\)" if args.with_key else "no (usa Samantha.exe key)")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
