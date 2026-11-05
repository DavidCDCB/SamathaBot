# -*- coding: utf-8 -*-
"""Samantha - un solo acceso a todo el asistente, en Python puro.

Desarrollo:
    python assistant.py start [--debug]   # arrancar (sin comando también arranca)
    python assistant.py key [-y]          # guardar la API key (enmascarada)
    python assistant.py check             # de donde sale la key + ping real a Go
    python assistant.py report            # latencia por turno desde el log
    python assistant.py say "texto"       # probar la voz (sintetiza y reproduce)
    python assistant.py facts             # ver lo que Samantha aprendió de ti
    python assistant.py facts delete 3 4  # borrar hechos por id (ver facts)
    python assistant.py orb               # prueba visual del orbe de estado

Carpeta portable:
    Samantha.exe [start] [--debug] | key | check | report | say "texto"
    (doble clic en Samantha.exe equivale a `start`, sin terminal)
    Samantha.exe --console [start]   # muestra/crea la terminal

No se imprimen claves nunca: longitudes y huellas SHA-256 (10 hex) nada mas.

Fuentes de la credencial, en orden (la primera con contenido gana):
  1. secret\\OPENCODE_GO_API_KEY   (recomendada; forrada por .gitignore)
  2. secret\\.env o .env           (linea OPENCODE_GO_API_KEY=...)
  3. variable de entorno del proceso
  4. HKCU\\Environment             (lo que escribe setx)
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import getpass
import hashlib
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
# En un exe congelado __file__ apunta a _internal: la raiz editable es la del exe.
ROOT = (
    Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
)
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
CLI = "Samantha.exe" if FROZEN else "python assistant.py"
CONFIG = ROOT / "assistant-config.yaml"
KEY_NAME = "OPENCODE_GO_API_KEY"
SECRET_FILE = ROOT / "secret" / "OPENCODE_GO_API_KEY"
FIRECRAWL_SECRET = ROOT / "secret" / "FIRECRAWL_API_KEY"


# ── bootstrap: si existe el venv y no estamos en el, re-ejecutar alli ────────
def _acquire_instance_lock():
    """Evita dos Samantha escuchando el mismo micro y escribiendo la misma BD.

    Devuelve el archivo abierto (hay que mantenerlo vivo) o False si ya hay
    otra instancia. En Windows el lock lo libera el sistema al terminar.
    """
    try:
        from openocto.config import USER_CONFIG_DIR
    except ImportError:
        return None
    lock_path = USER_CONFIG_DIR / "samantha.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return False
    return f


def _ensure_venv() -> None:
    if FROZEN:
        return  # dentro del exe no hay venv que relanzar
    if not VENV_PY.exists():
        return  # dejamos que falle bonito mas abajo, con mensaje propio
    if str(sys.executable).lower().startswith(str(VENV_PY).lower()):
        return
    args = [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        # En Windows execv crea el proceso nuevo y termina este: no queda un
        # padre esperando al hijo (se veia como dos python.exe en el Administrador).
        os.execv(str(VENV_PY), args)
    except OSError:
        # Respaldo por si el sistema no permite execv.
        r = subprocess.run(args, cwd=str(ROOT))
        sys.exit(r.returncode)


# ── credencial ───────────────────────────────────────────────────────────────
def _clean(txt) -> str | None:
    if txt is None:
        return None
    t = str(txt).strip().strip('"').strip("'").strip()
    return t or None


def _huella(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()[:10]


def _leer_archivo_clave(path: Path) -> tuple[str | None, str | None]:
    """(valor, problema). Distingue 'no existe' de 'existe vacio' (0 bytes)."""
    if not path.exists():
        return None, None
    if path.stat().st_size == 0:
        return None, f"{path} existe pero esta VACIO (0 bytes)"
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    return _clean(raw), None


def _leer_dotenv(path: Path) -> str | None:
    if not path.exists():
        return None
    for linea in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        m = re.match(r"^\s*(?:export\s+)?OPENCODE_GO_API_KEY\s*=(.*)$", linea)
        if m:
            return _clean(m.group(1))
    return None


def _reg() -> tuple[str | None, str | None]:
    """(valor actual, nota). Y guarda el handle para borrar si hace falta."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            v = _clean(winreg.QueryValueEx(h, KEY_NAME)[0])
            return v, None
    except FileNotFoundError:
        return None, None
    except OSError as e:
        return None, f"registro ilegible: {e}"


def resolve_key() -> dict:
    """Devuelve dict(valor, fuente, problemas:list, fuentes:[(fuente,valor)])."""
    problemas: list[str] = []
    fuentes: list[tuple[str, str]] = []

    v, p = _leer_archivo_clave(SECRET_FILE)
    if p:
        problemas.append(p)
    if v:
        fuentes.append((str(SECRET_FILE), v))
    for df in (ROOT / "secret" / ".env", ROOT / ".env"):
        x = _leer_dotenv(df)
        if x:
            fuentes.append((f"dotenv {df}", x))
    x = _clean(os.environ.get(KEY_NAME))
    if x:
        fuentes.append(("variable de entorno del proceso", x))
    rv, rp = _reg()
    if rp:
        problemas.append(rp)
    if rv:
        fuentes.append(("registro HKCU\\Environment", rv))

    electa = fuentes[0] if fuentes else (None, None)
    distintas = [f for f in fuentes[1:] if f[1] != electa[1]]
    if electa[1] and re.search(r"\s", electa[1]):
        problemas.append(
            "la key elegida contiene espacios/blancos internos: probablemente copiada mal"
        )
    return {
        "valor": electa[1],
        "fuente": electa[0],
        "fuentes": fuentes,
        "distintas": distintas,
        "problemas": problemas,
    }


def _aviso_fuentes(r: dict, verbose: bool = True) -> None:
    if verbose:
        for i, (fuente, val) in enumerate(r["fuentes"]):
            marca = "ELEGIDA" if i == 0 else "       "
            print(f"  {marca}  {fuente}  |  {len(val)} chars  huella {_huella(val)}")
    for d, val in r["distintas"]:
        print(
            f"  OJO: {d} tiene una key DISTINTA (huella {_huella(val)}); si la rotaste, limpiala."
        )
    for p in r["problemas"]:
        print(f"  aviso: {p}")


# ── start ────────────────────────────────────────────────────────────────────
def cmd_start(args: argparse.Namespace) -> int:
    r = resolve_key()
    if not r["valor"]:
        print("No hay API key util.")
        for p in r["problemas"]:
            print(f"  {p}")
        print(f"  Guardala con:  {CLI} key")
        return 1
    os.environ[KEY_NAME] = r["valor"]
    if FIRECRAWL_SECRET.exists():
        fc = _clean(FIRECRAWL_SECRET.read_text(encoding="utf-8-sig", errors="replace"))
        if fc:
            os.environ["FIRECRAWL_API_KEY"] = fc
    print(
        f"  key   {r['fuente']}  |  {len(r['valor'])} chars  |  "
        f"huella {_huella(r['valor'])}"
    )
    _aviso_fuentes(r, verbose=False)
    print(f"  log   logs\\assistant.log  ->  {CLI} report")
    print()
    try:
        from openocto.app import OpenOctoApp
        from openocto.config import load_config
        from openocto.utils.logging_setup import setup_logging
    except ImportError as e:
        print(
            f"Falta openocto en este interprete ({e}).\n"
            "  instala las dependencias en el venv:  "
            '.venv\\Scripts\\python.exe -m pip install -e ".[audio]"'
        )
        return 2

    config = load_config(str(CONFIG))
    if "web_search" in config.skills.enabled and not os.environ.get(
        "FIRECRAWL_API_KEY"
    ):
        print(
            "  aviso: sin FIRECRAWL_API_KEY; web_search queda desactivada "
            f"({CLI} key firecrawl)"
        )
    if getattr(args, "debug", False):
        os.environ["OCTO_DEBUG"] = "1"
        config.logging.console_level = "DEBUG"
    lock = _acquire_instance_lock()
    if lock is False:
        print("Ya hay otra instancia de Samantha usando esta carpeta.")
        print("  Cierra la otra (Ctrl+C) o revisa el Administrador de tareas.")
        return 1
    setup_logging(config.logging)
    app = OpenOctoApp(config)
    app._instance_lock = lock
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            app._cleanup()
        except Exception:
            pass
    print("\nbye.")
    return 0


# ── key ──────────────────────────────────────────────────────────────────────
def cmd_key(args: argparse.Namespace) -> int:
    provider = getattr(args, "provider", "opencode")
    secret_file = FIRECRAWL_SECRET if provider == "firecrawl" else SECRET_FILE
    label = "Firecrawl" if provider == "firecrawl" else "OpenCode Go"
    print(f"Pega aqui la key de {label} (no se ve al escribir):")
    txt = _clean(getpass.getpass("key: "))
    if not txt:
        print("No escribiste nada; no se guardo nada.")
        return 1
    if re.search(r"\s", txt):
        print("La key contiene espacios internos (copiado mal). No se guardo nada.")
        return 1

    secret_file.parent.mkdir(parents=True, exist_ok=True)
    # utf-8 SIN BOM: un BOM corromperia la key y el 401 no diria por que
    io.open(secret_file, "w", encoding="utf-8", newline="\n").write(txt)
    try:
        subprocess.run(
            [
                "icacls",
                str(secret_file),
                "/inheritance:r",
                "/grant",
                f"*{os.environ.get('USERNAME', '')}:F",
            ],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass
    print(f"guardada: {secret_file}  |  {len(txt)} chars  huella {_huella(txt)}")

    if provider == "firecrawl":
        if args.clear_registry:
            print("aviso: -y solo aplica a la key de OpenCode Go.")
        print(f"Comprueba:  {CLI} check")
        return 0

    if args.clear_registry:
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE
            ) as h:
                winreg.DeleteValue(h, KEY_NAME)
            # avisar al resto de sesiones (nuevo valor del entorno)
            HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x1A, 0x0002
            try:
                res = ctypes.c_ulong()
                ctypes.windll.user32.SendMessageTimeoutW(
                    HWND_BROADCAST,
                    WM_SETTINGCHANGE,
                    0,
                    "Environment",
                    SMTO_ABORTIFHUNG,
                    5000,
                    ctypes.byref(res),
                )
            except Exception:
                pass
            print(
                "registro HKCU\\Environment: variable eliminada (abre terminales nuevas)."
            )
        except FileNotFoundError:
            print("registro: no existia esa variable.")
        except OSError as e:
            print(f"registro: no se pudo borrar ({e}).")

    r = resolve_key()
    if r["valor"] != txt and r["fuente"] != str(SECRET_FILE):
        print(
            f"aviso: la fuente elegida pasaria a ser {r['fuente']} (huella {_huella(r['valor'] or '')})."
        )
    else:
        print(f"el asistente usara: {r['fuente']}")
    print(f"Comprueba:  {CLI} check")
    return 0


# ── check ────────────────────────────────────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> int:
    print("=== firecrawl ===")
    if FIRECRAWL_SECRET.exists():
        fc = _clean(FIRECRAWL_SECRET.read_text(encoding="utf-8-sig", errors="replace"))
        if fc:
            print(f"  OK  {FIRECRAWL_SECRET}  |  {len(fc)} chars  huella {_huella(fc)}")
        else:
            print(f"  {FIRECRAWL_SECRET} existe pero esta VACIO")
    else:
        print(f"  falta -> {CLI} key firecrawl")
    print()

    r = resolve_key()
    print("=== fuentes encontradas ===")
    if not r["fuentes"]:
        print(f"  ninguna. Guarda la key con:  {CLI} key")
        return 1
    _aviso_fuentes(r)
    if not r["valor"]:
        print("no hay ninguna fuente util.")
        return 1
    os.environ[KEY_NAME] = r["valor"]
    print(f"\n=== prueba real contra OpenCode Go (key de: {r['fuente']}) ===")

    try:
        import requests
        from openocto.config import load_config
    except ImportError as e:
        print(
            "FALLO: faltan dependencias (requests/openocto): ",
            e,
            "\n  corre con el venv:  .venv\\Scripts\\python.exe assistant.py check",
        )
        return 2

    cfg = load_config(str(CONFIG))
    go = cfg.ai.providers[cfg.ai.default_backend]
    url = go.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {r['valor']}",
        "Content-Type": "application/json",
        **go.headers,
    }

    def redactar(s: str) -> str:
        return s.replace(r["valor"], "«key-oculta»")

    print(
        f"endpoint : {url}\nmodelo   : {go.model}\ncabeceras: {go.headers}\nkey      : {len(r['valor'])} chars (oculta)"
    )
    try:
        from openocto.persona.manager import PersonaManager

        sysp = PersonaManager().activate(cfg.persona).system_prompt
        print(f"prompt   : system de persona {cfg.persona} ({len(sysp)} chars)")
    except Exception as e:  # noqa: BLE001
        sysp = "Respondes solo en español, en una frase corta."
        print(f"prompt   : generico (persona no cargable: {type(e).__name__})")

    body = {
        "model": go.model,
        "messages": [
            {"role": "system", "content": sysp},
            {
                "role": "user",
                "content": "Me oíste? Contesta solamente con la palabra: ok",
            },
        ],
        "max_tokens": int(os.environ.get("OCTO_PING_MAX_TOKENS", "200")),
        "stream": False,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"FALLO de red: {type(e).__name__}: {redactar(str(e))[:200]}")
        return 1
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {redactar(resp.text)[:300]}")
        try:
            code = (resp.json().get("error") or {}).get("code")
            if code == "api_key_expired":
                print(
                    f"-> key caducada: rota en https://opencode.ai/console y {CLI} key"
                )
        except ValueError:
            pass
        return 1
    j = resp.json()
    uso = j.get("usage") or {}
    contenido = ((j.get("choices") or [{}])[0].get("message") or {}).get(
        "content"
    ) or ""
    print(f"OK HTTP 200 | finish {(j.get('choices') or [{}])[0].get('finish_reason')}")
    print(
        f"    respuesta: {contenido[:60]!r} | tokens out={uso.get('completion_tokens')}"
    )
    return 0


# ── report ───────────────────────────────────────────────────────────────────
def cmd_report(args: argparse.Namespace) -> int:
    log = ROOT / "logs" / "assistant.log"
    if not log.exists():
        print(f"no existe {log}: arranca primero ({CLI} start)")
        return 1
    import datetime as _dt

    TS = re.compile(r"^(\d{2}:\d{2}:\d{2}),(\d{3}) ")
    STATE = re.compile(r"State: (\w+) -> (\w+) \(trigger: (\w+)\)")
    STT = re.compile(r"Transcribed in (\d+)ms \[(\w+)\]: (.*)")
    turnos: list[dict] = []
    cur: dict = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TS.match(line)
        tm = None
        if m:
            tm = (
                _dt.datetime.strptime(m.group(1), "%H:%M:%S").timestamp()
                + int(m.group(2)) / 1000
            )
        s = STATE.search(line)
        if s and tm is not None:
            frm, to, trig = s.groups()
            if to == "recording" and frm == "idle":
                cur = {"grabando": tm}
            elif trig == "stop_recording" and cur:
                cur["fin_habla"] = tm
            elif to == "transcribing" and cur:
                cur["stt_ini"] = tm
            elif to == "processing" and cur:
                cur["stt_done"] = tm
            elif to == "speaking" and cur:
                cur["ia_done"] = tm
            elif to == "idle" and cur and "ia_done" in cur:
                cur["fin_audio"] = tm
                turnos.append(cur)
                cur = {}
        t = STT.search(line)
        if t and cur is not None:
            cur["stt_ms"] = int(t.group(1))
            cur["texto"] = t.group(3)[:56]

    if not turnos:
        print("el log no tiene turnos completos (o logueabas INFO; hace falta DEBUG)")
        return 1
    print(
        f"turnos: {len(turnos)}   (silencio de corte: ver config, normalmente 0,8 s)\n"
    )
    print(
        f"{'#':>2} {'STT':>7} {'LLM':>7} {'audio':>7} {'a 1er audio (aprox)':>22}  oido"
    )
    lat = []
    for i, tu in enumerate(turnos, 1):
        d_stt = tu.get("stt_ms", 0) / 1000.0 or (
            tu["stt_done"] - tu.get("stt_ini", tu["stt_done"])
        )
        d_llm = tu["ia_done"] - tu["stt_done"]
        d_audio = tu["fin_audio"] - tu["ia_done"]
        est = d_stt + d_llm + 0.15
        lat.append(est)
        print(
            f"{i:>2} {d_stt:>6.2f}s {d_llm:>6.2f}s {d_audio:>6.2f}s {est:>20.2f}s  {tu.get('texto', '')}"
        )
    n = len(lat)
    print(
        f"\nestimacion media a primer audio: {sum(lat) / n:.2f}s "
        f"(+0,8s de deteccion de silencio ≈ {sum(lat) / n + 0.8:.2f}s) | "
        f"min {min(lat):.2f} max {max(lat):.2f} | umbral EVAL: <=4s"
    )
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    """Sintetiza y reproduce un texto con la voz configurada (prueba rapida)."""
    text = " ".join(args.text).strip()
    if not text:
        print("Nada que decir.")
        return 1

    try:
        from openocto.audio.player import AudioPlayer
        from openocto.config import load_config
        from openocto.tts.factory import create_tts_engine
    except ImportError as e:
        print(f"FALLO: faltan dependencias de audio: {e}")
        return 1

    cfg = load_config(str(CONFIG))
    lang = args.lang
    model = cfg.tts.models.get(lang) or next(iter(cfg.tts.models.values()), "")
    engine = cfg.tts.engines.get(lang, cfg.tts.engine)
    voice = cfg.tts.edge.voices.get(lang, "-") if engine == "edge" else model

    tts = create_tts_engine(lang, model, cfg.tts)
    t0 = time.perf_counter()
    seg = tts.synthesize(text)
    dt = time.perf_counter() - t0
    dur = len(seg.audio) / seg.sample_rate if seg.sample_rate else 0.0
    used = getattr(tts, "last_engine", engine)
    if used != engine:
        voice = model
    print(
        f"  motor={used} voz={voice} | {len(text)} chars -> "
        f"{dur:.2f}s de audio en {dt:.2f}s ({seg.sample_rate} Hz)"
    )
    AudioPlayer(cfg.audio).play(seg.audio, seg.sample_rate, gain=cfg.tts.gain)
    return 0


# ── facts ────────────────────────────────────────────────────────────────────
def cmd_facts(args: argparse.Namespace) -> int:
    """Lista o borra los hechos que Samantha guardó sobre el usuario."""
    try:
        from openocto.history import HistoryStore
    except ImportError as e:
        print(f"FALLO: faltan dependencias: {e}")
        return 2

    store = HistoryStore()
    if args.user:
        user = store.get_user_by_name(args.user)
        if user is None:
            print(f"No existe el usuario {args.user!r}.")
            return 1
    else:
        users = store.list_users()
        if not users:
            print("Todavía no hay usuarios: arranca el asistente una vez.")
            return 1
        if len(users) > 1:
            print("Hay varios usuarios: indica --user NOMBRE.")
            return 1
        user = users[0]

    facts = store.get_active_facts(user["id"])
    if args.action == "delete":
        if not args.all and not args.ids:
            print("Indica los ids a borrar (ver `facts list`) o usa --all.")
            return 1
        activos = {f["id"] for f in facts}
        objetivos = sorted(activos) if args.all else args.ids
        borrados = 0
        for fid in objetivos:
            if fid in activos:
                store.deactivate_fact(fid)
                borrados += 1
            else:
                print(f"  #{fid}: no es un hecho activo de {user['name']}.")
        print(f"Borrados {borrados} hecho(s) de {user['name']}.")
        return 0 if borrados else 1

    if not facts:
        print(f"{user['name']}: todavía no hay hechos guardados.")
        return 0
    print(f"Hechos de {user['name']}:")
    for f in facts:
        print(f"  #{f['id']:<4} [{f['category']}] {f['fact']}")
    return 0


# ── orb ──────────────────────────────────────────────────────────────────────
def cmd_orb(args: argparse.Namespace) -> int:
    """Muestra el orbe recorriendo sus estados (prueba visual, sin micrófono)."""
    import math

    from openocto.ui.orb import OrbController

    orbe = OrbController(enabled=True)
    orbe.start()
    orbe.show()
    for estado in (
        "listening",
        "hearing",
        "transcribing",
        "thinking",
        "speaking",
        "waiting",
    ):
        orbe.state(estado)
        for i in range(30):  # ~1,5 s por estado
            orbe.level(1500 * (0.5 + 0.5 * math.sin(i / 3)))
            time.sleep(0.05)
    orbe.end(error=True)
    time.sleep(1.5)
    orbe.close()
    return 0


def _argv_por_defecto(argv: list[str]) -> list[str]:
    """Sin argumentos, el comando es start: doble clic en el .exe."""
    return ["start", *argv] if not argv else argv


_CONSOLE_FLAGS = ("--console", "--terminal")
_CONSOLE_STREAMS: tuple | None = None


class _NullStream:
    """Sumidero para el .exe --windowed cuando no hay terminal."""

    encoding = "utf-8"
    errors = "replace"

    def write(self, _texto: str) -> int:
        return len(_texto)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("sin consola")

    def read(self, *_args, **_kwargs) -> str:
        return ""

    def readline(self, *_args, **_kwargs) -> str:
        return ""

    def reconfigure(self, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


_NULL_STREAM = _NullStream()


def _instalar_salida_nula() -> None:
    sys.stdout = _NULL_STREAM
    sys.stderr = _NULL_STREAM
    sys.stdin = _NULL_STREAM


def _conectar_stdio() -> None:
    """Conecta stdout/stderr/stdin a la consola de Windows ya adjuntada."""
    global _CONSOLE_STREAMS
    if sys.stdout is not None and sys.stderr is not None and sys.stdin is not None:
        return
    try:
        out = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        err = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        inp = open("CONIN$", "r", encoding="utf-8", buffering=1)
    except OSError:
        _instalar_salida_nula()
        return
    sys.stdout, sys.stderr, sys.stdin = out, err, inp
    _CONSOLE_STREAMS = (out, err, inp)


def _preparar_consola(forzar: bool, cmd: str) -> None:
    """En el .exe: start sin terminal; CLI y --console con terminal.

    - Doble clic (start, sin --console): no hay consola; stdout va a un
      sumidero para que print/logging no fallen.
    - key/check/report/say/facts/orb y --help: se adjuntan a la consola
      desde la que se lanzaron o crean una nueva.
    - --console: fuerza la consola.
    """
    if os.name != "nt":
        return
    try:
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        kernel32.AllocConsole.restype = wintypes.BOOL
        kernel32.AttachConsole.restype = wintypes.BOOL
        kernel32.AttachConsole.argtypes = [wintypes.DWORD]
    except Exception:
        _instalar_salida_nula()
        return

    hay_consola = bool(kernel32.GetConsoleWindow())
    creada = False
    if not hay_consola:
        if forzar or cmd not in ("start",):
            # Primero intenta la terminal desde la que se lanzó; si no,
            # crea una nueva (doble clic en check/key/etc.).
            if not kernel32.AttachConsole(0xFFFFFFFF):
                if not kernel32.AllocConsole():
                    _instalar_salida_nula()
                    return
                creada = True
        else:
            _instalar_salida_nula()
            return

    try:
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        if creada:  # no renombrar la terminal del usuario si solo se adjuntó
            kernel32.SetConsoleTitleW("Samantha")
    except Exception:
        pass
    _conectar_stdio()


def _quitar_flag_consola(argv: list[str]) -> list[str]:
    """Saca --console/--terminal antes de argparse."""
    return [arg for arg in argv if arg not in _CONSOLE_FLAGS]


def main() -> int:
    argv = sys.argv[1:]
    quiere_consola = any(arg in _CONSOLE_FLAGS for arg in argv)
    sys.argv[1:] = _argv_por_defecto(_quitar_flag_consola(argv))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if FROZEN or sys.stdout is None or quiere_consola:
        _preparar_consola(quiere_consola, cmd)
    _ensure_venv()
    ap = argparse.ArgumentParser(
        prog=CLI,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Windows: doble clic arranca en segundo plano, sin terminal. "
            "Usa --console para mostrar/crear la terminal."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start", help="arrancar el asistente")
    p.add_argument(
        "--debug",
        action="store_true",
        help="muestra DEBUG en consola (el archivo ya guarda DEBUG)",
    )
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser("key", help="guardar la API key (enmascarada)")
    p.add_argument(
        "provider",
        nargs="?",
        choices=["opencode", "firecrawl"],
        default="opencode",
        help="qué credencial guardar (por defecto: opencode)",
    )
    p.add_argument(
        "-y",
        "--clear-registry",
        action="store_true",
        help="y borrar la copia del registro",
    )
    p.set_defaults(fn=cmd_key)
    p = sub.add_parser("check", help="de donde sale la key + ping real a Go")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("report", help="latencia por turno desde el log")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("say", help="probar la voz: sintetiza y reproduce un texto")
    p.add_argument("text", nargs="+", help="texto a decir")
    p.add_argument("--lang", default="es", help="idioma de la voz (es, en)")
    p.set_defaults(fn=cmd_say)
    p = sub.add_parser("facts", help="ver o borrar lo que Samantha aprendió de ti")
    p.add_argument("action", nargs="?", choices=["list", "delete"], default="list")
    p.add_argument("ids", nargs="*", type=int, help="ids de hechos a borrar (ver list)")
    p.add_argument(
        "--all", action="store_true", help="borrar todos los hechos del usuario"
    )
    p.add_argument("--user", default=None, help="nombre del usuario, si hay varios")
    p.set_defaults(fn=cmd_facts)
    p = sub.add_parser("orb", help="prueba visual del orbe de estado (sin micrófono)")
    p.set_defaults(fn=cmd_orb)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
