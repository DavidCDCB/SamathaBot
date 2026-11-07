"""Main OpenOcto application orchestrator."""

from __future__ import annotations

import asyncio
import difflib
import logging
import time

import numpy as np

from openocto import __version__
from openocto.audio.capture import AudioCapture
from openocto.audio.player import AudioPlayer
from openocto.config import AppConfig
from openocto.history import HistoryStore
from openocto.memory import MemoryManager
from openocto.persona.manager import Persona, PersonaManager
from openocto.skills import SkillRegistry, build_default_registry
from openocto.state_machine import State, StateMachine
from openocto.utils import console as ui
from openocto.utils.dates import fecha_es, hora_es
from openocto.utils.icons import (
    CHECK,
    CROSS,
    MIC,
    WRENCH,
    USER,
)

from openocto.ui.orb import OrbController
from openocto.utils.hotkey import hotkey_label, parse_hotkey
from openocto.utils.names import extraer_nombre

logger = logging.getLogger(__name__)


def normalize_for_stt(
    audio: np.ndarray, target_rms: float = 0.10, max_gain: float = 20.0
) -> np.ndarray:
    """Normaliza la toma COMPLETA antes del STT (nunca por chunk).

    Medido (audio es, whisper base): el AGC por chunk subia el CER con ruido de
    3,2% a 10,6% porque amplificaba el silencio entre palabras; normalizar la
    toma entera deja el CER como el audio limpio y da margen a micros flojos.
    Solo amplifica (gain >= 1) y recorta a 0.95 para no saturar.
    """
    x = audio.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    if rms < 1e-6:
        return audio
    gain = min(max_gain, max(1.0, target_rms / rms))
    y = np.clip(x * gain, -0.95, 0.95)
    return (y * 32768.0).astype(np.int16)


class OpenOctoApp:
    """Main application — wires all components together and runs the main loop."""

    def __init__(self, config: AppConfig, user_name: str | None = None) -> None:
        self._config = config
        self._requested_user_name = user_name

        # Core infrastructure
        self._state_machine = StateMachine()

        # Audio
        self._capture = AudioCapture(config.audio, mic_gain=config.audio.mic_gain)
        self._player = AudioPlayer(config.audio)

        # P18: activacion. "vad" = una pulsacion empieza y el silencio corta.
        self._vad = None
        self._vad_task: asyncio.Task | None = None
        self._conversation_active = False
        self._auto_turn = False
        self._orb = OrbController(enabled=config.ui.orb)
        self._auto_listen_task: asyncio.Task | None = None
        self._ptt_mode = str(config.audio.ptt_mode or "hold").lower()
        if self._ptt_mode not in ("hold", "vad"):
            logger.warning("audio.ptt_mode=%r desconocido; uso 'hold'", self._ptt_mode)
            self._ptt_mode = "hold"
        if self._ptt_mode == "vad":
            try:
                from openocto.vad.silero import SileroVAD

                self._vad = SileroVAD(config.vad)
            except Exception as exc:
                logger.warning("VAD no disponible (%s); uso PTT de mantener", exc)
                self._ptt_mode = "hold"
        self._ptt_label = hotkey_label(parse_hotkey(config.audio.ptt_hotkey))
        logger.info(
            "PTT mode: %s | continuous: %s",
            self._ptt_mode,
            self._config.audio.continuous,
        )

        # Components initialized lazily (need model downloads)
        self._stt = None
        self._tts_engines: dict[str, object] = {}
        # P8 (enfoque A): buffer del hablado por frases del turno en curso
        self._tts_buf = ""
        self._tts_suppress = False
        # P21: pipeline sintesis <-> reproduccion (evita silencios en los puntos)
        self._tts_text_q: asyncio.Queue | None = None
        self._tts_audio_q: asyncio.Queue | None = None
        self._tts_synth_task: asyncio.Task | None = None
        self._tts_play_task: asyncio.Task | None = None
        self._ai_router = None
        self._persona_manager = PersonaManager()
        self._persona: Persona | None = None
        # Persistent history + memory
        self._history_store = HistoryStore()
        self._memory: MemoryManager | None = None
        self._current_user_id: int | None = None

        # Skills (LLM-callable tools)
        self._skills: SkillRegistry | None = None

        # Bloqueo de procesamiento (compartido por los modos PTT y VAD)
        self._processing = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # P22: si el STT entrega la misma frase dos veces seguidas, no repetir turno
        self._last_user_text = ""
        self._last_user_at = 0.0
        # Texto sintético que _handle_audio usa como "transcripción" (saludo).
        self._pending_text = None
        self._greet_task = None
        self._instance_lock = None
        # P23: preguntar el nombre una vez al inicio si aún no está en memoria
        self._name_unknown = False
        self._name_asked = False
        # P24: wake word local "Hola Samantha" con Vosk
        self._wakeword = None
        self._wakeword_enabled = False
        self._wakeword_guard_task: asyncio.Task | None = None

    def _nombre_conocido(self, uid: int) -> bool:
        """True si el perfil o un hecho guardado ya tienen un nombre real."""
        for user in self._history_store.list_users():
            if user["id"] == uid:
                nombre = str(user.get("name") or "").strip().casefold()
                if nombre not in ("", "user", "usuario"):
                    return True
        for fact in self._history_store.get_active_facts(uid):
            texto = str(fact.get("fact") or "").casefold()
            categoria = str(fact.get("category") or "").casefold()
            if (
                categoria in ("nombre", "name")
                or "se llama" in texto
                or "su nombre es" in texto
            ):
                return True
        return False

    def _captura_nombre(self, uid: int, persona: str, texto: str) -> None:
        """Guarda el nombre dictado por voz y lo deja en la memoria del perfil."""
        if not self._name_unknown:
            return
        nombre = extraer_nombre(texto, esperando=self._name_asked)
        if not nombre:
            return
        self._history_store.set_user_name(uid, nombre)
        self._history_store.add_fact(
            uid,
            f"Se llama {nombre}",
            category="nombre",
            source="user",
            persona=persona,
        )
        self._name_unknown = False
        self._name_asked = False
        logger.info("Nombre guardado en memoria: %s", nombre)
        print(ui.dim(f"  (nombre guardado: {nombre})"))

    async def _resolve_user(self) -> tuple[int, str]:
        """Return (user_id, user_name) based on --user flag or interactive selection."""
        import questionary

        users = self._history_store.list_users()

        # --user NAME given explicitly
        if self._requested_user_name:
            user = self._history_store.get_user_by_name(self._requested_user_name)
            if user is None:
                raise SystemExit(
                    f"{CROSS} User '{self._requested_user_name}' not found. "
                    f"Available: {', '.join(u['name'] for u in users) or 'none'}"
                )
            return user["id"], user["name"]

        # No users yet — create a default one silently
        if not users:
            uid = self._history_store.create_user("User", is_default=True)
            return uid, "User"

        # One user — auto-select
        if len(users) == 1:
            u = users[0]
            return u["id"], u["name"]

        # Multiple users — show last-active first, prompt with arrow keys
        last_active = self._history_store.get_last_active_user()
        if last_active:
            users = [last_active] + [u for u in users if u["id"] != last_active["id"]]

        choices = [
            questionary.Choice(
                title=u["name"]
                + (
                    " (last active)"
                    if last_active and u["id"] == last_active["id"]
                    else ""
                ),
                value=u,
            )
            for u in users
        ]
        selected = await questionary.select(
            f"{USER} Multiple users — who are you?",
            choices=choices,
        ).ask_async()

        if selected is None:
            raise SystemExit("Cancelled.")

        return selected["id"], selected["name"]

    def _init_components(self) -> None:
        """Initialize heavy components (downloads models if needed)."""
        print(ui.dim(f"{WRENCH} Preparando modelos y audio..."))

        # STT: faster-whisper grande + fallback whisper.cpp
        try:
            from openocto.stt.factory import create_stt_engine

            self._stt = create_stt_engine(self._config.stt)
        except Exception as exc:
            logger.exception("STT no disponible")
            print(ui.warn(f"  STT no disponible: {exc}"))
            print(ui.dim("   pip install -e .[audio]\n"))

        # Persona
        self._persona = self._persona_manager.activate(self._config.persona)

        # TTS — pre-load models for all configured languages
        from openocto.tts.factory import create_tts_engine

        for lang, model_name in self._config.tts.models.items():
            try:
                self._tts_engines[lang] = create_tts_engine(
                    lang, model_name, self._config.tts
                )
                engine_name = self._config.tts.engines.get(
                    lang, self._config.tts.engine
                )
                logger.info(
                    "TTS loaded for lang=%s: engine=%s, piper_model=%s",
                    lang,
                    engine_name,
                    model_name,
                )
            except Exception as e:
                logger.warning("Failed to load TTS for lang=%s: %s", lang, e)

        # AI Router
        from openocto.ai.router import AIRouter

        self._ai_router = AIRouter(self._config.ai)

        # Skills (LLM tool-use)
        try:
            self._skills = build_default_registry(self._config.skills)
            if len(self._skills) == 0:
                logger.info("No skills enabled")
                self._skills = None
            else:
                logger.info(
                    "Loaded %d skills: %s", len(self._skills), self._skills.names()
                )
        except Exception:
            logger.exception(
                "Failed to build skill registry; continuing without skills"
            )
            self._skills = None

        # Memory system
        if self._config.memory.enabled:
            self._memory = MemoryManager(self._history_store, self._config.memory)

        print(ui.ok(f"{CHECK} Lista.\n"))

    @staticmethod
    def _looks_like_error_response(text: str) -> bool:
        """Detect AI responses that are actually backend error payloads.

        Some proxies (notably claude-max-proxy when its session expires)
        return upstream errors as plain HTTP 200 chat completions instead
        of raising — so the text reaches us as if it were the model's
        answer.  We refuse to TTS these so the user doesn't hear an
        authentication error read aloud in Russian.
        """
        if not text:
            return False
        head = text.lstrip()[:200].lower()
        markers = (
            "failed to authenticate",
            "api error",
            "authentication_error",
            '"type":"error"',
            '"type": "error"',
            "unauthorized",
            "401 ",
            "403 ",
        )
        return any(m in head for m in markers)

    @staticmethod
    def _friendly_error(e: Exception) -> str:
        """Convert exceptions to user-friendly messages."""
        error_str = str(e).lower()
        backend_hint = ""

        # Connection errors (proxy not running, network issues)
        if "connection" in error_str or "connect" in error_str:
            backend_hint = (
                "AI backend is not reachable.\n"
                "   Possible causes:\n"
                "   - No internet connection\n"
                "   - API endpoint is down\n"
                "   Fix the issue and try again. Check the setup with: python assistant.py check"
            )
        # Auth errors
        elif (
            "auth" in error_str
            or "api key" in error_str
            or "401" in error_str
            or "403" in error_str
        ):
            backend_hint = (
                "AI backend rejected your credentials.\n"
                "   Check your API key, then run: python assistant.py key && python assistant.py check"
            )
        # Rate limits
        elif "rate" in error_str or "429" in error_str or "quota" in error_str:
            backend_hint = "AI backend rate limit reached. Wait a moment and try again."
        # Timeout
        elif "timeout" in error_str or "timed out" in error_str:
            backend_hint = "AI backend timed out. Check your connection and try again."

        if backend_hint:
            return backend_hint
        return f"Error: {e}"

    def _detect_response_lang(self, text: str, stt_lang: str | None = None) -> str:
        """Detect language of AI response text by script.

        SPIKE PATCH 4: el upstream solo distingue cirilico (ru) del resto (en),
        lo que haria hablar el espanol con voz inglesa. Cuando el texto no es
        cirilico se usa el idioma que detecto Whisper (result.language).
        """
        if not text:
            return stt_lang or "en"
        cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
        if cyrillic / len(text) > 0.2:
            return "ru"
        return stt_lang or "en"

    @staticmethod
    def _split_first_sentence(buf: str) -> tuple[str, str] | None:
        """P8: primera frase "cerrada" de buf -> (frase, resto) o None.

        Corta en . ! ? solo si van seguidos de espacio o fin de cadena
        ("1.500 pesos" NO corta; "Hola. ¿Qué tal?" si). Como fallback, corta
        en un salto de linea si hay al menos 20 caracteres antes.
        """
        for i in range(len(buf)):
            if buf[i] in ".!?" and (i + 1 == len(buf) or buf[i + 1] in " \n"):
                return buf[: i + 1].strip(), buf[i + 1 :].strip()
        nl = buf.find("\n")
        if nl >= 20:
            return buf[:nl].strip(), buf[nl + 1 :].strip()
        return None

    def _start_tts_pipeline(self) -> None:
        """P21: sintetiza la frase N+1 mientras suena la N.

        Antes se hacia `synthesize` + `play` en serie dentro de on_chunk: cada
        punto costaba ~1,3 s de silencio (lo que tarda Edge en la nueva frase).
        Ahora un worker sintetiza y otro reproduce; las colas conservan el orden.
        """
        if self._tts_text_q is not None:
            self._abort_tts()
        self._tts_suppress = False
        self._tts_text_q = asyncio.Queue()
        self._tts_audio_q = asyncio.Queue()
        self._tts_synth_task = asyncio.create_task(self._tts_synth_worker())
        self._tts_play_task = asyncio.create_task(self._tts_play_worker())

    def _enqueue_text(self, text: str, result) -> None:
        text = text.strip()
        if not text or self._tts_suppress or self._tts_text_q is None:
            return
        tts = self._get_tts(
            self._detect_response_lang(text, getattr(result, "language", None))
        )
        if tts is None:
            logger.warning(
                "No TTS engine for language: %s", getattr(result, "language", None)
            )
            return
        self._tts_text_q.put_nowait((text, tts))

    def _enqueue_closed_sentences(self, result) -> None:
        while True:
            cut = self._split_first_sentence(self._tts_buf)
            if not cut:
                return
            frase, self._tts_buf = cut
            if frase:
                self._enqueue_text(frase, result)

    async def _tts_synth_worker(self) -> None:
        text_q = self._tts_text_q
        audio_q = self._tts_audio_q
        if text_q is None or audio_q is None:
            return
        while True:
            item = await text_q.get()
            if item is None:
                await audio_q.put(None)
                return
            frase, tts = item
            if self._tts_suppress:
                continue
            try:
                audio = await asyncio.to_thread(tts.synthesize, frase)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "TTS: fallo al sintetizar una frase; sigo", exc_info=True
                )
                continue
            if self._tts_suppress:
                continue
            await audio_q.put((audio, len(frase)))

    async def _tts_play_worker(self) -> None:
        audio_q = self._tts_audio_q
        if audio_q is None:
            return
        while True:
            item = await audio_q.get()
            if item is None:
                return
            audio, _ = item
            if self._tts_suppress:
                continue
            self._orb.state("speaking")
            try:
                await self._player.play_async_awaitable(
                    audio.audio,
                    audio.sample_rate,
                    gain=self._config.tts.gain,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "TTS: fallo al reproducir una frase; sigo", exc_info=True
                )

    async def _finish_tts(self) -> None:
        """Cierra la cola y espera a que se reproduzca todo lo pendiente."""
        text_q = self._tts_text_q
        if text_q is None:
            return
        await text_q.put(None)
        for task in (self._tts_synth_task, self._tts_play_task):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("TTS worker error", exc_info=True)
        self._tts_text_q = None
        self._tts_audio_q = None
        self._tts_synth_task = None
        self._tts_play_task = None
        self._tts_buf = ""

    def _abort_tts(self) -> None:
        """Corta sintesis/reproduccion pendientes (barge-in o error)."""
        self._tts_suppress = True
        if self._tts_text_q is not None:
            while True:
                try:
                    self._tts_text_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        for task in (self._tts_synth_task, self._tts_play_task):
            if task is not None and not task.done():
                task.cancel()
        self._tts_text_q = None
        self._tts_audio_q = None
        self._tts_synth_task = None
        self._tts_play_task = None

    def _get_tts(self, language: str):
        """Get TTS engine for the given language code."""
        if language in self._tts_engines:
            return self._tts_engines[language]
        prefix = language.split("_")[0]
        if prefix in self._tts_engines:
            return self._tts_engines[prefix]
        return next(iter(self._tts_engines.values())) if self._tts_engines else None

    # --- PTT mode ---

    def _on_ptt_press(self) -> None:
        """Called from keyboard thread when PTT key is pressed."""
        if self._player.is_playing:
            self._player.stop()
            self._abort_tts()
            if self._wakeword_enabled:
                self._capture.stop_recording()
            else:
                self._capture.stop()
            if self._conversation_active:
                self._conversation_active = False
        state = self._state_machine.state
        if state == State.RECORDING and self._ptt_mode == "vad":
            if self._conversation_active and self._auto_turn:
                # En conversacion, pulsar mientras espera tu voz la termina.
                self._conversation_active = False
                asyncio.ensure_future(self._stop_recording(skip_stt=True))
            else:
                # Primera toma: pulsar de nuevo la cierra y procesa ya.
                asyncio.ensure_future(self._stop_recording())
            return
        if state == State.IDLE and not self._processing:
            # cada conversación nueva vuelve a preguntar solo si sigue sin nombre
            self._name_asked = False
            if self._wakeword is not None:
                self._wakeword.pause()
            self._conversation_active = bool(
                self._config.audio.continuous and self._ptt_mode == "vad"
            )
            asyncio.ensure_future(self._start_recording())

    def _on_ptt_release(self) -> None:
        """Called from keyboard thread when PTT key is released."""
        if self._ptt_mode == "vad":
            return  # en modo VAD soltar no corta: se corta al callar
        if self._state_machine.state == State.RECORDING:
            asyncio.ensure_future(self._stop_recording())

    def _start_wakeword(self) -> None:
        """Carga Vosk, abre el stream fijo y empieza a escuchar la frase."""
        cfg = self._config.wakeword
        if not cfg.enabled:
            return
        try:
            from openocto.utils.model_downloader import get_vosk_model
            from openocto.wakeword.vosk_detector import (
                VoskWakeWord,
                WakeWordController,
            )

            modelo = get_vosk_model(cfg.model)
            detector = VoskWakeWord(cfg.phrase, modelo, aliases=cfg.aliases)
            self._wakeword = WakeWordController(
                detector,
                lambda: self._loop.call_soon_threadsafe(self._on_wake_word),
            )
            self._wakeword.start()
            self._capture.set_chunk_callback(self._wakeword.feed)
            self._capture.start_stream()
            self._wakeword_guard_task = asyncio.create_task(self._wakeword_guard())
            self._wakeword_enabled = True
            logger.info("Wake word listo: %r (%s)", cfg.phrase, modelo.name)
        except Exception as exc:
            logger.warning("Wake word no disponible (%s); queda PTT", exc)
            print(ui.warn(f"  Wake word no disponible: {exc}"))
            self._wakeword_enabled = False
            self._capture.set_chunk_callback(None)
            if self._wakeword is not None:
                self._wakeword.stop()
                self._wakeword = None
            self._capture.stop()

    def _on_wake_word(self) -> None:
        """Callback en el loop cuando Vosk detecta la frase."""
        if not self._wakeword_enabled or self._wakeword is None:
            return
        if self._state_machine.state != State.IDLE or self._processing:
            return
        self._wakeword.pause()
        self._name_asked = False
        self._conversation_active = bool(
            self._config.audio.continuous and self._ptt_mode == "vad"
        )
        logger.info(
            "Wake word detectada: %r (oído %r)",
            self._config.wakeword.phrase,
            self._wakeword.last_text,
        )
        asyncio.ensure_future(self._start_recording())

    async def _wakeword_guard(self) -> None:
        """Pausa Vosk en grabación/LLM/TTS y lo reanuda al quedar en reposo."""
        try:
            while True:
                await asyncio.sleep(0.15)
                if self._wakeword is None:
                    continue
                libre = (
                    self._state_machine.state == State.IDLE
                    and not self._processing
                    and not self._conversation_active
                    and not self._player.is_playing
                )
                if libre:
                    self._wakeword.resume()
                else:
                    self._wakeword.pause()
        except asyncio.CancelledError:
            pass

    async def _start_recording(self, auto: bool = False) -> None:
        if self._state_machine.state != State.IDLE:
            return
        self._auto_turn = auto
        await self._state_machine.transition("start_recording")
        if auto:
            self._orb.state("listening")
        else:
            self._orb.show()
        # El stream de algunos micros (Bluetooth/MME) tarda ~0.7 s en
        # entregar el primer chunk. Si damos el chime antes, la persona habla
        # y el arranque se come la primera sílaba. Calentamos primero.
        self._capture.start_stream()
        print(ui.dim(f"{MIC} Preparando micrófono..."), end="\r", flush=True)
        if not await self._capture.wait_ready(timeout=2.0):
            logger.warning("El micrófono no entregó audio a tiempo")
            # Puede ser el dispositivo anterior (audífonos recién
            # desconectados): reabro en el predeterminado actual y reintento.
            self._capture.reopen_now()
            if not await self._capture.wait_ready(timeout=2.0):
                logger.warning("Sigue sin audio tras reabrir el micrófono")
        if self._state_machine.state != State.RECORDING:
            self._capture.stop()
            return
        self._player.chime(ascending=True)  # mic abierto
        await asyncio.sleep(0.18)  # el chime no se graba
        if self._state_machine.state != State.RECORDING:
            self._capture.stop()
            return
        self._capture.start_recording()
        if self._ptt_mode == "vad":
            print(
                ui.dim(f"{MIC} Escuchando... (habla; se corta al callar)"),
                end="\r",
                flush=True,
            )
            self._vad_task = asyncio.create_task(self._vad_watch())
        else:
            print(
                ui.dim(f"{MIC} Recording... (suelta {self._ptt_label})"),
                end="\r",
                flush=True,
            )

    async def _vad_watch(self) -> None:
        """Corta la toma cuando Silero detecta silencio tras hablar."""
        if self._vad is None:
            return
        if self._state_machine.state != State.RECORDING:
            return

        self._vad.reset()
        last_idx = self._capture.latest_chunk_index()
        t0 = time.monotonic()
        speech_detected = False
        last_ui = 0.0
        try:
            while self._state_machine.state == State.RECORDING:
                now = time.monotonic()
                elapsed = now - t0

                if elapsed >= self._config.vad.max_recording_duration:
                    logger.debug("VAD: tope de %.0fs alcanzado", elapsed)
                    break

                if (
                    not speech_detected
                    and elapsed >= self._config.vad.no_speech_timeout
                ):
                    logger.debug("VAD: sin voz en %.1fs; cancelo", elapsed)
                    await self._stop_recording(skip_stt=True)
                    return

                item = self._capture.get_latest_chunk(after=last_idx)
                if item is None:
                    await asyncio.sleep(0.04)
                    continue
                chunk, last_idx = item

                speech = self._vad.is_speech(chunk)
                self._orb.level(self._vad.last_rms)
                if speech and not speech_detected:
                    self._orb.state("hearing")
                if speech:
                    speech_detected = True

                if speech_detected and self._vad.should_stop_recording(
                    chunk, speech=speech
                ):
                    logger.debug("VAD: silencio tras voz (%.1fs)", elapsed)
                    break

                if now - last_ui >= 0.2:
                    last_ui = now
                    print(
                        ui.dim(
                            f"{MIC} {'Grabando' if speech_detected else 'Escuchando'}... "
                            f"{elapsed:4.1f}s "
                            f"(p={getattr(self._vad, 'last_prob', 0.0):.2f})"
                        ),
                        end="\r",
                        flush=True,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("VAD loop error")
        finally:
            if self._vad is not None:
                self._vad.reset()

        if self._state_machine.state == State.RECORDING:
            try:
                await self._stop_recording()
            except Exception:
                logger.exception("VAD: fallo al cerrar la toma")
                self._capture.stop()
                self._state_machine.reset()

    async def _auto_listen(self) -> None:
        """Reabre el micro tras una respuesta en modo conversacion."""
        if not self._config.audio.continuous or self._ptt_mode != "vad":
            return
        guard = max(0.1, float(self._config.audio.continuous_tail_guard))
        await asyncio.sleep(guard)
        if not self._conversation_active:
            self._orb.end()
            return
        if self._state_machine.state != State.IDLE or self._processing:
            return
        if self._player.is_playing:
            return
        await self._start_recording(auto=True)

    def _schedule_auto_listen(self) -> None:
        if not self._config.audio.continuous or not self._conversation_active:
            return
        if self._auto_listen_task and not self._auto_listen_task.done():
            return
        self._auto_listen_task = asyncio.create_task(self._auto_listen())

    async def _stop_recording(self, skip_stt: bool = False) -> None:
        if self._state_machine.state != State.RECORDING:
            return
        was_auto = self._auto_turn
        self._auto_turn = False
        task = self._vad_task
        self._vad_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

        # En conversación continua dejamos el stream caliente entre turnos
        # (sin grabar) para no pagar el arranque del Bluetooth cada frase.
        keep_warm = self._wakeword_enabled or (
            self._ptt_mode == "vad" and self._conversation_active and not skip_stt
        )
        if keep_warm:
            self._capture.stop_recording()
        else:
            self._capture.stop()
        self._player.chime(ascending=False)  # mic cerrado
        audio = self._capture.get_recording()
        ui.clear_line()

        if skip_stt:
            # Cualquier cierre sin transcripcion termina la conversacion. Si
            # no se limpia, el wake word queda pausado para siempre.
            self._conversation_active = False
            if was_auto and self._config.audio.continuous:
                print(ui.warn("  Conversación terminada."))
            else:
                print(ui.warn("  No te escuché."))
            await self._state_machine.transition("cancel")
            self._orb.end()
            return

        if audio.size == 0:
            self._conversation_active = False
            print(ui.warn("  No se capturo audio."))
            await self._state_machine.transition("cancel")
            self._orb.end()
            return

        if self._config.audio.normalize:
            audio = normalize_for_stt(
                audio,
                target_rms=self._config.audio.normalize_target_rms,
                max_gain=self._config.audio.normalize_max_gain,
            )

        await self._handle_audio(audio)

    def _fecha_block(self) -> str:
        """Fecha/hora reales para el system prompt (el modelo no las sabe)."""
        return (
            "\n\n## Fecha y hora actuales\n"
            f"Hoy es {fecha_es()}, hora local {hora_es()} (zona de Bogotá). "
            "Usa esto para 'hoy', 'ayer' o 'mañana'; no lo busques en la web."
        )

    async def _handle_audio(self, audio) -> None:
        """Run the full pipeline: transcribe → AI → TTS."""
        self._processing = True
        spinner = None
        try:
            # Transcribe
            await self._state_machine.transition("stop_recording")
            self._orb.state("transcribing")
            pending = self._pending_text
            self._pending_text = None
            if pending is not None:
                # Turno inyectado por texto (saludo de arranque): no hubo audio.
                result = pending
            elif self._stt is None:
                self._conversation_active = False
                print(ui.warn("  STT no disponible: pip install -e .[audio]"))
                await self._state_machine.transition("cancel")
                self._orb.end()
                return
            else:
                print(ui.dim(f"{WRENCH} Transcribiendo..."), end="\r", flush=True)
                result = await asyncio.to_thread(self._stt.transcribe, audio)
                ui.clear_line()

            if not result.text.strip():
                self._conversation_active = False
                print(ui.warn("  No pude transcribir el audio."))
                await self._state_machine.transition("cancel")
                self._orb.end()
                return

            # P22: dos capturas/instancias pueden entregar la misma frase casi
            # seguida; no la procesamos dos veces.
            normalized = " ".join(result.text.casefold().split())
            now = time.monotonic()
            similar = bool(
                normalized
                and self._last_user_text
                and difflib.SequenceMatcher(
                    None, normalized, self._last_user_text
                ).ratio()
                >= 0.90
            )
            if similar and now - self._last_user_at < 6.0:
                logger.info("STT duplicado ignorado: %s", result.text[:120])
                print(ui.warn("  (repetición inmediata ignorada)"))
                await self._state_machine.transition("cancel")
                self._orb.end()
                return
            self._last_user_text = normalized
            self._last_user_at = now

            print(ui.user(result.text))
            await self._state_machine.transition("transcription_done")
            self._orb.state("thinking")

            # Persist user message & build context
            persona_name = self._persona.name
            uid = self._current_user_id
            self._captura_nombre(uid, persona_name, result.text)

            # Build context (memory-enriched system prompt + recent history).
            # IMPORTANTE: construir el contexto ANTES de guardar el mensaje
            # actual. Si se guarda primero, el historial ya lo trae y
            # AIRouter.send_streaming lo vuelve a añadir: el modelo recibía
            # el mismo mensaje dos veces ("me estás escuchando doble").
            if self._memory:
                system_prompt, history = self._memory.build_context(
                    uid,
                    self._persona,
                    result.text,
                )
            else:
                system_prompt = self._persona.system_prompt
                history = self._history_store.get_recent_messages(
                    uid,
                    persona_name,
                    limit=self._config.ai.max_history,
                )

            self._history_store.add_message(
                uid,
                persona_name,
                "user",
                result.text,
                language=result.language,
            )

            # La IA no sabe qué día es: sin esto, "hoy" es alucinación o una
            # búsqueda web innecesaria (MiniMax respondió "24 de marzo").
            system_prompt += self._fecha_block()

            # P23: al inicio, si no hay nombre, pedirlo en voz alta una vez.
            if self._name_unknown and not self._name_asked:
                system_prompt += (
                    "\n\n## Nombre pendiente\n"
                    "Todavía no sabes el nombre de la persona. En esta respuesta, "
                    "pregúntale cómo se llama, una sola vez y con naturalidad; "
                    "responde también lo que te haya pedido. No inventes un nombre."
                )
                self._name_asked = True

            # AI response
            got_first_chunk = False
            self._tts_buf = ""  # P8 (A)
            self._tts_suppress = False  # P8 (A): el turno anterior pudo dejarlo True
            self._start_tts_pipeline()  # P21: sintesis y reproduccion en paralelo

            async def _spin() -> None:
                i = 0
                while True:
                    print(
                        "\r"
                        + ui.dim(f"{self._persona.display_name}: {'|/-\\'[i % 4]}"),
                        end="",
                        flush=True,
                    )
                    i += 1
                    await asyncio.sleep(0.1)

            spinner = asyncio.create_task(_spin())

            async def on_chunk(chunk: str) -> None:
                nonlocal got_first_chunk
                if not got_first_chunk:
                    got_first_chunk = True
                    spinner.cancel()
                    ui.clear_line()
                    print(ui.ai_label(self._persona.display_name), end="")
                print(chunk, end="", flush=True)
                # P21: encolar cada frase cerrada; la sintesis y el playback
                # corren en segundo plano, sin bloquear el stream del LLM.
                self._tts_buf += chunk
                self._enqueue_closed_sentences(result)

            # Bind per-request context for skills that need user state.
            if self._skills is not None:
                self._skills.bind_context(
                    history=self._history_store,
                    user_id=uid,
                    persona=persona_name,
                    player=self._player,
                    loop=self._loop,
                )

            response = await self._ai_router.send_streaming(
                result.text,
                history,
                system_prompt,
                on_chunk,
                skills=self._skills,
            )
            if not spinner.done():
                spinner.cancel()
            if not response.strip():
                # MiniMax y otros modelos de razonamiento pueden gastar todo
                # max_tokens pensando y terminar con finish=length y cero texto.
                logger.warning(
                    "Respuesta vacia del backend %s; uso aviso hablado",
                    self._ai_router.active_backend_name,
                )
                print(
                    ui.warn(
                        "  El modelo no devolvio texto "
                        "(posible limite de razonamiento)."
                    )
                )
                response = (
                    "Me quedé en blanco un momento. ¿Me repites la pregunta, por favor?"
                )
                self._enqueue_text(response, result)
            if not got_first_chunk:
                ui.clear_line()
                print(ui.ai_label(self._persona.display_name), end="")
            print()  # newline after streamed response

            # Persist assistant response
            self._history_store.add_message(
                uid,
                persona_name,
                "assistant",
                response,
                backend=self._ai_router.active_backend_name,
            )

            # Background memory processing (summarization, fact extraction)
            if self._memory:
                asyncio.create_task(
                    self._memory.maybe_process(uid, self._persona, self._ai_router)
                )

            await self._state_machine.transition("response_ready")

            # If the "response" is actually an upstream backend error
            # (e.g. claude-proxy returning a 401 body as chat content),
            # show it on screen but don't read it aloud — and treat the
            # turn as failed so wake-word stays disarmed for a moment.
            if self._looks_like_error_response(response):
                logger.warning(
                    "Suppressing TTS for error-shaped response: %s", response[:200]
                )
                print(ui.warn("  El backend devolvio un error: no lo leo en voz alta."))
                self._tts_suppress = True  # P8 (A): corta el hablado restante
                self._player.stop()
                self._abort_tts()
                self._conversation_active = False
                await self._state_machine.transition("speech_done")
                self._orb.end()
                return

            # P21: las frases cerradas ya estan en la cola del worker.
            # Aqui solo se encola el resto sin cerrar y se espera el playback.
            if self._tts_buf.strip() and not self._tts_suppress:
                self._enqueue_text(self._tts_buf, result)
            self._tts_buf = ""

            await self._finish_tts()
            await self._state_machine.transition("speech_done")
            if self._conversation_active:
                self._orb.state("waiting")
                self._schedule_auto_listen()
            else:
                self._orb.end()

        except Exception as e:
            self._abort_tts()
            self._player.stop()
            self._conversation_active = False
            if self._wakeword_enabled:
                self._capture.stop_recording()
            else:
                self._capture.stop()
            if spinner and not spinner.done():
                spinner.cancel()
                ui.clear_line()
            logger.debug("Pipeline error", exc_info=True)
            error_msg = self._friendly_error(e)
            print(ui.err(f"\n{CROSS} {error_msg}"))
            print(ui.dim(f"   Reintenta con {self._ptt_label}.\n"))
            self._state_machine.reset()
            self._orb.end(error=True)
            return  # no auto-listen tras error: espera otra pulsacion
        finally:
            self._processing = False

    async def _inject_text(self, text: str) -> None:
        """Turno sin micro: lleva un texto al modelo como si fuera una
        transcripcion del usuario (saludo de arranque)."""
        if self._processing or self._state_machine.state != State.IDLE:
            return
        if not text.strip():
            return

        # Como al despertar: pausa Vosk y abre la conversacion continua.
        if self._wakeword is not None:
            self._wakeword.pause()
        self._name_asked = False
        self._conversation_active = bool(
            self._config.audio.continuous and self._ptt_mode == "vad"
        )

        from openocto.stt.base import TranscriptionResult

        lang = self._config.stt.language
        if not lang or lang == "auto":
            lang = "es"
        self._pending_text = TranscriptionResult(
            text=text, language=lang, confidence=1.0, duration_ms=0.0
        )
        # _handle_audio consume _pending_text en lugar de transcribir.
        # El orbe solo aparece con show(); este camino no pasa por
        # _start_recording, así que lo mostramos aquí (igual que el primer
        # turno normal) para que el transcribing/thinking/speaking se vean.
        self._orb.show()
        await self._state_machine.transition("start_recording")
        await self._handle_audio(np.zeros(0, dtype=np.int16))

    async def _auto_greet(self) -> None:
        """Tras el arranque, saluda: inyecta la frase al modelo para que
        Samantha responda y la conversacion quede abierta."""
        cfg = self._config.wakeword
        try:
            await asyncio.sleep(max(0.0, float(cfg.greet_delay)))
        except asyncio.CancelledError:
            return
        text = (cfg.greet_text or cfg.phrase or "Hola Samantha").strip()
        logger.info("Saludo de arranque: %r", text)
        await self._inject_text(text)

    def _cleanup(self) -> None:
        """Nada de larga vida que detener en el modo terminal."""
        self._abort_tts()
        if self._greet_task is not None and not self._greet_task.done():
            self._greet_task.cancel()
        self._greet_task = None
        if self._wakeword_guard_task is not None:
            self._wakeword_guard_task.cancel()
            self._wakeword_guard_task = None
        if self._wakeword is not None:
            self._wakeword.stop()
            self._wakeword = None
        self._capture.set_chunk_callback(None)
        self._capture.stop()
        if self._vad_task is not None:
            self._vad_task.cancel()
            self._vad_task = None
        if self._auto_listen_task is not None:
            self._auto_listen_task.cancel()
            self._auto_listen_task = None
        self._orb.close()
        logger.debug("cleanup")

    def _show_orb_at_start(self) -> None:
        """Muestra el orbe en gris antes de cargar modelos.

        `start()` lanza la ventana en su hilo (empieza oculta); `state()` solo
        cambia el estado y `show()` es lo único que la hace visible. El orden
        importa: si se llama a `state("waiting")` después de `show()`, la
        ventana puede quedar un instante con el estilo por defecto.
        """
        if not self._config.ui.orb:
            return
        self._orb.start()
        self._orb.state("waiting")
        self._orb.show()

    async def run(self) -> None:
        """Main application loop."""
        self._loop = asyncio.get_running_loop()

        # El orbe aparece desde el primer instante, en gris ("waiting").
        # _init_components (whisper, Piper) tarda decenas de segundos y antes
        # no había ninguna señal visual hasta que terminaba.
        self._show_orb_at_start()

        # Ensure cleanup on SIGTERM/SIGHUP (terminal close)
        import signal

        # SPIKE PATCH 5: fallback para Windows (ver comentario arriba).
        try:
            for sig in (signal.SIGTERM, signal.SIGHUP):
                self._loop.add_signal_handler(sig, self._cleanup)
        except (AttributeError, NotImplementedError):
            for _name in ("SIGBREAK", "SIGTERM"):
                _sig = getattr(signal, _name, None)
                if _sig is not None:
                    try:
                        signal.signal(_sig, lambda *_: self._cleanup())
                    except (OSError, ValueError):
                        pass

        uid, uname = await self._resolve_user()
        self._current_user_id = uid
        self._name_unknown = not self._nombre_conocido(uid)
        logger.info(
            "Active user: %s (id=%d) | nombre conocido: %s",
            uname,
            uid,
            not self._name_unknown,
        )
        self._init_components()

        # Health check: verify AI backend responds before starting
        print(
            ui.dim(
                f"{WRENCH} Comprobando el backend de IA "
                f"({self._ai_router.active_backend_name})..."
            )
        )
        ok, msg = await self._ai_router.health_check()
        if ok:
            print(
                ui.ok(f"{CHECK} Conexión OK ({self._ai_router.active_backend_name})\n")
            )
        else:
            print(ui.warn(f"\n  AI backend no responde: {msg}"))
            bajo = str(msg).lower()
            if "reasoning effort" in bajo or "adaptive thinking" in bajo:
                print(
                    ui.warn(
                        "   El modelo rechazó reasoning_effort: usa 'low' "
                        "o comenta esa línea."
                    )
                )
            elif "invalid params" in bajo or "bad_request" in bajo or "400" in bajo:
                print(
                    ui.warn(
                        "   El proveedor rechazó parámetros del modelo "
                        "(reasoning_effort, max_tokens o modelo)."
                    )
                )
            elif "401" in bajo or "api key" in bajo or "unauthorized" in bajo:
                print(ui.warn("   Revisa la API key."))
            elif "timeout" in bajo or "connect" in bajo or "network" in bajo:
                print(ui.warn("   Revisa la red y el tiempo de espera."))
            else:
                print(
                    ui.warn("   Revisa la configuración del modelo, la key y la red.")
                )
            print()

        header = (
            f"{self._persona.display_name} {__version__} · "
            f"{self._ai_router.active_backend_name}"
        )

        await self._run_ptt_mode(header)

    async def _run_ptt_mode(self, header: str) -> None:
        """Push-to-talk global (combinación en audio.ptt_hotkey)."""
        try:
            from openocto.utils.keyboard import AsyncPushToTalkListener
        except ImportError as e:
            print(ui.warn(f"\n  Push-to-talk no disponible: {e}"))
            print(ui.dim("   Instala las dependencias de audio (pynput)."))
            return
        listener = AsyncPushToTalkListener(
            on_press=self._on_ptt_press,
            on_release=self._on_ptt_release,
            mode=self._ptt_mode,
            combo=self._config.audio.ptt_hotkey,
        )
        print(ui.header(header))
        tts_cfg = self._config.tts
        engine = tts_cfg.engines.get("es", tts_cfg.engine)
        if engine == "edge":
            voice = tts_cfg.edge.voices.get("es", "-")
            rate = f" {tts_cfg.edge.rate}"
            if tts_cfg.edge.trim_silence:
                rate += " · recorte"
        else:
            voice = tts_cfg.models.get("es", engine)
            rate = ""
        mode = "VAD" if self._ptt_mode == "vad" else "PTT"
        conv = (
            " · conversación continua"
            if self._ptt_mode == "vad" and self._config.audio.continuous
            else ""
        )
        # El oído real lo describe el motor cargado (large GPU/CPU o base),
        # no la config estática: si CUDA falló, aquí se ve "CPU int8".
        _oido = getattr(self._stt, "describe", None)
        oido = _oido() if callable(_oido) else f"whisper {self._config.stt.model_size}"
        print(
            ui.dim(
                f"   voz {engine} {voice}{rate} · oído {oido} · {mode} "
                f"{self._config.vad.silence_duration:.1f}s{conv}"
            )
        )
        if self._ptt_mode == "vad":
            if self._config.audio.continuous:
                hint = (
                    f"Pulsa [{self._ptt_label}] y conversa: habla y se corta al callar; "
                    "tras responder sigue escuchando. Otra pulsación termina."
                )
            else:
                hint = f"Pulsa [{self._ptt_label}] y habla: se corta al callar."
        else:
            hint = f"Mantén [{self._ptt_label}] mientras hablas; al soltar responde."
        if self._config.wakeword.enabled:
            print(
                ui.dim(
                    f'   Di "{self._config.wakeword.phrase}" para despertar, '
                    f"o usa [{self._ptt_label}]."
                )
            )
        print(ui.dim(f"   {hint} · [Ctrl+C] sale\n"))
        self._start_wakeword()
        listener.start()
        if self._config.wakeword.greet_on_start:
            self._greet_task = asyncio.create_task(self._auto_greet())
        try:
            while True:
                await asyncio.sleep(0.1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            listener.stop()
            self._capture.stop()
            self._player.stop()
            self._cleanup()
            print(ui.dim("\n Hasta luego."))
