# Whisper grande local con fallback — Change Spec

> **For implementer subagents:** REQUIRED SUB-SKILL — use
> `.opencode/skills/subagent-driven-development/SKILL.md` (recommended) or
> `.opencode/skills/executing-task-lists/SKILL.md` to implement task by task.
> Task steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Samantha transcribe con un Whisper grande local (`faster-whisper`
`large-v3-turbo`) y usa el motor actual `whisper.cpp base` como respaldo.

**Architecture:** `openocto/stt/factory.py` construye el motor primario y lo
envuelve con `FallbackSTTEngine`. `FasterWhisperEngine` carga
`large-v3-turbo` con CTranslate2; si CUDA falla, reintenta CPU int8. Si el
motor grande no carga o falla en un turno, se usa `WhisperCppEngine` con el
modelo `base` actual.

**Tech Stack:** Python 3.12, `faster-whisper 1.2.1`, `ctranslate2 4.8.5`,
`av` (FFmpeg embebido), `numpy`, `unittest`.

---

## Intent

Reducir errores de transcripción en español sin perder la posibilidad de
volver al motor liviano actual si el modelo grande no está disponible, no
cabe en el equipo o falla.

## Rationale

El `whisper.cpp base` es rápido (0.75 s en el i5) pero tiene errores en
nombres y audio difícil. `large-v3-turbo` es mucho más preciso y, en la RTX
4050, debería ser igual o más rápido. En Windows la GPU requiere CUDA 12 +
cuDNN 9 + cuBLAS; por eso el motor debe poder caer a CPU int8 y, en última
instancia, al `base`.

## Scope

- `openocto/stt/faster_whisper.py` — motor primario.
- `openocto/stt/fallback.py` — wrapper primario/fallback.
- `openocto/stt/factory.py` — selección por configuración.
- `openocto/app.py` — usar el factory en `_init_components`.
- `openocto/config.py` — `engine`, `fallback_engine`, `faster_whisper`.
- `assistant-config.yaml`, `config/default.yaml` — defaults.
- `pyproject.toml` — dependencia `faster-whisper>=1.2.1`.
- `build_portable.py` — hidden imports, collect-all, copia opcional del
  modelo y textos del LEEME.
- `tests/test_stt_faster.py` — pruebas con modelo falso.
- `README.md` — instalación, GPU, primer arranque.

## Constraints

- El motor actual `whisper.cpp base` debe seguir funcionando sin cambios.
- No exigir GPU: `device: "auto"` intenta CUDA y cae a CPU int8.
- El fallback se usa si el primario no carga, lanza una excepción o devuelve
  texto vacío.
- El modelo grande se descarga en `models/faster-whisper`; no se versiona.
- `faster-whisper` debe importarse de forma perezosa para que la app siga
  arrancando si la dependencia no está instalada.
- No cambiar VAD, captura, TTS, orbe ni pipeline de conversación.
- No descargar el modelo de 1,6 GB durante los tests.

## Config

```yaml
stt:
  engine: "faster_whisper"
  fallback_engine: "whisper_cpp"
  model_size: "base"
  language: "es"
  hotwords: ["Bogota", "Samantha"]
  n_threads: 8
  faster_whisper:
    model: "large-v3-turbo"
    device: "auto"
    compute_type: "default"
    beam_size: 5
    download_root: "models/faster-whisper"
```

- `device: "auto"` → intenta CUDA; si falla, CPU int8.
- `device: "cpu"` → CPU int8.
- `compute_type: "float16"` en GPU; `int8` en CPU.
- `beam_size: 1` acelera; `5` prioriza precisión.

## Acceptance Criteria

- AC1: `.venv\Scripts\python.exe -m py_compile openocto/stt/faster_whisper.py openocto/stt/factory.py openocto/stt/fallback.py openocto/app.py` exits 0.
- AC2: `.venv\Scripts\python.exe tests\test_stt_faster.py` passes (6 tests).
- AC3: `FasterWhisperEngine` pasa audio int16 a float32, usa `language`,
  `beam_size`, `initial_prompt` y `vad_filter=False`.
- AC4: si CUDA falla al cargar, se reintenta `device="cpu"` y
  `compute_type="int8"`.
- AC5: `FallbackSTTEngine` usa el fallback si el primario lanza o devuelve
  texto vacío.
- AC6: el factory usa `whisper.cpp` si `faster-whisper` no carga.
- AC7: con el modelo `tiny` real en CPU, un WAV de 8 s transcribe y devuelve
  texto (prueba manual ya realizada: 337 ms).
- AC8: `python build_portable.py` incluye `faster_whisper`, `ctranslate2`,
  `av`, `tokenizers`, `huggingface_hub`; el primer arranque descarga el
  modelo grande.

## Follow-up: fallback por latencia (2026-11-05)

**Intent:** si el motor grande responde bien pero tarda más de 4 s, no
sacrificar todos los turnos siguientes: ese turno se conserva y los próximos
usan `whisper.cpp base`.

**Implementación:**
- `STTConfig.fallback_after_ms: int = 4000` (`0` desactiva).
- `FallbackSTTEngine` mide `TranscriptionResult.duration_ms`; si supera el
  umbral, activa `_prefer_fallback` para los turnos siguientes.
- El turno lento se devuelve con el texto del motor grande; no se repite la
  transcripción con el base.
- `create_stt_engine()` pasa `config.fallback_after_ms`.
- `tests/test_stt_faster.py`: lento → el segundo turno usa base; rápido → no
  cambia; umbral 0 → no cambia.

**Acceptance Criteria adicionales:**
- AC9: `.venv\Scripts\python.exe tests\test_stt_faster.py` pasa (9 tests).
- AC10: con un primario de 5000 ms y umbral 4000, el primer turno devuelve
  el texto del primario y el segundo el del fallback.

## Follow-up 2: banner real + DLLs de CUDA (2026-11-05)

El log del portable mostró `large` cargado pero CUDA no operativo
(falta `cublas64_12.dll`) → CPU int8, mientras el banner seguía diciendo
"oído whisper base".

- `describe()` en `STTEngine` (base), `WhisperCppEngine`, `FasterWhisperEngine`
  y `FallbackSTTEngine`; el banner de `app.py` muestra el motor real,
  p. ej. `oído faster-whisper large-v3-turbo (CPU int8) -> whisper.cpp base`.
- `_register_cuda_dll_dirs()`: registra vía `os.add_dll_directory` las
  carpetas con cublas/cuDNN: config `cuda_dll_dirs`, `cuda\` junto a la app
  y paquetes pip `nvidia-*-cu12` del venv.
- `build_portable.py --gpu`: copia las DLLs `nvidia\**\*.dll` del venv a
  `cuda\` en el portable.
- AC11: `tests\test_stt_faster.py` cubre describe (CPU/GPU), cadena de
  fallback y registro de `cuda_dll_dirs` en Windows.

### corrección: ctranslate2 solo ve el PATH (mismo día)

Con los paquetes pip `nvidia-*-cu12` instalados, `os.add_dll_directory`
bastó para `ctranslate2.get_cuda_device_count()` pero NO para el
`LoadLibrary("cublas64_12.dll")` de `ctranslate2.dll` al crear el modelo
(usa la búsqueda estándar, que incluye PATH, no los registros de Python).
`_register_cuda_dll_dirs` ahora también **ante-pone las carpetas al
`os.environ["PATH"]`**.

Validado en vivo con el config real y el modelo cacheado (1.5 GB):

```text
CUDA: carpeta de DLLs registrada: ...nvidia\cublas\bin (+cuda_nvrtc, cudnn)
faster-whisper loaded: large-v3-turbo (device=cuda, compute=default)
DESCRIBE: faster-whisper large-v3-turbo (GPU default) -> whisper.cpp base
LOADED en 7.6 s | warmup 1 s de audio: 311.7 ms
```

AC12: `create_stt_engine(cfg.stt)` sobre el PC con RTX 4050 + las wheels
nvidia describe el motor como `(GPU default)`; el fallback base se carga
detrás y no interviene mientras el grande responda.

## Follow-up 3: saludo de arranque (2026-11-05)

**Intent:** al abrir el `.exe`, Samantha recibe "hola samantha" como si el
usuario la hubiera despertado, responde por voz sola y queda a la escucha
(conversación continua), sin decir la wake word ni pulsar el atajo.

**Implementación (sin duplicar el pipeline):**
- `WakeWordConfig.greet_on_start` (default false), `greet_text` (vacío =
  `phrase`) y `greet_delay` (1.5 s).
- `_inject_text(text)`: pausa Vosk, fija `_conversation_active`/`_name_asked`
  como `_on_wake_word`, mete un `TranscriptionResult` sintético en
  `_pending_text` y llama a `_handle_audio(np.zeros(0))`.
- `_handle_audio` consume `_pending_text` en vez de transcribir cuando existe
  (rama `pending` antes de la revisión de STT).
- `run()` agenda `_auto_greet()` tras `listener.start()` si la opción está
  activada; `_cleanup` cancela la tarea.
- `assistant-config.yaml` activa `greet_on_start: true`; `config/default.yaml`
  la documenta apagada.

**AC13:** `tests\test_greet.py` pasa: `_inject_text` alimenta el pipeline con
el texto y deja la conversación activa; pausa la wake word; es no-op si ya
hay un turno en curso; `_auto_greet` respeta el delay y prefiere `greet_text`
sobre la phrase; la config del asistente trae el saludo activado.

**Bug corregido (mismo día):** el saludo respondía sin orbe porque la ventana
solo se muestra con la acción `"show"` (`_start_recording`) y el camino
sintético no pasaba por ahí; `state()` cambia el estado pero no hace visible
la ventana. `_inject_text` ahora llama `self._orb.show()` antes del turno
**AC14:** `test_inject_text_feeds_pipeline_as_if_transcribed` verifica que el
show ocurre; el auto-listen posterior hereda el orbe visible vía `state`.
(El exe de `C:\SamanthaBuild` de las 14:30 NO trae el fix: sonda del PYZ
embebido → `_inject_text` sin llamada a `show()`; se analizó segundos antes
de aplicarlo. Pendiente recompilar para verlo en el `.exe`.)

## Follow-up 4: fecha real y validación de clave Firecrawl (tarde)

Hallazgo (log 14:32–14:43 y captura 15:07): con GPU y saludo ya funcionando,
el modelo **alucinaba la fecha** ("24 de marzo") porque nada se la inyectaba,
y las búsquedas fallaban: `secret\FIRECRAWL_API_KEY` medía 13 chars y no
empezaba por `fc-` (401 → "error de acceso").

- `openocto/utils/dates.py`: `fecha_es()`/`hora_es()` (español, sin locale).
- `OpenOctoApp._fecha_block()`: añade "## Fecha y hora actuales (Bogotá)" al
  system prompt en cada turno; regla de persona "## Fecha" en system_prompt.md.
- `web_search`: un 401 ahora cae al respaldo DuckDuckGo como el 402
  (`_duckduckgo(..., motivo=...)`); antes devolvía solo el error.
- `assistant.py`: `_fc_problema()` valida el prefijo `fc-` — `check` marca
  `MAL` + AVISO y el arranque avisa que se usará el respaldo.
- `build_portable.py --with-key` copia ahora también `FIRECRAWL_API_KEY`
  (antes solo la de OpenCode; el `secret\` viejo sobrevivía al swap).

**AC15:** `tests\test_dates.py` (4) verde; `test_web_search.py` actualizado
(`test_401_falls_back_to_duckduckgo`); suite completa 76 en verde.
**Pendiente usuario:** reponer la clave real (`assistant.py key firecrawl`,
de app.firecrawl.dev — la anterior posiblemente caducó/rotó) y recompilar.

## Non-goals

- No streaming parcial de Whisper.
- No diarización ni timestamps por palabra.
- No conversión de modelos ni fine-tuning.
- No instalar CUDA/cuDNN automáticamente.
- No cambiar el modelo `base` del fallback ni el pipeline de audio.
- No descargar el modelo grande en tests/CI.
