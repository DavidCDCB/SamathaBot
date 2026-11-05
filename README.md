# Samantha — asistente de voz personal (terminal)

Asistente local de voz con el personaje de **Samantha** (inspirado en la IA de
la película *Her*): cálida, curiosa y emocionalmente presente. **Solo terminal**:
sin interfaz web. Se activa con **Ctrl+Shift+Space**, diciendo
**“Hola Samantha”**, o sola al arrancar (saludo inicial).

Hablas, la toma se cierra al detectar silencio, Samantha responde por voz y
**vuelve a escuchar sola** para tu siguiente frase. Si callas
`vad.no_speech_timeout` segundos, la conversación termina; una segunda pulsación
del atajo también la termina.

```
Micrófono -> Silero VAD (corta al callar) -> conversación continua
  -> faster-whisper large-v3-turbo (GPU CUDA o CPU int8)
     -> fallback: whisper.cpp base
  -> memoria SQLite + fecha/hora real del turno
  -> OpenCode Go (streaming + tools: notas, web, Wikipedia, abrir búsquedas)
  -> Edge TTS frase a frase -> fallback Piper local -> altavoz
  -> reabre el micrófono (continuous)
```

## Novedades (2026-11)

- **STT grande local:** `large-v3-turbo` (~1,6 GB) con GPU CUDA real y caída
  automática a CPU `int8` y al `whisper.cpp base`.
- **Fallback por latencia:** si un turno con el modelo grande tarda más de 4 s,
  los siguientes usan `base` sin reiniciar.
- **Saludo de arranque:** al abrir el `.exe` Samantha te habla primero y deja
  la conversación abierta (incluido el orbe).
- **Fecha y hora reales** inyectadas en cada turno (sin búsquedas ni fechas
  alucinadas; nada quemado en el prompt).
- **Búsqueda robusta:** Firecrawl con validación de formato de la key y
  respaldo DuckDuckGo también ante `401` (key inválida), no solo `402`.
- **Portable:** `build_portable.py --gpu` incluye las DLLs de CUDA;
  `--with-key` copia OpenCode **y** Firecrawl.
- **Orbe rediseñado:** interior plano (sin brillos ni reflejos) con un símbolo
  de infinito crema centrado; se conservan colores por estado, halo y animación.

## Requisitos

| Requisito | Detalle |
|---|---|
| Windows 10/11 **x64** | Único sistema soportado por el `.exe` |
| Python 3.10–3.12 | Solo para desarrollo (el portable no lo necesita) |
| Internet | LLM OpenCode Go, Edge TTS y Firecrawl; sin red, Piper es local |
| GPU NVIDIA (opcional) | RTX 4050 probada; sin GPU, `large` corre en CPU int8 |
| ~8 GB RAM | Recomendado para el modelo grande en CPU |

## Instalación (desarrollo)

```powershell
cd E:\NewBackup2\BackupCDCB\Algoritmos\protoPython\Assistant\octo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[audio]"

python assistant.py key      # guarda la key de OpenCode Go (enmascarada)
python assistant.py check    # origen de la key + ping real al backend
python assistant.py start    # uso diario (sin comando también arranca)
```

`python` puede ser cualquiera: `assistant.py` se auto-relanza contra `.venv`.

### GPU NVIDIA sin instalar el CUDA Toolkit

```powershell
.\.venv\Scripts\python.exe -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

La app registra esas carpetas (más `cuda\` junto al `.exe` y
`stt.faster_whisper.cuda_dll_dirs`) en el `PATH` del proceso, que es lo que
`ctranslate2` necesita para cargar `cublas64_12.dll` / `cudnn64_9.dll`.
Verificación: el banner debe decir
`oído faster-whisper large-v3-turbo (GPU …)`. Si dice `CPU int8`, revisa
`logs\assistant.log` (la causa exacta aparece en el WARNING).

## Uso

```powershell
python assistant.py start [--debug]   # arranca (DEBUG en consola)
python assistant.py key [-y]          # OpenCode Go
python assistant.py key firecrawl     # Firecrawl (búsqueda web)
python assistant.py check             # keys + ping real + formato Firecrawl
python assistant.py report            # latencia por turno desde el log
python assistant.py say "Hola"        # probar la voz configurada
python assistant.py facts             # ver lo que Samantha aprendió
python assistant.py facts delete 3 4  # borrar hechos por id
python assistant.py orb               # prueba visual del orbe
```

Con el asistente corriendo: pulsa **Ctrl+Shift+Space** una vez y conversa.
Habla, espera la respuesta y sigue hablando sin volver a pulsar: tras responder,
el micrófono se reabre solo (chime al abrir y al cerrar cada toma).
`Ctrl+C` para salir.

- El atajo es **global** (funciona en segundo plano) y se cambia en
  `audio.ptt_hotkey`.
- El **orbe** aparece al activar la conversación y se puede arrastrar; muestra
  esperando/escuchando/transcribiendo/pensando/hablando. Interior plano con un
  **infinito crema** (sin brillos) y halo por estado. Se desactiva con
  `ui.orb: false`.
- La conversación va coloreada (Tú en cian, Samantha en verde); `NO_COLOR=1`
  la deja sin color.

## Activación: atajo, wake word y saludo

### Atajo y VAD

- `audio.ptt_mode: "vad"` (recomendado): una pulsación empieza a escuchar; el
  silencio cierra la toma (`vad.silence_duration: 0.9` s). Otra pulsación
  termina la conversación.
- `audio.ptt_mode: "hold"`: mantienes la combinación mientras hablas.
- `audio.continuous: true` reabre el micro tras cada respuesta;
  `continuous_tail_guard` (1.0 s) espera a que muera el eco del TTS.

### Wake word “Hola Samantha”

Con `wakeword.enabled: true` el micrófono queda abierto y **Vosk** escucha la
frase localmente (el audio no sale a la nube). El detector se pausa mientras
Samantha graba, piensa o habla, para no despertarse con su propia voz.

```yaml
wakeword:
  enabled: true
  phrase: "hola samantha"
  model: "vosk-model-small-es-0.42"   # ~38 MB comprimido
```

Si dices la frase y la orden en la misma respiración, la orden puede empezar
antes del chime; espera el tono para no perder la primera palabra.

### Saludo de arranque

```yaml
wakeword:
  greet_on_start: true   # saluda solo al iniciar (también el .exe)
  greet_text: ""         # vacío = usa la phrase ("hola samantha")
  greet_delay: 1.5       # segundos tras "Conexión OK"
```

Samantha recibe la frase como si la hubieras dicho: responde por voz, muestra
el orbe y (en modo continuo) se queda escuchando. Si aún no sabe tu nombre, te
lo pregunta en ese primer turno. `greet_on_start: false` vuelve al
comportamiento clásico (wake word o atajo).

## Modelo de voz local (STT)

| Pieza | Detalle |
|---|---|
| Principal | `faster-whisper` `large-v3-turbo` (~1,6 GB, CTranslate2) |
| GPU | CUDA 12 + cuDNN 9 + cuBLAS (vía pip o `cuda\` en el portable) |
| CPU | `int8` automático si CUDA no está operativo |
| Fallback | `whisper.cpp base` (rápido, ~0,75 s medido) |
| Fallback por latencia | turno > `fallback_after_ms` (4 s) → siguientes con `base` |

```yaml
stt:
  engine: "faster_whisper"
  fallback_engine: "whisper_cpp"
  fallback_after_ms: 4000     # 0 desactiva el cambio por latencia
  model_size: "base"          # modelo del fallback whisper.cpp
  language: "es"
  hotwords: ["Bogota", "Samantha"]
  n_threads: 8
  faster_whisper:
    model: "large-v3-turbo"
    device: "auto"            # auto | cuda | cpu
    compute_type: "default"   # float16/int8_float16/int8…
    beam_size: 5              # 1 = más rápido en CPU
    download_root: "models/faster-whisper"
    cuda_dll_dirs: []         # carpetas extra con cublas/cudnn
```

- El modelo grande se descarga en `models/faster-whisper` (desarrollo:
  `octo\models\faster-whisper`; portable: junto al `.exe`).
- Sin terminal no se ve el progreso de descarga: usa `--console` o mira
  `logs\assistant.log`.
- La primera inferencia en GPU se “calienta” al cargar (1 s de silencio). Si
  falta `cublas64_12.dll`, el motor detecta el fallo, recarga en CPU y sigue.
- El cambio por latencia es **permanente durante la sesión**: si instalas CUDA
  y quieres volver a probar el modelo grande, reinicia Samantha.

## Voz (TTS)

- `tts.engine: "edge"` → voz neuronal de Microsoft
  (`es-CO-SalomeNeural`, `rate: +25%`), con **Piper local de respaldo**
  (`es_MX-claude-high`). Requiere internet; el texto de las respuestas se envía
  a Microsoft.
- Edge agrega ~0,7 s de silencio por frase: `trim_silence: true` lo recorta y
  deja solo `sentence_silence` (0,25 s).
- **Pipeline frase a frase:** la frase N+1 se sintetiza mientras suena la N;
  el playback arranca antes y los puntos ya no dejan silencios largos.
- `gain: 1.6` (~+4 dB) con limitador suave; súbelo si el altavoz está bajo.
- Si Edge tarda o falla (`timeout: 10` s), esa frase se dice con Piper y Edge se
  reintenta tras `fallback_cooldown` (30 s).

## Buscar, informar y abrir búsquedas

1. Crea una cuenta en https://www.firecrawl.dev y copia tu API key.
2. Guárdala sin mostrarla: `Samantha.exe key firecrawl`
   (o `python assistant.py key firecrawl`).
3. Comprueba: `assistant.py check` — debe mostrar `OK` en Firecrawl.

- Las keys de Firecrawl empiezan por **`fc-`**. Si el archivo contiene otra
  cosa, `check` muestra `MAL` y el arranque avisa; las búsquedas usarán el
  respaldo.
- Plan gratuito: 1.000 créditos/mes, 10 req/min. Si Firecrawl responde `402`
  (sin créditos) **o `401`** (key inválida), Samantha intenta **DuckDuckGo**
  con `ddgs` (scraper no oficial; puede fallar por CAPTCHA/límite y va contra
  los términos de uso de DuckDuckGo). No inventa resultados.
- `wikipedia` no necesita key.
- `open_search` solo abre Google o YouTube con la consulta ya codificada; no
  abre URLs arbitrarias ni ejecuta programas.
- La **fecha y hora actuales** van en el prompt de cada turno: “¿qué día es
  hoy?” se responde sin buscar y sin alucinar.

## Memoria

- Di “recuerda que…” y lo guarda como nota (skill `manage_notes_and_facts`);
  notas y hechos se inyectan en cada turno.
- También puedes pedir “lista mis notas” o “resuelve la nota N”.
- `python assistant.py facts` / `facts delete ID` / `facts delete --all`.
- Base SQLite (sin FTS): usuarios, mensajes, resúmenes, notas y hechos.
  - Desarrollo: `~\.openocto\history.db`
  - Portable: `data\history.db` junto al `.exe`
- Resumen en segundo plano cuando la conversación crece
  (`memory.summarize_threshold`), ventana reciente configurable
  (`recent_window: 80`).

## Versión portable (.exe, sin Python)

```powershell
.\.venv\Scripts\python.exe build_portable.py --gpu --with-key
```

| Opción | Efecto |
|---|---|
| `--gpu` | Copia `cublas`/`cuDNN` del venv a `cuda\` (activa large en GPU) |
| `--with-key` | Copia OpenCode **y** Firecrawl a `secret\` |
| `--no-models` | No copia modelos (se descargan al primer arranque) |
| `--distpath C:\SamanthaBuild` | Construye en otra carpeta (útil si `dist\` está bloqueada o dañada) |

- Tamaños: **~2,3 GB** con el modelo grande sin GPU; **~4,2 GB** con `--gpu`.
- El build usa **staging + intercambio**: si `dist\Samantha` está abierta, el
  rename falla *antes* de borrar nada y la build nueva queda en
  `dist\_stage_Samantha\Samantha`. Conserva `data`, `logs` y `secret`, y deja
  el config anterior como `assistant-config.previous.yaml`.
- Contenido:

```
Samantha.exe          sin consola por defecto; --console para verla
_internal\            runtime + onnxruntime + piper + edge-tts + DLLs
assistant-config.yaml editable, en la raíz
personas\samantha\
models\               whisper.cpp, faster-whisper, Piper, VAD y Vosk
cuda\                 (solo --gpu) cublas/cuDNN/nvrtc
secret\               (solo --with-key o creado con Samantha.exe key)
LEEME.txt
```

### Arranque, consola y cierre

- Doble clic: arranca **sin ventana de consola**; el orbe y el saludo siguen
  funcionando. `--console` (o `--terminal`) muestra la terminal.
- `key`, `check`, `report`, `say`, `facts` y `--help` se adjuntan a la consola
  actual o crean una nueva.
- Para detener: `Ctrl+C` si usaste `--console`; si no, Administrador de
  tareas → `Samantha.exe`.
- Instancia única: si ya hay una corriendo, la segunda solo avisa (no se
  adjunta a la primera).
- Logs sin terminal: `Get-Content dist\Samantha\logs\assistant.log -Wait`.

> **Cuidado con discos exFAT.** Si `dist\` vive en un volumen exFAT con errores
> (`fsutil dirty query E:` → “El volumen contiene errores”), el build puede
> morir con `WinError 1392/145` en `_stage_Samantha`. No insistas: ejecuta
> `chkdsk E: /f` como administrador y, mientras tanto, compila con
> `--distpath C:\SamanthaBuild`.

## Ajustes frecuentes (`assistant-config.yaml`)

- **Micrófono/altavoz:** `audio.input_device` / `output_device` (`null` = los
  predeterminados de Windows; acepta índice o parte del nombre). Si cambias el
  dispositivo predeterminado, reinicia Samantha.
- **VAD:** `vad.threshold` (0,5), `silence_duration` (0,9 s),
  `rms_speech_threshold` (50: deja pasar voz baja; súbelo a 80–120 si el ruido
  dispara el VAD), `no_speech_timeout` (8 s), `max_recording_duration` (30 s).
- **Micrófono lento (Bluetooth/MME):** el stream se calienta antes del chime y
  no se graba hasta que el tono termina; si hablas antes, pierdes la primera
  sílaba. En conversación continua el stream queda caliente entre turnos.
- **Nombres que oye mal:** `stt.hotwords` (tu gente y tus ciudades).
- **Modelo LLM:** `ai.providers.go.model` (por defecto `minimax-m3.1-flash`);
  cambia solo esa línea y valida con `assistant.py check`.
  - `reasoning_effort`: MiniMax acepta `low|medium|high|xhigh|max`;
    `none|auto|adaptive` dan HTTP 400. Con `high`, sube `ai.max_tokens`
    (4000 en la config actual) porque el razonamiento consume el tope.
- **Timeout/tokens:** `ai.turn_timeout` (30 s), `ai.max_tokens` (4000).
- **Voz:** `tts.edge.voices.es`, `tts.edge.rate`, `tts.models.es` (Piper),
  `tts.gain`, `tts.engine` (`edge`/`piper`).
- **Activación:** `audio.ptt_mode`, `audio.continuous`,
  `wakeword.enabled`, `wakeword.greet_on_start`.
- **Orbe:** `ui.orb: false` lo desactiva.

Referencia técnica completa (módulos, diagramas, runbook):
[`AGENTS.md`](AGENTS.md).

## Estructura

| Ruta | Qué es |
|---|---|
| `assistant.py` | `start` / `key` / `check` / `report` / `say` / `facts` / `orb` — única puerta de entrada |
| `assistant-config.yaml` | **fuente de verdad** del comportamiento |
| `openocto/` | paquete: audio, STT, TTS, VAD, wake word, IA, memoria, skills, orbe |
| `personas/samantha/` | personaje (`persona.yaml` + `system_prompt.md`) |
| `secret/` | API keys (gitignored; nunca al repo) |
| `samples/` | 9 voces Piper en wav para escuchar |
| `logs/assistant.log` | DEBUG: transcripciones y tiempos (lo lee `report`) |
| `config/default.yaml` | defaults base del merge |
| `models/` | faster-whisper (desarrollo) |
| `~\.openocto\` | fuera del repo: modelos e `history.db` (memoria) |
| `docs/specs/` | change specs de las funciones grandes |
| `tests/` | suite (116 pruebas) |

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
# Ran 116 tests ... OK
```

## Privacidad

- **No hay servidor escuchando**: ningún puerto HTTP en este modo.
- **Teclado:** en modo `vad` la tecla global se registra con `RegisterHotKey`
  (sin hook); si otra app usa la combinación, cae a `pynput` y lo avisa. En
  `hold` se usa `pynput` (hook global).
- El audio no sale del PC. Los textos van a OpenCode Go y, con `tts.engine:
  edge`, también a Microsoft. Con `piper` la voz es 100 % local.
- Con `--debug`, las transcripciones quedan en `logs\assistant.log` y en
  `history.db` (local).
- La key vive en `secret\OPENCODE_GO_API_KEY`; rótala si crees que se filtró
  (https://opencode.ai/console) y repite `python assistant.py key`.
