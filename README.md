# Samantha — asistente de voz personal (terminal)

Asistente local de voz con el personaje de **Samantha** (inspirado en la IA de
la película *Her*): cálida, curiosa y emocionalmente presente. **Solo terminal**:
sin interfaz web. Se activa con **Ctrl+Shift+Space** o diciendo
**“Hola Samantha”**:
una pulsación inicia una conversación; hablas, la toma se cierra al detectar
silencio, Samantha responde y **vuelve a escuchar sola** para tu siguiente
frase. Si callas `vad.no_speech_timeout` segundos, la conversación termina; una
segunda pulsación también la termina. En `ptt_mode: hold` mantienes la
combinación mientras hablas. El turno se procesa con STT → memoria → LLM →
Edge TTS (Piper de respaldo).

```
Ctrl+Shift+Space (pulsar) -> Silero VAD corta al callar -> whisper.cpp (es + hotwords)
   -> memoria SQLite/FTS5
   -> OpenCode Go (glm-5.3-flash, streaming) -> Edge TTS frase a frase
   -> (sin red: Piper local) -> altavoz
   -> continuous: reabre el micro y repite (fin: silencio N s o 2a pulsacion)
```

## Uso

```powershell
cd E:\NewBackup2\BackupCDCB\Algoritmos\protoPython\Assistant\octo

python assistant.py key -y     # una sola vez: guarda la API key en secret\ (enmascarada)
python assistant.py check      # de donde sale la key + prueba real contra Go
python assistant.py start      # uso diario
python assistant.py report     # latencia medida por turno desde el log
python assistant.py say "Hola, soy Samantha"   # probar la voz configurada
python assistant.py facts                # ver lo que Samantha aprendió de ti
python assistant.py facts delete 3 4     # borrar hechos por id (ver facts)
```

Con el asistente corriendo: pulsa **Ctrl+Shift+Space** una vez y conversa. Habla,
espera la respuesta y sigue hablando sin volver a pulsar: tras responder, el
micrófono se reabre solo. Si callas 8 s, la conversación termina; una segunda
pulsación también la termina. En `audio.ptt_mode: "hold"` se mantiene el
comportamiento clásico de mantener la combinación. Escucharás un chime al abrir
y otro al cerrar cada toma. `Ctrl+C` para salir.
El atajo es **global**: funciona con la ventana del asistente en segundo plano. Se cambia en `audio.ptt_hotkey` (por defecto `ctrl+shift+space`).
`python` puede ser cualquiera: el script se auto-relanza contra `.venv`.

- Al activar la conversación aparece un orbe de colores (ventana pequeña que se puede arrastrar) con el estado: esperando tu voz, escuchándote, transcribiendo, pensando, hablando. Se desactiva con `ui.orb: false`. Prueba visual sin micrófono: `python assistant.py orb`.
- `--debug` en `start` muestra DEBUG en la consola (por defecto la consola solo
  enseña avisos; el archivo `logs\assistant.log` guarda DEBUG siempre).
- La conversación va coloreada: **Tú** en cian, **Samantha** en verde, avisos en
  amarillo y errores en rojo. `NO_COLOR=1` desactiva el color.

## Wake word: “Hola Samantha”

Con `wakeword.enabled: true` el micrófono queda abierto y **Vosk** escucha
localmente la frase (no sale audio a la nube):

1. Di **“Hola Samantha”**.
2. Espera el chime.
3. Habla con normalidad; el VAD cierra la toma al callar.

El detector se pausa mientras Samantha graba, piensa o habla, para no
despertarse con su propia voz. El atajo global sigue funcionando como respaldo.
El modelo español pequeño ocupa ~38 MB comprimido (~58 MB en disco) y se descarga la primera vez en
`~\.openocto\models\vosk\` (en el portable, `models\vosk\`).

```yaml
wakeword:
  enabled: true
  phrase: "Hola Samantha"
  model: "vosk-model-small-es-0.42"
```

Si dices la frase y la orden en la misma respiración, la orden puede empezar
antes del chime; espera el tono para que no se pierda la primera palabra.

## Versión portable (.exe, sin Python)

```powershell
.\.venv\Scripts\python.exe build_portable.py      # 3-5 min
# opciones: --with-key (incluye la API key) · --no-models (no copia modelos)
#           --distpath dist\otra (si dist\Samantha está abierta)
```

El build ya **no borra** `dist\Samantha` antes de tiempo: construye en staging y
solo intercambia si la carpeta está libre; conserva `data`, `logs` y `secret`, y
deja el config anterior como `assistant-config.previous.yaml`.

Genera `dist\Samantha\` (~559 MB) lista para copiar a cualquier Windows x64:

```
Samantha.exe          consola, hotkey global Ctrl+Shift+Space
_internal\            runtime + onnxruntime + piper/espeak + edge-tts + DLLs
assistant-config.yaml editable, en la raíz de la carpeta
personas\samantha\
models\               whisper + voces Piper + VAD Silero + Vosk
LEEME.txt
```

Uso: `Samantha.exe check` → `Samantha.exe key` (si falta) → `Samantha.exe start [--debug]`.
Para oir la voz sin hablarle: `Samantha.exe say "Hola, soy Samantha"`.
Los datos se crean dentro de la carpeta: `data\history.db` y `logs\assistant.log`.

`--with-key` copia la API key a `secret\` (**viaja con la carpeta: no compartas esa
copia**). Sin esa opción, la key se crea con `Samantha.exe key`. Sigue necesitando
internet para el LLM y para la voz Edge; sin red, Piper responde local. Si no
llevas `models\`, se descargan en el primer arranque.

## Estructura

| ruta | que es |
|---|---|
| `assistant.py` | start / key / check / report — única puerta de entrada |
| `assistant-config.yaml` | **fuente de verdad**: voz, modelo, hotwords, ganancia |
| `openocto/` | paquete (OpenOcto modificado: PTT por combo, memoria, TTS por frases) |
| `personas/samantha/` | personaje: cálida, curiosa, breve, para leer en voz alta |
| `secret/` | API key (gitignored; nunca al repo) |
| `samples/` | las 9 voces es en wav para audición |
| `logs/assistant.log` | DEBUG: transcripciones y tiempos (aquí lee `report`) |
| `config/` | defaults del upstream (`config/default.yaml`, base del merge) |
| `~\.openocto\` | fuera del repo: modelos (~1 GB) e `history.db` (memoria) |

## Instalar en un PC nuevo (solo Python)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[audio]"
python assistant.py key          # pega la key del Console (enmascarada)
python assistant.py check        # espera: OK HTTP 200
python assistant.py start        # la primera vez descarga ~260 MB de modelos
```

Requiere Python 3.10+ (probado con 3.12; los wheels de audio existen para 3.13
pero el pipeline ahí no está probado). Git y ffmpeg no hacen falta. Atajo:
copia `~\.openocto\models\` del PC que ya funciona (y `history.db` si quieres
llevarte la memoria).

## Ajustes frecuentes (`assistant-config.yaml`)

- **Voz**: `tts.engine` (`edge` = neuronal colombiana, requiere internet;
  `piper` = local). `tts.edge.voices.es` elige `es-CO-SalomeNeural` (femenina)
  o `es-CO-GonzaloNeural` (masculina). `tts.models.es` es la voz Piper de
  respaldo; escucha `samples\voz-*.wav`. Con `fallback_engine: ""` se desactiva
  el respaldo.
- **Velocidad de Edge**: `tts.edge.rate` (`+25%` actual; prueba `+10%` o
  `+30%`). `tts.edge.trim_silence: true` recorta los ~0,7 s de cola que Edge
  deja por frase y conserva solo `tts.sentence_silence` (0,25 s). Medido en 3
  frases: **12,55 s → 8,40 s (-33 %)**, con CER 1,2 % (no se comen palabras).
- **Volumen**: `tts.gain` (1.0 original, **1.6 actual ≈ +4 dB**). Se aplica a
  Edge y a Piper con un limitador suave; el pico no pasa de 1.0. Si aún suena
  bajo, sube el volumen de Windows/altavoz antes de pasar de 2.0.
- **Micrófono con arranque lento** (Bluetooth/MME): el primer chunk puede
  tardar ~0,65–1,1 s. Se calienta el stream antes del chime y no se empieza a
  grabar hasta que el chime termina; en conversación continua el stream queda
  caliente entre turnos (sin grabar), así el segundo turno arranca al instante.
  Si hablas antes del chime, esa primera sílaba aún se pierde: espera el tono.
- **Pausas en los puntos**: la frase siguiente se sintetiza **mientras suena
  la actual** (P21). Antes cada punto agregaba ~1,3 s de silencio; ahora queda
  solo `tts.sentence_silence` (0,25 s). Medido: inicios 0,42 / 1,42 / 2,44 s.
- **Si Edge se cuelga**: `tts.edge.timeout` (6 s) corta la espera y
  `tts.fallback_cooldown` (30 s) decide cuánto se queda en Piper antes de
  reintentar. El aviso `edge no disponible (TimeoutError: sin respuesta en 6s)`
  es normal: esa frase se dijo con Piper y Edge se reintenta al vencer el
  cooldown.
- **Nombres que oye mal**: `stt.hotwords` — tu gente y tus ciudades.
- **Modelo**: `ai.providers.go.model` (glm-5.3-flash, mimo-v2.6-flash,
  deepseek-v4.1-flash…). `reasoning_effort` solo lo respetan algunos.
- **Ganancia del micro**: `audio.mic_gain` (`null` = sin ganancia fija) y
  `audio.normalize: true` (normaliza cada toma antes de Whisper; medido: el
  AGC por chunk empeoraba el CER con ruido de 3,2% a 10,6%).
- **Hilos de Whisper**: `stt.n_threads` (8 en este equipo).
- **Activación**: `audio.ptt_mode` = `"vad"` (pulsar una vez; corta al callar)
  o `"hold"` (mantener). En `vad`, `vad.silence_duration` (0,9 s) decide cuánto
  silencio espera; `vad.no_speech_timeout` (8 s) cancela/termina si no hablas; y
  `vad.max_recording_duration` (30 s) es el tope. `vad.threshold` (0,5) sube
  si se corta con ruido de fondo. `vad.rms_speech_threshold` está en **50**
  para micro lejano: con 300 una voz a RMS 150 ni llegaba a Silero. Si el
  ruido de fondo dispara el VAD, súbelo a 80–120 (no vuelvas a 300).
- **Conversación continua**: `audio.continuous: true` reabre el micrófono tras
  cada respuesta; `continuous_tail_guard` (1,0 s) espera a que muera el eco del
  TTS. Con `false`, cada turno requiere una pulsación.

## Memoria

- Di "recuerda que…" y lo guarda como nota (skill `manage_notes_and_facts`);
  las notas y los hechos se inyectan en cada turno.
- También puedes pedirle "lista mis notas" o "resuelve la nota N" por voz.
- Para ver o borrar lo que aprendió: `python assistant.py facts`, `facts delete ID` o `facts delete --all`.
- Todo vive en `~\.openocto\history.db` en desarrollo, o en
  `data\history.db` dentro de la carpeta portable (SQLite con FTS5).

## Actualizar el upstream

El remote `origin` apunta a `openocto-dev/openocto`. Los cambios propios están
en los commits del fork; para traer upstream: `git fetch` y merge manual (git
mostrará los conflictos donde upstream toque las mismas líneas).

## Privacidad

- **No hay servidor escuchando**: ningún puerto HTTP en este modo.
- **Teclado**: con `audio.ptt_mode: "vad"` (por defecto) la tecla global se registra con RegisterHotKey y no se instala ningún hook de teclado. Si otra aplicación ya usa la combinación, el asistente cae a pynput y lo avisa en el log. Con `"hold"` se usa pynput, que sí instala un hook global que ve todas las pulsaciones.
- El audio no sale del PC. Los textos van a OpenCode Go y, con `tts.engine:
  edge`, también a Microsoft (voz). Con `piper` la voz es 100 % local.
- Con `DEBUG`, las transcripciones quedan en `logs\assistant.log` y en
  `history.db` (local).
- La key vive en `secret\OPENCODE_GO_API_KEY`; rotarla si crees que se filtró
  (https://opencode.ai/console) y repetir `python assistant.py key`.

## Parches propios sobre OpenOcto (resumen)

P1 voces Piper es · P2 cabeceras de Go (obligatorias) · P3 UTF-8 en YAML ·
P4 idioma del TTS = idioma del habla · P5 arranque en Windows ·
P7 reasoning_effort/extra_body · P8 TTS por frases concurrente ·
P9 eventos internos · P11 hotwords de Whisper · P17 Edge TTS es-CO con
fallback a Piper · P18 VAD Silero (ptt_mode: vad) · P19 conversación continua
(continuous) · P20 Edge rate + recorte de silencio · P21 TTS pipeline
(síntesis N+1 mientras suena N) · **P16 modo terminal**
(combo configurable en audio.ptt_hotkey, por defecto Ctrl+Shift+Space; fuera UI web, wake word, MCP, API).

Las etapas intermedias (wake word "hotword", pantalla /live, etc.) fueron
eliminadas en P16 y quedan en la historia de git (`git log --oneline`).

## Pendientes de dueño

- Elegir voz Piper de respaldo entre las 9 muestras (la principal ya es
  `es-CO-SalomeNeural`).
- Rotar la API key (quedó visible en el chat de su creación).
- Borrar `..\prototype-openocto\` cuando confirmes esta versión (~1,3 GB).
