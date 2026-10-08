# Transcripción local de grabaciones

Transcribe archivos WAV, MP3, M4A y FLAC con **Whisper large-v3 y faster-whisper**, en español o catalán. Procesa la carpeta de entrada secuencialmente, carga el modelo una vez y guarda TXT, SRT, JSON y un TXT por segmentos. Las grabaciones y resultados permanecen locales. No identifica hablantes.

## Requisitos y alcance validado

- **Windows 11 x64 y CPython 3.11 x64**. No Python ARM, 3.12 ni otras plataformas para este lock. [Descargas oficiales de Python para Windows](https://www.python.org/downloads/windows/). El instalador de Python debe incluir `venv` y `pip`; el launcher `py` es práctico, no obligatorio.
- **GPU NVIDIA en índice 0 con CUDA y float16**, controlador compatible con CUDA 12.9. Como referencia, CUDA 12.9 Update 1 documenta Windows driver >=576.57. No se instala ni actualiza el controlador desde este proyecto. [NVIDIA CUDA 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html).
- Runtime **Microsoft Visual C++ x64** compatible. Si falta, instalarlo desde [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist). Ningún script cambia permisos, PATH global ni configuración del sistema.
- Conexión a Internet solo para instalar paquetes y preparar el modelo. Modelo: aproximadamente **3.09 GB**, más espacio para `.venv`, descargas temporales y resultados. El mínimo total de espacio no se ha medido en otro equipo.

Equipo con transcripción real previamente comprobada: Windows 11 x64, Python 3.11.9, RTX 5060 Laptop de unos 8 GiB de VRAM, driver 596.13 y 32 GB RAM. **Otros ordenadores/GPU no están validados**. Esta entrega se ha comprobado en una copia limpia del mismo ordenador, con pruebas simuladas y sin reinstalar paquetes. No garantiza funcionar en cualquier NVIDIA ni con menos VRAM. No cambia automáticamente a CPU u otra precisión si falla.

No hacen falta Git, Git LFS, VSCode ni Codex. No se distribuyen modelos, DLL ni instaladores de terceros. Las ruedas del lock aportan las bibliotecas CUDA/cuDNN locales y PyAV para audio; no se exige instalar FFmpeg ni el Toolkit completo globalmente. [CTranslate2](https://opennmt.net/CTranslate2/installation.html), [cuDNN en Windows](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html).

## 1. Obtener e instalar

Descarga el ZIP de fuentes y **extráelo completo** en una carpeta donde puedas escribir, incluso con espacios. No ejecutes desde dentro del ZIP. Abre `Instalar.cmd` con doble clic. Detecta Python 3.11 mediante `py`, o utiliza `python` si no existe ese launcher; el script valida versión, arquitectura y sistema.

Alternativa desde PowerShell, situado en la carpeta del proyecto:

```powershell
py -3.11 -B .\instalar.py
```

Si Python no está en PATH, sustituye `py -3.11` por `& "ruta completa a python.exe"`. No necesitas activar el entorno ni cambiar políticas de PowerShell.

El instalador crea `.venv`, instala **únicamente las 28 versiones fijadas con hashes** de `requirements-win-py311.lock.txt`, ejecuta `pip check`, registra versiones reales en `logs/` y prepara `configuracion.json`, `audios/` y `transcripciones/`. No actualiza versiones ni regenera el lock. `pip` y `setuptools` iniciales proceden de `ensurepip` del Python usado; se registran, no se actualizan.

**Si `.venv` ya existe, se detiene sin modificarla.** No lo uses como actualizador. Para recuperar una instalación fallida, conserva el registro y aparta manualmente el entorno parcial antes de repetir, o instala las fuentes en otra carpeta. No borres audios, modelos ni resultados. Si ya tienes un entorno correcto pero falta configuración, copia `configuracion.example.json` como `configuracion.json` sin sobrescribir una configuración existente, y crea `audios/` y `transcripciones/`.

Verifica el entorno:

```powershell
& .\.venv\Scripts\python.exe -B .\verificar_entorno.py
```

Debe terminar con código 0 e indicar DLL locales, CUDA y float16. Esta comprobación no carga Whisper ni demuestra por sí sola que la inferencia funcionará. Las DLL se añaden solo al proceso y sus manejadores permanecen vivos mientras se utiliza el modelo.

## 2. Preparar el modelo una vez

```powershell
& .\.venv\Scripts\python.exe -B .\modelo.py --descargar
& .\.venv\Scripts\python.exe -B .\modelo.py --verificar
$LASTEXITCODE
```

La primera orden es la **única utilidad que descarga el modelo**. La segunda no usa red. Reutiliza archivos existentes válidos, descarga solo faltantes y verifica tamaños y hashes. Si existe un archivo incorrecto, lo conserva y pide apartarlo manualmente. Descargas interrumpidas quedan con sufijo `.parcial-*`; no se usan como modelo. Repetir la orden conserva archivos ya validados y vuelve a intentar los faltantes; no promete reanudación dentro de un archivo parcial.

Repositorio: [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3). Revisión fija: `edaa852ec7e145841d8ffdb056a99866b5f0a478`. Destino: `modelos/faster-whisper-large-v3/edaa852ec7e145841d8ffdb056a99866b5f0a478/`.

Se necesitan `model.bin`, `config.json`, `preprocessor_config.json`, `tokenizer.json` y `vocabulary.json`. El manifiesto público `recursos/modelo.json` conserva los tamaños y hashes: SHA256 para el binario LFS y hash del objeto Git para los JSON. Puedes descargarlos manualmente desde la revisión fija y ejecutar la verificación. Exige cinco OK, VERIFICADO y código 0. No uses el repositorio del modelo original que requiere conversión.

La transcripción carga exclusivamente esa carpeta con `local_files_only=True`, Hugging Face offline y conexiones Python bloqueadas. Nunca descarga recursos automáticamente ni crea otra copia del modelo en una caché.

## 3. Configurar y ejecutar

`configuracion.example.json` es el ejemplo público. `configuracion.json` es privado: puedes editar `input_directory` para usar una carpeta externa absoluta sin mover sus originales. Por defecto usa `audios/`; la salida es `transcripciones/`. **Todas las rutas relativas se resuelven desde la raíz del proyecto**, también cuando el programa se lanza desde otro directorio. No uses salidas dentro de la entrada.

Coloca audios en la carpeta configurada y haz doble clic en **`Transcribir audios.cmd`**. Busca solo archivos inmediatos (no subcarpetas) WAV/MP3/M4A/FLAC. Muestra estado, contadores y ruta del registro; espera una tecla al finalizar. Ejecuta una sola instancia y no cierres la ventana mientras trabaja.

Alternativas desde PowerShell:

```powershell
& .\.venv\Scripts\python.exe -B .\transcribir.py --listar
& .\.venv\Scripts\python.exe -B .\transcribir.py --archivo=ejemplo.wav
& .\.venv\Scripts\python.exe -B .\transcribir.py --todos
```

Usa `--archivo=nombre` para nombres que empiecen por guion. Para sesiones en catalán/valenciano, cambia `transcription.language` de `es` a `ca` antes de procesarlas. No se traduce ni reformula. Se mantienen beam 5, temperatura 0, VAD desactivado, sin condicionamiento por texto anterior y los demás parámetros del ejemplo; el JSON registra todos los parámetros efectivos.

## Salidas y reanudación

Cada audio genera una carpeta única con TXT continuo, SRT aproximado, JSON de segmentos y `<audio>_bloques.txt`. Este último muestra un segmento automático por bloque y su inicio, **no turnos ni frases garantizados**. Conserva el texto exacto sin correcciones. Su registro de integridad es independiente del manifiesto original.

Al repetir el lanzador se omiten resultados completos solo si coinciden audio, configuración de inferencia, versiones y modelo, y sus salidas conservan los hashes. Cambios únicamente de exportación no invalidan una inferencia compatible. Los intentos fallidos y resultados antiguos se conservan; se crean carpetas nuevas al reintentar. Modificar un audio o parámetro puede originar una nueva transcripción. No se modifica el original.

Para obtener bloques de un resultado ya completado, sin modelo ni audio:

```powershell
& .\.venv\Scripts\python.exe -B .\transcribir.py --bloques-json "transcripciones\carpeta-del-resultado\transcripcion.json"
```

El JSON distingue duración total decodificada y tiempo de procesamiento hasta agotar el generador; excluye carga, hashes y exportación del segundo valor. Agotar el generador o ver un segmento cerca del final **no prueba ausencia de omisiones**. Whisper puede omitir repeticiones, equivocarse o producir texto ausente. Revisa fidelidad contra el audio; no se garantiza literalidad perfecta ni se identifican interlocutores.

## Errores habituales y privacidad

- Python incorrecto: usa Python 3.11 x64; no cambies el lock para instalar en otra plataforma.
- DLL/CUDA ausente: ejecuta el verificador; revisa entorno local, Visual C++ y controlador. No copies DLL desconocidas al sistema.
- Modelo ausente o hash incorrecto: prepara/verifica los cinco archivos. No se usará CPU ni otra revisión para ocultar el fallo.
- GPU sin memoria: cierra otras cargas GPU y revisa el registro. No cambies automáticamente modelo o precisión.
- Carpeta ausente: revisa configuración y extrae la distribución completa.
- Código 1: consulta `logs/`; los resultados completos anteriores siguen disponibles. Código 0: ejecución sin errores detectados, no certificación de fidelidad.

No publiques `configuracion.json`, grabaciones, transcripciones, modelos, `.venv`, logs ni credenciales. Los logs pueden contener nombres de archivos. `.gitignore` protege esas rutas, pero no hace segura una compresión de todo tu directorio ni elimina datos ya versionados.

## Pruebas y licencia

```powershell
py -3.11 -B -m unittest discover -s tests -v
```

Las pruebas usan datos y motores simulados, sin participantes, red ni pesos. Incluyen reanudación, fallos tardíos, integridad, rutas, lanzador y preparación del modelo simulada. Una instalación de todas las dependencias y una transcripción en otro ordenador siguen pendientes.

**Licencia del código propio: pendiente de elección por su titular.** No se atribuye una licencia automáticamente. Véase `TERCEROS.md` para referencias de dependencias y modelo; sus licencias no se transfieren al código propio.
