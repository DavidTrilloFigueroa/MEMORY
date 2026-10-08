# 🚀 Guía de Instalación y Ejecución en un Nuevo Ordenador

Este documento contiene los pasos exactos para replicar y ejecutar el proyecto **MEMORY** desde cero en una nueva computadora.

---

## ⚠️ Requisitos previos del nuevo equipo
* **Sistema Operativo:** Windows 11 x64 con Python 3.11 x64 instalado.
* **Hardware:** Tarjeta gráfica NVIDIA (GPU) compatible con CUDA.

---

## Paso 1: Clonar el proyecto desde GitHub

Abre la terminal de tu nuevo ordenador (`PowerShell` o `Git Bash`) en la carpeta donde quieras guardar el proyecto y ejecuta:

```bash
git clone https://github.com/tu-usuario/MEMORY.git
```

Una vez clonado, abre la carpeta resultante (`MEMORY`) con **Visual Studio Code**.

---

## Paso 2: Crear el Entorno Virtual e Instalar Dependencias

1. Abre la Terminal Integrada de VS Code (`Ctrl + ~`) y asegúrate de estar usando **PowerShell**.
2. El proyecto cuenta con un sistema automatizado que creará el entorno `.venv` e instalará las 28 librerías de Python exactas (incluyendo componentes de NVIDIA y CUDA) sin necesidad de configuraciones manuales. Solo ejecuta:

```powershell
./Instalar.cmd
```

### 🔍 Verificar que el entorno es correcto
Para comprobar que las librerías de la tarjeta gráfica y dependencias se vincularon bien, ejecuta este comando de diagnóstico:

```powershell
& .\.venv\Scripts\python.exe -B .\verificar_entorno.py
```

> Debe finalizar con código `0` indicando que detecta CUDA y `float16`.

---

## Paso 3: Descargar y Verificar el Modelo de IA de forma automática

No necesitas buscar los archivos del modelo por internet. El script `modelo.py` se encargará de descargar los 5 archivos estructurados necesarios de Hugging Face (`model.bin`, `config.json`, etc.).

1. **Iniciar la descarga del modelo:**
   ```powershell
   & .\.venv\Scripts\python.exe -B .\modelo.py --descargar
   ```
   *Este paso puede demorar unos minutos ya que el modelo pesa aproximadamente 3.09 GB.*

2. **Verificar la integridad de la descarga:**
   Una vez termine la descarga, ejecuta esto para asegurarte de que ningún archivo se haya corrompido:
   ```powershell
   & .\.venv\Scripts\python.exe -B .\modelo.py --verificar 0
   ```
   *Deberás ver cinco mensajes de **OK** y la palabra **VERIFICADO**.*

---

## Paso 4: Configuración y Primera Ejecución

### 1. Crear el archivo de configuración
* En la raíz del proyecto verás un archivo llamado `configuracion.example.json`.
* Sácale una copia en esa misma carpeta y renómbralo exactamente como `configuracion.json`.
* *(Este archivo está en el `.gitignore` por lo que tus rutas privadas nunca se subirán a internet).*

### 2. Ejecutar la transcripción
* Coloca tus archivos de audio (`.wav`, `.mp3`, `.m4a` o `.flac`) dentro de la carpeta `audios/`.
* Haz doble clic en el archivo ejecutable `Transcribir audios.cmd`.
* Alternativamente, si prefieres lanzarlo desde la terminal de PowerShell, puedes usar:
  ```powershell
  & .\.venv\Scripts\python.exe -B .\transcribir.py --todos
  ```

¡Listo! Las transcripciones aparecerán automáticamente organizadas en carpetas únicas dentro de la ruta `transcripciones/`.