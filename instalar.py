"""Crear entorno nuevo con el lock fijado; nunca modifica otro Python."""
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def main():
    if (os.name != "nt" or sys.version_info[:2] != (3, 11)
            or struct.calcsize("P") != 8 or platform.machine().upper() not in ("AMD64", "X86_64")
            or sys.getwindowsversion().build < 22000):
        print("Se requiere Windows 11 x64 y Python 3.11 x64. Instale los requisitos del README.")
        return 1
    target = ROOT / ".venv"
    if target.exists():
        print(".venv ya existe: no se modifica. Use verificar_entorno.py; consulte README si la instalacion quedo incompleta.")
        return 1
    lock = ROOT / "requirements-win-py311.lock.txt"
    if not lock.is_file():
        print("Falta el archivo de dependencias bloqueadas. Extraiga la distribucion completa.")
        return 1
    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    logfile = logdir / ("instalacion-" + stamp + ".txt")

    def run(args):
        result = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        output = result.stdout.replace(str(ROOT), "<PROYECTO>").replace(os.environ.get("USERPROFILE", "<USUARIO>"), "<USUARIO>")
        command = subprocess.list2cmdline(args).replace(str(ROOT), "<PROYECTO>").replace(os.environ.get("USERPROFILE", "<USUARIO>"), "<USUARIO>")
        with logfile.open("a", encoding="utf-8") as stream:
            stream.write(command + "\n" + output + "\nCodigo: " + str(result.returncode) + "\n")
        if result.returncode:
            raise RuntimeError("Operacion fallida; consulte " + str(logfile))
        return result.stdout

    try:
        print("Creando .venv nueva e instalando las versiones fijadas. Espere...")
        run([sys.executable, "-B", "-m", "venv", str(target)])
        python = str(target / "Scripts/python.exe")
        run([python, "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-cache-dir",
             "install", "--only-binary=:all:", "--require-hashes", "--index-url", "https://pypi.org/simple",
             "-r", str(lock)])
        run([python, "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "check"])
        installed = run([python, "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "freeze", "--all"])
        (logdir / ("versiones-" + stamp + ".txt")).write_text(installed, encoding="utf-8")
        config = ROOT / "configuracion.json"
        if not config.exists():
            example = json.loads((ROOT / "configuracion.example.json").read_text(encoding="utf-8"))
            with config.open("x", encoding="utf-8") as stream:
                json.dump(example, stream, ensure_ascii=False, indent=2)
        (ROOT / "audios").mkdir(exist_ok=True)
        (ROOT / "transcripciones").mkdir(exist_ok=True)
        print("Instalacion completada. Ejecute verificar_entorno.py y prepare el modelo segun README.")
        print("Registro: " + str(logfile))
        return 0
    except Exception as exc:
        print(str(exc))
        print("No se borra el entorno parcial ni se modifica Python global. Consulte recuperacion en README.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
