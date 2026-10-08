"""DLL NVIDIA locales. Importar este modulo no configura CUDA ni carga modelos."""
import os
from pathlib import Path
import sys

# Identidad de comportamiento conservada para la reanudacion de resultados previos.
# No es un hash de este archivo: LF/CRLF o una reubicacion no cambian la inferencia.
DLL_COMPATIBILITY_ID = "a4a515436ec499cf68dadf288a303c746798af2cd541c3fe9df41ee15dd5a730"


def configure_cuda(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    expected = root / ".venv"
    if Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError("Ejecutar con .venv/Scripts/python.exe; primero ejecute Instalar.cmd")
    nvidia = expected / "Lib/site-packages/nvidia"
    directories = sorted({p.parent for p in nvidia.rglob("*.dll")})
    if not directories:
        raise RuntimeError("No hay DLL NVIDIA en .venv; revise el registro de instalacion")
    handles = []
    try:
        for directory in directories:
            handles.append(os.add_dll_directory(str(directory)))
        os.environ["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get("PATH", "")
        return nvidia, handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    def block_network(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("Conexion de red prohibida durante transcripcion local")
    sys.addaudithook(block_network)
