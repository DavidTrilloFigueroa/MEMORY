"""Verificar dependencias y CUDA local, sin cargar Whisper ni abrir audios."""
import ctypes
import importlib.metadata
from pathlib import Path
import re
import sys
from runtime_cuda import configure_cuda, offline

ROOT = Path(__file__).resolve().parent


def main():
    handles = []
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Se requiere Python 3.11")
        for line in (ROOT / "requirements-win-py311.lock.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, version = line.split(" --hash=", 1)[0].split("==")
            if importlib.metadata.version(name) != version:
                raise RuntimeError("Version instalada distinta del lock: " + name)
        nvidia, handles = configure_cuda(ROOT)
        offline()
        import ctranslate2
        import faster_whisper
        import av
        import onnxruntime
        import tokenizers
        loaded = []
        for name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
            paths = list(nvidia.rglob(name))
            if len(paths) != 1:
                raise RuntimeError("DLL ausente o ambigua: " + name)
            loaded.append(ctypes.WinDLL(str(paths[0])))
        count = ctranslate2.get_cuda_device_count()
        if count < 1:
            raise RuntimeError("No se detecta GPU CUDA; revisar controlador NVIDIA")
        types = ctranslate2.get_supported_compute_types("cuda", device_index=0)
        if "float16" not in types:
            raise RuntimeError("GPU 0 sin float16 compatible; no se sustituye por CPU")
        print("OK dependencias del lock e importaciones")
        print("OK DLL locales; dispositivos CUDA: " + str(count))
        print("Tipos admitidos GPU 0: " + ", ".join(sorted(types)))
        print("El modelo no se ha cargado; esto no valida aun la inferencia.")
        return 0
    except Exception as exc:
        print("ERROR: " + str(exc))
        print("Revise README: Python 3.11 x64, dependencias, controlador NVIDIA y Visual C++ x64.")
        return 1
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    sys.exit(main())
