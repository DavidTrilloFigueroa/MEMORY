"""Preparacion explicita del modelo: descargar faltantes o verificar sin red."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
EXPECTED_FILES = {"config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"}


def load_manifest(root=ROOT):
    manifest = json.loads((root / "recursos/modelo.json").read_text(encoding="utf-8"))
    if (manifest["repo"] != "Systran/faster-whisper-large-v3"
            or manifest["revision"] != "edaa852ec7e145841d8ffdb056a99866b5f0a478"
            or {item["rfilename"] for item in manifest["files"]} != EXPECTED_FILES
            or len(manifest["files"]) != 5):
        raise ValueError("Manifiesto inesperado o incompleto")
    return manifest


def file_valid(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    if "lfs" in item:
        digest = hashlib.sha256()
        expected = item["lfs"]["sha256"]
    else:
        digest = hashlib.sha1(("blob " + str(item["size"]) + "\0").encode("ascii"))
        expected = item["blobId"]
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def prepare(root, download=False, opener=None):
    manifest = load_manifest(root)
    folder = root / "modelos/faster-whisper-large-v3" / manifest["revision"]
    opener = opener or urllib.request.urlopen
    errors = 0
    for item in manifest["files"]:
        path = folder / item["rfilename"]
        if file_valid(path, item):
            print("OK " + item["rfilename"] + " (reutilizado)")
            continue
        if not download:
            print("ERROR " + item["rfilename"] + ": falta, tamano o hash incorrectos")
            errors += 1
            continue
        if path.exists():
            print("ERROR " + item["rfilename"] + ": archivo no valido conservado. Apartelo manualmente antes de reintentar.")
            errors += 1
            continue
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / (item["rfilename"] + ".parcial-" + uuid.uuid4().hex)
        url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{item['rfilename']}?download=true"
        try:
            print("Descargando " + item["rfilename"] + "...")
            request = urllib.request.Request(url, headers={"User-Agent": "transcripcion-local/1"})
            with opener(request, timeout=60) as response, partial.open("xb") as stream:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    stream.write(chunk)
            if not file_valid(partial, item):
                raise ValueError("Tamano o hash incorrecto; descarga parcial conservada")
            if path.exists():
                raise FileExistsError("Destino aparecio durante la descarga; no se sobrescribe")
            partial.rename(path)
            print("OK " + item["rfilename"])
        except Exception as exc:
            # No registrar URLs de redireccion firmadas ni credenciales en excepciones de red.
            print("ERROR " + item["rfilename"] + ": " + type(exc).__name__ + "; parcial conservado")
            errors += 1
    if errors:
        print("Verificacion incompleta; no utilice el modelo hasta resolver los errores.")
        return 1
    print("VERIFICADO: cinco archivos validos de la revision " + manifest["revision"])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--descargar", action="store_true", help="Solo aqui se permite descargar archivos faltantes")
    modes.add_argument("--verificar", action="store_true", help="Verificar exclusivamente archivos locales")
    args = parser.parse_args(argv)
    try:
        return prepare(ROOT, args.descargar)
    except Exception as exc:
        print("Error de preparacion: " + str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
