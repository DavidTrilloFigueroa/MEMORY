"""Preparar entrega solo desde una lista publica; nunca comprimir el proyecto entero."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
PUBLIC_FILES = (
    ".gitignore", ".gitattributes", "README.md", "TERCEROS.md", "configuracion.example.json",
    "requirements-win-py311.lock.txt", "transcribir.py", "runtime_cuda.py", "instalar.py",
    "Instalar.cmd", "Transcribir audios.cmd", "modelo.py", "verificar_entorno.py", "empaquetar.py",
    "recursos/modelo.json", "tests/test_flujo.py", "tests/test_publico.py",
)


def validate_public(name, content):
    if name not in PUBLIC_FILES and not Path(name).name.upper().startswith(("LICENSE", "COPYING")):
        raise ValueError("Archivo fuera de lista publica")
    text = content.decode("utf-8-sig")
    patterns = (r"[A-Za-z]:[\\/]+Users[\\/]", r"/" + r"home/[^/\s]+/", r"/" + r"Users/[^/\s]+/",
                r"MA-MP[J M]\d", r"(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}",
                r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError("Posible dato privado en archivo publico: " + name)


def main():
    destination = ROOT / "entrega_github"
    archive = ROOT / "entrega_github.zip"
    if destination.exists() or archive.exists():
        raise FileExistsError("La entrega ya existe; no se sobrescribe")
    names = list(PUBLIC_FILES)
    for path in ROOT.iterdir():
        if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING")):
            names.append(path.name)
    contents = {}
    for name in names:
        source = ROOT / name
        if source.is_symlink():
            raise ValueError("No empaquetar enlaces")
        data = source.read_bytes()
        validate_public(name, data)
        # Distribucion independiente de autocrlf: LF en fuentes, CRLF en lanzadores.
        data = data.replace(b"\r\n", b"\n")
        if name.endswith(".cmd"):
            data = data.replace(b"\n", b"\r\n")
        contents[name] = data
    destination.mkdir()
    hashes = {}
    for name, data in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = json.dumps({"files": hashes, "contains": "source-only; no data, weights, environments or credentials"}, indent=2).encode("utf-8") + b"\n"
    (destination / "MANIFIESTO_ENTREGA.json").write_bytes(manifest)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                z.write(path, "entrega_github/" + path.relative_to(destination).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP corrupto")
        expected = {"entrega_github/" + name for name in contents} | {"entrega_github/MANIFIESTO_ENTREGA.json"}
        if set(z.namelist()) != expected:
            raise ValueError("Contenido inesperado en ZIP")
        for name, digest in hashes.items():
            if hashlib.sha256(z.read("entrega_github/" + name)).hexdigest() != digest:
                raise ValueError("Hash incorrecto en ZIP")
    print("Entrega verificada: " + str(destination))
    print("ZIP: " + str(archive))
    print("Archivos publicos: " + str(len(hashes)) + " y manifiesto de integridad")


if __name__ == "__main__":
    main()
