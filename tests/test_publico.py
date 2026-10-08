"""Pruebas publicas sin modelo, audios reales ni red."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import instalar
import modelo
import runtime_cuda


class PublicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prueba-publica-", dir=PROJECT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = modelo.load_manifest()
        self.bytes = {}
        for item in self.manifest["files"]:
            content = ("contenido simulado " + item["rfilename"]).encode()
            self.bytes[item["rfilename"]] = content
            item["size"] = len(content)
            if "lfs" in item:
                item["lfs"]["sha256"] = hashlib.sha256(content).hexdigest()
                item["lfs"]["size"] = len(content)
            else:
                item["blobId"] = hashlib.sha1((f"blob {len(content)}\0").encode() + content).hexdigest()
        (self.root / "recursos").mkdir()
        (self.root / "recursos/modelo.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.folder = self.root / "modelos/faster-whisper-large-v3" / self.manifest["revision"]

    def opener(self, request, **kwargs):
        name = request.full_url.split("/")[-1].split("?")[0]
        return io.BytesIO(self.bytes[name])

    def test_download_reuse_and_offline_verification(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(modelo.prepare(self.root, True, self.opener), 0)
            self.assertEqual(modelo.prepare(self.root, True, lambda *a, **k: self.fail("Red inesperada")), 0)
            self.assertEqual(modelo.prepare(self.root, False, lambda *a, **k: self.fail("Red inesperada")), 0)
            path = self.folder / "model.bin"
            path.write_bytes(b"INVALIDO")
            self.assertEqual(modelo.prepare(self.root, True, lambda *a, **k: self.fail("No sobrescribir")), 1)
            self.assertEqual(path.read_bytes(), b"INVALIDO")

    def test_bad_download_not_published(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(modelo.prepare(self.root, True, lambda *a, **k: io.BytesIO(b"BAD")), 1)
        self.assertFalse((self.folder / "model.bin").exists())
        self.assertEqual(len(list(self.folder.glob("*.parcial-*"))), 5)

    def test_installer_refuses_existing_environment(self):
        (self.root / ".venv").mkdir()
        marker = self.root / ".venv/conservar.txt"
        marker.write_text("sin modificar")
        with patch.object(instalar, "ROOT", self.root), patch.object(instalar.platform, "machine", return_value="AMD64"), patch.object(instalar.subprocess, "run", side_effect=AssertionError("No instalar")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(instalar.main(), 1)
        self.assertEqual(marker.read_text(), "sin modificar")

    @unittest.skipUnless(os.name == "nt", "Objetivo Windows")
    def test_installer_uses_hash_lock_without_upgrades(self):
        (self.root / "requirements-win-py311.lock.txt").write_text("# test\n")
        (self.root / "configuracion.example.json").write_text('{"input_directory":"audios"}')
        calls = []

        def simulated(args, **kwargs):
            calls.append(args)
            if "venv" in args:
                (self.root / ".venv").mkdir()
            return subprocess.CompletedProcess(args, 0, stdout="OK\n")

        with patch.object(instalar, "ROOT", self.root), patch.object(instalar.platform, "machine", return_value="AMD64"), patch.object(instalar.subprocess, "run", side_effect=simulated), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(instalar.main(), 0)
        install_call = next(args for args in calls if "install" in args)
        self.assertIn("--require-hashes", install_call)
        self.assertIn("--only-binary=:all:", install_call)
        self.assertNotIn("--upgrade", install_call)
        self.assertEqual(install_call[-1], str(self.root / "requirements-win-py311.lock.txt"))

    def test_cuda_handles_and_line_endings(self):
        nvidia = self.root / ".venv/Lib/site-packages/nvidia/a/bin"
        nvidia.mkdir(parents=True)
        (nvidia / "simulada.dll").write_bytes(b"test")
        handle = object()
        for ending in (b"\n", b"\r\n"):
            source = (PROJECT / "runtime_cuda.py").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending)
            namespace = {"__file__": str(self.root / "runtime_cuda.py")}
            exec(compile(source, "runtime_cuda.py", "exec"), namespace)
            with patch.object(sys, "prefix", str(self.root / ".venv")), patch.object(os, "add_dll_directory", return_value=handle, create=True), patch.dict(os.environ):
                _, handles = namespace["configure_cuda"](self.root)
                self.assertEqual(handles, [handle])
                self.assertEqual(namespace["DLL_COMPATIBILITY_ID"], runtime_cuda.DLL_COMPATIBILITY_ID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
