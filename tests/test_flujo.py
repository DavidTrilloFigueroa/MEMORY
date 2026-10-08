"""Pruebas de flujo con motor simulado; no carga CUDA ni lee grabaciones reales."""
import contextlib
from dataclasses import dataclass
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location("transcribir_c04", PROJECT / "transcribir.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


@dataclass
class Segment:
    start: float = 0.5
    end: float = 1.5
    text: str = " prueba simulada"


@dataclass
class Info:
    duration: float = 12.0
    duration_after_vad: float = 12.0
    transcription_options: tuple = (0.0,)


class FakeModel:
    loads = 0
    calls = 0
    fail_generator = False

    def __init__(self, model_size_or_path, device="auto", device_index=0,
                 compute_type="default", local_files_only=False, num_workers=1):
        type(self).loads += 1
        self.model = SimpleNamespace(device=device, device_index=[device_index], compute_type=compute_type)

    def transcribe(self, audio, language=None, task="transcribe", beam_size=5,
                   temperature=0.0, condition_on_previous_text=True, vad_filter=False,
                   no_speech_threshold=0.6, log_prob_threshold=-1.0,
                   compression_ratio_threshold=2.4, word_timestamps=False,
                   without_timestamps=False, clip_timestamps="0", log_progress=False):
        type(self).calls += 1

        def generator():
            yield Segment()
            if type(self).fail_generator:
                raise RuntimeError("Fallo simulado despues de producir un segmento")

        return generator(), Info()


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="c04-prueba-", dir=PROJECT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "entrada"
        self.source.mkdir()
        (self.source / "elegido.wav").write_bytes(b"SIMULACION SIN AUDIO")
        (self.source / "otro.mp3").write_bytes(b"NO PROCESAR EN MODO UNICO")
        (self.root / "modelos/faster-whisper-large-v3" / app.REVISION).mkdir(parents=True)
        config = json.loads((PROJECT / "configuracion.example.json").read_text(encoding="utf-8"))
        config["input_directory"] = str(self.source)
        self.config = self.root / "configuracion.json"
        app.write_json(self.config, config)
        self.output = self.root / "transcripciones"
        FakeModel.loads = FakeModel.calls = 0
        FakeModel.fail_generator = False
        for target, value in (("ROOT", self.root), ("configure_runtime", lambda: []),
                              ("model_manifest", lambda: []), ("verify_local_model", lambda *args: None)):
            manager = patch.object(app, target, value)
            manager.start()
            self.addCleanup(manager.stop)
        manager = patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=FakeModel)})
        manager.start()
        self.addCleanup(manager.stop)

    def execute(self, all_files=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return app.run(self.config, "elegido.wav", all_files)

    def complete(self):
        return list(self.output.glob("*/completo.json")) if self.output.exists() else []

    def test_single_complete_skip_and_corruption(self):
        self.assertEqual(self.execute(), 0)
        self.assertEqual(FakeModel.calls, 1)
        folder = self.complete()[0].parent
        payload = json.loads((folder / "transcripcion.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["duration_total_audio_seconds"], 12.0)
        self.assertEqual(payload["segments"][-1]["end"], 1.5)
        self.assertTrue(payload["generator_exhausted"])
        self.assertGreaterEqual(payload["processing_seconds"], 0)
        self.assertEqual(self.execute(), 0)
        self.assertEqual(FakeModel.calls, 1)  # No carga ni inferencia al reanudar resultados integros.
        self.assertEqual(FakeModel.loads, 1)
        (folder / "transcripcion.txt").write_text("corrupto", encoding="utf-8")
        self.assertEqual(self.execute(), 0)
        self.assertEqual(FakeModel.calls, 2)
        self.assertEqual(len(self.complete()), 2)
        self.assertEqual((folder / "transcripcion.txt").read_text(encoding="utf-8"), "corrupto")
        self.assertEqual((self.source / "elegido.wav").read_bytes(), b"SIMULACION SIN AUDIO")

    def test_late_generator_failure_and_retry(self):
        FakeModel.fail_generator = True
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.complete(), [])
        self.assertEqual(len(list(self.output.glob(".parcial-*/error.json"))), 1)
        FakeModel.fail_generator = False
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.complete()), 1)
        self.assertEqual(len(list(self.output.glob(".parcial-*/error.json"))), 1)

    def test_explicit_folder_loads_once(self):
        self.assertEqual(self.execute(all_files=True), 0)
        self.assertEqual(FakeModel.calls, 2)
        self.assertEqual(FakeModel.loads, 1)

    def test_changed_configuration_does_not_skip(self):
        self.assertEqual(self.execute(), 0)
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["transcription"]["language"] = "ca"
        self.config.write_text(json.dumps(config), encoding="utf-8")
        self.assertEqual(self.execute(), 0)
        self.assertEqual(FakeModel.calls, 2)
        self.assertEqual(len(self.complete()), 2)

    def test_temporal_inconsistency_rejected(self):
        with self.assertRaises(ValueError):
            app.validate_segments([{"start": 3.0, "end": 2.0}], 12.0)
        self.assertEqual(app.timestamp(3661.234), "01:01:01,234")

    def test_legacy_c04_export_change_is_skipped(self):
        self.assertEqual(self.execute(), 0)
        folder = self.complete()[0].parent
        path = folder / "transcripcion.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["configuration"]["schema_version"] = 1
        data["configuration"].pop("inference_revision")
        data["configuration"]["script_sha256"] = next(iter(app.LEGACY_COMPATIBLE_SCRIPTS))
        data["configuration_sha256"] = app.digest_json(data["configuration"])
        path.write_text(json.dumps(data), encoding="utf-8")
        marker = folder / "completo.json"
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        manifest["configuration_sha256"] = data["configuration_sha256"]
        manifest["files"]["transcripcion.json"] = app.sha256(path)
        marker.write_text(json.dumps(manifest), encoding="utf-8")
        before = {name: app.sha256(folder / name) for name in (*app.OUTPUTS, "completo.json")}
        self.assertEqual(self.execute(), 0)
        self.assertEqual(FakeModel.calls, 1)
        self.assertEqual(FakeModel.loads, 1)
        self.assertEqual(before, {name: app.sha256(folder / name) for name in before})

    def test_unknown_legacy_or_changed_parameters_not_compatible(self):
        self.assertEqual(self.execute(), 0)
        data = json.loads((self.complete()[0].parent / "transcripcion.json").read_text(encoding="utf-8"))
        current = data["configuration"]
        changed = json.loads(json.dumps(current))
        changed["transcription_call"]["language"] = "ca"
        self.assertNotEqual(app.inference_identity(current), app.inference_identity(changed))
        changed["schema_version"] = 1
        changed["script_sha256"] = "unknown"
        self.assertIsNone(app.inference_identity(changed))

    def test_modified_audio_is_reprocessed(self):
        self.assertEqual(self.execute(), 0)
        (self.source / "elegido.wav").write_bytes(b"OTRO AUDIO SIMULADO")
        self.assertEqual(self.execute(), 0)
        self.assertEqual(FakeModel.calls, 2)
        self.assertEqual(len(self.complete()), 2)

    @unittest.skipUnless(os.name == "nt", "Lanzador Windows")
    def test_real_cmd_in_isolated_stub_project(self):
        sandbox = self.root / "proyecto con espacios"
        scripts = sandbox / ".venv/Scripts"
        scripts.mkdir(parents=True)
        import venv
        venv.EnvBuilder(with_pip=False).create(sandbox / ".venv")
        shutil.copyfile(PROJECT / "Transcribir audios.cmd", sandbox / "Transcribir audios.cmd")
        (sandbox / "configuracion.json").write_text("{}", encoding="utf-8")
        (sandbox / "transcribir.py").write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "Path('invocacion.json').write_text(json.dumps(sys.argv[1:]))\n"
            "sys.exit(int(Path('salida.txt').read_text()))\n", encoding="utf-8")
        for code in (0, 7):
            (sandbox / "salida.txt").write_text(str(code), encoding="utf-8")
            result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c",
                                     str(sandbox / "Transcribir audios.cmd")],
                                    input="\n", capture_output=True, text=True, cwd=PROJECT, timeout=30)
            self.assertEqual(result.returncode, code, result.stdout + result.stderr)
            self.assertEqual(json.loads((sandbox / "invocacion.json").read_text()), ["--todos"])

    def test_blocks_preserve_exact_text_and_originals_without_runtime(self):
        segment = Segment(text="  ¿Prueba?\r\nSí, exacta.  ")
        with patch.dict(globals(), {"Segment": lambda: segment}):
            self.assertEqual(self.execute(), 0)
        folder = self.complete()[0].parent
        originals = {name: app.sha256(folder / name) for name in (*app.OUTPUTS, "completo.json")}
        blocks = folder / "elegido_bloques.txt"
        receipt = json.loads(blocks.with_suffix(".registro.json").read_text(encoding="utf-8"))
        offset = receipt["blocks"][0]
        extracted = blocks.read_bytes()[offset["text_byte_start"]:offset["text_byte_end"]].decode("utf-8")
        self.assertEqual(extracted, segment.text)
        self.assertIn(b"[00:00:00,500]", blocks.read_bytes())
        # Simular un resultado antiguo que ya tiene sus salidas; la orden solo usa JSON.
        with patch.object(app, "configure_runtime", side_effect=AssertionError("No CUDA")), \
                patch.object(app, "run", side_effect=AssertionError("No inferencia")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(app.main(["--bloques-json", str(folder / "transcripcion.json")]), 0)
        self.assertEqual(len(list(folder.glob("*_bloques*.txt"))), 1)
        blocks.write_text("edicion manual", encoding="utf-8")
        new = app.export_blocks(folder / "transcripcion.json")
        self.assertNotEqual(new, blocks)
        self.assertEqual(blocks.read_text(encoding="utf-8"), "edicion manual")
        self.assertEqual(originals, {name: app.sha256(folder / name) for name in originals})


if __name__ == "__main__":
    unittest.main(verbosity=2)
