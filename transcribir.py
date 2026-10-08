"""Transcripcion local; no hace nada al importar. Seleccion explicita obligatoria."""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from runtime_cuda import DLL_COMPATIBILITY_ID, configure_cuda, offline

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
HELPER_SHA256 = DLL_COMPATIBILITY_ID
OUTPUTS = ("transcripcion.txt", "transcripcion.srt", "transcripcion.json")
# Cambiar esta revision si cambia la inferencia; los exportadores no la cambian.
INFERENCE_REVISION = "c04-v1"
LEGACY_COMPATIBLE_SCRIPTS = {"f34c9aed42f61d2ef6a5b0c1a99bd8cdfd802834a435de67ce451ae7d75e5125"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def list_audio(folder):
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p: (p.name.casefold(), p.name))


def fingerprint(path):
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("El audio cambio mientras se calculaba su huella")
    return {"sha256": digest, "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config["device"] != "cuda" or config["compute_type"] != "float16" or config["device_index"] != 0:
        raise ValueError("Se requiere CUDA:0/float16, sin fallback")
    if config["model_revision"] != REVISION or config["model_repository"] != "Systran/faster-whisper-large-v3":
        raise ValueError("Modelo o revision distintos de los verificados")
    options = config["transcription"]
    if options["language"] not in ("es", "ca") or options["task"] != "transcribe":
        raise ValueError("Solo transcripcion es/ca")
    if options.get("without_timestamps", False) or options.get("clip_timestamps", "0") != "0":
        raise ValueError("Se requieren tiempos del audio completo, sin recortes")
    if options.get("initial_prompt") or options.get("prefix") or options.get("hotwords"):
        raise ValueError("No se utilizan prompts ni reformulacion")
    if options.get("log_progress", False):
        raise ValueError("La salida de consola no debe mostrar contenido")
    source = (ROOT / config["input_directory"]).resolve()
    output = (ROOT / config["output_directory"]).resolve()
    model = (ROOT / config["model_directory"]).resolve()
    if not source.is_dir():
        raise ValueError("No existe la carpeta de entrada; revise input_directory en configuracion.json")
    if not model.is_dir():
        raise ValueError("Falta el modelo local; ejecute modelo.py --descargar y --verificar segun README")
    if not output.is_relative_to(ROOT) or output == ROOT or output.is_relative_to(source):
        raise ValueError("Los resultados deben estar dentro del proyecto y fuera de la entrada")
    if model != ROOT / "modelos" / "faster-whisper-large-v3" / REVISION:
        raise ValueError("Ruta de modelo distinta de la carpeta local verificada")
    return config, source, output, model


def configure_runtime():
    _, handles = configure_cuda(ROOT)
    offline()
    return handles


def model_manifest():
    metadata = json.loads((ROOT / "recursos/modelo.json").read_text(encoding="utf-8"))
    if metadata["revision"] != REVISION:
        raise ValueError("Revision del manifiesto inesperada")
    names = {"model.bin", "config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"}
    return [entry for entry in metadata["files"] if entry["rfilename"] in names]


def verify_local_model(folder, manifest):
    if len(manifest) != 5:
        raise ValueError("Manifiesto incompleto")
    for entry in manifest:
        path = folder / entry["rfilename"]
        if path.stat().st_size != entry["size"]:
            raise ValueError("Tamano del modelo incorrecto: " + entry["rfilename"])
        if "lfs" in entry:
            valid = sha256(path) == entry["lfs"]["sha256"]
        else:
            digest = hashlib.sha1(("blob " + str(entry["size"]) + "\0").encode("ascii"))
            digest.update(path.read_bytes())
            valid = digest.hexdigest() == entry["blobId"]
        if not valid:
            raise ValueError("Hash del modelo incorrecto: " + entry["rfilename"])


def effective_parameters(function, overrides, exclude):
    signature = inspect.signature(function)
    result = {key: param.default for key, param in signature.parameters.items()
              if key not in exclude and param.default is not inspect.Parameter.empty}
    if set(overrides) - set(result):
        raise ValueError("Parametros desconocidos: " + repr(sorted(set(overrides) - set(result))))
    result.update(overrides)
    canonical(result)  # Rechazar valores no reproducibles antes de leer audio.
    return result


def inference_identity(configuration):
    """Separar formato/codigo de exportacion de la inferencia, sin relajar parametros."""
    if configuration.get("schema_version") == 1:
        if configuration.get("script_sha256") not in LEGACY_COMPATIBLE_SCRIPTS:
            return None
    elif (configuration.get("schema_version") != 2
          or configuration.get("inference_revision") != INFERENCE_REVISION):
        return None
    keys = ("python_version", "packages", "model_repository", "model_revision", "model_files",
            "model_constructor", "transcription_call", "dll_helper_sha256")
    return {"inference_revision": INFERENCE_REVISION, **{key: configuration[key] for key in keys}}


def find_complete(output, audio_hash, config_hash, configuration=None):
    if not output.exists():
        return None
    for folder in sorted(output.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        try:
            manifest = json.loads((folder / "completo.json").read_text(encoding="utf-8"))
            if manifest["audio_sha256"] != audio_hash:
                continue
            if manifest["status"] != "complete" or set(manifest["files"]) != set(OUTPUTS):
                continue
            if not all((folder / name).is_file() and not (folder / name).is_symlink()
                       and sha256(folder / name) == manifest["files"][name] for name in OUTPUTS):
                continue
            data = json.loads((folder / "transcripcion.json").read_text(encoding="utf-8"))
            stored_hash = manifest["configuration_sha256"]
            matches = stored_hash == config_hash
            if not matches and configuration is not None:
                previous = inference_identity(data["configuration"])
                current = inference_identity(configuration)
                matches = previous is not None and current is not None and canonical(previous) == canonical(current)
            if (data["generator_exhausted"] is True and data["status"] == "complete"
                    and data["audio"]["sha256"] == audio_hash
                    and data["configuration_sha256"] == stored_hash
                    and digest_json(data["configuration"]) == stored_hash and matches):
                return folder
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, milliseconds = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def validate_segments(segments, duration):
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Duracion total invalida")
    warnings = []
    last_start = -1.0
    last_end = 0.0
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if not all(math.isfinite(value) for value in (start, end)) or start < 0 or end < start or start < last_start:
            raise ValueError("Marcas temporales incoherentes")
        if end > duration + 1.0:
            raise ValueError("Segmento excede la duracion total en mas de un segundo")
        if start < last_end:
            warnings.append("Segmentos con solapamiento temporal; conservar y revisar")
        last_start, last_end = start, end
    if not segments:
        warnings.append("No se genero texto; puede ser silencio u omision y requiere revision")
    return sorted(set(warnings))


def export_results(stage, payload):
    payload = json.loads(canonical(payload))
    segments = payload["segments"]
    text = "".join(item["text"] for item in segments)
    with (stage / OUTPUTS[0]).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text + ("\n" if text else ""))
    with (stage / OUTPUTS[1]).open("x", encoding="utf-8", newline="\n") as stream:
        for index, item in enumerate(segments, 1):
            stream.write(f"{index}\n{timestamp(item['start'])} --> {timestamp(item['end'])}\n{item['text'].strip()}\n\n")
    write_json(stage / OUTPUTS[2], payload)
    # Relectura UTF-8/JSON y coherencia de los tres formatos, sin imprimir contenido.
    if (stage / OUTPUTS[0]).read_bytes().decode("utf-8") != text + ("\n" if text else ""):
        raise ValueError("TXT no coincide con segmentos")
    expected_srt = "".join(f"{i}\n{timestamp(s['start'])} --> {timestamp(s['end'])}\n{s['text'].strip()}\n\n"
                           for i, s in enumerate(segments, 1))
    if (stage / OUTPUTS[1]).read_bytes().decode("utf-8") != expected_srt:
        raise ValueError("SRT no coincide con segmentos")
    if json.loads((stage / OUTPUTS[2]).read_text(encoding="utf-8")) != payload:
        raise ValueError("JSON no coincide con resultado")
    write_json(stage / "completo.json", {
        "status": "complete", "audio_sha256": payload["audio"]["sha256"],
        "configuration_sha256": payload["configuration_sha256"],
        "files": {name: sha256(stage / name) for name in OUTPUTS}, "completed_utc": utc(),
    })
    export_blocks(stage / "transcripcion.json")


def export_blocks(json_path):
    """Salida derivada desde un resultado completo; solo biblioteca estandar, sin audio."""
    json_path = Path(json_path).resolve(strict=True)
    folder = json_path.parent
    if json_path.name != "transcripcion.json":
        raise ValueError("Seleccionar el transcripcion.json del resultado completo")
    original_names = (*OUTPUTS, "completo.json")
    before = {name: sha256(folder / name) for name in original_names}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    manifest = json.loads((folder / "completo.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or data.get("status") != "complete"
            or data.get("generator_exhausted") is not True
            or set(manifest.get("files", {})) != set(OUTPUTS)
            or any(manifest["files"][name] != before[name] for name in OUTPUTS)
            or manifest["audio_sha256"] != data["audio"]["sha256"]
            or manifest["configuration_sha256"] != data["configuration_sha256"]
            or digest_json(data["configuration"]) != data["configuration_sha256"]):
        raise ValueError("Resultado original incompleto o con integridad incorrecta")
    segments = data["segments"]
    validate_segments(segments, data["duration_total_audio_seconds"])
    body = bytearray(("Segmentos automaticos de Whisper. No equivalen necesariamente a turnos ni frases.\n"
                      "Tiempos de inicio aproximados respecto al audio original.\n\n").encode("utf-8"))
    offsets = []
    for index, segment in enumerate(segments, 1):
        body.extend(f"[{timestamp(segment['start'])}] Segmento {index}\n".encode("utf-8"))
        start = len(body)
        text_bytes = segment["text"].encode("utf-8")
        body.extend(text_bytes)  # Sin strip, normalizacion, correccion ni sustitucion de saltos.
        offsets.append({"segment_number": index, "start_seconds": segment["start"],
                        "text_byte_start": start, "text_byte_end": len(body),
                        "text_sha256": hashlib.sha256(text_bytes).hexdigest()})
        body.extend(b"\n\n")
    expected = bytes(body)
    stem = Path(data["audio"]["filename"]).stem
    candidate = folder / (stem + "_bloques.txt")
    receipt = candidate.with_suffix(".registro.json")
    reused = False
    if candidate.is_file() and not candidate.is_symlink() and receipt.is_file():
        try:
            old = json.loads(receipt.read_text(encoding="utf-8"))
            reused = (candidate.read_bytes() == expected and old["source_json_sha256"] == before["transcripcion.json"]
                      and old["blocks"] == offsets and old["output_sha256"] == hashlib.sha256(expected).hexdigest())
        except (OSError, ValueError, KeyError):
            pass
    if not reused:
        if candidate.exists() or receipt.exists():
            candidate = folder / (stem + "_bloques_" + uuid.uuid4().hex[:12] + ".txt")
            receipt = candidate.with_suffix(".registro.json")
        with candidate.open("xb") as stream:
            stream.write(expected)
    actual = candidate.read_bytes()
    if actual != expected:
        raise ValueError("Salida de bloques no coincide con formato esperado")
    for segment, offset in zip(segments, offsets):
        if actual[offset["text_byte_start"]:offset["text_byte_end"]].decode("utf-8") != segment["text"]:
            raise ValueError("Texto de segmento alterado en salida de bloques")
    after = {name: sha256(folder / name) for name in original_names}
    if before != after:
        raise RuntimeError("Los originales cambiaron durante la exportacion")
    if not reused:
        write_json(receipt, {
            "format_version": 1, "created_utc": utc(), "audio_filename": data["audio"]["filename"],
            "audio_sha256": data["audio"]["sha256"], "source_json_sha256": before["transcripcion.json"],
            "output_filename": candidate.name, "output_sha256": sha256(candidate),
            "original_hashes_before": before, "original_hashes_after": after, "blocks": offsets,
            "verification": "Exact segment text including whitespace; originals unchanged; no inference",
        })
    return candidate


def run(config_path, chosen, all_files):
    config, source, output, model_folder = read_config(config_path)
    files = list_audio(source)
    if not all_files:
        files = [p for p in files if p.name == chosen]
        if len(files) != 1:
            raise ValueError("Seleccion no encontrada: debe ser un nombre exacto de la carpeta")
    if not files:
        print("No hay archivos de audio compatibles. Completados: 0 | Omitidos: 0 | Errores: 0")
        return 0
    print("Entrada: " + str(source))
    print("Resultados: " + str(output))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    log_folder = ROOT / "logs" / "ejecuciones"
    log_folder.mkdir(parents=True, exist_ok=True)
    log_path = log_folder / (run_id + ".jsonl")

    def record(event, **values):
        value = {"time_utc": utc(), "event": event, **values}
        text = json.dumps(value, ensure_ascii=False)
        text = text.replace(str(ROOT).replace("\\", "\\\\"), "<WORKSPACE>")
        text = text.replace(os.environ.get("USERPROFILE", "<USERPROFILE>").replace("\\", "\\\\"), "<USERPROFILE>")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    model, handles = None, []
    failed = False
    counts = {"completed": 0, "skipped": 0, "errors": 0}
    model_load_seconds = None
    record("start", selection=[p.name for p in files], mode="folder" if all_files else "single")
    try:
        handles = configure_runtime()
        from faster_whisper import WhisperModel
        transcribe_parameters = effective_parameters(WhisperModel.transcribe, config["transcription"], {"self", "audio"})
        constructor = effective_parameters(WhisperModel.__init__, {
            "device": "cuda", "device_index": 0, "compute_type": "float16",
            "local_files_only": True, "num_workers": 1,
        }, {"self", "model_size_or_path", "model_kwargs"})
        versions = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}
        manifest = model_manifest()
        identity = {
            "schema_version": 2, "inference_revision": INFERENCE_REVISION,
            "python_version": sys.version, "packages": versions,
            "model_repository": config["model_repository"], "model_revision": REVISION,
            "model_files": manifest, "model_constructor": constructor,
            "transcription_call": transcribe_parameters, "script_sha256": sha256(Path(__file__)),
            "dll_helper_sha256": HELPER_SHA256,
        }
        config_hash = digest_json(identity)
        record("configuration", configuration=identity, configuration_sha256=config_hash,
               model_directory=config["model_directory"])
        for index, audio in enumerate(files, 1):
            print(f"Archivo {index}/{len(files)}: {audio.name}", flush=True)
            stage = None
            try:
                audio_identity = fingerprint(audio)
                existing = find_complete(output, audio_identity["sha256"], config_hash, identity)
                if existing:
                    blocks_path = export_blocks(existing / "transcripcion.json")
                    record("blocks_exported", result=str(blocks_path.relative_to(ROOT)))
                    record("skipped_complete", filename=audio.name, result=str(existing.relative_to(ROOT)))
                    print("Resultado completo e integro coincidente: omitido.", flush=True)
                    counts["skipped"] += 1
                    continue
                if model is None:
                    verify_local_model(model_folder, manifest)
                    started = time.perf_counter()
                    model = WhisperModel(str(model_folder), **constructor)
                    model_load_seconds = time.perf_counter() - started
                    if model.model.device != "cuda" or model.model.compute_type != "float16" or model.model.device_index != [0]:
                        model = None
                        raise RuntimeError("El modelo no esta en CUDA:0/float16")
                    record("model_loaded", device=model.model.device, compute_type=model.model.compute_type,
                           device_index=model.model.device_index, load_seconds=model_load_seconds)
                output.mkdir(parents=True, exist_ok=True)
                result_name = audio.stem + "-" + audio_identity["sha256"][:10] + "-" + config_hash[:10] + "-" + uuid.uuid4().hex[:12]
                stage = output / (".parcial-" + result_name)
                stage.mkdir(exist_ok=False)
                started = time.perf_counter()
                generator, info = model.transcribe(str(audio), **transcribe_parameters)
                segments = []
                for segment in generator:
                    segments.append(dataclasses.asdict(segment))
                processing_seconds = time.perf_counter() - started
                warnings = validate_segments(segments, float(info.duration))
                after = fingerprint(audio)
                if audio_identity != after:
                    raise RuntimeError("El original cambio durante la transcripcion; no publicar resultado")
                payload = {
                    "schema_version": 1, "status": "complete", "created_utc": utc(),
                    "audio": {"filename": audio.name, **audio_identity},
                    "configuration_sha256": config_hash, "configuration": identity,
                    "effective_device": model.model.device, "effective_compute_type": model.model.compute_type,
                    "effective_device_index": model.model.device_index,
                    "duration_total_audio_seconds": float(info.duration),
                    "duration_after_vad_seconds": float(info.duration_after_vad),
                    "processing_seconds": processing_seconds,
                    "processing_time_definition": "Decodificacion y transcripcion hasta agotar el generador; excluye carga de modelo, hashes y exportacion",
                    "model_load_seconds_shared": model_load_seconds,
                    "generator_exhausted": True, "transcription_info": dataclasses.asdict(info),
                    "segments": segments, "warnings": warnings, "errors": [],
                    "fidelity_note": "Salida automatica sin corregir. Agotar el generador y el final del ultimo segmento no prueban ausencia de omisiones. Tiempos aproximados respecto al original.",
                }
                export_results(stage, payload)
                destination = output / result_name
                if destination.exists():
                    raise FileExistsError("No sobrescribir resultados existentes")
                stage.rename(destination)
                stage = None
                record("complete", filename=audio.name, audio_sha256=audio_identity["sha256"],
                       configuration_sha256=config_hash, result=str(destination.relative_to(ROOT)),
                       duration_total_audio_seconds=info.duration, processing_seconds=processing_seconds,
                       segment_count=len(segments), warnings=warnings)
                print("Completado: " + str(destination.relative_to(ROOT)), flush=True)
                counts["completed"] += 1
            except Exception:
                failed = True
                counts["errors"] += 1
                error = traceback.format_exc()
                record("audio_error", filename=audio.name, traceback=error,
                       partial_directory=str(stage.relative_to(ROOT)) if stage else None)
                if stage is not None:
                    write_json(stage / "error.json", {"status": "failed", "time_utc": utc(), "error": error})
                print("Fallo; detalles en el registro local. No se declara resultado completo.", flush=True)
                # Fallo de carga afecta a todos: detener; fallo de un audio permite continuar el lote explicito.
                if model is None:
                    break
    except Exception:
        failed = True
        counts["errors"] += 1
        record("fatal_error", traceback=traceback.format_exc())
        print("Fallo de preparacion; consultar registro local.", flush=True)
    finally:
        model = None
        gc.collect()
        for handle in handles:
            handle.close()
        record("end", exit_code=1 if failed else 0, **counts)
        print(f"Resumen: Completados: {counts['completed']} | Omitidos: {counts['skipped']} | Errores: {counts['errors']}", flush=True)
        print("Registro: " + str(log_path.relative_to(ROOT)), flush=True)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configuracion.json")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--listar", action="store_true", help="Solo enumerar; no leer contenido de audios")
    selection.add_argument("--archivo", help="Nombre exacto de UN audio; usar --archivo=nombre si empieza por guion")
    selection.add_argument("--todos", action="store_true", help="Procesar secuencialmente toda la carpeta; requiere autorizacion del usuario")
    selection.add_argument("--bloques-json", type=Path, help="Exportar bloques de un resultado completo sin cargar Whisper ni leer audio")
    args = parser.parse_args(argv)
    args.config = (ROOT / args.config).resolve()
    if args.listar:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        for index, path in enumerate(list_audio((ROOT / config["input_directory"]).resolve()), 1):
            print(f"{index}. {path.name}")
        return 0
    try:
        if args.bloques_json is not None:
            result = export_blocks(ROOT / args.bloques_json)
            print("Bloques verificados: " + str(result))
            return 0
        return run(args.config, args.archivo, args.todos)
    except Exception as exc:
        # Preflight: ninguna grabacion ni contenido se ha abierto todavia.
        print("Error previo a la transcripcion: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
