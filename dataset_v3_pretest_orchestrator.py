#!/usr/bin/env python3
"""Sequential fixed-config Docker orchestration for Dataset V3 pre-test."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


IMAGE = "vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
VLLM_VERSION = "0.28.0"
ROUTES_SHA256 = "cfdff764daf9e962e49fb05003148c1469462fc1e0558573cf08a02430da4561"
DATASET_HASHES = {
    "development": "be113c460c0534fcd434206000281fbaed9eb356a2b8b2ff6c710d38eef569af",
    "validation": "22cfce2bebe1a99754605c1776a480edd575640d1639ee6638aecfb71ccc78dc",
}
SMOKE_IDS = [
    "v3_development_0003", "v3_development_0006",
    "v3_development_0002", "v3_development_0001",
    "v3_development_0011", "v3_development_0005",
]
MODEL_SPECS = [
    {
        "label": "M4", "model_id": "Qwen/Qwen3-4B-AWQ", "served_model_name": "qwen3-4b",
        "gpu_memory_utilization": 0.90, "cpu_offload_gb": 0, "enforce_eager": False,
        "max_model_len": 2048, "max_num_seqs": 1, "quantization": "awq",
        "splits": ["development", "validation"],
    },
    {
        "label": "M8", "model_id": "Qwen/Qwen3-8B-AWQ", "served_model_name": "qwen3-8b",
        "gpu_memory_utilization": 0.85, "cpu_offload_gb": 0, "enforce_eager": True,
        "max_model_len": 2048, "max_num_seqs": 1, "quantization": "awq",
        "splits": ["development", "validation"],
    },
    {
        "label": "M14", "model_id": "Qwen/Qwen3-14B-AWQ", "served_model_name": "qwen3-14b",
        "gpu_memory_utilization": 0.85, "cpu_offload_gb": 4, "enforce_eager": True,
        "max_model_len": 2048, "max_num_seqs": 1, "quantization": "awq",
        "splits": ["validation"],
    },
]


def build_docker_command(spec, *, image, cache_dir, container_name):
    command = [
        "docker", "run", "--detach", "--name", container_name,
        "--gpus", "all", "--ipc", "host", "--publish", "8000:8000",
        "--volume", f"{cache_dir}:/root/.cache/huggingface:rw",
        image,
        "--model", spec["model_id"],
        "--served-model-name", spec["served_model_name"],
        "--host", "0.0.0.0", "--port", "8000",
        "--quantization", spec["quantization"],
        "--gpu-memory-utilization", str(spec["gpu_memory_utilization"]),
        "--cpu-offload-gb", str(spec["cpu_offload_gb"]),
        "--max-model-len", str(spec["max_model_len"]),
        "--max-num-seqs", str(spec["max_num_seqs"]),
        "--generation-config", "vllm",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
    ]
    if spec["enforce_eager"]:
        command.append("--enforce-eager")
    return command


def run(command, check=True):
    return subprocess.run(command, text=True, capture_output=True, check=check)


def capture(command, path):
    result = run(command, check=False)
    Path(path).write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def stop_container(name):
    run(["docker", "stop", "--time", "30", name], check=False)
    run(["docker", "rm", "--force", name], check=False)


def wait_for_model(base_url, served_name, container_name, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        status = run(["docker", "inspect", "--format", "{{.State.Status}}", container_name], check=False)
        if status.returncode != 0 or status.stdout.strip() == "exited":
            return False, "container exited"
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            response.raise_for_status()
            names = [item["id"] for item in response.json().get("data", [])]
            if served_name in names:
                return True, None
            last_error = f"served models: {names}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(5)
    return False, f"startup timeout: {last_error}"


def run_router(args, spec, model_dir, metadata_path, *, smoke=False, split="development"):
    output_stem = f"{spec['label'].lower()}_smoke" if smoke else f"{spec['label'].lower()}_{split}"
    output_parent = model_dir if smoke else Path(args.output_dir)
    command = [
        sys.executable,
        str(Path(args.project_dir) / "dataset_v3_pretest_routing.py"),
        "--routes", str(Path(args.project_dir) / "routes_continue_first.json"),
        "--dataset", str(Path(args.project_dir) / f"dataset_v3_{split}.jsonl"),
        "--expected-dataset-sha256", DATASET_HASHES[split],
        "--expected-routes-sha256", ROUTES_SHA256,
        "--model", spec["served_model_name"],
        "--base-url", args.base_url,
        "--server-metadata", str(metadata_path),
        "--output", str(output_parent / f"{output_stem}.csv"),
        "--json-output", str(output_parent / f"{output_stem}.json"),
    ]
    if smoke:
        command.extend(["--case-ids", *SMOKE_IDS])
    return command, run(command, check=False), output_parent / f"{output_stem}.json"


def classify_runner_result(result, json_path, expected_rows):
    path = Path(json_path)
    if not path.exists():
        return {"acceptable": False, "status": "missing_json_output", "response_error_count": None}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"acceptable": False, "status": "invalid_json_output", "response_error_count": None}
    rows = document.get("rows") or []
    if len(rows) != expected_rows:
        return {"acceptable": False, "status": "incomplete_rows", "response_error_count": None}
    errors = [row for row in rows if row.get("error")]
    if not errors:
        return {"acceptable": True, "status": "complete", "response_error_count": 0}
    raw_keys = {
        (raw.get("condition"), raw.get("id"))
        for raw in (document.get("raw_responses") or [])
        if raw.get("response") is not None
    }
    response_level = all((row.get("condition"), row.get("id")) in raw_keys for row in errors)
    if errors and response_level:
        return {
            "acceptable": True,
            "status": "complete_with_response_errors",
            "response_error_count": len(errors),
        }
    return {"acceptable": False, "status": "runner_failed", "response_error_count": len(errors)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_model_dir(output_dir, label):
    base = Path(output_dir) / "servers" / label.lower()
    if not base.exists():
        return base
    index = 1
    while (Path(output_dir) / "servers" / f"{label.lower()}_resume{index}").exists():
        index += 1
    return Path(output_dir) / "servers" / f"{label.lower()}_resume{index}"


def execute_model(args, spec, pending_splits):
    label = spec["label"]
    model_dir = _next_model_dir(args.output_dir, label)
    model_dir.mkdir(parents=True, exist_ok=False)
    container_name = f"dataset-v3-pretest-20260909-{model_dir.name}"
    record = {
        "configuration": spec,
        "container_name": container_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
    }
    capture(["nvidia-smi"], model_dir / "nvidia_before.txt")
    command = build_docker_command(spec, image=args.image, cache_dir=args.cache_dir, container_name=container_name)
    record["docker_command"] = command
    started = run(command, check=False)
    record["docker_run_stdout"] = started.stdout.strip()
    record["docker_run_stderr"] = started.stderr.strip()
    if started.returncode != 0:
        record["status"] = "docker_run_failed"
        return record, False
    try:
        ready, error = wait_for_model(args.base_url, spec["served_model_name"], container_name, args.startup_timeout)
        record["startup_error"] = error
        (model_dir / "server.log").write_text(run(["docker", "logs", container_name], check=False).stdout + run(["docker", "logs", container_name], check=False).stderr, encoding="utf-8")
        if not ready:
            record["status"] = "startup_failed"
            return record, False
        capture(["nvidia-smi"], model_dir / "nvidia_after_load.txt")
        version = run(["docker", "exec", container_name, "python3", "-c", "import vllm; print(vllm.__version__)"], check=False)
        inspect = run(["docker", "inspect", container_name], check=False)
        metadata = {
            **spec,
            "docker_image": args.image,
            "container_name": container_name,
            "vllm_version": version.stdout.strip() or VLLM_VERSION,
            "docker_inspect": json.loads(inspect.stdout)[0] if inspect.returncode == 0 else None,
            "nvidia_smi_before": (model_dir / "nvidia_before.txt").read_text(encoding="utf-8"),
            "nvidia_smi_after_load": (model_dir / "nvidia_after_load.txt").read_text(encoding="utf-8"),
        }
        metadata_path = model_dir / "server_metadata.json"
        write_json(metadata_path, metadata)

        smoke_command, smoke, smoke_json = run_router(args, spec, model_dir, metadata_path, smoke=True)
        record["smoke_command"] = smoke_command
        record["smoke_exit_code"] = smoke.returncode
        (model_dir / "smoke_runner.log").write_text(smoke.stdout + smoke.stderr, encoding="utf-8")
        if smoke.returncode != 0:
            record["status"] = "smoke_failed"
            return record, False

        record["split_runs"] = {}
        for split in pending_splits:
            full_command, full, full_json = run_router(args, spec, model_dir, metadata_path, split=split)
            (model_dir / f"{split}_runner.log").write_text(full.stdout + full.stderr, encoding="utf-8")
            classification = classify_runner_result(full, full_json, expected_rows=(240 if split == "development" else 120))
            record["split_runs"][split] = {"command": full_command, "exit_code": full.returncode, **classification}
            if not classification["acceptable"]:
                record["status"] = f"{split}_failed"
                return record, False
        capture(["nvidia-smi"], model_dir / "nvidia_after_experiment.txt")
        record["status"] = "complete"
        return record, True
    finally:
        logs = run(["docker", "logs", container_name], check=False)
        (model_dir / "server.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        stop_container(container_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--cache-dir", default="/home/surious-type/.cache/huggingface")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"Manifest уже существует: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if args.resume else {
        "experiment": "dataset-v3-pretest-v1",
        "protocol_sha256": "75e3542f0c8fbc8d2e3c1aef3a5558802b62e86566aa15811d286289a71687da",
        "docker_image": args.image,
        "vllm_version": VLLM_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "test_status": "NOT RUN",
    }
    run_number = len(manifest.get("resume_events", [])) + (1 if args.resume else 0)
    snapshot_prefix = f"resume{run_number}_" if args.resume else ""
    capture(["nvidia-smi"], output_dir / f"{snapshot_prefix}nvidia_initial.txt")
    active = run(["docker", "ps", "--filter", "publish=8000", "--format", "{{.ID}}"], check=True)
    preexisting = [line for line in active.stdout.splitlines() if line]
    manifest["stopped_preexisting_containers"] = preexisting
    for container_id in preexisting:
        run(["docker", "stop", "--time", "30", container_id], check=True)
    write_json(manifest_path, manifest)
    try:
        for spec in MODEL_SPECS:
            pending_splits = []
            reused = {}
            for split in spec["splits"]:
                json_path = output_dir / f"{spec['label'].lower()}_{split}.json"
                classification = classify_runner_result(type("Result", (), {"returncode": 1})(), json_path, 240 if split == "development" else 120)
                if classification["acceptable"]:
                    reused[split] = classification
                else:
                    pending_splits.append(split)
            if not pending_splits:
                continue
            record, success = execute_model(args, spec, pending_splits)
            record["reused_existing_splits"] = reused
            if spec["label"] in manifest["models"]:
                record["previous_record"] = manifest["models"][spec["label"]]
            manifest["models"][spec["label"]] = record
            write_json(manifest_path, manifest)
            if not success:
                raise RuntimeError(f"{spec['label']} condition failed: {record['status']}")
    finally:
        capture(["nvidia-smi"], output_dir / f"{snapshot_prefix}nvidia_final.txt")
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if args.resume:
            manifest.setdefault("resume_events", []).append({"run_number": run_number, "finished_at_utc": manifest["finished_at_utc"]})
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
