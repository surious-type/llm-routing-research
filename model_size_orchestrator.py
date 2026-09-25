#!/usr/bin/env python3
"""Sequential Docker orchestration for the reason+route model-size experiment."""

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


IMAGE = "vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
VLLM_VERSION = "0.28.0"
IMPORTANT_CASES = [
    "case_001", "case_014", "case_020", "case_021",
    "case_023", "case_024", "case_029",
]
MODEL_SPECS = [
    {
        "label": "M4",
        "model_id": "Qwen/Qwen3-4B-AWQ",
        "served_model_name": "qwen3-4b",
        "gpu_memory_utilization": 0.90,
        "cpu_offload_gb": 0,
        "max_model_len": 2048,
        "max_num_seqs": 1,
        "enforce_eager": False,
        "quantization": "awq",
    },
    {
        "label": "M8",
        "model_id": "Qwen/Qwen3-8B-AWQ",
        "served_model_name": "qwen3-8b",
        "gpu_memory_utilization": 0.85,
        "cpu_offload_gb": 0,
        "max_model_len": 2048,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "quantization": "awq",
    },
    {
        "label": "M14",
        "model_id": "Qwen/Qwen3-14B-AWQ",
        "served_model_name": "qwen3-14b",
        "gpu_memory_utilization": 0.85,
        "cpu_offload_gb": 4,
        "max_model_len": 2048,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "quantization": "awq",
    },
]


def with_cpu_offload(spec, value):
    updated = copy.deepcopy(spec)
    updated["cpu_offload_gb"] = value
    return updated


def is_gpu_oom(log_text):
    lowered = log_text.lower()
    return "cuda out of memory" in lowered or "outofmemoryerror" in lowered


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


def run(command, *, check=True):
    return subprocess.run(command, text=True, capture_output=True, check=check)


def capture(command, path):
    result = run(command, check=False)
    Path(path).write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def container_logs(name):
    result = run(["docker", "logs", name], check=False)
    return result.stdout + result.stderr


def stop_container(name):
    run(["docker", "stop", "--time", "30", name], check=False)


def remove_container(name):
    run(["docker", "rm", "--force", name], check=False)


def wait_for_model(base_url, served_name, container_name, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        status = run(
            ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
            check=False,
        )
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
    return False, f"startup timeout; last endpoint error: {last_error}"


def stop_port_8000_containers(manifest):
    result = run(["docker", "ps", "--filter", "publish=8000", "--format", "{{.ID}}"], check=True)
    ids = [line for line in result.stdout.splitlines() if line]
    manifest["stopped_preexisting_containers"] = ids
    for container_id in ids:
        stop_container(container_id)


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_router(project_dir, output_dir, spec, metadata_path, smoke):
    stem = f"{spec['label'].lower()}_{'smoke' if smoke else 'full'}"
    command = [
        sys.executable, str(Path(project_dir) / "model_size_routing.py"),
        "--model", spec["served_model_name"],
        "--server-metadata", str(metadata_path),
        "--output", str(Path(output_dir) / f"{stem}.csv"),
        "--json-output", str(Path(output_dir) / f"{stem}.json"),
    ]
    if smoke:
        command.extend(["--case-ids", *IMPORTANT_CASES])
    return run(command, check=False)


def attempt_model(spec, attempt_number, args, output_dir, manifest):
    label = spec["label"]
    name = f"routing-model-size-20260907-{label.lower()}-a{attempt_number}"
    attempt = {
        "attempt": attempt_number,
        "configuration": spec,
        "container_name": name,
        "image": args.image,
        "vllm_version": VLLM_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_dir = output_dir / label.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    before_path = model_dir / f"attempt{attempt_number}_nvidia_before.txt"
    capture(["nvidia-smi"], before_path)
    attempt["nvidia_smi_before"] = str(before_path)
    command = build_docker_command(
        spec, image=args.image, cache_dir=args.cache_dir, container_name=name
    )
    attempt["docker_command"] = command
    started = run(command, check=False)
    attempt["docker_run_stdout"] = started.stdout.strip()
    attempt["docker_run_stderr"] = started.stderr.strip()
    if started.returncode != 0:
        attempt["status"] = "docker_run_failed"
        return attempt, False
    ready, startup_error = wait_for_model(
        args.base_url, spec["served_model_name"], name, args.startup_timeout
    )
    logs = container_logs(name)
    log_path = model_dir / f"attempt{attempt_number}_server.log"
    log_path.write_text(logs, encoding="utf-8")
    attempt["server_log"] = str(log_path)
    attempt["startup_error"] = startup_error
    attempt["gpu_oom"] = is_gpu_oom(logs)
    if not ready:
        attempt["status"] = "startup_failed"
        stop_container(name)
        remove_container(name)
        return attempt, False

    after_load_path = model_dir / f"attempt{attempt_number}_nvidia_after_load.txt"
    capture(["nvidia-smi"], after_load_path)
    attempt["nvidia_smi_after_load"] = str(after_load_path)
    version = run(
        ["docker", "exec", name, "python3", "-c", "import vllm; print(vllm.__version__)"],
        check=False,
    )
    inspect = run(["docker", "inspect", name], check=False)
    metadata = {
        **spec,
        "docker_image": args.image,
        "container_name": name,
        "vllm_version": version.stdout.strip() or VLLM_VERSION,
        "docker_inspect": json.loads(inspect.stdout)[0] if inspect.returncode == 0 else None,
        "nvidia_smi_before": before_path.read_text(encoding="utf-8"),
        "nvidia_smi_after_load": after_load_path.read_text(encoding="utf-8"),
    }
    metadata_path = model_dir / "server_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    attempt["server_metadata"] = str(metadata_path)

    smoke = run_router(args.project_dir, model_dir, spec, metadata_path, True)
    (model_dir / "smoke_runner.log").write_text(smoke.stdout + smoke.stderr, encoding="utf-8")
    attempt["smoke_exit_code"] = smoke.returncode
    if smoke.returncode != 0:
        attempt["status"] = "smoke_failed"
        stop_container(name)
        remove_container(name)
        return attempt, False

    full = run_router(args.project_dir, model_dir, spec, metadata_path, False)
    (model_dir / "full_runner.log").write_text(full.stdout + full.stderr, encoding="utf-8")
    attempt["experiment_exit_code"] = full.returncode
    final_logs = container_logs(name)
    log_path.write_text(final_logs, encoding="utf-8")
    after_experiment_path = model_dir / f"attempt{attempt_number}_nvidia_after_experiment.txt"
    capture(["nvidia-smi"], after_experiment_path)
    attempt["nvidia_smi_after_experiment"] = str(after_experiment_path)
    attempt["status"] = "complete" if full.returncode == 0 else "experiment_failed"
    stop_container(name)
    remove_container(name)
    return attempt, full.returncode == 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--cache-dir", default="/home/surious-type/.cache/huggingface")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "experiment_family": "reason-route-model-size-v1",
        "image": args.image,
        "expected_vllm_version": VLLM_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    capture(["nvidia-smi"], output_dir / "nvidia_initial.txt")
    stop_port_8000_containers(manifest)
    write_manifest(manifest_path, manifest)
    try:
        for original_spec in MODEL_SPECS:
            attempts = []
            attempt, success = attempt_model(
                original_spec, 1, args, output_dir, manifest
            )
            attempts.append(attempt)
            if (
                original_spec["label"] == "M8"
                and not success
                and attempt.get("gpu_oom")
            ):
                fallback = with_cpu_offload(original_spec, 1)
                attempt, success = attempt_model(
                    fallback, 2, args, output_dir, manifest
                )
                attempts.append(attempt)
            manifest["models"][original_spec["label"]] = {
                "success": success,
                "attempts": attempts,
            }
            write_manifest(manifest_path, manifest)
    finally:
        capture(["nvidia-smi"], output_dir / "nvidia_final.txt")
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
