#!/usr/bin/env python3
"""Run fixed P1/P2 structured reason+route on one Dataset V3 split."""

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from structured_factorial_routing import build_payload, route_case
from structured_routing import load_routes, verify_model


PROMPT_VERSION = "structured-factorial-v1"
EVALUATION_FIELDS = (
    "id", "expected_route", "split", "category", "difficulty", "ambiguity",
    "domain", "family_id", "policy_version", "annotation_notes",
)
OUTPUT_EVALUATION_FIELDS = (
    "split", "category", "difficulty", "ambiguity", "domain", "family_id",
    "policy_version",
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl_strict(path):
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                raise ValueError(f"Пустая строка {line_number} в {path}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Невалидный JSON, строка {line_number} в {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Строка {line_number} не является объектом")
            rows.append(row)
    return rows


def build_conditions(route_names):
    forward = list(route_names)
    return {
        "P1": {"prompt_order": forward, "enum_order": forward},
        "P2": {"prompt_order": list(reversed(forward)), "enum_order": forward},
    }


def select_conditions(conditions, requested):
    if not requested:
        return conditions
    unknown = sorted(set(requested) - set(conditions))
    if unknown:
        raise ValueError(f"Неизвестные conditions: {unknown}")
    return {name: conditions[name] for name in requested}


def prompt_bytes(model, row, routes, condition):
    payload = build_payload(
        model,
        row,
        routes,
        condition,
        "reason-route",
        max_completion_tokens=128,
        enable_thinking=False,
    )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def attach_evaluation_metadata(result, source_row):
    attached = dict(result)
    for field in OUTPUT_EVALUATION_FIELDS:
        attached[field] = source_row[field]
    attached.pop("annotation_notes", None)
    return attached


def _ensure_new(*paths):
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(f"Outputs уже существуют: {existing}")


def _save_csv(path, metadata, rows):
    fields = list(metadata) + list(rows[0])
    with Path(path).open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**metadata, **row})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", default="routes_continue_first.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-routes-sha256", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-metadata", required=True)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    _ensure_new(args.output, args.json_output)
    actual_dataset_hash = file_sha256(args.dataset)
    actual_routes_hash = file_sha256(args.routes)
    if actual_dataset_hash != args.expected_dataset_sha256:
        raise RuntimeError("Dataset SHA-256 не совпадает с protocol")
    if actual_routes_hash != args.expected_routes_sha256:
        raise RuntimeError("Routes SHA-256 не совпадает с protocol")

    dataset = load_jsonl_strict(args.dataset)
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [row for row in dataset if row["id"] in wanted]
        missing = sorted(wanted - {row["id"] for row in cases})
        if missing:
            raise ValueError(f"Неизвестные case ids: {missing}")
    else:
        cases = dataset
    routes = load_routes(args.routes)
    route_names = [route["name"] for route in routes]
    conditions = select_conditions(build_conditions(route_names), args.conditions)
    server_metadata = json.loads(Path(args.server_metadata).read_text(encoding="utf-8"))

    session = requests.Session()
    served_models = verify_model(session, args.base_url, args.model, args.timeout)
    metadata = {
        "experiment": "dataset-v3-pretest-v1",
        "model": args.model,
        "served_models_json": json.dumps(served_models, ensure_ascii=False),
        "base_url": args.base_url,
        "prompt_version": PROMPT_VERSION,
        "mode": "reason-route",
        "temperature": 0,
        "seed": 0,
        "thinking_enabled": False,
        "max_completion_tokens": 128,
        "structured_output": "json_schema",
        "route_names_json": json.dumps(route_names, ensure_ascii=False),
        "conditions_json": json.dumps(conditions, ensure_ascii=False),
        "dataset": args.dataset,
        "dataset_sha256": actual_dataset_hash,
        "routes": args.routes,
        "routes_sha256": actual_routes_hash,
        "case_count_per_condition": len(cases),
        "request_count": len(cases) * len(conditions),
        "server_metadata_file": args.server_metadata,
        "server_configuration_json": json.dumps(server_metadata, ensure_ascii=False),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    rows = []
    raw_responses = []
    for condition_name, condition in conditions.items():
        for source_row in cases:
            started = time.perf_counter()
            try:
                result, raw = route_case(
                    session,
                    args.base_url,
                    args.model,
                    source_row,
                    routes,
                    condition_name,
                    condition,
                    "reason-route",
                    args.timeout,
                    128,
                    False,
                )
                raw_responses.append(raw)
            except Exception as exc:
                result = {
                    "condition": condition_name,
                    "mode": "reason-route",
                    "id": source_row["id"],
                    "message": source_row["message"],
                    "expected_route": source_row["expected_route"],
                    "predicted_route": None,
                    "correct": None,
                    "reason": None,
                    "prompt_route_order": json.dumps(condition["prompt_order"], ensure_ascii=False),
                    "enum_route_order": json.dumps(condition["enum_order"], ensure_ascii=False),
                    "structured_response": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result["request_latency_sec"] = time.perf_counter() - started
            rows.append(attach_evaluation_metadata(result, source_row))
            print(
                f"{condition_name} {source_row['id']}: "
                f"predicted={result.get('predicted_route')} "
                f"expected={source_row['expected_route']} error={result.get('error')}"
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    _save_csv(args.output, metadata, rows)
    Path(args.json_output).write_text(
        json.dumps({"metadata": metadata, "rows": rows, "raw_responses": raw_responses}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise RuntimeError(f"Эксперимент неполный: ошибок {len(errors)} из {len(rows)}")
    for condition_name in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition_name]
        correct = sum(bool(row["correct"]) for row in condition_rows)
        mean_latency = statistics.mean(row["request_latency_sec"] for row in condition_rows)
        print(f"{condition_name}: accuracy={correct}/{len(condition_rows)}, mean_latency={mean_latency:.6f}s")


if __name__ == "__main__":
    main()
