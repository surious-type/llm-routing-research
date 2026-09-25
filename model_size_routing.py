#!/usr/bin/env python3
"""P1/P2 reason+route experiment for one already served model."""

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from structured_factorial_routing import ensure_new_outputs, route_case
from structured_routing import load_jsonl, load_routes, select_cases, verify_model


PROMPT_VERSION = "structured-factorial-v1"


def build_conditions(route_names):
    forward = list(route_names)
    return {
        "P1": {"prompt_order": forward, "enum_order": forward},
        "P2": {"prompt_order": list(reversed(forward)), "enum_order": forward},
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", default="routes_continue_first.json")
    parser.add_argument("--dataset", default="dataset_v2.jsonl")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--server-metadata", required=True)
    return parser.parse_args()


def save_csv(path, metadata, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(metadata) + list(rows[0])
    with output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**metadata, **row})


def main():
    args = parse_args()
    ensure_new_outputs(args.output, args.json_output)
    dataset_path = Path(args.dataset)
    routes_path = Path(args.routes)
    server_metadata_path = Path(args.server_metadata)
    dataset = load_jsonl(dataset_path)
    routes = load_routes(routes_path)
    cases = select_cases(dataset, args.case_ids, 0, None)
    if not cases:
        raise ValueError("После фильтрации не осталось случаев")
    route_names = [route["name"] for route in routes]
    conditions = build_conditions(route_names)
    server_metadata = json.loads(server_metadata_path.read_text(encoding="utf-8"))

    session = requests.Session()
    served_models = verify_model(session, args.base_url, args.model, args.timeout)
    metadata = {
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
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "routes": args.routes,
        "routes_sha256": hashlib.sha256(routes_path.read_bytes()).hexdigest(),
        "case_count_per_condition": len(cases),
        "request_count": len(cases) * len(conditions),
        "server_metadata_file": str(server_metadata_path),
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
                    session, args.base_url, args.model, source_row, routes,
                    condition_name, condition, "reason-route", args.timeout,
                    128, False,
                )
                raw_responses.append(raw)
            except Exception as exc:
                result = {
                    "condition": condition_name,
                    "mode": "reason-route",
                    "id": source_row["id"],
                    "difficulty": source_row.get("difficulty"),
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
            rows.append(result)
            print(
                f"{condition_name} {result['id']}: "
                f"predicted={result['predicted_route']} "
                f"expected={result['expected_route']} error={result['error']}"
            )

    save_csv(args.output, metadata, rows)
    Path(args.json_output).write_text(
        json.dumps(
            {"metadata": metadata, "rows": rows, "raw_responses": raw_responses},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    errors = [row for row in rows if row["error"]]
    if errors:
        raise RuntimeError(f"Эксперимент неполный: ошибок {len(errors)} из {len(rows)}")
    for condition_name in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition_name]
        correct = sum(row["correct"] for row in condition_rows)
        latency = statistics.mean(row["request_latency_sec"] for row in condition_rows)
        print(f"{condition_name}: accuracy={correct}/{len(condition_rows)}, mean_latency={latency:.6f}s")


if __name__ == "__main__":
    main()
