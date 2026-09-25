#!/usr/bin/env python3
"""Сравнение двух запусков прямой структурированной маршрутизации."""

import argparse
import csv
import json
import statistics
from pathlib import Path


IMPORTANT_CASES = {
    "case_017", "case_020", "case_021", "case_023", "case_024", "case_029"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _optional_bool(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["correct"] = _optional_bool(row.get("correct"))
        row["error"] = row.get("error") or None
        if row.get("request_latency_sec"):
            row["request_latency_sec"] = float(row["request_latency_sec"])
    return rows


def summarize_run(rows, labels):
    usable = [row for row in rows if not row.get("error")]
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in usable:
        matrix[index[row["expected_route"]]][index[row["predicted_route"]]] += 1
    correct = sum(bool(row["correct"]) for row in usable)
    latencies = [row["request_latency_sec"] for row in usable if row.get("request_latency_sec") is not None]
    return {
        "correct": correct,
        "evaluated": len(usable),
        "accuracy": correct / len(usable) if usable else None,
        "errors": [row["id"] for row in rows if row.get("error")],
        "incorrect_cases": [row["id"] for row in usable if not row["correct"]],
        "confusion_matrix": {"labels": labels, "matrix": matrix},
        "mean_latency_sec": statistics.mean(latencies) if latencies else None,
        "median_latency_sec": statistics.median(latencies) if latencies else None,
    }


def compare_runs(run_a, run_b, labels):
    by_a = {row["id"]: row for row in run_a}
    by_b = {row["id"]: row for row in run_b}
    if set(by_a) != set(by_b):
        raise ValueError("Наборы case id в запусках не совпадают")
    comparable = [
        case_id for case_id in by_a
        if not by_a[case_id].get("error") and not by_b[case_id].get("error")
    ]
    changed = [
        case_id for case_id in comparable
        if by_a[case_id]["predicted_route"] != by_b[case_id]["predicted_route"]
    ]
    agreed = len(comparable) - len(changed)
    return {
        "run_a": summarize_run(run_a, labels),
        "run_b": summarize_run(run_b, labels),
        "semantic_agreement": {
            "count": agreed,
            "total": len(comparable),
            "rate": agreed / len(comparable) if comparable else None,
        },
        "order_sensitive_cases": changed,
        "uncompared_error_cases": [
            case_id for case_id in by_a
            if by_a[case_id].get("error") or by_b[case_id].get("error")
        ],
        "important_cases": [
            {
                "id": case_id,
                "expected_route": by_a[case_id]["expected_route"],
                "predicted_route_a": by_a[case_id]["predicted_route"],
                "predicted_route_b": by_b[case_id]["predicted_route"],
                "correct_a": by_a[case_id]["correct"],
                "correct_b": by_b[case_id]["correct"],
                "error_a": by_a[case_id].get("error"),
                "error_b": by_b[case_id].get("error"),
            }
            for case_id in sorted(IMPORTANT_CASES & set(by_a))
        ],
    }


def route_labels(rows):
    raw = rows[0].get("route_names_json") if rows else None
    if raw:
        return json.loads(raw)
    labels = []
    for row in rows:
        for key in ("expected_route", "predicted_route"):
            value = row.get(key)
            if value and value not in labels:
                labels.append(value)
    return labels


def main():
    args = parse_args()
    run_a = load_csv(args.run_a)
    run_b = load_csv(args.run_b)
    labels = route_labels(run_a)
    result = compare_runs(run_a, run_b, labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
