#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


IMPORTANT_CASES = ["case_017", "case_020", "case_021", "case_023", "case_024", "case_029"]


def load_csv(path):
    with open(path, encoding="utf-8") as file:
        return {row["id"]: row for row in csv.DictReader(file)}


def route_scores(row):
    return {item["name"]: float(item["score"]) for item in json.loads(row["routes_json"])}


def is_true(value):
    return value.strip().lower() == "true"


def run_summary(rows):
    valid = [row for row in rows.values() if not row.get("error")]
    errors = [row["id"] for row in rows.values() if row.get("error")]
    return {
        "cases": len(rows),
        "successful": len(valid),
        "error_count": len(errors),
        "error_cases": sorted(errors),
        "correct": sum(is_true(row["correct"]) for row in valid),
        "accuracy": sum(is_true(row["correct"]) for row in valid) / len(valid) if valid else None,
        "mean_latency_sec": statistics.mean(float(row["request_latency_sec"]) for row in valid) if valid else None,
        "median_latency_sec": statistics.median(float(row["request_latency_sec"]) for row in valid) if valid else None,
        "mean_margin": statistics.mean(float(row["margin"]) for row in valid) if valid else None,
        "median_margin": statistics.median(float(row["margin"]) for row in valid) if valid else None,
    }


def compare_rows(run_a, run_b):
    if set(run_a) != set(run_b):
        raise ValueError("Наборы идентификаторов Run A и Run B не совпадают")
    ids = sorted(run_a)
    error_cases = [case_id for case_id in ids if run_a[case_id].get("error") or run_b[case_id].get("error")]
    valid_ids = [case_id for case_id in ids if case_id not in error_cases]
    disagreements = [case_id for case_id in valid_ids if run_a[case_id]["predicted_route"] != run_b[case_id]["predicted_route"]]
    score_differences = []
    for case_id in valid_ids:
        scores_a = route_scores(run_a[case_id])
        scores_b = route_scores(run_b[case_id])
        if set(scores_a) != set(scores_b):
            raise ValueError(f"Наборы маршрутов не совпадают для {case_id}")
        score_differences.append({
            "id": case_id,
            "score_differences": {name: scores_a[name] - scores_b[name] for name in sorted(scores_a)},
            "run_a_scores": scores_a,
            "run_b_scores": scores_b,
        })
    important = {}
    for case_id in IMPORTANT_CASES:
        if case_id in run_a and case_id in valid_ids:
            important[case_id] = {
                "expected_route": run_a[case_id]["expected_route"],
                "run_a_prediction": run_a[case_id]["predicted_route"],
                "run_b_prediction": run_b[case_id]["predicted_route"],
                "run_a_scores": route_scores(run_a[case_id]),
                "run_b_scores": route_scores(run_b[case_id]),
                "run_a_margin": float(run_a[case_id]["margin"]),
                "run_b_margin": float(run_b[case_id]["margin"]),
            }
    return {
        "run_a": run_summary(run_a),
        "run_b": run_summary(run_b),
        "semantic_agreement": {"count": len(valid_ids) - len(disagreements), "total": len(valid_ids), "rate": (len(valid_ids) - len(disagreements)) / len(valid_ids) if valid_ids else None},
        "order_sensitive_cases": disagreements,
        "uncompared_error_cases": error_cases,
        "per_case_score_differences": score_differences,
        "important_cases": important,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_rows(load_csv(args.run_a), load_csv(args.run_b))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
