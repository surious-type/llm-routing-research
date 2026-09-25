#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path

IMPORTANT = ["case_017", "case_020", "case_021", "case_023", "case_024", "case_029"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def as_bool(value):
    return value.strip().lower() == "true"


def candidate_map(row):
    return {x["name"]: x for x in json.loads(row["candidates_json"])}


def summarize(label, rows):
    correct = sum(as_bool(r["correct"]) for r in rows.values())
    return {"label": label, "cases": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None,
            "mean_latency_sec": sum(float(r["latency_sec"]) for r in rows.values()) / len(rows) if rows else None,
            "median_latency_sec": statistics.median(float(r["latency_sec"]) for r in rows.values()) if rows else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    a, b = load(args.run_a), load(args.run_b)
    ids = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise RuntimeError("Runs do not contain the same case ids")
    disagreements = [i for i in ids if a[i]["predicted_route"] != b[i]["predicted_route"]]
    errors = {"run_a": [i for i in ids if not as_bool(a[i]["correct"])], "run_b": [i for i in ids if not as_bool(b[i]["correct"])]}
    required = {}
    for i in IMPORTANT:
        if i in a:
            required[i] = {"expected": a[i]["expected_route"], "run_a": a[i]["predicted_route"], "run_b": b[i]["predicted_route"],
                           "run_a_candidates": candidate_map(a[i]), "run_b_candidates": candidate_map(b[i]),
                           "run_a_margin": float(a[i]["score_margin"]), "run_b_margin": float(b[i]["score_margin"])}
    result = {"run_a": summarize("A", a), "run_b": summarize("B", b), "semantic_agreement": {"count": len(ids) - len(disagreements), "total": len(ids), "rate": (len(ids) - len(disagreements)) / len(ids)}, "disagreements": disagreements, "errors": errors, "important_cases": required, "inputs": {"run_a": args.run_a, "run_b": args.run_b}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
