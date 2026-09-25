#!/usr/bin/env python3
"""Analyze P1/P2 reason+route results across model sizes."""

import argparse
import itertools
import json
from pathlib import Path

from analyze_structured_factorial import (
    compare_condition_pair,
    labels_from_rows,
    load_csv,
    summarize_condition,
)


IMPORTANT_CASES = (
    "case_001", "case_014", "case_020", "case_021",
    "case_023", "case_024", "case_029",
)


def _group(rows):
    return {name: [row for row in rows if row["condition"] == name] for name in ("P1", "P2")}


def summarize_model(rows, labels):
    grouped = _group(rows)
    p1 = summarize_condition(grouped["P1"], labels)
    p2 = summarize_condition(grouped["P2"], labels)
    agreement = compare_condition_pair(grouped["P1"], grouped["P2"])
    accuracies = [value for value in (p1["accuracy"], p2["accuracy"]) if value is not None]
    return {
        "P1": p1,
        "P2": p2,
        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "semantic_agreement": agreement,
        "order_sensitive_case_count": agreement["changed_count"],
        "order_sensitive_cases": agreement["changed_cases"],
    }


def _case_errors(rows):
    errors = {}
    for row in rows:
        if row.get("error") or not row.get("correct"):
            errors.setdefault(row["id"], []).append(row["condition"])
    return errors


def compare_models(smaller_rows, larger_rows):
    smaller_errors = _case_errors(smaller_rows)
    larger_errors = _case_errors(larger_rows)
    return {
        "fixed_cases": sorted(set(smaller_errors) - set(larger_errors)),
        "new_error_cases": sorted(set(larger_errors) - set(smaller_errors)),
        "persistent_error_cases": sorted(set(smaller_errors) & set(larger_errors)),
        "smaller_errors_by_case": smaller_errors,
        "larger_errors_by_case": larger_errors,
    }


def parse_model(value):
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Формат: LABEL=results.csv") from exc
    return label, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Файл уже существует: {output}")
    models = {label: load_csv(path) for label, path in args.model}
    labels = labels_from_rows(next(iter(models.values())))
    summaries = {label: summarize_model(rows, labels) for label, rows in models.items()}
    comparisons = {
        f"{left}→{right}": compare_models(models[left], models[right])
        for left, right in itertools.combinations(models, 2)
    }
    important = {}
    for case_id in IMPORTANT_CASES:
        important[case_id] = {}
        for label, rows in models.items():
            important[case_id][label] = {
                condition: next(
                    (
                        {
                            "expected_route": row["expected_route"],
                            "predicted_route": row["predicted_route"],
                            "correct": row["correct"],
                            "error": row.get("error"),
                        }
                        for row in rows
                        if row["id"] == case_id and row["condition"] == condition
                    ),
                    None,
                )
                for condition in ("P1", "P2")
            }
    result = {
        "manifest": json.loads(Path(args.manifest).read_text(encoding="utf-8")),
        "models": summaries,
        "comparisons": comparisons,
        "important_cases": important,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        label: {
            "P1_accuracy": summary["P1"]["accuracy"],
            "P2_accuracy": summary["P2"]["accuracy"],
            "mean_accuracy": summary["mean_accuracy"],
            "agreement": summary["semantic_agreement"]["agreement_rate"],
            "order_sensitive_cases": summary["order_sensitive_case_count"],
        }
        for label, summary in summaries.items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
