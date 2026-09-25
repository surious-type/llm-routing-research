#!/usr/bin/env python3
"""Анализ факторного structured routing и сравнение route-only/reason+route."""

import argparse
import csv
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path


CONDITIONS = ("E1", "E2", "E3", "E4")
IMPORTANT_CASES = (
    "case_017", "case_020", "case_021", "case_023", "case_024", "case_029"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-only", required=True)
    parser.add_argument("--reason-route", required=True)
    parser.add_argument("--factorial-output", required=True)
    parser.add_argument("--reason-output", required=True)
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
        row["reason"] = row.get("reason") or None
        for field in ("request_latency_sec", "prompt_tokens", "completion_tokens"):
            if row.get(field) not in (None, ""):
                row[field] = float(row[field])
    return rows


def labels_from_rows(rows):
    if rows and rows[0].get("route_names_json"):
        return json.loads(rows[0]["route_names_json"])
    labels = []
    for row in rows:
        for key in ("expected_route", "predicted_route"):
            value = row.get(key)
            if value and value not in labels:
                labels.append(value)
    return labels


def summarize_condition(rows, labels):
    usable = [row for row in rows if not row.get("error")]
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in usable:
        matrix[positions[row["expected_route"]]][positions[row["predicted_route"]]] += 1
    correct = sum(bool(row["correct"]) for row in usable)
    prediction_counts = Counter(row["predicted_route"] for row in usable)

    def stats(field):
        values = [row[field] for row in rows if row.get(field) is not None]
        return {
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }

    return {
        "correct": correct,
        "evaluated": len(usable),
        "accuracy": correct / len(usable) if usable else None,
        "prediction_counts": {label: prediction_counts.get(label, 0) for label in labels},
        "confusion_matrix": {"labels": labels, "matrix": matrix},
        "errors": [row["id"] for row in rows if row.get("error")],
        "incorrect_cases": [row["id"] for row in usable if not row["correct"]],
        "latency_sec": stats("request_latency_sec"),
        "prompt_tokens": stats("prompt_tokens"),
        "completion_tokens": stats("completion_tokens"),
    }


def compare_condition_pair(left_rows, right_rows):
    left = {row["id"]: row for row in left_rows}
    right = {row["id"]: row for row in right_rows}
    if set(left) != set(right):
        raise ValueError("Наборы case id в условиях не совпадают")
    comparable = [
        case_id for case_id in left
        if not left[case_id].get("error") and not right[case_id].get("error")
    ]
    changed = [
        case_id for case_id in comparable
        if left[case_id]["predicted_route"] != right[case_id]["predicted_route"]
    ]
    agreement = len(comparable) - len(changed)
    return {
        "agreement_count": agreement,
        "total": len(comparable),
        "agreement_rate": agreement / len(comparable) if comparable else None,
        "changed_count": len(changed),
        "changed_cases": changed,
        "uncompared_error_cases": [
            case_id for case_id in left
            if left[case_id].get("error") or right[case_id].get("error")
        ],
    }


def _group_conditions(rows):
    return {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in CONDITIONS
    }


def analyze_factorial(rows, labels):
    grouped = _group_conditions(rows)
    summaries = {
        condition: summarize_condition(grouped[condition], labels)
        for condition in CONDITIONS
    }
    pairwise = {}
    for left, right in itertools.combinations(CONDITIONS, 2):
        pairwise[f"{left}↔{right}"] = compare_condition_pair(
            grouped[left], grouped[right]
        )
    prompt_effect = {
        pair: pairwise[pair] for pair in ("E1↔E2", "E3↔E4")
    }
    enum_effect = {
        pair: pairwise[pair] for pair in ("E1↔E3", "E2↔E4")
    }
    prompt_first = set(pairwise["E1↔E2"]["changed_cases"])
    prompt_second = set(pairwise["E3↔E4"]["changed_cases"])
    enum_first = set(pairwise["E1↔E3"]["changed_cases"])
    enum_second = set(pairwise["E2↔E4"]["changed_cases"])
    accuracies = [summaries[condition]["accuracy"] for condition in CONDITIONS]
    did = None
    if all(value is not None for value in accuracies):
        did = (
            summaries["E4"]["accuracy"] - summaries["E3"]["accuracy"]
            - summaries["E2"]["accuracy"] + summaries["E1"]["accuracy"]
        )

    by_case = {}
    for row in rows:
        by_case.setdefault(row["id"], {})[row["condition"]] = row
    order_sensitive = []
    robustness_evaluated = 0
    for case_id, condition_rows in by_case.items():
        valid = [condition_rows.get(condition) for condition in CONDITIONS]
        if all(item and not item.get("error") for item in valid):
            robustness_evaluated += 1
            if len({item["predicted_route"] for item in valid}) > 1:
                order_sensitive.append(case_id)
    return {
        "conditions": summaries,
        "pairwise_agreement": pairwise,
        "prompt_order_effect": prompt_effect,
        "enum_order_effect": enum_effect,
        "interaction": {
            "accuracy_difference_in_differences": did,
            "case_level_prompt_effect_symmetric_difference": sorted(prompt_first ^ prompt_second),
            "case_level_enum_effect_symmetric_difference": sorted(enum_first ^ enum_second),
        },
        "order_robustness_evaluated_cases": robustness_evaluated,
        "order_sensitive_case_count": len(order_sensitive) if robustness_evaluated else None,
        "order_sensitive_cases": order_sensitive,
    }


def compare_modes(s1_rows, s2_rows, labels):
    s1_grouped = _group_conditions(s1_rows)
    s2_grouped = _group_conditions(s2_rows)
    by_condition = {}
    all_changed = []
    all_corrected = []
    all_broken = []
    comparable_count = 0
    for condition in CONDITIONS:
        s1 = {row["id"]: row for row in s1_grouped[condition]}
        s2 = {row["id"]: row for row in s2_grouped[condition]}
        if not s1 and not s2:
            continue
        if set(s1) != set(s2):
            raise ValueError(f"Наборы case id S1/S2 не совпадают для {condition}")
        comparable = [
            case_id for case_id in s1
            if not s1[case_id].get("error") and not s2[case_id].get("error")
        ]
        changed = [
            case_id for case_id in comparable
            if s1[case_id]["predicted_route"] != s2[case_id]["predicted_route"]
        ]
        corrected = [case_id for case_id in changed if not s1[case_id]["correct"] and s2[case_id]["correct"]]
        broken = [case_id for case_id in changed if s1[case_id]["correct"] and not s2[case_id]["correct"]]
        comparable_count += len(comparable)
        by_condition[condition] = {
            "changed_count": len(changed) if comparable else None,
            "changed_cases": changed,
            "corrected_count": len(corrected) if comparable else None,
            "corrected_cases": corrected,
            "broken_count": len(broken) if comparable else None,
            "broken_cases": broken,
            "agreement_count": len(comparable) - len(changed) if comparable else None,
            "total": len(comparable),
        }
        all_changed.extend(f"{condition}:{case_id}" for case_id in changed)
        all_corrected.extend(f"{condition}:{case_id}" for case_id in corrected)
        all_broken.extend(f"{condition}:{case_id}" for case_id in broken)

    complete_factorial = all(s1_grouped[item] and s2_grouped[item] for item in CONDITIONS)
    if complete_factorial:
        s1_factorial = analyze_factorial(s1_rows, labels)
        s2_factorial = analyze_factorial(s2_rows, labels)
    else:
        s1_factorial = {
            "conditions": {
                condition: summarize_condition(s1_grouped[condition], labels)
                for condition in by_condition
            },
            "pairwise_agreement": {},
            "order_sensitive_case_count": None,
            "order_sensitive_cases": [],
        }
        s2_factorial = {
            "conditions": {
                condition: summarize_condition(s2_grouped[condition], labels)
                for condition in by_condition
            },
            "pairwise_agreement": {},
            "order_sensitive_case_count": None,
            "order_sensitive_cases": [],
        }
    s1_by_key = {(row["condition"], row["id"]): row for row in s1_rows}
    s2_by_key = {(row["condition"], row["id"]): row for row in s2_rows}
    important = []
    for case_id in IMPORTANT_CASES:
        condition_results = []
        for condition in CONDITIONS:
            first = s1_by_key.get((condition, case_id))
            second = s2_by_key.get((condition, case_id))
            if first and second:
                condition_results.append({
                    "condition": condition,
                    "expected_route": first["expected_route"],
                    "s1_prediction": first["predicted_route"],
                    "s2_prediction": second["predicted_route"],
                    "s2_reason": second.get("reason"),
                    "s1_correct": first["correct"],
                    "s2_correct": second["correct"],
                    "s1_error": first.get("error"),
                    "s2_error": second.get("error"),
                })
        important.append({"id": case_id, "conditions": condition_results})
    return {
        "s1_conditions": s1_factorial["conditions"],
        "s2_conditions": s2_factorial["conditions"],
        "s1_order_robustness": {
            "pairwise_agreement": s1_factorial["pairwise_agreement"],
            "evaluated_cases": s1_factorial.get("order_robustness_evaluated_cases"),
            "order_sensitive_case_count": s1_factorial["order_sensitive_case_count"],
            "order_sensitive_cases": s1_factorial["order_sensitive_cases"],
        },
        "s2_order_robustness": {
            "pairwise_agreement": s2_factorial["pairwise_agreement"],
            "evaluated_cases": s2_factorial.get("order_robustness_evaluated_cases"),
            "order_sensitive_case_count": s2_factorial["order_sensitive_case_count"],
            "order_sensitive_cases": s2_factorial["order_sensitive_cases"],
        },
        "by_condition": by_condition,
        "aggregate_condition_case_changes": {
            "comparable_count": comparable_count,
            "changed_count": len(all_changed) if comparable_count else None,
            "changed": all_changed,
            "corrected_count": len(all_corrected) if comparable_count else None,
            "corrected": all_corrected,
            "broken_count": len(all_broken) if comparable_count else None,
            "broken": all_broken,
            "unique_changed_cases": sorted({item.split(":", 1)[1] for item in all_changed}),
        },
        "important_cases": important,
    }


def write_new_json(path, data):
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Файл уже существует и не будет перезаписан: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    s1_rows = load_csv(args.route_only)
    s2_rows = load_csv(args.reason_route)
    labels = labels_from_rows(s1_rows)
    factorial = analyze_factorial(s1_rows, labels)
    reason_comparison = compare_modes(s1_rows, s2_rows, labels)
    write_new_json(args.factorial_output, factorial)
    write_new_json(args.reason_output, reason_comparison)
    print(json.dumps({
        "factorial_accuracies": {
            key: value["accuracy"] for key, value in factorial["conditions"].items()
        },
        "s1_order_sensitive_cases": factorial["order_sensitive_case_count"],
        "s2_accuracies": {
            key: value["accuracy"] for key, value in reason_comparison["s2_conditions"].items()
        },
        "s2_order_sensitive_cases": reason_comparison["s2_order_robustness"]["order_sensitive_case_count"],
        "mode_changes": reason_comparison["aggregate_condition_case_changes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
