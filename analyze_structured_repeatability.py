#!/usr/bin/env python3
"""Сравнение повторов факторного structured routing."""

import argparse
import itertools
import json
from pathlib import Path

from analyze_structured_factorial import (
    CONDITIONS,
    IMPORTANT_CASES,
    analyze_factorial,
    compare_condition_pair,
    labels_from_rows,
    load_csv,
    summarize_condition,
)


def _by_condition(rows):
    return {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in CONDITIONS
    }


def analyze_repeats(repeats, labels):
    if len(repeats) < 2:
        raise ValueError("Для анализа повторяемости нужны минимум два прогона")
    grouped = {name: _by_condition(rows) for name, rows in repeats.items()}
    run_results = {}
    for name, rows in repeats.items():
        run_grouped = grouped[name]
        if all(run_grouped[condition] for condition in CONDITIONS):
            run_results[name] = analyze_factorial(rows, labels)
        else:
            run_results[name] = {
                "conditions": {
                    condition: summarize_condition(condition_rows, labels)
                    for condition, condition_rows in run_grouped.items()
                    if condition_rows
                },
                "pairwise_agreement": {},
                "order_robustness_evaluated_cases": 0,
                "order_sensitive_case_count": None,
                "order_sensitive_cases": [],
            }
    by_condition = {}
    for condition in CONDITIONS:
        available = {
            name: rows[condition] for name, rows in grouped.items() if rows[condition]
        }
        if not available:
            continue
        case_maps = {
            name: {row["id"]: row for row in rows}
            for name, rows in available.items()
        }
        case_sets = [set(case_map) for case_map in case_maps.values()]
        if any(case_set != case_sets[0] for case_set in case_sets[1:]):
            raise ValueError(f"Наборы case id повторов не совпадают для {condition}")
        all_cases = sorted(case_sets[0])
        comparable = [
            case_id for case_id in all_cases
            if all(not case_map[case_id].get("error") for case_map in case_maps.values())
        ]
        unstable = [
            case_id for case_id in comparable
            if len({case_map[case_id]["predicted_route"] for case_map in case_maps.values()}) > 1
        ]
        pairwise = {}
        for left, right in itertools.combinations(available, 2):
            pairwise[f"{left}↔{right}"] = compare_condition_pair(
                available[left], available[right]
            )
        by_condition[condition] = {
            "all_repeat_comparable_count": len(comparable),
            "all_repeat_agreement_count": len(comparable) - len(unstable),
            "all_repeat_agreement_rate": (
                (len(comparable) - len(unstable)) / len(comparable)
                if comparable else None
            ),
            "unstable_case_count": len(unstable) if comparable else None,
            "unstable_cases": unstable,
            "uncompared_error_cases": [
                case_id for case_id in all_cases if case_id not in comparable
            ],
            "pairwise": pairwise,
        }

    important = {}
    for case_id in IMPORTANT_CASES:
        important[case_id] = {
            name: {
                condition: next(
                    (
                        {
                            "predicted_route": row["predicted_route"],
                            "correct": row["correct"],
                            "error": row.get("error"),
                            "reason": row.get("reason"),
                        }
                        for row in grouped[name][condition] if row["id"] == case_id
                    ),
                    None,
                )
                for condition in CONDITIONS
            }
            for name in repeats
        }
    return {
        "runs": run_results,
        "repeatability_by_condition": by_condition,
        "important_cases": important,
    }


def compare_families(left, right):
    if list(left) != list(right):
        raise ValueError("Имена повторов сравниваемых групп должны совпадать")
    by_repeat = {}
    changed = []
    corrected = []
    broken = []
    comparable_count = 0
    for repeat_name in left:
        left_map = {(row["condition"], row["id"]): row for row in left[repeat_name]}
        right_map = {(row["condition"], row["id"]): row for row in right[repeat_name]}
        if set(left_map) != set(right_map):
            raise ValueError(f"Наборы condition/case не совпадают для {repeat_name}")
        current_changed = []
        current_corrected = []
        current_broken = []
        current_comparable = 0
        for key in sorted(left_map):
            first, second = left_map[key], right_map[key]
            if first.get("error") or second.get("error"):
                continue
            current_comparable += 1
            if first["predicted_route"] != second["predicted_route"]:
                item = f"{key[0]}:{key[1]}"
                current_changed.append(item)
                if not first["correct"] and second["correct"]:
                    current_corrected.append(item)
                elif first["correct"] and not second["correct"]:
                    current_broken.append(item)
        by_repeat[repeat_name] = {
            "comparable_count": current_comparable,
            "changed_count": len(current_changed),
            "changed": current_changed,
            "corrected_count": len(current_corrected),
            "corrected": current_corrected,
            "broken_count": len(current_broken),
            "broken": current_broken,
        }
        comparable_count += current_comparable
        changed.extend(f"{repeat_name}:{item}" for item in current_changed)
        corrected.extend(f"{repeat_name}:{item}" for item in current_corrected)
        broken.extend(f"{repeat_name}:{item}" for item in current_broken)
    return {
        "by_repeat": by_repeat,
        "aggregate": {
            "comparable_count": comparable_count,
            "changed_count": len(changed),
            "changed": changed,
            "corrected_count": len(corrected),
            "corrected": corrected,
            "broken_count": len(broken),
            "broken": broken,
        },
    }


def parse_group(value):
    try:
        name, paths = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Формат группы: ИМЯ=run1.csv,run2.csv,run3.csv") from exc
    files = paths.split(",")
    if len(files) != 3 or not name:
        raise argparse.ArgumentTypeError("Группа должна содержать имя и ровно три CSV")
    return name, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", type=parse_group, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Файл уже существует и не будет перезаписан: {output}")
    groups = {}
    for group_name, files in args.group:
        if group_name in groups:
            raise ValueError(f"Повторяющееся имя группы: {group_name}")
        groups[group_name] = {
            f"R{index}": load_csv(path) for index, path in enumerate(files, start=1)
        }
    first_rows = next(iter(next(iter(groups.values())).values()))
    labels = labels_from_rows(first_rows)
    result = {
        "groups": {
            name: analyze_repeats(repeats, labels) for name, repeats in groups.items()
        },
        "group_comparisons": {},
    }
    for left, right in itertools.combinations(groups, 2):
        result["group_comparisons"][f"{left}↔{right}"] = compare_families(
            groups[left], groups[right]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            repeat: {
                condition: metrics["accuracy"]
                for condition, metrics in run["conditions"].items()
            }
            for repeat, run in data["runs"].items()
        }
        for name, data in result["groups"].items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
