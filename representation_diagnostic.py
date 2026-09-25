#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import requests

from independent_candidate_scoring import (
    discover_model,
    load_jsonl,
    score_candidate,
    tokenize,
)


SEMANTICS = ("CONTINUE", "NEW")


def distribution(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "std_sample": statistics.stdev(values) if len(values) > 1 else None,
    }


def pearson(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("Pearson vectors must have equal length")
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def validate_conditions(conditions):
    ids = [condition["id"] for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("Condition ids must be unique")
    for condition in conditions:
        semantics = {candidate["semantic"] for candidate in condition["candidates"]}
        if semantics != set(SEMANTICS):
            raise ValueError("Each condition must contain CONTINUE and NEW")
        for candidate in condition["candidates"]:
            if not candidate.get("name") or not candidate.get("description"):
                raise ValueError("Each candidate needs name and description")


def resolve_conditions(config):
    descriptions = config["descriptions"]
    resolved = []
    for raw in config["conditions"]:
        variant = raw["description_variant"]
        resolved.append({
            "id": raw["id"],
            "factor": raw["factor"],
            "description_variant": variant,
            "candidates": [
                {
                    "semantic": semantic,
                    "name": raw["surface_names"][semantic],
                    "description": descriptions[variant][semantic],
                }
                for semantic in SEMANTICS
            ],
        })
    validate_conditions(resolved)
    return resolved


def summarize_condition(rows, baseline_predictions):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["id"], {})[row["semantic"]] = row
    cases = []
    groups = {
        "expected_CONTINUE_candidate_CONTINUE": [],
        "expected_CONTINUE_candidate_NEW": [],
        "expected_NEW_candidate_CONTINUE": [],
        "expected_NEW_candidate_NEW": [],
    }
    for case_id in sorted(grouped):
        pair = grouped[case_id]
        c_score = float(pair["CONTINUE"]["score"])
        n_score = float(pair["NEW"]["score"])
        expected = pair["CONTINUE"]["expected_route"]
        predicted = "CONTINUE" if c_score >= n_score else "NEW"
        delta = c_score - n_score
        groups[f"expected_{expected}_candidate_CONTINUE"].append(c_score)
        groups[f"expected_{expected}_candidate_NEW"].append(n_score)
        cases.append({
            "id": case_id,
            "expected_route": expected,
            "predicted_route": predicted,
            "baseline_prediction": baseline_predictions[case_id],
            "correct": predicted == expected,
            "continue_score": c_score,
            "new_score": n_score,
            "delta_continue_minus_new": delta,
            "prediction_changed_from_baseline": predicted != baseline_predictions[case_id],
        })
    deltas = [case["delta_continue_minus_new"] for case in cases]
    changed = [case["id"] for case in cases if case["prediction_changed_from_baseline"]]
    summary = {
        "case_count": len(cases),
        "accuracy": sum(case["correct"] for case in cases) / len(cases),
        "candidate_score_distributions": {
            "CONTINUE": distribution([case["continue_score"] for case in cases]),
            "NEW": distribution([case["new_score"] for case in cases]),
        },
        "delta": distribution(deltas),
        "delta_positive_count": sum(value > 0 for value in deltas),
        "delta_negative_count": sum(value < 0 for value in deltas),
        "delta_zero_count": sum(value == 0 for value in deltas),
        "semantic_agreement_with_baseline": 1 - len(changed) / len(cases),
        "changed_prediction_cases": changed,
        "score_distributions_by_expected_and_candidate": {
            key: distribution(values) for key, values in groups.items()
        },
    }
    return summary, cases


def pairwise_analysis(condition_rows, condition_ids, semantic):
    result = []
    vectors = {
        condition_id: [
            float(row["score"])
            for row in sorted(condition_rows[condition_id], key=lambda item: item["id"])
            if row["semantic"] == semantic
        ]
        for condition_id in condition_ids
    }
    for index, left in enumerate(condition_ids):
        for right in condition_ids[index + 1:]:
            differences = [a - b for a, b in zip(vectors[left], vectors[right])]
            result.append({
                "left": left,
                "right": right,
                "semantic": semantic,
                "pearson_correlation": pearson(vectors[left], vectors[right]),
                "left_minus_right": distribution(differences),
                "per_case_differences": differences,
            })
    return result


def pairwise_prediction_analysis(cases_by_condition, condition_ids):
    result = []
    for index, left in enumerate(condition_ids):
        left_predictions = {row["id"]: row["predicted_route"] for row in cases_by_condition[left]}
        for right in condition_ids[index + 1:]:
            right_predictions = {row["id"]: row["predicted_route"] for row in cases_by_condition[right]}
            ids = sorted(set(left_predictions) & set(right_predictions))
            changed = [case_id for case_id in ids if left_predictions[case_id] != right_predictions[case_id]]
            result.append({
                "left": left,
                "right": right,
                "semantic_agreement": 1 - len(changed) / len(ids),
                "changed_prediction_cases": changed,
            })
    return result


def select_description_condition_ids(conditions):
    description_ids = [condition["id"] for condition in conditions if condition["factor"] == "description"]
    if not description_ids:
        return []
    for condition in conditions:
        names = {candidate["name"] for candidate in condition.get("candidates", [])}
        if (
            condition["factor"] == "route_name"
            and condition.get("description_variant") == "original"
            and names == {"CANDIDATE"}
        ):
            return [condition["id"], *description_ids]
    return description_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="representation_conditions.json")
    parser.add_argument("--dataset", default="dataset_v2.jsonl")
    parser.add_argument("--baseline", default="results/independent_A_20260904_1445.csv")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_baseline(path):
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["predicted_route"] for row in csv.DictReader(f)}


def main():
    args = parse_args()
    config_path = Path(args.conditions)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    conditions = resolve_conditions(config)
    dataset = load_jsonl(args.dataset)
    if args.limit is not None:
        dataset = dataset[:args.limit]
    baseline_predictions = load_baseline(args.baseline)
    session = requests.Session()
    model = args.model or discover_model(session, args.base_url, args.timeout)
    token_one = tokenize(session, args.base_url, model, "1", args.timeout)
    token_zero = tokenize(session, args.base_url, model, "0", args.timeout)
    raw_rows = []
    for condition in conditions:
        print(f"Condition: {condition['id']}")
        for row in dataset:
            for candidate in condition["candidates"]:
                scored = score_candidate(
                    session, args.base_url, model, row, candidate, token_one, token_zero,
                    args.timeout, args.temperature, args.seed, False,
                )
                raw_rows.append({
                    "condition_id": condition["id"],
                    "factor": condition["factor"],
                    "description_variant": condition["description_variant"],
                    "id": row["id"],
                    "expected_route": row["expected_route"],
                    "semantic": candidate["semantic"],
                    "surface_name": candidate["name"],
                    "description": candidate["description"],
                    **{key: value for key, value in scored.items() if key != "name"},
                })
    raw_path = Path(args.raw_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    condition_rows = {
        condition["id"]: [row for row in raw_rows if row["condition_id"] == condition["id"]]
        for condition in conditions
    }
    summaries, case_rows, cases_by_condition = {}, [], {}
    for condition in conditions:
        summary, cases = summarize_condition(condition_rows[condition["id"]], baseline_predictions)
        summaries[condition["id"]] = {**summary, "factor": condition["factor"], "description_variant": condition["description_variant"]}
        cases_by_condition[condition["id"]] = cases
        case_rows.extend({"condition_id": condition["id"], **case} for case in cases)
    with Path(args.cases_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    name_ids = [condition["id"] for condition in conditions if condition["factor"] == "route_name"]
    description_ids = select_description_condition_ids(conditions)
    analysis = {
        "metadata": {
            "version": config["version"],
            "model": model,
            "temperature": args.temperature,
            "seed": args.seed,
            "scoring_formula": "logP(1)-logP(0)",
            "dataset": args.dataset,
            "case_count": len(dataset),
            "baseline": args.baseline,
            "conditions_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "conditions": summaries,
        "name_effect_pairwise": [
            item for semantic in SEMANTICS for item in pairwise_analysis(condition_rows, name_ids, semantic)
        ],
        "description_effect_pairwise": [
            item for semantic in SEMANTICS for item in pairwise_analysis(condition_rows, description_ids, semantic)
        ],
        "name_effect_prediction_agreement": pairwise_prediction_analysis(cases_by_condition, name_ids),
        "description_effect_prediction_agreement": pairwise_prediction_analysis(cases_by_condition, description_ids),
        "case_results": case_rows,
    }
    Path(args.analysis_json).write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    for condition_id, summary in summaries.items():
        print(condition_id, f"accuracy={summary['accuracy']:.3%}",
              f"delta_mean={summary['delta']['mean']:.6f}",
              f"agreement={summary['semantic_agreement_with_baseline']:.3%}",
              f"changed={len(summary['changed_prediction_cases'])}")


if __name__ == "__main__":
    main()
