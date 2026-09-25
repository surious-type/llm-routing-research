#!/usr/bin/env python3
"""Analyze Dataset V3 pre-test model runs and select the validation winner."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


LABELS = ("CONTINUE", "NEW")


def _number(value):
    return float(value) if value is not None else None


def _stats(values):
    clean = [_number(value) for value in values if value is not None]
    return {
        "mean": statistics.mean(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
    }


def summarize_order(rows):
    valid = [row for row in rows if not row.get("error") and row.get("predicted_route") in LABELS]
    matrix = [[0, 0], [0, 0]]
    errors = []
    unresolved_by_expected = {label: 0 for label in LABELS}
    for row in rows:
        if row not in valid:
            unresolved_by_expected[row["expected_route"]] += 1
    for row in valid:
        expected_index = LABELS.index(row["expected_route"])
        predicted_index = LABELS.index(row["predicted_route"])
        matrix[expected_index][predicted_index] += 1
        if row["expected_route"] != row["predicted_route"]:
            errors.append(row["id"])
    recalls = {}
    for index, label in enumerate(LABELS):
        support = sum(matrix[index]) + unresolved_by_expected[label]
        recalls[label] = matrix[index][index] / support if support else None
    accuracy = sum(matrix[index][index] for index in range(2)) / len(rows) if rows else None
    recall_values = [value for value in recalls.values() if value is not None]
    return {
        "count": len(rows),
        "valid_prediction_count": len(valid),
        "accuracy": accuracy,
        "balanced_accuracy": statistics.mean(recall_values) if recall_values else None,
        "confusion_matrix": matrix,
        "label_order": list(LABELS),
        "recall": recalls,
        "unresolved_by_expected": unresolved_by_expected,
        "error_count": len(errors) + len(rows) - len(valid),
        "error_ids": sorted(errors),
        "request_error_count": len(rows) - len(valid),
        "latency_sec": _stats([row.get("request_latency_sec") for row in rows]),
        "prompt_tokens": _stats([row.get("prompt_tokens") for row in rows]),
        "completion_tokens": _stats([row.get("completion_tokens") for row in rows]),
    }


def _paired(rows):
    by_id = defaultdict(dict)
    for row in rows:
        by_id[row["id"]][row["condition"]] = row
    missing = sorted(case_id for case_id, pair in by_id.items() if set(pair) != {"P1", "P2"})
    if missing:
        raise ValueError(f"Неполные P1/P2 пары: {missing}")
    return by_id


def dual_order_metrics(rows):
    pairs = _paired(rows)
    invalid = []
    agreed = []
    disagreed = []
    for case_id, pair in pairs.items():
        if any(
            item.get("error") or item.get("predicted_route") not in LABELS
            for item in pair.values()
        ):
            invalid.append((case_id, pair))
            continue
        if pair["P1"].get("predicted_route") == pair["P2"].get("predicted_route"):
            agreed.append((case_id, pair))
        else:
            disagreed.append((case_id, pair))
    agreed_correct = [pair["P1"]["predicted_route"] == pair["P1"]["expected_route"] for _, pair in agreed]
    p1_disagreement = [pair["P1"]["predicted_route"] == pair["P1"]["expected_route"] for _, pair in disagreed]
    p2_disagreement = [pair["P2"]["predicted_route"] == pair["P2"]["expected_route"] for _, pair in disagreed]
    valid_total = len(agreed) + len(disagreed)
    all_total = len(pairs)
    return {
        "valid_pair_count": valid_total,
        "invalid_pair_count": len(invalid),
        "invalid_pair_ids": sorted(case_id for case_id, _ in invalid),
        "semantic_agreement": len(agreed) / valid_total if valid_total else None,
        "coverage": len(agreed) / all_total if all_total else None,
        "agreed_count": len(agreed),
        "accuracy_when_agree": sum(agreed_correct) / len(agreed_correct) if agreed_correct else None,
        "agreed_but_wrong_count": sum(not value for value in agreed_correct),
        "agreed_but_wrong_ids": sorted(case_id for (case_id, _), correct in zip(agreed, agreed_correct) if not correct),
        "disagreement_count": len(disagreed),
        "order_sensitive_ids": sorted(case_id for case_id, _ in disagreed),
        "P1_accuracy_on_disagreement": sum(p1_disagreement) / len(disagreed) if disagreed else None,
        "P2_accuracy_on_disagreement": sum(p2_disagreement) / len(disagreed) if disagreed else None,
    }


def summarize_run(rows):
    p1 = summarize_order([row for row in rows if row["condition"] == "P1"])
    p2 = summarize_order([row for row in rows if row["condition"] == "P2"])
    dual = dual_order_metrics(rows)
    return {
        "P1": p1,
        "P2": p2,
        "mean_accuracy": statistics.mean([p1["accuracy"], p2["accuracy"]]),
        "dual": dual,
        "mean_latency_sec": statistics.mean([p1["latency_sec"]["mean"], p2["latency_sec"]["mean"]]),
    }


def breakdown(rows, dimension):
    grouped = defaultdict(list)
    for row in rows:
        value = "standalone" if dimension == "family_status" and row.get("family_id") is None else (
            "family" if dimension == "family_status" else row[dimension]
        )
        grouped[str(value)].append(row)
    result = {}
    for value, group_rows in sorted(grouped.items()):
        result[value] = {
            condition: summarize_order([row for row in group_rows if row["condition"] == condition])
            for condition in ("P1", "P2")
        }
        result[value]["dual"] = dual_order_metrics(group_rows)
    return result


def _family_order(rows, condition):
    families = defaultdict(list)
    for row in rows:
        if row["condition"] == condition and row.get("family_id") is not None:
            families[row["family_id"]].append(row)
    accuracies = {
        family_id: sum(row["expected_route"] == row["predicted_route"] for row in members) / len(members)
        for family_id, members in families.items()
    }
    exact = [family_id for family_id, value in accuracies.items() if value == 1.0]
    errors = [family_id for family_id, value in accuracies.items() if value < 1.0]
    return {
        "family_count": len(families),
        "family_exact_accuracy": len(exact) / len(families) if families else None,
        "mean_per_family_accuracy": statistics.mean(accuracies.values()) if accuracies else None,
        "families_with_any_error_count": len(errors),
        "families_with_any_error": sorted(errors),
        "per_family_accuracy": dict(sorted(accuracies.items())),
    }


def analyze_family(rows):
    pairs = _paired(rows)
    disagreements = sorted({
        pair["P1"]["family_id"]
        for pair in pairs.values()
        if pair["P1"].get("family_id") is not None
        and not pair["P1"].get("error") and not pair["P2"].get("error")
        and pair["P1"].get("predicted_route") in LABELS and pair["P2"].get("predicted_route") in LABELS
        and pair["P1"].get("predicted_route") != pair["P2"].get("predicted_route")
    })
    invalid_families = sorted({
        pair["P1"].get("family_id")
        for pair in pairs.values()
        if pair["P1"].get("family_id") is not None
        and any(item.get("error") or item.get("predicted_route") not in LABELS for item in pair.values())
    })
    return {
        "P1": _family_order(rows, "P1"),
        "P2": _family_order(rows, "P2"),
        "families_with_order_disagreement_count": len(disagreements),
        "families_with_order_disagreement": disagreements,
        "families_with_invalid_pair_count": len(invalid_families),
        "families_with_invalid_pair": invalid_families,
        "standalone": breakdown([row for row in rows if row.get("family_id") is None], "family_status") if any(row.get("family_id") is None for row in rows) else {},
    }


def choose_winner(validation_summaries):
    def ranking_key(item):
        _, summary = item
        return (
            summary["mean_accuracy"],
            summary["dual"]["semantic_agreement"],
            -summary["dual"]["agreed_but_wrong_count"],
            -summary["mean_latency_sec"],
        )
    ordered = sorted(validation_summaries.items(), key=ranking_key, reverse=True)
    return {
        "winner": ordered[0][0],
        "ranking": [label for label, _ in ordered],
        "rule": ["mean_validation_accuracy", "semantic_agreement", "fewer_agreed_but_wrong", "lower_operational_latency"],
        "values": {
            label: {
                "mean_accuracy": summary["mean_accuracy"],
                "semantic_agreement": summary["dual"]["semantic_agreement"],
                "agreed_but_wrong_count": summary["dual"]["agreed_but_wrong_count"],
                "mean_latency_sec": summary["mean_latency_sec"],
            }
            for label, summary in ordered
        },
    }


def apply_repairs(original_rows, repair_rows):
    merged = [dict(row) for row in original_rows]
    positions = {(row["condition"], row["id"]): index for index, row in enumerate(merged)}
    for repair in repair_rows:
        key = (repair["condition"], repair["id"])
        if key not in positions:
            raise ValueError(f"Repair не соответствует исходной строке: {key}")
        index = positions[key]
        original = merged[index]
        if not original.get("error"):
            raise ValueError(f"Исходная строка не была ошибочной: {key}")
        if repair.get("error") or repair.get("predicted_route") not in LABELS:
            raise ValueError(f"Repair не содержит валидный prediction: {key}")
        replacement = dict(repair)
        replacement["repaired_from_error"] = original["error"]
        merged[index] = replacement
    return merged


def parse_run(value):
    try:
        key, path = value.split("=", 1)
        model, split = key.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Формат --run MODEL:SPLIT=path.json") from exc
    return model, split, path


def parse_repair(value):
    return parse_run(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--repair", action="append", type=parse_repair, default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    paths = [
        output_dir / "comparison.json",
        output_dir / "breakdown_by_category.json",
        output_dir / "breakdown_by_difficulty.json",
        output_dir / "breakdown_by_domain.json",
        output_dir / "family_analysis.json",
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("Один или несколько analysis outputs уже существуют")

    runs = {}
    for model, split, path in args.run:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        runs[(model, split)] = document["rows"]
    repair_summary = {}
    for model, split, path in args.repair:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        key = (model, split)
        if key not in runs:
            raise ValueError(f"Repair не имеет основного run: {model}:{split}")
        runs[key] = apply_repairs(runs[key], document["rows"])
        repair_summary[f"{model}:{split}"] = {
            "path": path,
            "repaired_row_count": len(document["rows"]),
            "keys": sorted(f"{row['condition']}:{row['id']}" for row in document["rows"]),
        }
    summaries = {f"{model}:{split}": summarize_run(rows) for (model, split), rows in runs.items()}
    validation_summaries = {model: summarize_run(rows) for (model, split), rows in runs.items() if split == "validation"}
    selection = choose_winner(validation_summaries)

    comparison = {
        "experiment": "dataset-v3-pretest-v1",
        "summaries": summaries,
        "validation_selection": selection,
        "technical_repairs": repair_summary,
        "historical_dataset_v2_reference": {
            "M4": {"mean_accuracy": 0.90, "agreement": 0.80},
            "M8": {"mean_accuracy": 0.9333, "agreement": 0.8667},
            "M14": {"mean_accuracy": 0.90, "agreement": 0.9333},
            "independent_test": False,
        },
        "breakdown_by_expected_route": {f"{model}:{split}": breakdown(rows, "expected_route") for (model, split), rows in runs.items()},
        "breakdown_by_ambiguity": {f"{model}:{split}": breakdown(rows, "ambiguity") for (model, split), rows in runs.items()},
        "breakdown_by_family_status": {f"{model}:{split}": breakdown(rows, "family_status") for (model, split), rows in runs.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, dimension in (
        ("breakdown_by_category.json", "category"),
        ("breakdown_by_difficulty.json", "difficulty"),
        ("breakdown_by_domain.json", "domain"),
    ):
        content = {f"{model}:{split}": breakdown(rows, dimension) for (model, split), rows in runs.items()}
        (output_dir / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    family = {f"{model}:{split}": analyze_family(rows) for (model, split), rows in runs.items()}
    (output_dir / "family_analysis.json").write_text(json.dumps(family, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"winner": selection["winner"], "summaries": {key: value["mean_accuracy"] for key, value in summaries.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
