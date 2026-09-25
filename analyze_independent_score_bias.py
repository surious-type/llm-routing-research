#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "std_sample": statistics.stdev(values) if len(values) > 1 else None,
    }


def analyze(path):
    rows = load_rows(path)
    observations = []
    score_groups = {"CONTINUE": [], "NEW": [], "correct_candidate": [], "incorrect_candidate": []}
    expected_groups = {"CONTINUE": {"CONTINUE": [], "NEW": []}, "NEW": {"CONTINUE": [], "NEW": []}}
    deltas = []
    both_positive = []
    both_match_gt_90 = []
    for row in rows:
        candidates = {x["name"]: x for x in json.loads(row["candidates_json"])}
        c, n = candidates["CONTINUE"], candidates["NEW"]
        c_score, n_score = float(c["score"]), float(n["score"])
        delta = c_score - n_score
        deltas.append(delta)
        expected = row["expected_route"]
        score_groups["CONTINUE"].append(c_score)
        score_groups["NEW"].append(n_score)
        score_groups["correct_candidate"].append(float(candidates[expected]["score"]))
        score_groups["incorrect_candidate"].append(float(candidates["NEW" if expected == "CONTINUE" else "CONTINUE"]["score"]))
        expected_groups[expected]["CONTINUE"].append(c_score)
        expected_groups[expected]["NEW"].append(n_score)
        if c_score > 0 and n_score > 0:
            both_positive.append(row["id"])
        if float(c["match_probability_binary"]) > 0.9 and float(n["match_probability_binary"]) > 0.9:
            both_match_gt_90.append(row["id"])
        observations.append({
            "id": row["id"], "expected_route": expected, "predicted_route": row["predicted_route"],
            "continue_score": c_score, "new_score": n_score, "delta_continue_minus_new": delta,
            "continue_match_probability": float(c["match_probability_binary"]),
            "new_match_probability": float(n["match_probability_binary"]),
            "both_scores_positive": c_score > 0 and n_score > 0,
            "both_match_probability_gt_0_9": float(c["match_probability_binary"]) > 0.9 and float(n["match_probability_binary"]) > 0.9,
            "correct_candidate_score": float(candidates[expected]["score"]),
            "incorrect_candidate_score": float(candidates["NEW" if expected == "CONTINUE" else "CONTINUE"]["score"]),
        })
    return {
        "input": path,
        "case_count": len(rows),
        "score_statistics": {key: stats(value) for key, value in score_groups.items()},
        "score_statistics_by_expected_route": {
            expected: {route: stats(values) for route, values in routes.items()}
            for expected, routes in expected_groups.items()
        },
        "delta_continue_minus_new": stats(deltas),
        "delta_positive_count": sum(x > 0 for x in deltas),
        "delta_negative_count": sum(x < 0 for x in deltas),
        "delta_zero_count": sum(x == 0 for x in deltas),
        "both_positive_match_score_count": len(both_positive),
        "both_positive_match_score_cases": both_positive,
        "both_match_probability_gt_0_9_count": len(both_match_gt_90),
        "both_match_probability_gt_0_9_cases": both_match_gt_90,
        "observations": observations,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--json-output", required=True)
    p.add_argument("--csv-output", required=True)
    args = p.parse_args()
    result = {"analysis": "independent_score_bias", "runs": {"A": analyze(args.run_a), "B": analyze(args.run_b)}}
    Path(args.json_output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    observations = []
    for label, run in result["runs"].items():
        for row in run["observations"]:
            observations.append({"run": label, **row})
    with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    for label, run in result["runs"].items():
        print(label, "delta_mean=%.6f" % run["delta_continue_minus_new"]["mean"],
              "delta_median=%.6f" % run["delta_continue_minus_new"]["median"],
              "both_positive=%d" % run["both_positive_match_score_count"],
              "both_p_gt_0.9=%d" % run["both_match_probability_gt_0_9_count"])


if __name__ == "__main__":
    main()
