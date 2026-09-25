import unittest

from dataset_v3_pretest_analyze import (
    analyze_family,
    apply_repairs,
    choose_winner,
    dual_order_metrics,
    summarize_order,
)


def result(case_id, condition, expected, predicted, family="f1", latency=1.0):
    return {
        "id": case_id,
        "condition": condition,
        "expected_route": expected,
        "predicted_route": predicted,
        "correct": expected == predicted,
        "error": "",
        "request_latency_sec": latency,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "difficulty": "easy",
        "ambiguity": "low",
        "category": "x",
        "domain": "d",
        "family_id": family,
        "split": "validation",
    }


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            result("a", "P1", "CONTINUE", "CONTINUE", latency=1),
            result("b", "P1", "NEW", "CONTINUE", latency=3),
            result("c", "P1", "NEW", "NEW", family=None, latency=5),
            result("a", "P2", "CONTINUE", "CONTINUE", latency=2),
            result("b", "P2", "NEW", "NEW", latency=4),
            result("c", "P2", "NEW", "CONTINUE", family=None, latency=6),
        ]

    def test_order_summary_reports_confusion_recalls_and_latency(self):
        summary = summarize_order([row for row in self.rows if row["condition"] == "P1"])
        self.assertEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["balanced_accuracy"], 0.75)
        self.assertEqual(summary["confusion_matrix"], [[1, 0], [1, 1]])
        self.assertEqual(summary["recall"], {"CONTINUE": 1.0, "NEW": 0.5})
        self.assertEqual(summary["error_ids"], ["b"])
        self.assertEqual(summary["latency_sec"]["mean"], 3.0)
        self.assertEqual(summary["latency_sec"]["median"], 3.0)

    def test_dual_order_metrics_separate_agreement_from_correctness(self):
        dual = dual_order_metrics(self.rows)
        self.assertEqual(dual["coverage"], 1 / 3)
        self.assertEqual(dual["accuracy_when_agree"], 1.0)
        self.assertEqual(dual["agreed_but_wrong_count"], 0)
        self.assertEqual(dual["disagreement_count"], 2)
        self.assertEqual(dual["order_sensitive_ids"], ["b", "c"])
        self.assertEqual(dual["P1_accuracy_on_disagreement"], 0.5)
        self.assertEqual(dual["P2_accuracy_on_disagreement"], 0.5)

    def test_family_analysis_counts_exact_and_disagreeing_families(self):
        family = analyze_family(self.rows)
        self.assertEqual(family["P1"]["family_count"], 1)
        self.assertEqual(family["P1"]["family_exact_accuracy"], 0.0)
        self.assertEqual(family["P1"]["mean_per_family_accuracy"], 0.5)
        self.assertEqual(family["families_with_order_disagreement"], ["f1"])

    def test_dual_order_excludes_invalid_pairs_and_reports_them(self):
        rows = [
            result("a", "P1", "CONTINUE", "CONTINUE"),
            result("a", "P2", "CONTINUE", "CONTINUE"),
            result("b", "P1", "NEW", "NEW"),
            {**result("b", "P2", "NEW", None), "error": "ValueError: truncated"},
        ]
        dual = dual_order_metrics(rows)
        self.assertEqual(dual["valid_pair_count"], 1)
        self.assertEqual(dual["invalid_pair_count"], 1)
        self.assertEqual(dual["invalid_pair_ids"], ["b"])
        self.assertEqual(dual["semantic_agreement"], 1.0)
        self.assertEqual(dual["coverage"], 0.5)

    def test_order_summary_counts_invalid_response_as_incorrect(self):
        rows = [
            result("a", "P1", "CONTINUE", "CONTINUE"),
            {**result("b", "P1", "NEW", None), "error": "ValueError: truncated"},
        ]
        summary = summarize_order(rows)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["valid_prediction_count"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["recall"], {"CONTINUE": 1.0, "NEW": 0.0})
        self.assertEqual(summary["balanced_accuracy"], 0.5)
        self.assertEqual(summary["request_error_count"], 1)
        self.assertEqual(summary["unresolved_by_expected"], {"CONTINUE": 0, "NEW": 1})

    def test_winner_ranking_uses_all_predeclared_tie_breakers(self):
        summaries = {
            "M4": {"mean_accuracy": 0.9, "dual": {"semantic_agreement": 0.8, "agreed_but_wrong_count": 1}, "mean_latency_sec": 1.0},
            "M8": {"mean_accuracy": 0.9, "dual": {"semantic_agreement": 0.9, "agreed_but_wrong_count": 2}, "mean_latency_sec": 2.0},
            "M14": {"mean_accuracy": 0.9, "dual": {"semantic_agreement": 0.9, "agreed_but_wrong_count": 1}, "mean_latency_sec": 3.0},
        }
        ranking = choose_winner(summaries)
        self.assertEqual(ranking["winner"], "M14")
        self.assertEqual(ranking["ranking"], ["M14", "M8", "M4"])

    def test_repair_replaces_only_matching_infrastructure_error(self):
        original = [
            {**result("a", "P1", "CONTINUE", "CONTINUE"), "reason": "old good"},
            {**result("a", "P2", "CONTINUE", None), "error": "ConnectionError", "reason": None},
        ]
        repair = [{**result("a", "P2", "CONTINUE", "CONTINUE"), "reason": "recovered"}]
        merged = apply_repairs(original, repair)
        self.assertEqual(merged[0]["reason"], "old good")
        self.assertEqual(merged[1]["reason"], "recovered")
        self.assertEqual(merged[1]["repaired_from_error"], "ConnectionError")
        with self.assertRaisesRegex(ValueError, "не была ошибочной"):
            apply_repairs(original, [{**result("a", "P1", "CONTINUE", "CONTINUE")}])


if __name__ == "__main__":
    unittest.main()
