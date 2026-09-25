import unittest

import analyze_structured_factorial as analysis


def row(condition, case_id, expected, predicted, reason=None):
    return {
        "condition": condition,
        "id": case_id,
        "expected_route": expected,
        "predicted_route": predicted,
        "correct": expected == predicted,
        "reason": reason,
        "request_latency_sec": 0.1,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "error": None,
    }


class AnalyzeStructuredFactorialTests(unittest.TestCase):
    def test_factorial_analysis_reports_pairwise_changes_and_interaction(self):
        rows = [
            row("E1", "one", "X", "X"), row("E1", "two", "Y", "Y"),
            row("E2", "one", "X", "Y"), row("E2", "two", "Y", "Y"),
            row("E3", "one", "X", "X"), row("E3", "two", "Y", "X"),
            row("E4", "one", "X", "X"), row("E4", "two", "Y", "X"),
        ]

        result = analysis.analyze_factorial(rows, ["X", "Y"])

        self.assertEqual(result["pairwise_agreement"]["E1↔E2"]["changed_cases"], ["one"])
        self.assertEqual(result["prompt_order_effect"]["E1↔E2"]["changed_count"], 1)
        self.assertEqual(result["enum_order_effect"]["E1↔E3"]["changed_count"], 1)
        self.assertEqual(result["interaction"]["case_level_prompt_effect_symmetric_difference"], ["one"])
        self.assertEqual(result["interaction"]["accuracy_difference_in_differences"], 0.5)

    def test_reason_comparison_counts_corrected_and_broken_changes(self):
        s1 = [
            row("E1", "one", "X", "Y"),
            row("E1", "two", "Y", "Y"),
        ]
        s2 = [
            row("E1", "one", "X", "X", "исправлено"),
            row("E1", "two", "Y", "X", "сломано"),
        ]

        result = analysis.compare_modes(s1, s2, ["X", "Y"])

        comparison = result["by_condition"]["E1"]
        self.assertEqual(comparison["changed_count"], 2)
        self.assertEqual(comparison["corrected_cases"], ["one"])
        self.assertEqual(comparison["broken_cases"], ["two"])

    def test_factorial_analysis_handles_conditions_with_only_errors(self):
        rows = []
        for condition in ("E1", "E2", "E3", "E4"):
            failed = row(condition, "one", "X", None)
            failed["correct"] = None
            failed["error"] = "обрезанный JSON"
            rows.append(failed)

        result = analysis.analyze_factorial(rows, ["X", "Y"])

        self.assertIsNone(result["conditions"]["E1"]["accuracy"])
        self.assertIsNone(result["interaction"]["accuracy_difference_in_differences"])
        self.assertEqual(result["pairwise_agreement"]["E1↔E2"]["total"], 0)
        self.assertEqual(result["order_robustness_evaluated_cases"], 0)
        self.assertIsNone(result["order_sensitive_case_count"])
        self.assertEqual(result["conditions"]["E1"]["latency_sec"]["mean"], 0.1)
        self.assertEqual(result["conditions"]["E1"]["completion_tokens"]["mean"], 2)

    def test_reason_comparison_marks_no_comparable_predictions_as_unavailable(self):
        s1 = [row("E1", "one", "X", "X")]
        failed = row("E1", "one", "X", None)
        failed["correct"] = None
        failed["error"] = "обрезанный JSON"

        result = analysis.compare_modes(s1, [failed], ["X", "Y"])

        self.assertEqual(result["by_condition"]["E1"]["total"], 0)
        self.assertIsNone(result["by_condition"]["E1"]["changed_count"])
        aggregate = result["aggregate_condition_case_changes"]
        self.assertEqual(aggregate["comparable_count"], 0)
        self.assertIsNone(aggregate["changed_count"])


if __name__ == "__main__":
    unittest.main()
