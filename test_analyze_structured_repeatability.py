import unittest

import analyze_structured_repeatability as repeatability


def row(condition, case_id, expected, predicted, *, error=None, reason="основание"):
    return {
        "condition": condition,
        "id": case_id,
        "expected_route": expected,
        "predicted_route": predicted,
        "correct": None if error else expected == predicted,
        "reason": reason,
        "request_latency_sec": 0.2,
        "prompt_tokens": 10.0,
        "completion_tokens": 3.0,
        "error": error,
    }


class AnalyzeStructuredRepeatabilityTests(unittest.TestCase):
    def test_three_repeat_analysis_reports_unanimity_and_changed_cases_per_condition(self):
        repeats = {
            "R1": [row("E1", "one", "X", "X"), row("E1", "two", "Y", "Y")],
            "R2": [row("E1", "one", "X", "X"), row("E1", "two", "Y", "X")],
            "R3": [row("E1", "one", "X", "X"), row("E1", "two", "Y", "Y")],
        }

        result = repeatability.analyze_repeats(repeats, ["X", "Y"])

        condition = result["repeatability_by_condition"]["E1"]
        self.assertEqual(condition["all_repeat_agreement_count"], 1)
        self.assertEqual(condition["all_repeat_agreement_rate"], 0.5)
        self.assertEqual(condition["unstable_cases"], ["two"])
        self.assertEqual(condition["pairwise"]["R1↔R2"]["changed_cases"], ["two"])
        self.assertEqual(condition["pairwise"]["R1↔R3"]["changed_cases"], [])

    def test_repeat_analysis_keeps_order_sensitivity_separate_for_each_repeat(self):
        repeats = {}
        for name, e2_prediction in (("R1", "X"), ("R2", "Y"), ("R3", "X")):
            rows = []
            for condition in ("E1", "E2", "E3", "E4"):
                prediction = e2_prediction if condition == "E2" else "X"
                rows.append(row(condition, "one", "X", prediction))
            repeats[name] = rows

        result = repeatability.analyze_repeats(repeats, ["X", "Y"])

        self.assertEqual(result["runs"]["R1"]["order_sensitive_cases"], [])
        self.assertEqual(result["runs"]["R2"]["order_sensitive_cases"], ["one"])
        self.assertEqual(result["runs"]["R3"]["order_sensitive_cases"], [])

    def test_repeat_analysis_excludes_errors_from_agreement_denominator(self):
        repeats = {
            "R1": [row("E1", "one", "X", "X")],
            "R2": [row("E1", "one", "X", None, error="обрезано")],
            "R3": [row("E1", "one", "X", "X")],
        }

        result = repeatability.analyze_repeats(repeats, ["X", "Y"])

        condition = result["repeatability_by_condition"]["E1"]
        self.assertEqual(condition["all_repeat_comparable_count"], 0)
        self.assertIsNone(condition["all_repeat_agreement_rate"])
        self.assertEqual(condition["uncompared_error_cases"], ["one"])

    def test_family_comparison_counts_corrected_and_broken_decisions(self):
        left = {
            "R1": [row("E1", "one", "X", "Y"), row("E1", "two", "Y", "Y")],
        }
        right = {
            "R1": [row("E1", "one", "X", "X"), row("E1", "two", "Y", "X")],
        }

        result = repeatability.compare_families(left, right)

        self.assertEqual(result["aggregate"]["changed_count"], 2)
        self.assertEqual(result["aggregate"]["corrected"], ["R1:E1:one"])
        self.assertEqual(result["aggregate"]["broken"], ["R1:E1:two"])


if __name__ == "__main__":
    unittest.main()
