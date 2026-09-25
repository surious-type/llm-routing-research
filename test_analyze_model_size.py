import unittest

import analyze_model_size as analysis


def row(condition, case_id, expected, predicted, *, latency=0.2):
    return {
        "condition": condition,
        "id": case_id,
        "expected_route": expected,
        "predicted_route": predicted,
        "correct": expected == predicted,
        "request_latency_sec": latency,
        "prompt_tokens": 10.0,
        "completion_tokens": 4.0,
        "error": None,
    }


class AnalyzeModelSizeTests(unittest.TestCase):
    def test_model_summary_reports_accuracy_agreement_and_changed_cases(self):
        rows = [
            row("P1", "one", "C", "C"),
            row("P1", "two", "N", "N"),
            row("P2", "one", "C", "N"),
            row("P2", "two", "N", "N"),
        ]

        result = analysis.summarize_model(rows, ["C", "N"])

        self.assertEqual(result["P1"]["accuracy"], 1.0)
        self.assertEqual(result["P2"]["accuracy"], 0.5)
        self.assertEqual(result["mean_accuracy"], 0.75)
        self.assertEqual(result["semantic_agreement"]["agreement_rate"], 0.5)
        self.assertEqual(result["order_sensitive_cases"], ["one"])

    def test_cross_model_comparison_lists_fixed_and_persistent_errors(self):
        small = [
            row("P1", "one", "C", "N"), row("P2", "one", "C", "N"),
            row("P1", "two", "N", "C"), row("P2", "two", "N", "C"),
        ]
        large = [
            row("P1", "one", "C", "C"), row("P2", "one", "C", "C"),
            row("P1", "two", "N", "C"), row("P2", "two", "N", "C"),
        ]

        result = analysis.compare_models(small, large)

        self.assertEqual(result["fixed_cases"], ["one"])
        self.assertEqual(result["persistent_error_cases"], ["two"])


if __name__ == "__main__":
    unittest.main()
