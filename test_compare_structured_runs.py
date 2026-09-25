import unittest

import compare_structured_runs as comparison


class CompareStructuredRunsTests(unittest.TestCase):
    def test_summary_builds_generic_confusion_matrix(self):
        rows = [
            {"id": "one", "expected_route": "X", "predicted_route": "X", "correct": True, "error": None},
            {"id": "two", "expected_route": "Y", "predicted_route": "X", "correct": False, "error": None},
            {"id": "three", "expected_route": "X", "predicted_route": None, "correct": None, "error": "timeout"},
        ]

        summary = comparison.summarize_run(rows, ["X", "Y"])

        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["evaluated"], 2)
        self.assertEqual(summary["errors"], ["three"])
        self.assertEqual(
            summary["confusion_matrix"],
            {
                "labels": ["X", "Y"],
                "matrix": [[1, 0], [1, 0]],
            },
        )

    def test_comparison_uses_semantic_route_identity(self):
        run_a = [
            {"id": "one", "expected_route": "X", "predicted_route": "X", "correct": True, "error": None},
            {"id": "two", "expected_route": "Y", "predicted_route": "Y", "correct": True, "error": None},
        ]
        run_b = [
            {"id": "one", "expected_route": "X", "predicted_route": "X", "correct": True, "error": None},
            {"id": "two", "expected_route": "Y", "predicted_route": "X", "correct": False, "error": None},
        ]

        result = comparison.compare_runs(run_a, run_b, ["X", "Y"])

        self.assertEqual(result["semantic_agreement"], {"count": 1, "total": 2, "rate": 0.5})
        self.assertEqual(result["order_sensitive_cases"], ["two"])


if __name__ == "__main__":
    unittest.main()
