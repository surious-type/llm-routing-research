import unittest

from representation_diagnostic import (
    pairwise_prediction_analysis,
    pearson,
    select_description_condition_ids,
    summarize_condition,
    validate_conditions,
)


class RepresentationDiagnosticTests(unittest.TestCase):
    def test_validate_conditions_requires_both_semantics(self):
        conditions = [{
            "id": "broken",
            "factor": "name",
            "candidates": [{"semantic": "CONTINUE", "name": "X", "description": "d"}],
        }]
        with self.assertRaisesRegex(ValueError, "CONTINUE and NEW"):
            validate_conditions(conditions)

    def test_pearson_returns_one_for_affine_vectors(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [4, 6, 8]), 1.0)

    def test_pearson_is_undefined_for_single_observation(self):
        self.assertIsNone(pearson([1], [2]))

    def test_summarize_condition_uses_semantic_scores(self):
        rows = [
            {"id": "a", "expected_route": "CONTINUE", "semantic": "CONTINUE", "score": 3.0},
            {"id": "a", "expected_route": "CONTINUE", "semantic": "NEW", "score": 1.0},
            {"id": "b", "expected_route": "NEW", "semantic": "CONTINUE", "score": 4.0},
            {"id": "b", "expected_route": "NEW", "semantic": "NEW", "score": 5.0},
        ]
        summary, cases = summarize_condition(rows, {"a": "CONTINUE", "b": "CONTINUE"})
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["delta_positive_count"], 1)
        self.assertEqual(summary["semantic_agreement_with_baseline"], 0.5)
        self.assertEqual(summary["changed_prediction_cases"], ["b"])
        self.assertEqual([x["delta_continue_minus_new"] for x in cases], [2.0, -1.0])
        self.assertEqual(summary["candidate_score_distributions"]["CONTINUE"]["mean"], 3.5)
        self.assertEqual(summary["candidate_score_distributions"]["NEW"]["mean"], 3.0)

    def test_pairwise_prediction_analysis_lists_changed_cases(self):
        cases = {
            "left": [{"id": "a", "predicted_route": "CONTINUE"}, {"id": "b", "predicted_route": "NEW"}],
            "right": [{"id": "a", "predicted_route": "CONTINUE"}, {"id": "b", "predicted_route": "CONTINUE"}],
        }
        result = pairwise_prediction_analysis(cases, ["left", "right"])
        self.assertEqual(result[0]["semantic_agreement"], 0.5)
        self.assertEqual(result[0]["changed_prediction_cases"], ["b"])

    def test_swap_only_config_has_no_description_conditions(self):
        conditions = [
            {"id": "normal", "factor": "route_name"},
            {"id": "swapped", "factor": "route_name"},
        ]
        self.assertEqual(select_description_condition_ids(conditions), [])


if __name__ == "__main__":
    unittest.main()
