import json
import unittest

from compare_reranker_runs import compare_rows


class CompareRerankerRunsTests(unittest.TestCase):
    def test_compare_rows_maps_scores_by_route_name(self):
        run_a = {
            "case_1": {
                "id": "case_1",
                "expected_route": "CONTINUE",
                "predicted_route": "CONTINUE",
                "correct": "True",
                "error": "",
                "margin": "0.7",
                "request_latency_sec": "0.01",
                "routes_json": json.dumps([
                    {"name": "CONTINUE", "score": 0.8},
                    {"name": "NEW", "score": 0.1},
                ]),
            }
        }
        run_b = {
            "case_1": {
                "id": "case_1",
                "expected_route": "CONTINUE",
                "predicted_route": "CONTINUE",
                "correct": "True",
                "error": "",
                "margin": "0.7",
                "request_latency_sec": "0.01",
                "routes_json": json.dumps([
                    {"name": "NEW", "score": 0.1},
                    {"name": "CONTINUE", "score": 0.8},
                ]),
            }
        }
        result = compare_rows(run_a, run_b)
        self.assertEqual(result["semantic_agreement"]["count"], 1)
        self.assertEqual(result["per_case_score_differences"][0]["score_differences"], {"CONTINUE": 0.0, "NEW": 0.0})

    def test_compare_rows_keeps_error_cases_without_parsing_scores(self):
        failed = {
            "id": "case_1",
            "expected_route": "NEW",
            "predicted_route": "",
            "correct": "",
            "error": "HTTPError",
            "routes_json": "",
            "margin": "",
            "request_latency_sec": "",
        }
        result = compare_rows({"case_1": failed}, {"case_1": failed})
        self.assertEqual(result["run_a"]["error_count"], 1)
        self.assertEqual(result["uncompared_error_cases"], ["case_1"])
        self.assertEqual(result["per_case_score_differences"], [])


if __name__ == "__main__":
    unittest.main()
