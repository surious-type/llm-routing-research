import unittest

from llm_routing.common import (
    extract_reranker_scores,
    normalize_logprobs,
    structured_response_format,
)


class CommonTests(unittest.TestCase):
    def test_normalize_logprobs(self):
        values = normalize_logprobs({"A": 0.0, "B": 0.0})
        self.assertAlmostEqual(values["A"], 0.5)
        self.assertAlmostEqual(values["B"], 0.5)

    def test_reranker_response_is_restored_by_index(self):
        scores = extract_reranker_scores(
            [{"index": 1, "score": 0.2}, {"index": 0, "score": 0.9}],
            2,
        )
        self.assertEqual(scores, [0.9, 0.2])

    def test_structured_schema_uses_requested_enum_order(self):
        schema = structured_response_format(["NEW", "CONTINUE"], True)
        route = schema["json_schema"]["schema"]["properties"]["route"]
        self.assertEqual(route["enum"], ["NEW", "CONTINUE"])
        self.assertEqual(
            schema["json_schema"]["schema"]["required"],
            ["reason", "route"],
        )


if __name__ == "__main__":
    unittest.main()
