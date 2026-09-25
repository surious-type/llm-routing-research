import unittest

from reranker_routing import build_candidate_text, build_query, extract_indexed_scores


class RerankerRoutingTests(unittest.TestCase):
    def test_build_query_contains_only_history_and_new_message(self):
        row = {
            "history": [
                {"role": "user", "content": "Вопрос"},
                {"role": "assistant", "content": "Ответ"},
            ],
            "message": "Новое сообщение",
        }
        self.assertEqual(
            build_query(row),
            "ТЕКУЩИЙ ДИАЛОГ:\nПользователь: Вопрос\nАссистент: Ответ\n\n"
            "НОВОЕ СООБЩЕНИЕ:\nНовое сообщение",
        )

    def test_build_candidate_text_uses_name_and_description(self):
        route = {"name": "ROUTE", "description": "Описание"}
        self.assertEqual(build_candidate_text(route), "ROUTE: Описание")

    def test_extract_indexed_scores_maps_results_by_index(self):
        response = {
            "results": [
                {"index": 1, "relevance_score": -0.2},
                {"index": 0, "relevance_score": 0.7},
            ]
        }
        self.assertEqual(extract_indexed_scores(response, 2), [0.7, -0.2])

    def test_extract_indexed_scores_maps_actual_list_response(self):
        response = [
            {"index": 1, "score": -0.2},
            {"index": 0, "score": 0.7},
        ]
        self.assertEqual(extract_indexed_scores(response, 2), [0.7, -0.2])

    def test_extract_indexed_scores_maps_parallel_indices_and_scores(self):
        response = {"indices": [1, 0], "scores": [-0.2, 0.7]}
        self.assertEqual(extract_indexed_scores(response, 2), [0.7, -0.2])

    def test_extract_indexed_scores_rejects_duplicate_index(self):
        response = {
            "results": [
                {"index": 0, "score": 0.7},
                {"index": 0, "score": -0.2},
            ]
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            extract_indexed_scores(response, 2)


if __name__ == "__main__":
    unittest.main()
