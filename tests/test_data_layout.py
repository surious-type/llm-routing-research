import json
import unittest
from pathlib import Path


class DataLayoutTests(unittest.TestCase):
    def test_route_configs_only_change_order(self):
        continue_first = json.loads(
            Path("data/routes/continue_first.json").read_text(encoding="utf-8")
        )
        new_first = json.loads(
            Path("data/routes/new_first.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            [item["name"] for item in continue_first],
            ["CONTINUE", "NEW"],
        )
        self.assertEqual(
            [item["name"] for item in new_first],
            ["NEW", "CONTINUE"],
        )

        by_name_a = {item["name"]: item["description"] for item in continue_first}
        by_name_b = {item["name"]: item["description"] for item in new_first}
        self.assertEqual(by_name_a, by_name_b)

    def test_diagnostic_dataset_uses_semantic_labels(self):
        rows = [
            json.loads(line)
            for line in Path("data/diagnostic/routing_v2.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 30)
        self.assertEqual(
            {row["expected_route"] for row in rows},
            {"CONTINUE", "NEW"},
        )


if __name__ == "__main__":
    unittest.main()
