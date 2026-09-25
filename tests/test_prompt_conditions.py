import json
import unittest
from pathlib import Path


class PromptConditionsTests(unittest.TestCase):
    def test_e1_e4_matrix_is_fixed(self):
        conditions = json.loads(
            Path("experiments/prompt_conditions/conditions.json").read_text(
                encoding="utf-8"
            )
        )
        actual = {
            item["id"]: (
                item["semantic_A"],
                item["semantic_B"],
                item["position_A"],
            )
            for item in conditions
        }
        self.assertEqual(
            actual,
            {
                "E1": ("CONTINUE", "NEW", "first"),
                "E2": ("NEW", "CONTINUE", "first"),
                "E3": ("CONTINUE", "NEW", "second"),
                "E4": ("NEW", "CONTINUE", "second"),
            },
        )


if __name__ == "__main__":
    unittest.main()
