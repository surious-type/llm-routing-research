import copy
import json
import unittest

from dataset_v3_pretest_routing import (
    EVALUATION_FIELDS,
    attach_evaluation_metadata,
    build_conditions,
    prompt_bytes,
    select_conditions,
)


ROUTES = [
    {"name": "CONTINUE", "description": "Описание продолжения"},
    {"name": "NEW", "description": "Описание новой задачи"},
]


def sample_row():
    return {
        "id": "v3_development_0001",
        "history": [
            {"role": "user", "content": "Сделай текущую задачу"},
            {"role": "assistant", "content": "Уточните параметр"},
        ],
        "message": "Используй второй вариант",
        "expected_route": "CONTINUE",
        "split": "development",
        "category": "clarification_answer",
        "difficulty": "easy",
        "ambiguity": "low",
        "domain": "programming",
        "family_id": "family_1",
        "policy_version": "policy-1",
        "annotation_notes": "секретное основание",
    }


class MetadataSafeRoutingTests(unittest.TestCase):
    def test_all_evaluation_metadata_changes_leave_prompt_bytes_identical(self):
        original = sample_row()
        mutated = copy.deepcopy(original)
        for field in EVALUATION_FIELDS:
            if field == "family_id":
                mutated[field] = None
            elif field == "expected_route":
                mutated[field] = "NEW"
            else:
                mutated[field] = f"CHANGED_{field}"
        for name, condition in build_conditions([route["name"] for route in ROUTES]).items():
            self.assertEqual(
                prompt_bytes("served-model", original, ROUTES, condition),
                prompt_bytes("served-model", mutated, ROUTES, condition),
                name,
            )

    def test_prompt_payload_fixes_orders_schema_and_decoding(self):
        conditions = build_conditions(["CONTINUE", "NEW"])
        self.assertEqual(conditions["P1"], {"prompt_order": ["CONTINUE", "NEW"], "enum_order": ["CONTINUE", "NEW"]})
        self.assertEqual(conditions["P2"], {"prompt_order": ["NEW", "CONTINUE"], "enum_order": ["CONTINUE", "NEW"]})
        payload = json.loads(prompt_bytes("qwen3-8b", sample_row(), ROUTES, conditions["P2"]))
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(payload["max_completion_tokens"], 128)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["route"]["enum"], ["CONTINUE", "NEW"])
        self.assertEqual(schema["required"], ["reason", "route"])
        self.assertLess(payload["messages"][1]["content"].index("NEW:"), payload["messages"][1]["content"].index("CONTINUE:"))

    def test_output_metadata_is_attached_after_prediction_without_annotation_notes(self):
        result = {"id": "v3_development_0001", "predicted_route": "CONTINUE"}
        attached = attach_evaluation_metadata(result, sample_row())
        self.assertEqual(attached["category"], "clarification_answer")
        self.assertEqual(attached["family_id"], "family_1")
        self.assertNotIn("annotation_notes", attached)

    def test_recovery_can_select_only_failed_order_without_changing_it(self):
        conditions = build_conditions(["CONTINUE", "NEW"])
        selected = select_conditions(conditions, ["P2"])
        self.assertEqual(selected, {"P2": {"prompt_order": ["NEW", "CONTINUE"], "enum_order": ["CONTINUE", "NEW"]}})
        with self.assertRaisesRegex(ValueError, "Неизвестные conditions"):
            select_conditions(conditions, ["P3"])


if __name__ == "__main__":
    unittest.main()
