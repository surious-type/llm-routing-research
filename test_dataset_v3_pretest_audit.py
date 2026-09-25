import json
import tempfile
import unittest
from pathlib import Path

from dataset_v3_pretest_audit import (
    REQUIRED_FIELDS,
    audit_datasets,
    build_lexical_text,
    load_jsonl_strict,
)


def row(case_id, split, label, family_id=None):
    return {
        "id": case_id,
        "history": [
            {"role": "user", "content": "Исходная задача"},
            {"role": "assistant", "content": "Нужно уточнение"},
        ],
        "message": "Новый текст",
        "expected_route": label,
        "split": split,
        "category": "explicit_continuation" if label == "CONTINUE" else "same_topic_new_operation",
        "difficulty": "easy",
        "ambiguity": "low",
        "domain": "programming",
        "family_id": family_id,
        "policy_version": "v1",
        "annotation_notes": "основание",
    }


class DatasetAuditTests(unittest.TestCase):
    def test_load_jsonl_strict_reports_invalid_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"id":"ok"}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "строка 2"):
                load_jsonl_strict(path)

    def test_audit_rejects_family_crossing_splits(self):
        development = [row("d1", "development", "CONTINUE", "family_x")]
        validation = [row("v1", "validation", "NEW", "family_x")]
        with self.assertRaisesRegex(ValueError, "family_id пересекает"):
            audit_datasets(development, validation, expected_counts=None)

    def test_audit_rejects_duplicate_router_input_even_with_different_metadata(self):
        development = [row("d1", "development", "CONTINUE")]
        validation_row = row("v1", "validation", "NEW")
        validation_row["category"] = "same_entity_new_operation"
        with self.assertRaisesRegex(ValueError, "дубликат модельного входа"):
            audit_datasets(development, [validation_row], expected_counts=None)

    def test_audit_counts_metadata_and_classes(self):
        development = [
            row("d1", "development", "CONTINUE", "fd"),
            {**row("d2", "development", "NEW", "fd"), "message": "Иная операция"},
        ]
        validation = [
            {**row("v1", "validation", "CONTINUE", "fv"), "message": "Уточнение"},
            {**row("v2", "validation", "NEW", "fv"), "message": "Новая цель"},
        ]
        result = audit_datasets(
            development,
            validation,
            expected_counts={
                "development": {"total": 2, "CONTINUE": 1, "NEW": 1},
                "validation": {"total": 2, "CONTINUE": 1, "NEW": 1},
            },
        )
        self.assertEqual(set(REQUIRED_FIELDS), set(result["required_fields"]))
        self.assertEqual(result["splits"]["development"]["class_counts"], {"CONTINUE": 1, "NEW": 1})
        self.assertEqual(result["splits"]["validation"]["difficulty_counts"], {"easy": 2})
        self.assertEqual(result["family_overlap"], [])
        self.assertEqual(result["exact_duplicate_full_cases"], 0)

    def test_lexical_views_are_exact_and_metadata_free(self):
        sample = row("secret-id", "development", "NEW")
        sample["annotation_notes"] = "секрет"
        self.assertEqual(build_lexical_text(sample, "message"), "Новый текст")
        self.assertEqual(
            build_lexical_text(sample, "history"),
            "Пользователь: Исходная задача\nАссистент: Нужно уточнение",
        )
        combined = build_lexical_text(sample, "history_message")
        self.assertEqual(
            combined,
            "ТЕКУЩИЙ ДИАЛОГ:\nПользователь: Исходная задача\nАссистент: Нужно уточнение\n\nНОВОЕ СООБЩЕНИЕ:\nНовый текст",
        )
        self.assertNotIn("secret-id", combined)
        self.assertNotIn("секрет", combined)


if __name__ == "__main__":
    unittest.main()
