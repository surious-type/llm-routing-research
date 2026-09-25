#!/usr/bin/env python3
"""Audit Dataset V3 development/validation and run lexical diagnostics."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from structured_routing import render_history


REQUIRED_FIELDS = (
    "id", "history", "message", "expected_route", "split", "category",
    "difficulty", "ambiguity", "domain", "family_id", "policy_version",
    "annotation_notes",
)
LABELS = ("CONTINUE", "NEW")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl_strict(path):
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                raise ValueError(f"Пустая строка {line_number} в {path}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Невалидный JSON, строка {line_number} в {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Строка {line_number} не является объектом")
            rows.append(value)
    return rows


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def router_input(row):
    return {"history": row["history"], "message": row["message"]}


def _validate_row(row, expected_split):
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{row.get('id', '<без id>')}: отсутствуют поля {missing}")
    if row["split"] != expected_split:
        raise ValueError(f"{row['id']}: split={row['split']}, ожидался {expected_split}")
    if row["expected_route"] not in LABELS:
        raise ValueError(f"{row['id']}: неизвестный expected_route")
    if row["difficulty"] not in {"easy", "medium", "hard"}:
        raise ValueError(f"{row['id']}: неизвестная difficulty")
    if row["ambiguity"] not in {"low", "medium", "high"}:
        raise ValueError(f"{row['id']}: неизвестная ambiguity")
    if not isinstance(row["history"], list) or not row["history"]:
        raise ValueError(f"{row['id']}: history должен быть непустым списком")
    if not isinstance(row["message"], str) or not row["message"].strip():
        raise ValueError(f"{row['id']}: message должен быть непустой строкой")
    for item in row["history"]:
        if set(item) != {"role", "content"}:
            raise ValueError(f"{row['id']}: history item содержит неверные поля")
        if item["role"] not in {"user", "assistant"} or not isinstance(item["content"], str):
            raise ValueError(f"{row['id']}: неверный history item")


def _duplicate_count(values):
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _split_summary(rows):
    return {
        "count": len(rows),
        "class_counts": dict(sorted(Counter(row["expected_route"] for row in rows).items())),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "ambiguity_counts": dict(sorted(Counter(row["ambiguity"] for row in rows).items())),
        "family_count": len({row["family_id"] for row in rows if row["family_id"] is not None}),
        "standalone_count": sum(row["family_id"] is None for row in rows),
    }


def audit_datasets(development, validation, expected_counts=None):
    for row in development:
        _validate_row(row, "development")
    for row in validation:
        _validate_row(row, "validation")
    all_rows = development + validation
    ids = [row["id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("id не уникальны между development/validation")

    development_families = {row["family_id"] for row in development if row["family_id"] is not None}
    validation_families = {row["family_id"] for row in validation if row["family_id"] is not None}
    overlap = sorted(development_families & validation_families)
    if overlap:
        raise ValueError(f"family_id пересекает splits: {overlap}")

    model_inputs = [_canonical(router_input(row)) for row in all_rows]
    duplicate_inputs = _duplicate_count(model_inputs)
    if duplicate_inputs:
        raise ValueError(f"Найден дубликат модельного входа: {duplicate_inputs}")
    full_without_id = [
        _canonical({key: value for key, value in row.items() if key != "id"})
        for row in all_rows
    ]
    duplicate_full = _duplicate_count(full_without_id)
    if duplicate_full:
        raise ValueError(f"Найден exact duplicate full case: {duplicate_full}")

    summaries = {
        "development": _split_summary(development),
        "validation": _split_summary(validation),
    }
    if expected_counts:
        for split, expected in expected_counts.items():
            actual = summaries[split]
            if actual["count"] != expected["total"]:
                raise ValueError(f"{split}: ожидалось {expected['total']}, получено {actual['count']}")
            for label in LABELS:
                if actual["class_counts"].get(label, 0) != expected[label]:
                    raise ValueError(f"{split}: неверный баланс {label}")
    return {
        "status": "passed",
        "required_fields": list(REQUIRED_FIELDS),
        "router_input_allowlist": ["history[].role", "history[].content", "message"],
        "splits": summaries,
        "unique_ids": len(ids),
        "family_overlap": overlap,
        "exact_duplicate_full_cases": duplicate_full,
        "exact_duplicate_router_inputs": duplicate_inputs,
    }


def build_lexical_text(row, view):
    history = render_history(row["history"])
    if view == "message":
        return row["message"]
    if view == "history":
        return history
    if view == "history_message":
        return f"ТЕКУЩИЙ ДИАЛОГ:\n{history}\n\nНОВОЕ СООБЩЕНИЕ:\n{row['message']}"
    raise ValueError(f"Неизвестное представление: {view}")


def run_lexical_baselines(development, validation):
    y_train = [row["expected_route"] for row in development]
    y_eval = [row["expected_route"] for row in validation]
    results = {}
    for view in ("message", "history", "history_message"):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)),
            ("logreg", LogisticRegression(random_state=0, max_iter=2000)),
        ])
        pipeline.fit([build_lexical_text(row, view) for row in development], y_train)
        predictions = pipeline.predict([build_lexical_text(row, view) for row in validation])
        results[view] = {
            "accuracy": float(accuracy_score(y_eval, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_eval, predictions)),
            "confusion_matrix": confusion_matrix(y_eval, predictions, labels=list(LABELS)).tolist(),
            "label_order": list(LABELS),
            "predictions": [
                {"id": row["id"], "expected_route": expected, "predicted_route": predicted, "correct": bool(expected == predicted)}
                for row, expected, predicted in zip(validation, y_eval, predictions)
            ],
        }
    return {
        "method": "TfidfVectorizer(word,1-2gram)+LogisticRegression",
        "scikit_learn_version": __import__("sklearn").__version__,
        "train_split": "development",
        "eval_split": "validation",
        "conditions": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", default="dataset_v3_development.jsonl")
    parser.add_argument("--validation", default="dataset_v3_validation.jsonl")
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--lexical-output", required=True)
    args = parser.parse_args()
    for path in (args.audit_output, args.lexical_output):
        if Path(path).exists():
            raise FileExistsError(f"Файл уже существует: {path}")
    development = load_jsonl_strict(args.development)
    validation = load_jsonl_strict(args.validation)
    audit = audit_datasets(
        development,
        validation,
        expected_counts={
            "development": {"total": 120, "CONTINUE": 60, "NEW": 60},
            "validation": {"total": 60, "CONTINUE": 30, "NEW": 30},
        },
    )
    audit["files"] = {
        "development": {"path": args.development, "sha256": sha256(args.development)},
        "validation": {"path": args.validation, "sha256": sha256(args.validation)},
    }
    lexical = run_lexical_baselines(development, validation)
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.lexical_output).write_text(json.dumps(lexical, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit["status"], "lexical": {key: value["accuracy"] for key, value in lexical["conditions"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
