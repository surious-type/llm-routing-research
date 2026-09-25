#!/usr/bin/env python3
"""Прямой выбор маршрута LLM через структурированный JSON-ответ."""

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "qwen3-4b"
PROMPT_VERSION = "structured-routing-v1"
SYSTEM_PROMPT = (
    "Ты выполняешь маршрутизацию диалога. Выбери один наиболее подходящий "
    "маршрут согласно описаниям доступных маршрутов. Верни выбор в "
    "требуемом структурированном формате."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--dataset", default="dataset_v2.jsonl")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    return parser.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_routes(path):
    with open(path, encoding="utf-8") as file:
        routes = json.load(file)
    if not routes:
        raise ValueError("Список маршрутов пуст")
    names = [route.get("name") for route in routes]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Имена маршрутов должны быть непустыми и уникальными")
    if any(not route.get("description") for route in routes):
        raise ValueError("У каждого маршрута должно быть описание")
    return routes


def render_history(history):
    role_names = {"user": "Пользователь", "assistant": "Ассистент"}
    return "\n".join(
        f"{role_names.get(item['role'], item['role'])}: {item['content']}"
        for item in history
    )


def build_user_prompt(row, routes):
    route_lines = "\n".join(
        f"{route['name']}: {route['description']}" for route in routes
    )
    return (
        f"ТЕКУЩИЙ ДИАЛОГ:\n{render_history(row['history'])}\n\n"
        f"НОВОЕ СООБЩЕНИЕ:\n{row['message']}\n\n"
        f"ДОСТУПНЫЕ МАРШРУТЫ:\n{route_lines}"
    )


def build_response_format(routes):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "route_selection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "enum": [route["name"] for route in routes],
                    }
                },
                "required": ["route"],
                "additionalProperties": False,
            },
        },
    }


def build_payload(model, row, routes, temperature=0.0, seed=0):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row, routes)},
        ],
        "temperature": temperature,
        "seed": seed,
        "max_completion_tokens": 32,
        "response_format": build_response_format(routes),
        "chat_template_kwargs": {"enable_thinking": False},
    }


def extract_structured_response(data, routes):
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("В ответе отсутствует structured content") from exc
    if isinstance(content, str):
        try:
            structured = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Структурированный ответ не является JSON") from exc
    elif isinstance(content, dict):
        structured = content
    else:
        raise ValueError("Неизвестный формат structured content")
    if set(structured) != {"route"}:
        raise ValueError("Структурированный ответ должен содержать только route")
    route = structured["route"]
    allowed = {item["name"] for item in routes}
    if route not in allowed:
        raise ValueError(f"Ответ содержит неизвестный маршрут: {route}")
    usage = data.get("usage") or {}
    return {
        "route": route,
        "structured_response": structured,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def verify_model(session, base_url, model, timeout):
    response = session.get(f"{base_url}/v1/models", timeout=timeout)
    response.raise_for_status()
    model_ids = [item["id"] for item in response.json().get("data", [])]
    if model not in model_ids:
        raise RuntimeError(
            f"Модель {model!r} не найдена на vLLM; доступны: {model_ids}"
        )
    return model_ids


def route_case(session, base_url, model, row, routes, timeout, temperature, seed):
    payload = build_payload(model, row, routes, temperature, seed)
    response = session.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=timeout
    )
    response.raise_for_status()
    raw_response = response.json()
    parsed = extract_structured_response(raw_response, routes)
    result = {
        "id": row["id"],
        "difficulty": row.get("difficulty"),
        "message": row["message"],
        "expected_route": row["expected_route"],
        "predicted_route": parsed["route"],
        "correct": parsed["route"] == row["expected_route"],
        "structured_response": json.dumps(
            parsed["structured_response"], ensure_ascii=False
        ),
        "prompt_tokens": parsed["prompt_tokens"],
        "completion_tokens": parsed["completion_tokens"],
        "error": None,
    }
    raw = {
        "id": row["id"],
        "payload": payload,
        "response": raw_response,
        "parsed_response": parsed["structured_response"],
    }
    return result, raw


def select_cases(dataset, case_ids, start, limit):
    if case_ids:
        requested = set(case_ids)
        cases = [row for row in dataset if row["id"] in requested]
        missing = requested - {row["id"] for row in cases}
        if missing:
            raise ValueError(f"Не найдены случаи: {sorted(missing)}")
        return cases
    return dataset[start:] if limit is None else dataset[start:start + limit]


def save_csv(path, metadata, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(metadata) + list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**metadata, **row})


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    routes_path = Path(args.routes)
    dataset = load_jsonl(dataset_path)
    routes = load_routes(routes_path)
    cases = select_cases(dataset, args.case_ids, args.start, args.limit)
    if not cases:
        raise ValueError("После фильтрации не осталось случаев")

    session = requests.Session()
    served_models = verify_model(session, args.base_url, args.model, args.timeout)
    metadata = {
        "model": args.model,
        "served_models_json": json.dumps(served_models, ensure_ascii=False),
        "base_url": args.base_url,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": args.temperature,
        "seed": args.seed,
        "thinking_enabled": False,
        "structured_output": "json_schema",
        "route_names_json": json.dumps(
            [route["name"] for route in routes], ensure_ascii=False
        ),
        "dataset": args.dataset,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "routes": args.routes,
        "routes_sha256": hashlib.sha256(routes_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    rows = []
    raw_responses = []
    for row in cases:
        started = time.perf_counter()
        try:
            result, raw = route_case(
                session, args.base_url, args.model, row, routes,
                args.timeout, args.temperature, args.seed,
            )
            raw_responses.append(raw)
        except Exception as exc:
            result = {
                "id": row["id"],
                "difficulty": row.get("difficulty"),
                "message": row["message"],
                "expected_route": row["expected_route"],
                "predicted_route": None,
                "correct": None,
                "structured_response": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["request_latency_sec"] = time.perf_counter() - started
        rows.append(result)
        print(
            f"{result['id']}: predicted={result['predicted_route']} "
            f"expected={result['expected_route']} error={result['error']}"
        )

    save_csv(args.output, metadata, rows)
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            {"metadata": metadata, "rows": rows, "raw_responses": raw_responses},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    errors = [row for row in rows if row["error"]]
    if errors:
        raise RuntimeError(f"Эксперимент неполный: ошибок {len(errors)} из {len(rows)}")
    correct = sum(row["correct"] for row in rows)
    latencies = [row["request_latency_sec"] for row in rows]
    print(f"Accuracy: {correct}/{len(rows)} = {correct / len(rows):.3%}")
    print(f"Mean latency: {statistics.mean(latencies):.6f} s")
    print(f"Median latency: {statistics.median(latencies):.6f} s")


if __name__ == "__main__":
    main()
