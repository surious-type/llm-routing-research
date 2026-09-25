#!/usr/bin/env python3
"""Факторный запуск direct structured routing с независимыми порядками."""

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from structured_routing import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    load_jsonl,
    load_routes,
    render_history,
    select_cases,
    verify_model,
)


PROMPT_VERSION = "structured-factorial-v1"
MODES = ("route-only", "reason-route")
REASON_INSTRUCTION = (
    " Сначала кратко сформулируй основание выбора в поле reason, затем укажи "
    "выбранный маршрут в поле route."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", default="routes_continue_first.json")
    parser.add_argument("--dataset", default="dataset_v2.jsonl")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-completion-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    return parser.parse_args()


def build_conditions(route_names):
    forward = list(route_names)
    reverse = list(reversed(route_names))
    return {
        "E1": {"prompt_order": forward, "enum_order": forward},
        "E2": {"prompt_order": reverse, "enum_order": forward},
        "E3": {"prompt_order": forward, "enum_order": reverse},
        "E4": {"prompt_order": reverse, "enum_order": reverse},
    }


def _route_map(routes):
    return {route["name"]: route for route in routes}


def validate_order(routes, order):
    names = [route["name"] for route in routes]
    if len(order) != len(names) or set(order) != set(names):
        raise ValueError("Порядок должен содержать каждый маршрут ровно один раз")


def build_user_prompt(row, routes, prompt_order):
    validate_order(routes, prompt_order)
    by_name = _route_map(routes)
    route_lines = "\n".join(
        f"{name}: {by_name[name]['description']}" for name in prompt_order
    )
    return (
        f"ТЕКУЩИЙ ДИАЛОГ:\n{render_history(row['history'])}\n\n"
        f"НОВОЕ СООБЩЕНИЕ:\n{row['message']}\n\n"
        f"ДОСТУПНЫЕ МАРШРУТЫ:\n{route_lines}"
    )


def build_response_format(enum_order, mode):
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим: {mode}")
    route_property = {"type": "string", "enum": list(enum_order)}
    if mode == "route-only":
        properties = {"route": route_property}
        required = ["route"]
    else:
        properties = {
            "reason": {"type": "string"},
            "route": route_property,
        }
        required = ["reason", "route"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "route_selection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def build_payload(
    model, row, routes, condition, mode, *, max_completion_tokens=32,
    enable_thinking=False,
):
    validate_order(routes, condition["prompt_order"])
    validate_order(routes, condition["enum_order"])
    system_prompt = SYSTEM_PROMPT + (REASON_INSTRUCTION if mode == "reason-route" else "")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_prompt(row, routes, condition["prompt_order"]),
            },
        ],
        "temperature": 0,
        "seed": 0,
        "max_completion_tokens": max_completion_tokens,
        "response_format": build_response_format(condition["enum_order"], mode),
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


def extract_structured_response(data, allowed_routes, mode):
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

    required = {"route"} if mode == "route-only" else {"reason", "route"}
    if set(structured) != required:
        raise ValueError(f"Поля structured response не совпадают со схемой {sorted(required)}")
    route = structured["route"]
    if route not in set(allowed_routes):
        raise ValueError(f"Ответ содержит неизвестный маршрут: {route}")
    reason = structured.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("Поле reason должно быть строкой")
    usage = data.get("usage") or {}
    return {
        "route": route,
        "reason": reason,
        "structured_response": structured,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def route_case(
    session, base_url, model, row, routes, condition_name, condition, mode,
    timeout, max_completion_tokens=32, enable_thinking=False,
):
    payload = build_payload(
        model, row, routes, condition, mode,
        max_completion_tokens=max_completion_tokens,
        enable_thinking=enable_thinking,
    )
    response = session.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=timeout
    )
    response.raise_for_status()
    raw_response = response.json()
    raw = {
        "condition": condition_name,
        "mode": mode,
        "id": row["id"],
        "prompt_route_order": condition["prompt_order"],
        "enum_route_order": condition["enum_order"],
        "payload": payload,
        "response": raw_response,
    }
    try:
        parsed = extract_structured_response(raw_response, condition["enum_order"], mode)
    except ValueError as exc:
        content = None
        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        usage = raw_response.get("usage") or {}
        error = f"{type(exc).__name__}: {exc}"
        raw["parse_error"] = error
        return {
            "condition": condition_name,
            "mode": mode,
            "id": row["id"],
            "difficulty": row.get("difficulty"),
            "message": row["message"],
            "expected_route": row["expected_route"],
            "predicted_route": None,
            "correct": None,
            "reason": None,
            "prompt_route_order": json.dumps(condition["prompt_order"], ensure_ascii=False),
            "enum_route_order": json.dumps(condition["enum_order"], ensure_ascii=False),
            "structured_response": content,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "error": error,
        }, raw
    result = {
        "condition": condition_name,
        "mode": mode,
        "id": row["id"],
        "difficulty": row.get("difficulty"),
        "message": row["message"],
        "expected_route": row["expected_route"],
        "predicted_route": parsed["route"],
        "correct": parsed["route"] == row["expected_route"],
        "reason": parsed["reason"],
        "prompt_route_order": json.dumps(condition["prompt_order"], ensure_ascii=False),
        "enum_route_order": json.dumps(condition["enum_order"], ensure_ascii=False),
        "structured_response": json.dumps(parsed["structured_response"], ensure_ascii=False),
        "prompt_tokens": parsed["prompt_tokens"],
        "completion_tokens": parsed["completion_tokens"],
        "error": None,
    }
    raw["parsed_response"] = parsed["structured_response"]
    return result, raw


def ensure_new_outputs(*paths):
    existing = [str(path) for path in map(Path, paths) if path.exists()]
    if existing:
        raise FileExistsError(f"Файлы уже существуют и не будут перезаписаны: {existing}")


def save_csv(path, metadata, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(metadata) + list(rows[0])
    with output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**metadata, **row})


def main():
    args = parse_args()
    ensure_new_outputs(args.output, args.json_output)
    dataset_path = Path(args.dataset)
    routes_path = Path(args.routes)
    dataset = load_jsonl(dataset_path)
    routes = load_routes(routes_path)
    route_names = [route["name"] for route in routes]
    conditions = build_conditions(route_names)
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
        "mode": args.mode,
        "temperature": 0,
        "seed": 0,
        "thinking_enabled": args.enable_thinking,
        "max_completion_tokens": args.max_completion_tokens,
        "structured_output": "json_schema",
        "route_names_json": json.dumps(route_names, ensure_ascii=False),
        "conditions_json": json.dumps(conditions, ensure_ascii=False),
        "dataset": args.dataset,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "routes": args.routes,
        "routes_sha256": hashlib.sha256(routes_path.read_bytes()).hexdigest(),
        "case_count_per_condition": len(cases),
        "request_count": len(cases) * len(conditions),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    rows = []
    raw_responses = []
    for condition_name, condition in conditions.items():
        for source_row in cases:
            started = time.perf_counter()
            try:
                result, raw = route_case(
                    session, args.base_url, args.model, source_row, routes,
                    condition_name, condition, args.mode, args.timeout,
                    args.max_completion_tokens, args.enable_thinking,
                )
                raw_responses.append(raw)
            except Exception as exc:
                result = {
                    "condition": condition_name,
                    "mode": args.mode,
                    "id": source_row["id"],
                    "difficulty": source_row.get("difficulty"),
                    "message": source_row["message"],
                    "expected_route": source_row["expected_route"],
                    "predicted_route": None,
                    "correct": None,
                    "reason": None,
                    "prompt_route_order": json.dumps(condition["prompt_order"], ensure_ascii=False),
                    "enum_route_order": json.dumps(condition["enum_order"], ensure_ascii=False),
                    "structured_response": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result["request_latency_sec"] = time.perf_counter() - started
            rows.append(result)
            print(
                f"{condition_name} {result['id']}: "
                f"predicted={result['predicted_route']} "
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
    for condition_name in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition_name]
        correct = sum(row["correct"] for row in condition_rows)
        latencies = [row["request_latency_sec"] for row in condition_rows]
        print(
            f"{condition_name}: accuracy={correct}/{len(condition_rows)} "
            f"({correct / len(condition_rows):.3%}), "
            f"mean_latency={statistics.mean(latencies):.6f}s"
        )


if __name__ == "__main__":
    main()
