#!/usr/bin/env python3
"""Маршрутизация диалогов через специализированный reranker."""

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


DEFAULT_ENDPOINT = "http://localhost:8011/rerank"
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
QUERY_VERSION = "reranker-query-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--dataset", default="dataset_v2.jsonl")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
    names = {"user": "Пользователь", "assistant": "Ассистент"}
    return "\n".join(
        f"{names.get(item['role'], item['role'])}: {item['content']}"
        for item in history
    )


def build_query(row):
    return (
        f"ТЕКУЩИЙ ДИАЛОГ:\n{render_history(row['history'])}\n\n"
        f"НОВОЕ СООБЩЕНИЕ:\n{row['message']}"
    )


def build_candidate_text(route):
    return f"{route['name']}: {route['description']}"


def _score_from_item(item):
    for key in ("score", "raw_score", "relevance_score"):
        if key in item:
            return float(item[key])
    raise ValueError(f"В результате отсутствует score: {item}")


def extract_indexed_scores(data, expected_count):
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        items = data["results"]
    elif isinstance(data, dict) and "indices" in data and "scores" in data:
        if len(data["indices"]) != len(data["scores"]):
            raise ValueError("Длины indices и scores не совпадают")
        items = [
            {"index": index, "score": score}
            for index, score in zip(data["indices"], data["scores"])
        ]
    else:
        raise ValueError(f"Неизвестный формат ответа reranker: {data}")

    mapped = {}
    for item in items:
        index = int(item["index"])
        if index in mapped:
            raise ValueError(f"duplicate index: {index}")
        if index < 0 or index >= expected_count:
            raise ValueError(f"Индекс вне диапазона: {index}")
        mapped[index] = _score_from_item(item)
    expected_indices = set(range(expected_count))
    if set(mapped) != expected_indices:
        missing = sorted(expected_indices - set(mapped))
        raise ValueError(f"Отсутствуют индексы маршрутов: {missing}")
    return [mapped[index] for index in range(expected_count)]


def route_case(session, endpoint, row, routes, timeout):
    query = build_query(row)
    texts = [build_candidate_text(route) for route in routes]
    payload = {"query": query, "texts": texts, "raw_scores": True}
    started = time.perf_counter()
    response = session.post(endpoint, json=payload, timeout=timeout)
    latency = time.perf_counter() - started
    response.raise_for_status()
    raw_response = response.json()
    scores = extract_indexed_scores(raw_response, len(routes))
    route_scores = [
        {"input_index": index, "name": route["name"], "score": scores[index]}
        for index, route in enumerate(routes)
    ]
    ranked = sorted(route_scores, key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else None
    margin = best["score"] - second_score if second_score is not None else None
    return {
        "id": row["id"],
        "difficulty": row.get("difficulty"),
        "message": row["message"],
        "expected_route": row["expected_route"],
        "predicted_route": best["name"],
        "correct": best["name"] == row["expected_route"],
        "best_score": best["score"],
        "second_score": second_score,
        "margin": margin,
        "routes_json": json.dumps(route_scores, ensure_ascii=False),
        "request_latency_sec": latency,
        "error": None,
    }, {"id": row["id"], "query": query, "texts": texts, "response": raw_response}


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
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [row for row in dataset if row["id"] in selected]
        missing = selected - {row["id"] for row in cases}
        if missing:
            raise ValueError(f"Не найдены случаи: {sorted(missing)}")
    else:
        cases = dataset[args.start:] if args.limit is None else dataset[args.start:args.start + args.limit]
    if not cases:
        raise ValueError("После фильтрации не осталось случаев")

    metadata = {
        "model": args.model,
        "endpoint": args.endpoint,
        "query_version": QUERY_VERSION,
        "dataset": args.dataset,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "routes": args.routes,
        "routes_sha256": hashlib.sha256(routes_path.read_bytes()).hexdigest(),
        "candidate_format": "<route name>: <route description>",
        "raw_scores": True,
        "case_count": len(cases),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    session = requests.Session()
    rows = []
    raw_responses = []
    for row in cases:
        try:
            result, raw = route_case(session, args.endpoint, row, routes, args.timeout)
            raw_responses.append(raw)
        except Exception as exc:
            result = {
                "id": row["id"], "difficulty": row.get("difficulty"), "message": row["message"],
                "expected_route": row["expected_route"], "predicted_route": None, "correct": None,
                "best_score": None, "second_score": None, "margin": None, "routes_json": None,
                "request_latency_sec": None, "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(result)
        print(f"{result['id']}: predicted={result['predicted_route']} expected={result['expected_route']} error={result['error']}")

    save_csv(args.output, metadata, rows)
    Path(args.json_output).write_text(
        json.dumps({"metadata": metadata, "rows": rows, "raw_responses": raw_responses}, ensure_ascii=False, indent=2),
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
