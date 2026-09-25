"""Эксперимент с cross-encoder reranker."""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

import requests

from llm_routing.common import (
    extract_reranker_scores,
    load_jsonl,
    load_routes,
    render_history,
    save_csv,
    save_json,
    select_cases,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/diagnostic/routing_v2.jsonl")
parser.add_argument("--routes", default="data/routes/continue_first.json")
parser.add_argument("--endpoint", default="http://localhost:8011/rerank")
parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--case-ids", nargs="*", default=None)
parser.add_argument("--timeout", type=float, default=120)
parser.add_argument("--output", default="results/reranker/latest.csv")
parser.add_argument("--json-output", default="results/reranker/latest.json")
args = parser.parse_args()

# 1. Загружаем один датасет и один явно заданный порядок маршрутов.
dataset = load_jsonl(args.dataset)
cases = select_cases(dataset, args.start, args.limit, args.case_ids)
routes = load_routes(args.routes)
if not cases:
    raise ValueError("После фильтрации не осталось случаев")

# 2. Каждый case отправляем reranker одним query + списком candidate texts.
session = requests.Session()
rows = []
raw_responses = []

for row in cases:
    query = (
        f"ТЕКУЩИЙ ДИАЛОГ:\n{render_history(row['history'])}\n\n"
        f"НОВОЕ СООБЩЕНИЕ:\n{row['message']}"
    )
    texts = [
        f"{route['name']}: {route['description']}"
        for route in routes
    ]
    payload = {
        "query": query,
        "texts": texts,
        "raw_scores": True,
    }

    started = time.perf_counter()
    error = None
    predicted_route = None
    best_score = None
    second_score = None
    margin = None
    route_scores = []
    raw = None

    try:
        response = session.post(
            args.endpoint,
            json=payload,
            timeout=args.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        scores = extract_reranker_scores(raw, len(routes))
        route_scores = [
            {
                "index": index,
                "name": route["name"],
                "score": scores[index],
            }
            for index, route in enumerate(routes)
        ]
        ranked = sorted(
            route_scores,
            key=lambda item: item["score"],
            reverse=True,
        )
        predicted_route = ranked[0]["name"]
        best_score = ranked[0]["score"]
        second_score = ranked[1]["score"] if len(ranked) > 1 else None
        margin = (
            best_score - second_score
            if second_score is not None
            else None
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - started
    rows.append({
        "id": row["id"],
        "expected_route": row["expected_route"],
        "predicted_route": predicted_route,
        "correct": (
            predicted_route == row["expected_route"]
            if predicted_route
            else None
        ),
        "best_score": best_score,
        "second_score": second_score,
        "margin": margin,
        "routes_json": json.dumps(route_scores, ensure_ascii=False),
        "latency_sec": latency,
        "error": error,
    })
    raw_responses.append({
        "id": row["id"],
        "payload": payload,
        "response": raw,
        "error": error,
    })
    print(
        f"{row['id']}: expected={row['expected_route']} "
        f"predicted={predicted_route} error={error}"
    )

# 3. Сохраняем результат. Для второго порядка запускаем тот же файл с другим --routes.
save_csv(args.output, rows)
valid = [row for row in rows if not row["error"]]
correct = sum(bool(row["correct"]) for row in valid)

save_json(args.json_output, {
    "metadata": {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "routes": args.routes,
        "endpoint": args.endpoint,
        "model": args.model,
    },
    "summary": {
        "evaluated": len(valid),
        "correct": correct,
        "accuracy": correct / len(valid) if valid else None,
        "mean_latency_sec": (
            statistics.mean(row["latency_sec"] for row in valid)
            if valid
            else None
        ),
    },
    "rows": rows,
    "raw_responses": raw_responses,
})
