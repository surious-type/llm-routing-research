"""Независимая оценка каждого маршрута через logP(1)-logP(0)."""

import argparse
import json
import math
import time
from datetime import datetime, timezone

import requests

from llm_routing.common import (
    discover_model,
    extract_named_logprobs,
    load_jsonl,
    load_routes,
    normalize_logprobs,
    render_history,
    save_csv,
    save_json,
    select_cases,
    tokenize_single,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/diagnostic/routing_v2.jsonl")
parser.add_argument("--routes", default="data/routes/continue_first.json")
parser.add_argument("--base-url", default="http://localhost:8000")
parser.add_argument("--model", default=None)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--case-ids", nargs="*", default=None)
parser.add_argument("--timeout", type=float, default=120)
parser.add_argument(
    "--output",
    default="results/independent_scoring/latest.csv",
)
parser.add_argument(
    "--json-output",
    default="results/independent_scoring/latest.json",
)
args = parser.parse_args()

# 1. Читаем датасет и описания кандидатов.
dataset = load_jsonl(args.dataset)
cases = select_cases(dataset, args.start, args.limit, args.case_ids)
routes = load_routes(args.routes)
if not cases:
    raise ValueError("После фильтрации не осталось случаев")

# 2. Подключаемся к модели и заранее определяем token ids для 1 и 0.
session = requests.Session()
model = args.model or discover_model(session, args.base_url, args.timeout)
token_one = tokenize_single(
    session,
    args.base_url,
    model,
    "1",
    args.timeout,
)
token_zero = tokenize_single(
    session,
    args.base_url,
    model,
    "0",
    args.timeout,
)

# 3. Для каждого case оцениваем каждый маршрут отдельным запросом.
rows = []
raw_responses = []
for row in cases:
    candidate_results = []
    total_latency = 0.0
    case_error = None

    for route in routes:
        user_prompt = (
            f"ТЕКУЩИЙ ДИАЛОГ:\n{render_history(row['history'])}\n\n"
            f"НОВОЕ СООБЩЕНИЕ:\n{row['message']}\n\n"
            f"КАНДИДАТНЫЙ МАРШРУТ:\n{route['name']}\n"
            f"Описание:\n{route['description']}\n\n"
            "Соответствует ли сообщение этому маршруту? Ответь 1 или 0."
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты выполняешь независимую проверку соответствия маршруту.\n"
                        "Ответь только одним символом:\n"
                        "1 = MATCH: новое сообщение относится к описанной задаче "
                        "маршрута с учётом диалога.\n"
                        "0 = NO_MATCH: новое сообщение не относится к описанной "
                        "задаче маршрута.\n"
                        "Не объясняй решение и не выполняй запрос пользователя."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_completion_tokens": 1,
            "logprobs": True,
            "top_logprobs": 2,
            "logprob_token_ids": [token_one, token_zero],
            "allowed_token_ids": [token_one, token_zero],
            "chat_template_kwargs": {"enable_thinking": False},
        }

        started = time.perf_counter()
        try:
            response = session.post(
                f"{args.base_url}/v1/chat/completions",
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            raw = response.json()
            logprobs = extract_named_logprobs(
                raw,
                ["1", "0"],
            )
            binary_probability = normalize_logprobs(
                logprobs
            )["1"]
            score = logprobs["1"] - logprobs["0"]
            candidate_results.append({
                "name": route["name"],
                "score": score,
                "match_probability": binary_probability,
                "logprob_1": logprobs["1"],
                "logprob_0": logprobs["0"],
            })
            raw_responses.append({
                "id": row["id"],
                "route": route["name"],
                "payload": payload,
                "response": raw,
            })
        except Exception as exc:
            case_error = f"{type(exc).__name__}: {exc}"
            break
        finally:
            total_latency += time.perf_counter() - started

    predicted_route = None
    margin = None
    if not case_error:
        ordered = sorted(
            candidate_results,
            key=lambda item: item["score"],
            reverse=True,
        )
        predicted_route = ordered[0]["name"]
        margin = (
            ordered[0]["score"] - ordered[1]["score"]
            if len(ordered) > 1
            else math.inf
        )

    rows.append({
        "id": row["id"],
        "expected_route": row["expected_route"],
        "predicted_route": predicted_route,
        "correct": (
            predicted_route == row["expected_route"]
            if predicted_route
            else None
        ),
        "margin": margin,
        "candidates_json": json.dumps(
            candidate_results,
            ensure_ascii=False,
        ),
        "latency_sec": total_latency,
        "error": case_error,
    })
    print(
        f"{row['id']}: expected={row['expected_route']} "
        f"predicted={predicted_route} error={case_error}"
    )

# 4. Сохраняем результаты; порядок routes остаётся параметром запуска.
save_csv(args.output, rows)
valid = [row for row in rows if not row["error"]]
correct = sum(bool(row["correct"]) for row in valid)
save_json(args.json_output, {
    "metadata": {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "routes": args.routes,
        "model": model,
        "base_url": args.base_url,
        "score": "logP(1)-logP(0)",
    },
    "summary": {
        "evaluated": len(valid),
        "correct": correct,
        "accuracy": (
            correct / len(valid)
            if valid
            else None
        ),
    },
    "rows": rows,
    "raw_responses": raw_responses,
})
