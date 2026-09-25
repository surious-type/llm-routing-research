"""E1-E4: влияние семантики A/B и позиции A в prompt."""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

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
parser.add_argument(
    "--conditions",
    default="experiments/prompt_conditions/conditions.json",
)
parser.add_argument("--base-url", default="http://localhost:8000")
parser.add_argument("--model", default=None)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--case-ids", nargs="*", default=None)
parser.add_argument("--timeout", type=float, default=120)
parser.add_argument("--output", default="results/prompt_conditions/latest.csv")
parser.add_argument(
    "--json-output",
    default="results/prompt_conditions/latest.json",
)
parser.add_argument("--show-prompt", action="store_true")
args = parser.parse_args()

# 1. Загружаем входные данные и четыре фиксированных условия E1-E4.
dataset = load_jsonl(args.dataset)
cases = select_cases(dataset, args.start, args.limit, args.case_ids)
routes = load_routes(args.routes)
conditions = json.loads(Path(args.conditions).read_text(encoding="utf-8"))
route_descriptions = {
    route["name"]: route["description"]
    for route in routes
}

if set(route_descriptions) != {"CONTINUE", "NEW"}:
    raise ValueError("Эксперимент E1-E4 рассчитан ровно на CONTINUE и NEW")
if not cases:
    raise ValueError("После фильтрации не осталось случаев")

# 2. Подключаемся к уже запущенному OpenAI-compatible серверу.
session = requests.Session()
model = args.model or discover_model(session, args.base_url, args.timeout)
token_a = tokenize_single(session, args.base_url, model, "A", args.timeout)
token_b = tokenize_single(session, args.base_url, model, "B", args.timeout)

# 3. Последовательно прогоняем каждый case во всех четырёх условиях.
rows = []
raw_responses = []
for condition in conditions:
    if condition["semantic_A"] == condition["semantic_B"]:
        raise ValueError(f"{condition['id']}: A и B должны иметь разную семантику")
    if {
        condition["semantic_A"],
        condition["semantic_B"],
    } != {"CONTINUE", "NEW"}:
        raise ValueError(f"{condition['id']}: неизвестная семантика")
    if condition["position_A"] not in {"first", "second"}:
        raise ValueError(
            f"{condition['id']}: position_A должен быть first или second"
        )

    definitions = [
        ("A", condition["semantic_A"]),
        ("B", condition["semantic_B"]),
    ]
    if condition["position_A"] == "second":
        definitions.reverse()

    definitions_text = "\n\n".join(
        f"{label} = {semantic}\n{route_descriptions[semantic]}"
        for label, semantic in definitions
    )
    system_prompt = (
        "Ты выполняешь только маршрутизацию диалога.\n\n"
        "Нужно определить, относится ли НОВОЕ сообщение пользователя "
        "к текущей задаче диалога.\n\n"
        f"{definitions_text}\n\n"
        "Не решай запрос пользователя. Не объясняй решение. "
        "Ответ должен быть только A или B."
    )

    for row in cases:
        user_prompt = (
            f"ТЕКУЩИЙ ДИАЛОГ:\n{render_history(row['history'])}\n\n"
            f"НОВОЕ СООБЩЕНИЕ:\n{row['message']}\n\n"
            "Выбери A или B."
        )
        if args.show_prompt:
            print(
                f"\n[{condition['id']} / {row['id']}]\n"
                f"{system_prompt}\n\n{user_prompt}\n"
            )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_completion_tokens": 1,
            "logprobs": True,
            "top_logprobs": 0,
            "logprob_token_ids": [token_a, token_b],
            "allowed_token_ids": [token_a, token_b],
            "chat_template_kwargs": {"enable_thinking": False},
        }

        started = time.perf_counter()
        error = None
        predicted_token = None
        predicted_route = None
        logp_a = None
        logp_b = None
        p_a = None
        p_b = None
        usage = {}
        raw_response = None

        try:
            response = session.post(
                f"{args.base_url}/v1/chat/completions",
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            raw_response = response.json()
            logprobs = extract_named_logprobs(
                raw_response,
                ["A", "B"],
            )
            probabilities = normalize_logprobs(logprobs)
            logp_a, logp_b = logprobs["A"], logprobs["B"]
            p_a, p_b = probabilities["A"], probabilities["B"]
            predicted_token = "A" if p_a >= p_b else "B"
            predicted_route = condition[f"semantic_{predicted_token}"]
            usage = raw_response.get("usage") or {}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        latency = time.perf_counter() - started
        expected_route = row["expected_route"]
        expected_token = (
            "A"
            if condition["semantic_A"] == expected_route
            else "B"
        )

        rows.append({
            "condition": condition["id"],
            "semantic_A": condition["semantic_A"],
            "semantic_B": condition["semantic_B"],
            "position_A": condition["position_A"],
            "id": row["id"],
            "expected_route": expected_route,
            "expected_token": expected_token,
            "predicted_route": predicted_route,
            "predicted_token": predicted_token,
            "correct": (
                predicted_route == expected_route
                if predicted_route
                else None
            ),
            "logprob_A": logp_a,
            "logprob_B": logp_b,
            "p_A": p_a,
            "p_B": p_b,
            "margin": (
                abs(p_a - p_b)
                if p_a is not None and p_b is not None
                else None
            ),
            "latency_sec": latency,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "error": error,
        })
        raw_responses.append({
            "condition": condition["id"],
            "id": row["id"],
            "payload": payload,
            "response": raw_response,
            "error": error,
        })
        print(
            f"{condition['id']} {row['id']}: "
            f"expected={expected_route} "
            f"predicted={predicted_route} error={error}"
        )

# 4. Сохраняем один набор результатов, где condition явно равен E1-E4.
save_csv(args.output, rows)
summary = {}
for condition in conditions:
    condition_rows = [
        row
        for row in rows
        if row["condition"] == condition["id"] and not row["error"]
    ]
    correct = sum(bool(row["correct"]) for row in condition_rows)
    summary[condition["id"]] = {
        "evaluated": len(condition_rows),
        "correct": correct,
        "accuracy": (
            correct / len(condition_rows)
            if condition_rows
            else None
        ),
        "mean_latency_sec": (
            statistics.mean(
                row["latency_sec"]
                for row in condition_rows
            )
            if condition_rows
            else None
        ),
    }

save_json(args.json_output, {
    "metadata": {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "routes": args.routes,
        "model": model,
        "base_url": args.base_url,
        "conditions": conditions,
    },
    "summary": summary,
    "rows": rows,
    "raw_responses": raw_responses,
})
print(json.dumps(summary, ensure_ascii=False, indent=2))
