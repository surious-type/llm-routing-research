"""Сравнение одной модели в двух порядках описаний P1/P2."""

import argparse
import json
import time
from pathlib import Path

import requests

from llm_routing.common import load_jsonl, load_routes, render_history, structured_response_format

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/diagnostic/routing_v2.jsonl")
parser.add_argument("--routes-p1", default="data/routes/continue_first.json")
parser.add_argument("--routes-p2", default="data/routes/new_first.json")
parser.add_argument("--base-url", default="http://localhost:8000")
parser.add_argument("--model", required=True)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--output", required=True)
args = parser.parse_args()

dataset = load_jsonl(args.dataset)
if args.limit is not None:
    dataset = dataset[:args.limit]

conditions = {
    "P1": load_routes(args.routes_p1),
    "P2": load_routes(args.routes_p2),
}

route_names = [route["name"] for route in conditions["P1"]]
session = requests.Session()

models_response = session.get(f"{args.base_url}/v1/models", timeout=120)
models_response.raise_for_status()
served_models = [item["id"] for item in models_response.json().get("data", [])]
if args.model not in served_models:
    raise RuntimeError(f"Модель {args.model!r} не найдена; доступны {served_models}")

rows = []

for condition, routes in conditions.items():
    for row in dataset:
        route_lines = "\n".join(
            f"{route['name']}: {route['description']}"
            for route in routes
        )
        prompt = (
            f"ТЕКУЩИЙ ДИАЛОГ:\n{render_history(row['history'])}\n\n"
            f"НОВОЕ СООБЩЕНИЕ:\n{row['message']}\n\n"
            f"ДОСТУПНЫЕ МАРШРУТЫ:\n{route_lines}"
        )
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты выполняешь маршрутизацию диалога. Выбери один наиболее "
                        "подходящий маршрут согласно описаниям доступных маршрутов. "
                        "Верни выбор в требуемом структурированном формате. "
                        "Сначала кратко сформулируй основание выбора в поле reason, "
                        "затем укажи выбранный маршрут в поле route."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "seed": 0,
            "max_completion_tokens": 128,
            "response_format": structured_response_format(route_names, True),
            "chat_template_kwargs": {"enable_thinking": False},
        }

        started = time.perf_counter()
        response = session.post(
            f"{args.base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        content_value = data["choices"][0]["message"]["content"]
        parsed = json.loads(content_value) if isinstance(content_value, str) else content_value
        predicted = parsed["route"]

        rows.append({
            "condition": condition,
            "id": row["id"],
            "expected_route": row["expected_route"],
            "predicted_route": predicted,
            "correct": predicted == row["expected_route"],
            "reason": parsed["reason"],
            "latency_sec": time.perf_counter() - started,
        })
        print(condition, row["id"], row["expected_route"], "->", predicted)

summary = {}
for condition in conditions:
    group = [row for row in rows if row["condition"] == condition]
    correct = sum(row["correct"] for row in group)
    summary[condition] = {
        "correct": correct,
        "total": len(group),
        "accuracy": correct / len(group) if group else None,
    }

p1 = {row["id"]: row for row in rows if row["condition"] == "P1"}
p2 = {row["id"]: row for row in rows if row["condition"] == "P2"}
ids = sorted(set(p1) & set(p2))
changed = [case_id for case_id in ids if p1[case_id]["predicted_route"] != p2[case_id]["predicted_route"]]
summary["order_agreement"] = {
    "count": len(ids) - len(changed),
    "total": len(ids),
    "rate": (len(ids) - len(changed)) / len(ids) if ids else None,
    "changed_cases": changed,
}

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(
        {
            "metadata": {
                "model": args.model,
                "dataset": args.dataset,
                "base_url": args.base_url,
            },
            "summary": summary,
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
