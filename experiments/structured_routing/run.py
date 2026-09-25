"""Простой structured routing: один запрос -> один JSON route."""

import argparse
import json
import time
from pathlib import Path

import requests

from llm_routing.common import load_jsonl, load_routes, render_history, structured_response_format

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/diagnostic/routing_v2.jsonl")
parser.add_argument("--routes", default="data/routes/continue_first.json")
parser.add_argument("--base-url", default="http://localhost:8000")
parser.add_argument("--model", required=True)
parser.add_argument("--mode", choices=["route-only", "reason-route"], default="reason-route")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--output", required=True)
args = parser.parse_args()

dataset = load_jsonl(args.dataset)
if args.limit is not None:
    dataset = dataset[:args.limit]
routes = load_routes(args.routes)
route_names = [route["name"] for route in routes]
with_reason = args.mode == "reason-route"

session = requests.Session()
rows = []

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
    instruction = (
        "Ты выполняешь маршрутизацию диалога. Выбери один наиболее "
        "подходящий маршрут согласно описаниям доступных маршрутов. "
        "Верни выбор в требуемом структурированном формате."
    )
    if with_reason:
        instruction += (
            " Сначала кратко сформулируй основание выбора в поле reason, "
            "затем укажи выбранный маршрут в поле route."
        )

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_completion_tokens": 128,
        "response_format": structured_response_format(route_names, with_reason),
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
        "id": row["id"],
        "expected_route": row["expected_route"],
        "predicted_route": predicted,
        "correct": predicted == row["expected_route"],
        "reason": parsed.get("reason"),
        "latency_sec": time.perf_counter() - started,
    })
    print(row["id"], row["expected_route"], "->", predicted)

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
