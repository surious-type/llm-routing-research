#!/usr/bin/env python3
"""Independent per-route binary scoring experiment for vLLM."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "http://localhost:8000"
PROMPT_VERSION = "independent-v1"
TEMPERATURE = 0.0
SEED = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--routes", required=True)
    p.add_argument("--dataset", default="dataset_v2.jsonl")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--prompt-version", default=PROMPT_VERSION)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--output", required=True)
    p.add_argument("--raw-output", default=None)
    p.add_argument("--debug-response", action="store_true")
    return p.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_routes(path):
    with open(path, encoding="utf-8") as f:
        routes = json.load(f)
    if not routes or any(not r.get("name") or not r.get("description") for r in routes):
        raise ValueError("Routes must be a non-empty list with name and description")
    names = [r["name"] for r in routes]
    if len(names) != len(set(names)):
        raise ValueError("Route names must be unique")
    return routes


def discover_model(session, base_url, timeout):
    response = session.get(f"{base_url}/v1/models", timeout=timeout)
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models:
        raise RuntimeError("vLLM returned no models")
    return models[0]["id"]


def tokenize(session, base_url, model, text, timeout):
    response = session.post(
        f"{base_url}/tokenize",
        json={"model": model, "prompt": text, "add_special_tokens": False},
        timeout=timeout,
    )
    response.raise_for_status()
    tokens = response.json().get("tokens", [])
    if len(tokens) != 1:
        raise RuntimeError(f"{text!r} must be one token, got {tokens}")
    return tokens[0]


def render_history(history):
    labels = {"user": "Пользователь", "assistant": "Ассистент"}
    return "\n".join(f"{labels.get(x['role'], x['role'])}: {x['content']}" for x in history)


def build_system_prompt():
    return """Ты выполняешь независимую проверку соответствия маршруту.
Ответь только одним символом:
1 = MATCH: новое сообщение относится к описанной задаче маршрута с учётом диалога.
0 = NO_MATCH: новое сообщение не относится к описанной задаче маршрута.
Не объясняй решение и не выполняй запрос пользователя."""


def build_prompt(row, route):
    return f"""ТЕКУЩИЙ ДИАЛОГ:
{render_history(row['history'])}

НОВОЕ СООБЩЕНИЕ:
{row['message']}

КАНДИДАТНЫЙ МАРШРУТ:
{route['name']}
Описание:
{route['description']}

Соответствует ли сообщение этому маршруту? Ответь 1 или 0."""


def find_logprob(entry, token_id):
    if not isinstance(entry, dict):
        return None
    for key, value in entry.items():
        try:
            if int(key) == token_id:
                return float(value["logprob"] if isinstance(value, dict) else value)
        except (TypeError, ValueError, KeyError):
            continue
    return None


def extract_binary_logprobs(data, token_one, token_zero):
    try:
        position = data["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Missing choices[0].logprobs.content[0]") from exc

    entries = []
    if position.get("token_logprob") is not None:
        entries.append(position)
    entries.extend(position.get("top_logprobs") or [])

    logp_one = logp_zero = None
    for entry in entries:
        token_id = entry.get("token_id")
        logprob = entry.get("logprob", entry.get("token_logprob"))
        if token_id == token_one and logprob is not None:
            logp_one = float(logprob)
        if token_id == token_zero and logprob is not None:
            logp_zero = float(logprob)

    # Some vLLM versions expose token ids only in top_logprobs keys.
    if logp_one is None or logp_zero is None:
        for entry in entries:
            token = entry.get("token")
            value = entry.get("logprob", entry.get("token_logprob"))
            if value is None:
                continue
            if token.strip() == "1":
                logp_one = float(value)
            elif token.strip() == "0":
                logp_zero = float(value)

    if logp_one is None or logp_zero is None:
        raise RuntimeError("Could not obtain logP(1) and logP(0) from token-level logprobs")
    return logp_one, logp_zero


def normalize_binary(logp_one, logp_zero):
    maximum = max(logp_one, logp_zero)
    one = math.exp(logp_one - maximum)
    zero = math.exp(logp_zero - maximum)
    return one / (one + zero)


def score_candidate(session, base_url, model, row, route, token_one, token_zero, timeout, temperature, seed, debug):
    prompt = build_prompt(row, route)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": build_system_prompt()}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_completion_tokens": 1,
        "logprobs": True,
        "top_logprobs": 2,
        "logprob_token_ids": [token_one, token_zero],
        "allowed_token_ids": [token_one, token_zero],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    response = session.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    latency = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    if debug:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    logp_match, logp_no_match = extract_binary_logprobs(data, token_one, token_zero)
    score = logp_match - logp_no_match
    usage = data.get("usage", {})
    return {
        "name": route["name"],
        "logp_match": logp_match,
        "logp_no_match": logp_no_match,
        "score": score,
        "match_probability_binary": normalize_binary(logp_match, logp_no_match),
        "latency_sec": latency,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "sampled_token": data["choices"][0].get("message", {}).get("content"),
    }


def relative_softmax(scores):
    maximum = max(scores.values())
    values = {k: math.exp(v - maximum) for k, v in scores.items()}
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def route_case(session, base_url, model, row, routes, token_one, token_zero, args):
    candidates = [score_candidate(session, base_url, model, row, route, token_one, token_zero, args.timeout, args.temperature, args.seed, args.debug_response) for route in routes]
    scores = {c["name"]: c["score"] for c in candidates}
    ranking = relative_softmax(scores)
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    predicted = ordered[0]["name"]
    margin = ordered[0]["score"] - ordered[1]["score"] if len(ordered) > 1 else None
    for candidate in candidates:
        candidate["relative_ranking_softmax"] = ranking[candidate["name"]]
    return {
        "id": row["id"],
        "difficulty": row.get("difficulty"),
        "message": row["message"],
        "expected_route": row["expected_route"],
        "predicted_route": predicted,
        "correct": predicted == row["expected_route"],
        "score_margin": margin,
        "ranking_confidence_relative": ranking[predicted],
        "candidates_json": json.dumps(candidates, ensure_ascii=False, sort_keys=True),
        "latency_sec": sum(c["latency_sec"] for c in candidates),
        "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in candidates),
        "completion_tokens": sum(c["completion_tokens"] or 0 for c in candidates),
        "error": None,
    }


def main():
    args = parse_args()
    dataset = load_jsonl(args.dataset)
    routes = load_routes(args.routes)
    cases = dataset[args.start:] if args.limit is None else dataset[args.start:args.start + args.limit]
    if not cases:
        raise RuntimeError("No cases selected")
    session = requests.Session()
    model = args.model or discover_model(session, args.base_url, args.timeout)
    token_one = tokenize(session, args.base_url, model, "1", args.timeout)
    token_zero = tokenize(session, args.base_url, model, "0", args.timeout)
    route_config_text = Path(args.routes).read_bytes()
    metadata = {
        "model": model, "base_url": args.base_url, "temperature": args.temperature,
        "seed": args.seed, "dataset": str(Path(args.dataset)), "case_count": len(cases),
        "routes": str(Path(args.routes)), "route_config_sha256": hashlib.sha256(route_config_text).hexdigest(),
        "prompt_version": args.prompt_version, "scoring_formula": "logP(1)-logP(0)",
        "token_one": token_one, "token_zero": token_zero, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    results = []
    raw = {"metadata": metadata, "responses": []}
    for row in cases:
        try:
            result = route_case(session, args.base_url, model, row, routes, token_one, token_zero, args)
        except Exception as exc:
            result = {"id": row["id"], "difficulty": row.get("difficulty"), "message": row["message"], "expected_route": row["expected_route"], "predicted_route": None, "correct": None, "score_margin": None, "ranking_confidence_relative": None, "candidates_json": None, "latency_sec": None, "prompt_tokens": None, "completion_tokens": None, "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(f"{result['id']}: predicted={result['predicted_route']} expected={result['expected_route']} error={result['error']}")
    if any(r["error"] for r in results) or len(results) != len(cases):
        raise RuntimeError("Experiment incomplete; refusing to write a successful result")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(metadata.keys()) + list(results[0].keys())
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({**metadata, **result})
    if args.raw_output:
        raw["rows"] = results
        Path(args.raw_output).write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    correct = sum(r["correct"] for r in results)
    print(f"Accuracy: {correct}/{len(results)} = {correct / len(results):.3%}")
    print(f"Mean latency: {statistics.mean(r['latency_sec'] for r in results):.3f}s")
    print(f"Median latency: {statistics.median(r['latency_sec'] for r in results):.3f}s")


if __name__ == "__main__":
    main()
