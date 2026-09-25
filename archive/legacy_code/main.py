#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests


BASE_URL = "http://localhost:8000"
DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
RESULTS_DIR = Path(__file__).with_name("results")

SYSTEM_PROMPT = """Ты выполняешь только маршрутизацию диалога.

Нужно определить, относится ли НОВОЕ сообщение пользователя
к текущей задаче диалога.

B = NEW
Сообщение вводит новую задачу

Если сообщение одновременно продолжает текущую тему
и содержит новый независимый запрос, выбирай A.

A = CONTINUE
Сообщение продолжает текущую задачу: отвечает на вопрос ассистента,
уточняет её, меняет параметры или задаёт непосредственно связанный вопрос.

Не решай запрос пользователя.
Не объясняй решение.
Ответ должен быть только A или B.
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Сколько примеров прогнать. Например --limit 1",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="С какого элемента начать, 0-based",
    )

    parser.add_argument(
        "--base-url",
        default=BASE_URL,
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Имя модели. Если не указано, определяется через /v1/models",
    )

    parser.add_argument(
        "--show-prompt",
        action="store_true",
    )

    parser.add_argument(
        "--debug-response",
        action="store_true",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
    )

    return parser.parse_args()


def load_dataset():
    rows = []

    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def discover_model(session, base_url, timeout):
    response = session.get(
        f"{base_url}/v1/models",
        timeout=timeout,
    )
    response.raise_for_status()

    models = response.json()["data"]

    if not models:
        raise RuntimeError("vLLM не вернул моделей")

    return models[0]["id"]


def tokenize(session, base_url, model, text, timeout):
    response = session.post(
        f"{base_url}/tokenize",
        json={
            "model": model,
            "prompt": text,
            "add_special_tokens": False,
        },
        timeout=timeout,
    )

    response.raise_for_status()

    tokens = response.json()["tokens"]

    if len(tokens) != 1:
        raise RuntimeError(f"{text!r} должен быть одним токеном. Получено: {tokens}")

    return tokens[0]


def render_history(history):
    names = {
        "user": "Пользователь",
        "assistant": "Ассистент",
    }

    return "\n".join(f"{names[item['role']]}: {item['content']}" for item in history)


def build_prompt(row):
    return f"""ТЕКУЩИЙ ДИАЛОГ:
{render_history(row["history"])}

НОВОЕ СООБЩЕНИЕ:
{row["message"]}

Выбери A или B."""


def normalize(logp_a, logp_b):
    maximum = max(logp_a, logp_b)

    a = math.exp(logp_a - maximum)
    b = math.exp(logp_b - maximum)

    total = a + b

    return a / total, b / total


def extract_logprobs(data):
    try:
        position = data["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Не найден choices[0].logprobs.content[0]") from exc

    values = {}

    token = position.get("token")
    logprob = position.get("logprob")

    if token is not None and logprob is not None:
        values[token.strip()] = float(logprob)

    for item in position.get("top_logprobs") or []:
        token = item.get("token")
        logprob = item.get("logprob")

        if token is not None and logprob is not None:
            values[token.strip()] = float(logprob)

    if "A" not in values or "B" not in values:
        raise RuntimeError(
            "Не удалось получить logprobs одновременно для A и B. "
            f"Доступные токены: {values}"
        )

    return values["A"], values["B"]


def route(
    session,
    base_url,
    model,
    token_a,
    token_b,
    row,
    timeout,
    debug_response,
):
    prompt = build_prompt(row)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_completion_tokens": 1,
        "logprobs": True,
        "top_logprobs": 0,
        # vLLM-specific:
        "logprob_token_ids": [
            token_a,
            token_b,
        ],
        "allowed_token_ids": [
            token_a,
            token_b,
        ],
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    started = time.perf_counter()

    response = session.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    latency = time.perf_counter() - started

    data = response.json()

    if debug_response:
        print("\nRAW RESPONSE:")
        print(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )
        )

    logp_a, logp_b = extract_logprobs(data)

    p_a, p_b = normalize(
        logp_a,
        logp_b,
    )

    predicted = "A" if p_a >= p_b else "B"

    confidence = max(p_a, p_b)
    margin = abs(p_a - p_b)

    usage = data.get("usage", {})

    sampled = data["choices"][0].get("message", {}).get("content")

    return {
        "id": row["id"],
        "difficulty": row["difficulty"],
        "message": row["message"],
        "expected": row["expected"],
        "predicted": predicted,
        "correct": predicted == row["expected"],
        "sampled_token": sampled,
        "logprob_a": logp_a,
        "logprob_b": logp_b,
        "p_a": p_a,
        "p_b": p_b,
        "confidence": confidence,
        "margin": margin,
        "latency_sec": latency,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "error": None,
        "logprob_margin": abs(logp_a - logp_b),
    }


def print_result(result):
    if result["error"]:
        print("ERROR:", result["error"])
        return

    correct_symbol = "✓" if result["correct"] else "✗"

    print()
    print(f"Sampled token : {result['sampled_token']!r}")
    print(f"Predicted     : {result['predicted']}")
    print(f"Expected      : {result['expected']}")
    print()
    print(f"logP(A)       : {result['logprob_a']:.6f}")
    print(f"logP(B)       : {result['logprob_b']:.6f}")
    print()
    print(f"P(A | A,B)    : {result['p_a']:.10g}")
    print(f"P(B | A,B)    : {result['p_b']:.10g}")
    print(f"Confidence    : {result['confidence']:.10g}")
    print(f"Margin        : {result['margin']:.10g}")
    print(f"LogP margin   : {result['logprob_margin']:.6f}")
    print()
    print(
        f"Result        : {correct_symbol} "
        f"{'CORRECT' if result['correct'] else 'WRONG'}"
    )
    print(f"Latency       : {result['latency_sec']:.3f} s")


def save_results(results):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = RESULTS_DIR / f"run_{timestamp}.csv"

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    return path


def print_summary(results):
    valid = [r for r in results if r["error"] is None]

    print()
    print("#" * 70)
    print("SUMMARY")
    print("#" * 70)

    print(f"Cases total       : {len(results)}")
    print(f"Successful        : {len(valid)}")
    print(f"Errors            : {len(results) - len(valid)}")

    if not valid:
        return

    correct = sum(r["correct"] for r in valid)

    accuracy = correct / len(valid)

    print(f"Accuracy          : {accuracy:.4f} ({correct}/{len(valid)})")

    print(f"Mean confidence   : {statistics.mean(r['confidence'] for r in valid):.4f}")

    print(f"Mean margin       : {statistics.mean(r['margin'] for r in valid):.4f}")

    print(
        f"Mean latency      : {statistics.mean(r['latency_sec'] for r in valid):.3f} s"
    )

    correct_rows = [r for r in valid if r["correct"]]

    wrong_rows = [r for r in valid if not r["correct"]]

    if correct_rows:
        print(
            f"Conf. correct     : "
            f"{statistics.mean(r['confidence'] for r in correct_rows):.4f}"
        )

    if wrong_rows:
        print(
            f"Conf. wrong       : "
            f"{statistics.mean(r['confidence'] for r in wrong_rows):.4f}"
        )

        print(
            f"Margin wrong      : "
            f"{statistics.mean(r['margin'] for r in wrong_rows):.4f}"
        )

    print()
    print("Accuracy by difficulty:")

    difficulties = sorted(set(r["difficulty"] for r in valid))

    for difficulty in difficulties:
        group = [r for r in valid if r["difficulty"] == difficulty]

        n_correct = sum(r["correct"] for r in group)

        print(
            f"  {difficulty:<8}: "
            f"{n_correct / len(group):.4f} "
            f"({n_correct}/{len(group)})"
        )


def main():
    args = parse_args()

    dataset = load_dataset()

    if args.limit is None:
        cases = dataset[args.start :]
    else:
        cases = dataset[args.start : args.start + args.limit]

    if not cases:
        raise RuntimeError("После применения --start/--limit не осталось примеров")

    session = requests.Session()

    model = args.model or discover_model(
        session,
        args.base_url,
        args.timeout,
    )

    print("Определяю token id A/B...")

    token_a = tokenize(
        session,
        args.base_url,
        model,
        "A",
        args.timeout,
    )

    token_b = tokenize(
        session,
        args.base_url,
        model,
        "B",
        args.timeout,
    )

    print()
    print("LLM Routing Research")
    print(f"vLLM    : {args.base_url}")
    print(f"Model   : {model}")
    print(f"Cases   : {len(cases)}")
    print(f"Token A : {token_a}")
    print(f"Token B : {token_b}")

    results = []

    try:
        for i, row in enumerate(
            cases,
            start=1,
        ):
            print()
            print("=" * 70)
            print(f"[{i}/{len(cases)}] {row['id']} | difficulty={row['difficulty']}")
            print("=" * 70)

            print(render_history(row["history"]))

            print(f"НОВОЕ: {row['message']}")

            print(f"EXPECTED: {row['expected']}")

            if args.show_prompt:
                print()
                print("--- PROMPT ---")
                print(build_prompt(row))
                print("--- END PROMPT ---")

            try:
                result = route(
                    session=session,
                    base_url=args.base_url,
                    model=model,
                    token_a=token_a,
                    token_b=token_b,
                    row=row,
                    timeout=args.timeout,
                    debug_response=args.debug_response,
                )

            except Exception as exc:
                result = {
                    "id": row["id"],
                    "difficulty": row["difficulty"],
                    "message": row["message"],
                    "expected": row["expected"],
                    "predicted": None,
                    "correct": None,
                    "sampled_token": None,
                    "logprob_a": None,
                    "logprob_b": None,
                    "p_a": None,
                    "p_b": None,
                    "confidence": None,
                    "margin": None,
                    "latency_sec": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "error": (f"{type(exc).__name__}: {exc}"),
                }

            results.append(result)

            print_result(result)

    except KeyboardInterrupt:
        print()
        print("Прервано. Сохраняю уже полученные результаты.")

    if not results:
        return

    print_summary(results)

    output = save_results(results)

    print()
    print(f"CSV saved: {output}")


if __name__ == "__main__":
    main()
