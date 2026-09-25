import csv
import json
import math
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_routes(path):
    routes = json.loads(Path(path).read_text(encoding="utf-8"))
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


def select_cases(dataset, start=0, limit=None, case_ids=None):
    if case_ids:
        requested = set(case_ids)
        rows = [row for row in dataset if row["id"] in requested]
        missing = requested - {row["id"] for row in rows}
        if missing:
            raise ValueError(f"Не найдены случаи: {sorted(missing)}")
        return rows
    if limit is None:
        return dataset[start:]
    return dataset[start:start + limit]


def discover_model(session, base_url, timeout):
    response = session.get(f"{base_url}/v1/models", timeout=timeout)
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models:
        raise RuntimeError("Сервер не вернул ни одной модели")
    return models[0]["id"]


def tokenize_single(session, base_url, model, text, timeout):
    response = session.post(
        f"{base_url}/tokenize",
        json={"model": model, "prompt": text, "add_special_tokens": False},
        timeout=timeout,
    )
    response.raise_for_status()
    tokens = response.json()["tokens"]
    if len(tokens) != 1:
        raise RuntimeError(f"{text!r} должен быть одним токеном, получено: {tokens}")
    return tokens[0]


def extract_named_logprobs(data, names):
    try:
        position = data["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Не найден choices[0].logprobs.content[0]") from exc

    values = {}
    items = [position] + list(position.get("top_logprobs") or [])
    for item in items:
        token = item.get("token")
        logprob = item.get("logprob")
        if token is not None and logprob is not None:
            values[token.strip()] = float(logprob)

    missing = [name for name in names if name not in values]
    if missing:
        raise RuntimeError(f"Не найдены logprobs для {missing}; доступны: {values}")
    return {name: values[name] for name in names}


def normalize_logprobs(values):
    maximum = max(values.values())
    exponentials = {key: math.exp(value - maximum) for key, value in values.items()}
    total = sum(exponentials.values())
    return {key: value / total for key, value in exponentials.items()}


def extract_reranker_scores(data, expected_count):
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
            raise ValueError(f"Повторяющийся index: {index}")
        if index < 0 or index >= expected_count:
            raise ValueError(f"Индекс вне диапазона: {index}")
        score = item.get("score", item.get("raw_score", item.get("relevance_score")))
        if score is None:
            raise ValueError(f"В результате отсутствует score: {item}")
        mapped[index] = float(score)

    expected = set(range(expected_count))
    if set(mapped) != expected:
        raise ValueError(f"Отсутствуют индексы: {sorted(expected - set(mapped))}")
    return [mapped[index] for index in range(expected_count)]


def structured_response_format(route_names, with_reason):
    properties = {
        "route": {"type": "string", "enum": list(route_names)},
    }
    required = ["route"]
    if with_reason:
        properties = {
            "reason": {"type": "string"},
            "route": {"type": "string", "enum": list(route_names)},
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


def parse_structured_route(data, route_names, with_reason):
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content) if isinstance(content, str) else content
    expected_fields = {"route", "reason"} if with_reason else {"route"}
    if set(parsed) != expected_fields:
        raise ValueError(
            f"Ожидались поля {sorted(expected_fields)}, получено {sorted(parsed)}"
        )
    if parsed["route"] not in set(route_names):
        raise ValueError(f"Неизвестный маршрут: {parsed['route']}")
    return parsed


def save_csv(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Нет строк для сохранения")
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
