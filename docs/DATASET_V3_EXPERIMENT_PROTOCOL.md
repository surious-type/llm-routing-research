# Dataset V3: протокол предэксперимента development + validation

Статус: **зафиксирован до первого model prediction**.  
Идентификатор: `dataset-v3-pretest-v1`.  
Дата фиксации: 2026-09-09.  

После первого model prediction этот документ запрещено изменять, кроме исправления документированной технической ошибки, не связанной с качеством ответов. Любое такое исправление требует новой версии документа и явного журнала изменений.

## 1. Границы эксперимента

Используются только:

- `dataset_v3_development.jsonl`, 120 случаев;
- `dataset_v3_validation.jsonl`, 60 случаев;
- `dataset_v3_manifest.json`;
- `quality_audit_dev_validation.json` как внешний предварительный аудит.

Закрытые test-входы, test-метки, trusted full test и secret-архивы не читаются, не передаются моделям и не включаются в анализ. Этот этап не является финальным test run.

Метки development/validation считаются предварительными синтетическими метками. Они не называются человеческим gold benchmark до независимой ручной проверки.

## 2. Зафиксированные хеши данных

| Файл | SHA-256 |
|---|---|
| `dataset_v3_development.jsonl` | `be113c460c0534fcd434206000281fbaed9eb356a2b8b2ff6c710d38eef569af` |
| `dataset_v3_validation.jsonl` | `22cfce2bebe1a99754605c1776a480edd575640d1639ee6638aecfb71ccc78dc` |
| `dataset_v3_manifest.json` | `3a92d6db689c276741cf95adc3824661161666c34ffaa6315bb7ab185d78e2c2` |
| `quality_audit_dev_validation.json` | `bde7b4f29fa6b973a8642e2d9d83075cac63cb4ce2aa711216e4b5155eb07702` |
| `routes_continue_first.json` | `cfdff764daf9e962e49fb05003148c1469462fc1e0558573cf08a02430da4561` |

До запуска проверяется совпадение фактических хешей. Несовпадение останавливает эксперимент.

## 3. Dataset audit до LLM

До первого LLM-запроса проверяются:

- валидность каждой JSONL-строки;
- ровно 120 development и 60 validation;
- уникальность `id` внутри и между частями;
- баланс 60/60 и 30/30;
- отсутствие пересечения ненулевых `family_id` между частями;
- наличие `id`, `history`, `message`, `expected_route`, `split`, `category`, `difficulty`, `ambiguity`, `domain`, `family_id`, `policy_version`, `annotation_notes`;
- допустимые значения меток, сложности, неоднозначности и ролей;
- распределения category/domain/difficulty/ambiguity;
- отсутствие точного дубликата полного случая;
- отсутствие точного дубликата модельного входа;
- соответствие declared split имени файла.

Построитель модельного prompt использует белый список: только `history[].role`, `history[].content`, `message` и внешнюю конфигурацию маршрутов. В prompt запрещены `id`, `expected_route`, `split`, `category`, `difficulty`, `ambiguity`, `domain`, `family_id`, `policy_version`, `annotation_notes`.

Регрессионный тест обязан доказать: замена всех evaluation metadata при неизменных `history` и `message` оставляет байты system/user prompt идентичными.

## 4. Лексические диагностические baseline

Обучение: development. Оценка: validation. Реализация: scikit-learn `1.7.2`, `TfidfVectorizer(ngram_range=(1, 2), analyzer="word", lowercase=True)` и `LogisticRegression(random_state=0, max_iter=2000)`. Дополнительная настройка гиперпараметров запрещена.

Условия:

- A: только `message`;
- B: только сериализованная `history`;
- C: `history + message` с фиксированными заголовками полей.

Для каждого условия сохраняются accuracy, balanced accuracy и confusion matrix в порядке классов `CONTINUE`, `NEW`. Эти модели являются только диагностикой возможных поверхностных подсказок.

## 5. LLM-метод

Метод: structured `reason + route`, без few-shot, logprobs, reranker, A/B, 1/0 и MATCH/NO_MATCH.

Системный prompt, версия `structured-factorial-v1`:

```text
Ты выполняешь маршрутизацию диалога. Выбери один наиболее подходящий маршрут согласно описаниям доступных маршрутов. Верни выбор в требуемом структурированном формате. Сначала кратко сформулируй основание выбора в поле reason, затем укажи выбранный маршрут в поле route.
```

Descriptions загружаются без изменений из `routes_continue_first.json` с указанным выше хешем.

JSON Schema:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "route_selection",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "reason": {"type": "string"},
        "route": {"type": "string", "enum": ["CONTINUE", "NEW"]}
      },
      "required": ["reason", "route"],
      "additionalProperties": false
    }
  }
}
```

Параметры всех запросов:

- `temperature=0`;
- `seed=0`;
- `enable_thinking=false`;
- `max_completion_tokens=128`;
- `max_model_len=2048`;
- `max_num_seqs=1`;
- enum order: `CONTINUE`, `NEW`.

Порядки descriptions:

- P1: `CONTINUE`, затем `NEW`;
- P2: `NEW`, затем `CONTINUE`.

## 6. Модели и инфраструктура

Docker image для всех условий: `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`. Фактически проверенная версия vLLM: `0.28.0`.

| Условие | Exact model id | Served id | GPU memory | CPU offload | Eager | Development | Validation |
|---|---|---|---:|---:|---|---|---|
| M4 | `Qwen/Qwen3-4B-AWQ` | `qwen3-4b` | 0.90 | 0 GB | false | да | да |
| M8 | `Qwen/Qwen3-8B-AWQ` | `qwen3-8b` | 0.85 | 0 GB | true | да | да |
| M14 | `Qwen/Qwen3-14B-AWQ` | `qwen3-14b` | 0.85 | 4 GB | true | нет | да |

Для всех: AWQ, `max_model_len=2048`, `max_num_seqs=1`. Новые memory fallback запрещены. Если M8 или M14 не запускается в указанной конфигурации, соответствующее условие фиксируется как infrastructure failure и не перенастраивается.

Одновременно в VRAM находится не более одной модели. Для каждой сохраняются Docker command, server log, `nvidia-smi` до загрузки, после загрузки и после эксперимента.

## 7. Smoke test

Правило выбрано до model output: для каждой комбинации `difficulty × expected_route` взять первый development-id в лексикографическом порядке.

Зафиксированные случаи:

- easy / `CONTINUE`: `v3_development_0003`;
- easy / `NEW`: `v3_development_0006`;
- medium / `CONTINUE`: `v3_development_0002`;
- medium / `NEW`: `v3_development_0001`;
- hard / `CONTINUE`: `v3_development_0011`;
- hard / `NEW`: `v3_development_0005`.

Smoke выполняется в P1 и P2. Full run разрешён только после 12 завершённых валидных JSON без HTTP/schema errors. Точность smoke не является критерием допуска.

## 8. Сохраняемые поля

Для каждого model/case/order: `id`, split, expected route, predicted route, correct, reason, полный structured response, error, request latency, prompt tokens, completion tokens, prompt order, enum order, model id и server configuration. Metadata используется только после ответа для анализа и не передаётся модели.

## 9. Показатели

Для P1 и P2 отдельно:

- accuracy и balanced accuracy;
- confusion matrix с порядком классов `CONTINUE`, `NEW`;
- recall `CONTINUE` и recall `NEW`;
- ошибки;
- mean/median latency;
- mean/median prompt и completion tokens.

Между P1/P2:

- semantic agreement и число/список order-sensitive ids;
- coverage согласованных решений;
- accuracy среди согласованных;
- число согласованных, но неправильных;
- размер disagreement subset;
- P1 и P2 accuracy на disagreement subset.

Согласие не называется вероятностью или уверенностью.

Breakdown для development и validation: expected route, difficulty, ambiguity, category, domain, family/standalone. Для семей: exact family accuracy, mean per-family accuracy, семьи хотя бы с одной ошибкой и семьи хотя бы с одним P1/P2 disagreement.

## 10. Выбор model condition

Validation ranking фиксируется заранее:

1. максимальная mean validation accuracy `(accuracy_P1 + accuracy_P2) / 2`;
2. при равенстве — максимальное P1/P2 semantic agreement;
3. затем — меньше agreed-but-wrong cases;
4. затем — меньшая operational cost/latency; latency M14 с CPU offload не трактуется как чистый эффект размера.

Нельзя выбирать по одному порядку, development, Dataset V2 или test. После анализа объявляется один frozen winner. Основное route-решение не меняется; threshold и selective routing не вводятся.

## 11. Запрет адаптации

После первого prediction запрещено менять dataset, labels, system/user prompt, route descriptions, schema, параметры, модельные flags, smoke cases, показатели и правило выбора ради улучшения результатов. Prompt tuning и few-shot запрещены в этом experiment family.

## 12. Остановка перед test

После development + validation создаётся `FROZEN_TEST_PROTOCOL.json`, содержащий winner, exact Docker/model config, хеши prompt/routes/evaluator, P1/P2 и правило отчётности. После этого работа останавливается. Test inputs и test labels не открываются; test requests не выполняются без отдельного явного подтверждения пользователя.

