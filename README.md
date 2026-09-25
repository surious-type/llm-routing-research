# LLM routing research

Исследовательский репозиторий по маршрутизации нового сообщения пользователя в один из двух семантических маршрутов: CONTINUE или NEW.

Главный принцип текущей структуры: один эксперимент — одна папка, один линейный run.py, явные входы и явный результат. Исторические скрипты не удалены: они перенесены в archive/ и нужны только для воспроизводимости старых запусков.

## Структура

```
llm-routing-research/
├── data/
│   ├── diagnostic/        # исходный 30-case benchmark
│   ├── v3/                # development/validation и manifest Dataset V3
│   └── routes/            # два порядка одинаковых route descriptions
├── experiments/
│   ├── prompt_conditions/ # E1-E4: семантика A/B + позиция A
│   ├── independent_scoring/
│   ├── reranker/
│   ├── structured_routing/
│   └── model_comparison/
├── llm_routing/common.py  # только переиспользуемая техническая логика
├── notebooks/
├── tests/
├── results/               # исторические результаты разложены по семействам
├── reports/
├── docs/
└── archive/               # старый код, старые тесты и промежуточные данные
```

## Эксперименты

### 1. Prompt conditions E1-E4

E1-E4 — четыре условия одного исходного эксперимента с буквенными ответами A/B.

| Условие | Семантика A | Семантика B | Позиция A |
| ------- | ----------- | ----------- | --------- |
| E1      | CONTINUE    | NEW         | первая    |
| E2      | NEW         | CONTINUE    | первая    |
| E3      | CONTINUE    | NEW         | вторая    |
| E4      | NEW         | CONTINUE    | вторая    |

Запуск:

```
python -m experiments.prompt_conditions.run \
  --dataset data/diagnostic/routing_v2.jsonl \
  --model Qwen/Qwen3-4B-AWQ
```

### 2. Independent candidate scoring

Каждый маршрут оценивается отдельным LLM-запросом; score равен logP(1)-logP(0).

```
python -m experiments.independent_scoring.run \
  --dataset data/diagnostic/routing_v2.jsonl \
  --routes data/routes/continue_first.json \
  --model Qwen/Qwen3-4B-AWQ
```

### 3. Reranker

Cross-encoder получает query и оба candidate texts, после чего выбирается максимальный score.

```
python -m experiments.reranker.run \
  --dataset data/diagnostic/routing_v2.jsonl \
  --routes data/routes/continue_first.json \
  --endpoint http://localhost:8011/rerank
```

### 4. Structured LLM routing

Модель непосредственно возвращает route через JSON Schema. Доступны route-only и reason-route.

```
python -m experiments.structured_routing.run \
  --dataset data/diagnostic/routing_v2.jsonl \
  --routes data/routes/continue_first.json \
  --model Qwen/Qwen3-4B-AWQ \
  --mode reason-route \
  --output results/structured_routing/qwen3-4b.json
```

Обозначения E1-E4 здесь намеренно не используются: они закреплены за исходным A/B prompt experiment.

### 5. Model comparison

Фиксированный pipeline reason + route, но меняется модель. Один запуск обслуживает одну уже поднятую модель и автоматически проверяет два порядка descriptions: P1 и P2.

```
python -m experiments.model_comparison.run \
  --model Qwen/Qwen3-8B-AWQ \
  --output results/model_comparison/m8.json
```

После смены модели:

```
python -m experiments.model_comparison.analyze \
  --run M4=results/model_comparison/m4.json \
  --run M8=results/model_comparison/m8.json \
  --run M14=results/model_comparison/m14.json
```

## Датасеты

data/diagnostic/routing_v2.jsonl — основной исторический набор из 30 случаев. Эталон хранится как expected_route, поэтому датасет не зависит от того, какой букве A/B назначена семантика.

data/v3/development.jsonl и data/v3/validation.jsonl — более крупный Dataset V3. Manifest лежит рядом.

Резервные копии и промежуточные версии перенесены в archive/data/. Их не следует выбирать по умолчанию для новых прогонов.

## Результаты

Новые результаты следует сохранять в папку соответствующего эксперимента. Исторические файлы также сгруппированы по семействам в подпапках legacy; их содержимое не менялось.

## Тесты

```
python -m unittest discover -s tests -v
```

Unit tests не вызывают LLM. Они проверяют E1-E4, нормализацию logprobs, восстановление reranker scores по index и порядок enum в structured schema.

## Исторический код

archive/legacy_code/ и archive/legacy_tests/ содержат исходные скрипты и тесты до рефакторинга. Для новых запусков используйте experiments/*/run.py.
