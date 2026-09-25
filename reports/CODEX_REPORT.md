# Codex Research Report

## 1. Goal

На этом этапе завершено заранее зафиксированное сравнение S1 route-only и S2 reason+route при единственном новом параметре `max_completion_tokens=128`, одинаковом для обоих режимов и всех E1–E4. Предыдущий неуспешный S2 с лимитом 32 сохранён без изменений. Модель, dataset, descriptions, prompts, JSON schemas, temperature, seed, thinking mode и annotation policy не менялись.

## 2. Starting state

В репозитории существовали:

- `main.py` — бинарный A/B baseline;
- `main_sequence.py` — listwise sequence-likelihood baseline;
- `routes_continue_first.json`, `routes_new_first.json` — декларативные конфигурации маршрутов;
- `dataset_v2.jsonl` — 30 размеченных случаев;
- исторические CSV в `results/`.
- завершённый BGE reranker baseline и его неизменённые artifacts;
- локальный vLLM с served model id `qwen3-4b`.
- direct structured route-only baseline: 22/30 при порядке CONTINUE→NEW и 20/30 при порядке NEW→CONTINUE; совместная перестановка prompt и enum ранее меняла 10/30 решений.

Изучение `main_sequence.py` выявило, что он оценивает все маршруты в одном prompt и потому не является independent scoring. Кроме того, `mean_scores` ошибочно использовал `sum_logprob`, а поле `mean_logprob` также заполнялось суммой. Исторические файлы не изменялись и не удалялись.

## 3. Changes made

- `structured_factorial_routing.py` — добавлен явный параметр `--max-completion-tokens`; значение по умолчанию оставлено 32 для воспроизводимости прежнего условия. Новое условие запускалось только с явно указанным значением 128.
- `test_structured_factorial_routing.py` — добавлены regression tests: прежний лимит 32 сохраняется по умолчанию, а заранее заданный лимит 128 без других изменений попадает в request payload.
- `structured_factorial_routing.py` — добавлен отдельный runner E1–E4 с независимыми `prompt_route_order` и `enum_route_order`, а также режимами `route-only` и `reason-route`. Он импортирует общие неизменённые функции baseline, не изменяя `structured_routing.py`. Это новый набор экспериментальных условий.
- `analyze_structured_factorial.py` — добавлен анализ условий, шести попарных agreement, отдельных prompt/enum effects, описательного взаимодействия факторов и сравнения S1/S2. Accuracy и устойчивость не вычисляются при отсутствии допустимых ответов.
- `test_structured_factorial_routing.py`, `test_analyze_structured_factorial.py` — добавлены проверки независимости порядков, сохранности semantic descriptions, обеих схем, динамического enum, отсутствия logprobs, извлечения route только из structured response и корректной обработки обрезанных ответов.
- `structured_routing.py` — добавлен отдельный generic runner прямой структурированной маршрутизации. JSON Schema строится из имён текущего routes config; один запрос содержит все маршруты. Изменение вводит новый экспериментальный метод и не затрагивает предыдущие baseline implementations.
- `compare_structured_runs.py` — добавлено generic сравнение двух порядков: accuracy, ошибки, confusion matrix, semantic agreement и order-sensitive cases.
- `test_structured_routing.py`, `test_compare_structured_runs.py` — добавлены проверки динамического enum, точного состава prompt, отсутствия logprobs, строгого разбора route и сравнения по semantic route identity.
- `reranker_routing.py` — добавлен отдельный generic runner для `/rerank`. Он отправляет все маршруты одним запросом, сопоставляет оценки с исходными маршрутами по `index`, выбирает максимальную исходную оценку и сохраняет сырые ответы. Предыдущие способы маршрутизации не изменены.
- `compare_reranker_runs.py` — добавлено сравнение Run A/Run B по semantic route names после индексного сопоставления, включая расхождения оценок и ошибки.
- `test_reranker_routing.py`, `test_compare_reranker_runs.py` — добавлены проверки точного формата query/candidate, нескольких форматов индексированного ответа, переставленного ответа, повторяющихся индексов и строк с ошибками.
- `semantic_name_swap_conditions.json` — добавлены normal и swapped conditions с точными существующими descriptions; semantic identity хранится отдельно от surface name.
- `representation_diagnostic.py` — устранена hard-coded зависимость от `name_candidate`, чтобы route-name-only configs анализировались без description conditions. Методология scoring не изменена.
- `test_representation_diagnostic.py` — добавлен regression test для swap-only condition matrix.
- `representation_conditions.json` — добавлена декларативная матрица трёх route-name conditions и двух description formulations (одна condition используется совместно обоими экспериментами).
- `representation_diagnostic.py` — добавлен generic condition runner и анализ distributions, deltas, correlations, per-case changes и semantic agreement. Каждый candidate оценивается отдельным запросом.
- `test_representation_diagnostic.py` — добавлены unit/regression tests для condition validation, semantic scoring, Pearson correlation и pairwise prediction agreement.
- `analyze_independent_score_bias.py` — добавлен воспроизводимый анализатор score distributions и route-specific offset.
- `results/independent_score_bias_20260904.json` и `results/independent_score_bias_20260904.csv` — созданы analysis artifacts с агрегатами и 30 построчными наблюдениями для каждого run.
- `independent_candidate_scoring.py` — добавлен generic runner: отдельный prompt на каждый route, конфигурационные routes, проверка single-token `1`/`0`, token-level vLLM logprobs, score `logP(1)-logP(0)`, binary match probability, relative ranking softmax, margins, latency, token usage и metadata. Это меняет методологию только для нового independent эксперимента; baseline не изменён.
- `compare_independent_runs.py` — добавлено сравнение run-файлов по accuracy, semantic agreement, disagreements, errors и required cases.
- `docs/superpowers/plans/2026-09-04-independent-candidate-scoring.md` — зафиксирован план.
- `reports/CODEX_REPORT.md` — этот отчёт.

Prompt новой реализации зафиксирован как `independent-v1` и не изменялся после попытки запуска.

## 4. Experiment configuration

- Model: `qwen3-4b`; root `Qwen/Qwen3-4B-AWQ`; `max_model_len=4096`; logprobs разрешены.
- Backend: локальный vLLM OpenAI-compatible API.
- Endpoint: `http://localhost:8000`.
- Temperature: `0.0`.
- Seed: `0`.
- Dataset: `dataset_v2.jsonl`, 30 cases.
- Route configurations: `routes_continue_first.json`, `routes_new_first.json`.
- Prompt version: `independent-v1`.
- Scoring formula: `score(route) = logP(1 | candidate) - logP(0 | candidate)`; prediction is `argmax score`.
- Binary match probability is normalized only between `1` and `0` for one candidate.
- Softmax over route scores is retained only as a relative ranking diagnostic, not as calibrated probability of correctness.
- Representation diagnostic conditions:
  - `name_semantic`: names `CONTINUE`/`NEW`, original descriptions;
  - `name_candidate`: neutral name `CANDIDATE` for both semantics, original descriptions;
  - `name_route_x`: neutral name `ROUTE_X` for both semantics, original descriptions;
  - `description_equivalent_v2`: name `CANDIDATE` for both semantics, equivalent reworded descriptions without new rules or test examples.
- Total model requests: 30 cases × 4 unique conditions × 2 independent candidates = 240.
- Semantic-name swap conditions:
  - `semantic_name_normal`: semantic CONTINUE/name CONTINUE; semantic NEW/name NEW;
  - `semantic_name_swapped`: semantic CONTINUE/name NEW; semantic NEW/name CONTINUE;
  - descriptions в обеих conditions побайтно совпадают с `routes_continue_first.json`;
  - total requests: 30 × 2 conditions × 2 independent candidates = 120.
- Reranker model: `BAAI/bge-reranker-v2-m3`.
- Endpoint: `http://localhost:8011/rerank`.
- Температура и seed: неприменимы, поскольку reranker не генерирует текст и не использует декодирование.
- Query version: `reranker-query-v1`.
- Query для каждого случая:

```text
ТЕКУЩИЙ ДИАЛОГ:
<serialized history>

НОВОЕ СООБЩЕНИЕ:
<message>
```

- Текст каждого кандидата: `<route name>: <route description>`.
- Один HTTP-запрос на случай, содержащий все маршруты: `{"query": "...", "texts": [...], "raw_scores": true}`.
- Выбор: маршрут с максимальной исходной оценкой reranker; margin = лучшая оценка − вторая оценка.
- Исходная оценка и margin не интерпретируются как вероятность правильности.
- Run A: `routes_continue_first.json`; Run B: `routes_new_first.json`.
- Direct structured LLM model: served id `qwen3-4b`, root `Qwen/Qwen3-4B-AWQ`, `max_model_len=4096`.
- Backend: локальный vLLM OpenAI-compatible API, `http://localhost:8000/v1/chat/completions`.
- Dataset: `dataset_v2.jsonl`, 30 случаев; descriptions и annotations не менялись.
- Run A: `routes_continue_first.json`; Run B: `routes_new_first.json`.
- Temperature: `0.0`; seed: `0`; `enable_thinking=false`.
- Prompt version: `structured-routing-v1`.
- Системная инструкция:

```text
Ты выполняешь маршрутизацию диалога. Выбери один наиболее подходящий маршрут согласно описаниям доступных маршрутов. Верни выбор в требуемом структурированном формате.
```

- Пользовательская часть содержит только историю, новое сообщение и строки `<route id>: <description>` в порядке текущего файла.
- Structured output: JSON Schema с объектом `{"route": "<route id>"}`; `enum` динамически строится из всех имён routes config, `additionalProperties=false`.
- Механизм выбора: значение поля `route`, сгенерированное самой моделью. Формула scoring отсутствует; token-level logprobs и sequence likelihood не запрашиваются.
- Factorial runner: `structured_factorial_routing.py`; canonical descriptions загружаются из `routes_continue_first.json` и привязываются к semantic route id до любой перестановки.
- E1: prompt `CONTINUE, NEW`; enum `CONTINUE, NEW`.
- E2: prompt `NEW, CONTINUE`; enum `CONTINUE, NEW`.
- E3: prompt `CONTINUE, NEW`; enum `NEW, CONTINUE`.
- E4: prompt `NEW, CONTINUE`; enum `NEW, CONTINUE`.
- S1 schema: только обязательное поле `route`.
- S2 schema: обязательные поля `reason`, `route`; дополнительные поля запрещены; route enum строится динамически.
- Для S2 системная инструкция сохраняет исходную задачу маршрутизации и добавляет только требование сначала кратко сформулировать основание в `reason`, затем вернуть `route` в том же ответе.
- Для S1 и S2: `temperature=0`, `seed=0`, `enable_thinking=false`, `max_completion_tokens=32`, один вызов на case, без logprobs.
- Всего выполнено 120 запросов S1 и 120 запросов S2. Результаты S1 из эксперимента 1 повторно использованы в сравнении S1/S2; лишние S1-вызовы не выполнялись.
- Новое условие `structured-max128-v1`: единственное изменение от предыдущего factorial experiment — `max_completion_tokens=128` вместо 32. Значение 128 одинаково применено к S1, S2, E1, E2, E3 и E4.
- В новом условии выполнено 120 новых S1-запросов и 120 новых S2-запросов. Перед ними S2 smoke test на `case_001`, `case_020`, `case_024`, `case_029` во всех E1–E4 дал 16/16 завершённых JSON с `finish_reason=stop`.

## 5. Results

Run A: accuracy 16/30 = 53.33%; Run B: accuracy 16/30 = 53.33%. Semantic agreement: 30/30 = 100.0%; disagreements: 0.

Одинаковые ошибки в обоих runs: `case_003`, `case_005`, `case_007`, `case_009`, `case_011`, `case_014`, `case_018`, `case_020`, `case_021`, `case_022`, `case_024`, `case_026`, `case_028`, `case_030` — expected=`NEW`, predicted=`CONTINUE`.

Latency Run A: mean 0.113785 s, median 0.095594 s. Run B: mean 0.034107 s, median 0.034462 s. Средний score margin: A=8.911979, B=8.904167; median: A=6.906250, B=6.906250. Средний relative ranking confidence: A=0.994379, B=0.994491.

Средний token usage на case: 600.67 prompt tokens суммарно по двум independent requests и 2 completion tokens. Все 30 cases успешно обработаны.

Score distributions (mean / median / sample std):

| group | Run A | Run B |
|---|---:|---:|
| candidate CONTINUE | 14.705 / 17.875 / 6.609 | 14.697 / 17.883 / 6.626 |
| candidate NEW | 6.324 / 10.547 / 8.966 | 6.324 / 10.563 / 8.963 |
| correct candidate | 9.556 / 15.297 / 11.070 | 9.564 / 15.289 / 11.070 |
| incorrect candidate | 11.473 / 12.828 / 5.995 | 11.458 / 12.828 / 6.008 |

Distributions by expected class:

| expected class / candidate | Run A mean / median / std | Run B mean / median / std |
|---|---:|---:|
| expected CONTINUE, candidate CONTINUE (n=15) | 18.442 / 18.500 / 0.718 | 18.448 / 18.516 / 0.722 |
| expected CONTINUE, candidate NEW (n=15) | 11.978 / 13.188 / 3.725 | 11.969 / 13.188 / 3.730 |
| expected NEW, candidate CONTINUE (n=15) | 10.968 / 12.031 / 7.748 | 10.947 / 12.094 / 7.765 |
| expected NEW, candidate NEW (n=15) | 0.670 / -1.656 / 9.172 | 0.680 / -1.656 / 9.177 |

Route-specific offset `delta = score_CONTINUE - score_NEW`:

- Run A: mean 8.380729, median 6.789063, sample std 6.162545; delta > 0 in 29/30 cases, delta < 0 in 1/30 (`case_016`).
- Run B: mean 8.372917, median 6.796875, sample std 6.164236; delta > 0 in 29/30 cases, delta < 0 in 1/30 (`case_016`).
- Both candidates had positive MATCH score in 20/30 cases in each run.
- Both candidates had binary `P(match)>0.9` in 20/30 cases in each run.

The full 30-case tables, including scores, binary probabilities, deltas, and correctness-relative scores, are in the analysis JSON/CSV artifacts listed below.

### Controlled representation diagnostic

Condition-level results:

| condition | accuracy | delta mean / median / std | delta > 0 | agreement with independent baseline | changed cases |
|---|---:|---:|---:|---:|---|
| name_semantic | 53.33% | 8.373 / 6.797 / 6.164 | 29/30 | 100.00% | none |
| name_candidate | 50.00% | 12.949 / 13.109 / 5.185 | 30/30 | 96.67% | case_016 |
| name_route_x | 50.00% | 11.815 / 10.992 / 5.616 | 30/30 | 96.67% | case_016 |
| description_equivalent_v2 | 50.00% | 13.345 / 13.336 / 7.539 | 30/30 | 96.67% | case_016 |

Overall candidate score distributions (mean / median / sample std):

| condition | CONTINUE | NEW |
|---|---:|---:|
| name_semantic | 14.697 / 17.883 / 6.626 | 6.324 / 10.563 / 8.963 |
| name_candidate | 14.400 / 17.164 / 6.628 | 1.451 / 3.250 / 8.183 |
| name_route_x | 14.605 / 17.852 / 6.974 | 2.791 / 6.094 / 8.884 |
| description_equivalent_v2 | 9.374 / 16.156 / 11.628 | -3.970 / -4.000 / 7.950 |

Required expected-class/candidate distributions (mean / median / sample std):

| condition | exp CONTINUE / cand CONTINUE | exp CONTINUE / cand NEW | exp NEW / cand CONTINUE | exp NEW / cand NEW |
|---|---:|---:|---:|---:|
| name_semantic | 18.448 / 18.516 / 0.722 | 11.969 / 13.188 / 3.730 | 10.947 / 12.094 / 7.765 | 0.680 / -1.656 / 9.177 |
| name_candidate | 18.010 / 18.188 / 0.722 | 7.523 / 8.281 / 3.377 | 10.790 / 13.344 / 7.909 | -4.621 / -7.063 / 6.950 |
| name_route_x | 18.354 / 18.563 / 0.778 | 9.394 / 9.906 / 3.802 | 10.856 / 13.766 / 8.367 | -3.813 / -5.781 / 7.456 |
| description_equivalent_v2 | 16.801 / 16.656 / 0.700 | 0.550 / 2.375 / 6.528 | 1.948 / 7.844 / 12.704 | -8.491 / -10.250 / 6.673 |

Route-name score-vector correlations:

- CONTINUE: semantic↔CANDIDATE r=0.9640; semantic↔ROUTE_X r=0.9642; CANDIDATE↔ROUTE_X r=0.9875.
- NEW: semantic↔CANDIDATE r=0.9219; semantic↔ROUTE_X r=0.9144; CANDIDATE↔ROUTE_X r=0.9691.
- Mean score change semantic-name minus CANDIDATE: CONTINUE +0.297, NEW +4.873.
- Mean score change semantic-name minus ROUTE_X: CONTINUE +0.092, NEW +3.534.
- Prediction agreement semantic names↔CANDIDATE: 29/30; semantic names↔ROUTE_X: 29/30; CANDIDATE↔ROUTE_X: 30/30. The only changed case was `case_016`.

Description-representation comparison (`name_candidate` original description versus `description_equivalent_v2`):

- CONTINUE score-vector correlation r=0.8596; original-minus-v2 mean +5.026, median +1.711, std 6.829.
- NEW score-vector correlation r=0.6411; original-minus-v2 mean +5.421, median +4.047, std 6.837.
- Semantic prediction agreement: 30/30; both conditions predicted CONTINUE for all cases.

### Semantic route-name swap diagnostic

Normal condition reproduced all 60 candidate scores from the earlier `name_semantic` condition exactly (`max_abs_score_diff=0`).

| condition | accuracy | delta mean / median / std | delta > 0 | predictions | agreement normal↔swap |
|---|---:|---:|---:|---|---:|
| semantic_name_normal | 53.33% | 8.373 / 6.797 / 6.164 | 29/30 | 29 CONTINUE, 1 NEW | — |
| semantic_name_swapped | 50.00% | 12.267 / 11.273 / 5.597 | 30/30 | 30 CONTINUE | 29/30 = 96.67% |

Candidate score distributions (mean / median / sample std):

| semantic candidate | normal name | swapped name | mean shift swapped − normal | Pearson r |
|---|---:|---:|---:|---:|
| CONTINUE | 14.697 / 17.883 / 6.626 | 15.270 / 17.891 / 6.075 | +0.572 | 0.9602 |
| NEW | 6.324 / 10.563 / 8.963 | 3.003 / 6.031 / 8.352 | −3.322 | 0.8750 |

Expected-class/candidate distributions (mean / median / sample std):

| condition | exp CONTINUE / cand CONTINUE | exp CONTINUE / cand NEW | exp NEW / cand CONTINUE | exp NEW / cand NEW |
|---|---:|---:|---:|---:|
| normal | 18.448 / 18.516 / 0.722 | 11.969 / 13.188 / 3.730 | 10.947 / 12.094 / 7.765 | 0.680 / −1.656 / 9.177 |
| swapped | 18.394 / 18.641 / 0.686 | 8.569 / 9.344 / 3.922 | 12.146 / 13.172 / 7.420 | −2.564 / −3.406 / 7.919 |

The only changed prediction was `case_016`: normal delta −7.969 predicted NEW; swapped delta +4.344 predicted CONTINUE.

### BGE reranker baseline

Reranker в этом эксперименте — специализированная cross-encoder модель, которая совместно обрабатывает query и каждый текст-кандидат и возвращает оценку их соответствия. Она не генерирует ответ, не использует `1/0`, `A/B`, MATCH/NO_MATCH или token-level logprobs.

Обязательная предварительная проверка на `case_001`, `case_020`, `case_024`, `case_029` успешно обработала 4/4 запросов в каждом порядке. Решения совпали между порядками; правильными были 2/4.

| показатель | Run A | Run B |
|---|---:|---:|
| обработано без ошибок | 30/30 | 30/30 |
| accuracy | 15/30 = 50.00% | 15/30 = 50.00% |
| предсказаний `CONTINUE` | 30/30 | 30/30 |
| предсказаний `NEW` | 0/30 | 0/30 |
| средний margin | 5.2563 | 5.2556 |
| медианный margin | 5.2610 | 5.2583 |
| средняя задержка запроса | 0.01140 s | 0.01111 s |
| медианная задержка | 0.01127 s | 0.01111 s |

Semantic agreement между Run A и Run B: 30/30 = 100%. Случаев изменения решения из-за порядка нет.

Ошибочны все 15 случаев с expected=`NEW`: `case_003`, `case_005`, `case_007`, `case_009`, `case_011`, `case_014`, `case_016`, `case_018`, `case_020`, `case_021`, `case_022`, `case_024`, `case_026`, `case_028`, `case_030`.

После корректного сопоставления по semantic route name 13 из 60 оценок имели небольшие численные различия между порядками. Среднее абсолютное различие: 0.000916; максимальное: 0.011719 (`case_017`, route `NEW`). Эти изменения не повлияли ни на одно решение.

Сравнение с ранее зафиксированными результатами; старые методы не перезапускались:

| метод | зафиксированная accuracy | устойчивость к порядку | основной результат |
|---|---:|---:|---|
| A/B listwise routing | 50.00%–93.33% в E1–E4 | сильная зависимость от представления | результаты существенно менялись между условиями |
| semantic-name listwise routing | 90.00% / 73.33% | 21/30 = 70% agreement | 9 решений изменились после перестановки |
| independent MATCH/NO_MATCH scoring | 53.33% / 53.33% | 30/30 = 100% agreement | 29/30 решений `CONTINUE` |
| BGE reranker | 50.00% / 50.00% | 30/30 = 100% agreement | 30/30 решений `CONTINUE` |
| direct structured LLM routing | 73.33% / 66.67% | 20/30 = 66.67% agreement | 10 решений изменились после перестановки |

Offline smoke проверил:

- syntax compilation обоих новых Python-файлов;
- извлечение `logP(1)` и `logP(0)` из representative vLLM-style response;
- binary probability normalization;
- наличие ровно одного candidate block в prompt builder.

### Direct structured LLM routing baseline

Предварительная проверка на `case_001`, `case_020`, `case_024`, `case_029` успешно обработала 4/4 запроса в каждом порядке. Все ответы соответствовали JSON Schema, thinking был отключён, параметры logprobs отсутствовали. Accuracy была 2/4 для Run A и 1/4 для Run B; решения различались в 3/4 случаев.

| показатель | Run A | Run B |
|---|---:|---:|
| обработано без ошибок | 30/30 | 30/30 |
| accuracy | 22/30 = 73.33% | 20/30 = 66.67% |
| предсказаний `CONTINUE` | 23/30 | 15/30 |
| предсказаний `NEW` | 7/30 | 15/30 |
| средняя задержка | 0.17420 s | 0.15719 s |
| медианная задержка | 0.18326 s | 0.16077 s |
| среднее число prompt tokens | 319.33 | 319.33 |
| среднее число completion tokens | 10.83 | 10.60 |

Semantic agreement между Run A и Run B: 20/30 = 66.67%. Order-sensitive cases: `case_001`, `case_002`, `case_012`, `case_014`, `case_017`, `case_018`, `case_020`, `case_026`, `case_029`, `case_030`.

Ошибки Run A: `case_018`, `case_020`, `case_021`, `case_022`, `case_024`, `case_026`, `case_028`, `case_030`. Ошибки Run B: `case_001`, `case_002`, `case_012`, `case_014`, `case_017`, `case_021`, `case_022`, `case_024`, `case_028`, `case_029`. Ошибок HTTP, JSON Schema или разбора ответа не было.

Confusion matrix; строки — expected, столбцы — predicted, порядок меток `CONTINUE`, `NEW`:

| run | expected CONTINUE → CONTINUE / NEW | expected NEW → CONTINUE / NEW |
|---|---:|---:|
| A | 15 / 0 | 8 / 7 |
| B | 10 / 5 | 5 / 10 |

### 1. Direct structured 2×2 order experiment

Все 120 запросов S1 завершились без технических ошибок. E1 побайтно повторяет условие предыдущего Run A по prompt и schema, E2 — предыдущего Run B. Semantic predictions нового E1 совпали с предыдущим Run A в 30/30 случаях; E2 совпал с предыдущим Run B в 30/30 случаях.

| условие | prompt order | enum order | accuracy | CONTINUE / NEW | confusion matrix: exp C→C/N; exp N→C/N |
|---|---|---|---:|---:|---:|
| E1 | CONTINUE→NEW | CONTINUE→NEW | 22/30 = 73.33% | 23 / 7 | 15/0; 8/7 |
| E2 | NEW→CONTINUE | CONTINUE→NEW | 20/30 = 66.67% | 15 / 15 | 10/5; 5/10 |
| E3 | CONTINUE→NEW | NEW→CONTINUE | 22/30 = 73.33% | 23 / 7 | 15/0; 8/7 |
| E4 | NEW→CONTINUE | NEW→CONTINUE | 20/30 = 66.67% | 15 / 15 | 10/5; 5/10 |

Попарное semantic agreement:

| сравнение | agreement | изменившиеся случаи |
|---|---:|---|
| E1↔E2 | 20/30 = 66.67% | case_001, case_002, case_012, case_014, case_017, case_018, case_020, case_026, case_029, case_030 |
| E1↔E3 | 30/30 = 100% | нет |
| E1↔E4 | 20/30 = 66.67% | те же 10 случаев |
| E2↔E3 | 20/30 = 66.67% | те же 10 случаев |
| E2↔E4 | 30/30 = 100% | нет |
| E3↔E4 | 20/30 = 66.67% | те же 10 случаев |

Влияние prompt order: E1↔E2 и E3↔E4 изменили одинаковые 10/30 решений. Влияние enum order: E1↔E3 и E2↔E4 не изменило ни одного решения. Описательная difference-in-differences для accuracy равна 0.0; case-level symmetric differences для prompt effect и enum effect пусты. В этих данных признаков взаимодействия факторов не наблюдалось.

Средняя задержка E1/E2/E3/E4: 0.11545 / 0.11313 / 0.11641 / 0.11397 s. Среднее число prompt tokens во всех условиях: 319.33. Среднее число completion tokens: 10.83 / 10.60 / 10.83 / 10.60.

### 2. Reason + route experiment

S2 был выполнен на всех 30 cases во всех четырёх условиях: 120/120 HTTP-запросов получили ответ сервера. Однако 120/120 генераций завершились с `finish_reason=length` ровно на `completion_tokens=32`. Каждый ответ содержал только начало JSON и обрывался внутри текста `reason` до появления поля `route`. Поэтому допустимых structured responses: 0/120; predicted route, accuracy, confusion matrix, semantic agreement, corrected/broken changes и устойчивость к порядку не определены.

Сырые обрезанные ответы сохранены полностью. Код не извлекал route из текста `reason` и не пытался угадывать его из незавершённого JSON. После обнаружения проблемы `max_completion_tokens`, prompt и schema не менялись.

| условие | допустимые ответы | ошибки | средняя задержка | prompt tokens, mean | completion tokens, mean |
|---|---:|---:|---:|---:|---:|
| S2-E1 | 0/30 | 30 | 0.36033 s | 353.33 | 32.0 |
| S2-E2 | 0/30 | 30 | 0.35477 s | 353.33 | 32.0 |
| S2-E3 | 0/30 | 30 | 0.31233 s | 353.33 | 32.0 |
| S2-E4 | 0/30 | 30 | 0.31280 s | 353.33 | 32.0 |

### 3. Comparison

| метод / условие | accuracy | устойчивость к порядку | примечание |
|---|---:|---:|---|
| semantic-name listwise | 90.00% / 73.33% | 21/30 = 70% | два зафиксированных порядка |
| independent MATCH/NO_MATCH | 53.33% / 53.33% | 30/30 = 100% | 29/30 решений CONTINUE |
| BGE reranker | 50.00% / 50.00% | 30/30 = 100% | 30/30 решений CONTINUE |
| direct structured route-only E1 | 73.33% | — | prompt C→N; enum C→N |
| direct structured route-only E2 | 66.67% | E1↔E2: 20/30 | prompt N→C; enum C→N |
| direct structured route-only E3 | 73.33% | E1↔E3: 30/30 | prompt C→N; enum N→C |
| direct structured route-only E4 | 66.67% | E2↔E4: 30/30; E3↔E4: 20/30 | prompt N→C; enum N→C |
| direct structured reason+route E1–E4 | не определена | не определена | 0/120 допустимых JSON из-за лимита 32 |

Ответ на вопрос 1: в этом 30-case эксперименте сильнее влияет порядок descriptions в prompt. Его перестановка изменила 10/30 решений при каждом фиксированном enum order. Перестановка только enum не изменила 0/30 решений при каждом фиксированном prompt order.

Ответ для исторического условия с лимитом 32: определить, улучшает ли reason+route accuracy или устойчивость, было невозможно, поскольку S2 не выдал ни одного завершённого structured response. Это ограничение устранено только в новом отдельно зафиксированном условии max128 ниже.

### 4. Completed S1 vs S2 at max_completion_tokens=128

Обязательный S2 smoke test успешно вернул 16/16 завершённых JSON с обоими полями `reason` и `route`. После этого выполнены полные S1 и S2 по 120 запросов. Ошибок HTTP, JSON Schema, разбора или обрезки не было.

Точность и confusion matrix; строки матрицы — expected `CONTINUE`, `NEW`, столбцы — predicted `CONTINUE`, `NEW`:

| условие | S1 accuracy | S1 matrix | S2 accuracy | S2 matrix |
|---|---:|---:|---:|---:|
| E1 | 22/30 = 73.33% | [[15,0],[8,7]] | 27/30 = 90.00% | [[14,1],[2,13]] |
| E2 | 20/30 = 66.67% | [[10,5],[5,10]] | 27/30 = 90.00% | [[14,1],[2,13]] |
| E3 | 22/30 = 73.33% | [[15,0],[8,7]] | 27/30 = 90.00% | [[14,1],[2,13]] |
| E4 | 20/30 = 66.67% | [[10,5],[5,10]] | 27/30 = 90.00% | [[14,1],[2,13]] |

S1 повторил прежние route-only predictions во всех четырёх условиях. Повышение лимита с 32 до 128 не изменило S1 route decisions.

Semantic agreement между порядками:

| сравнение | S1 | S2 |
|---|---:|---:|
| E1↔E2 | 20/30 = 66.67% | 24/30 = 80.00% |
| E1↔E3 | 30/30 = 100% | 30/30 = 100% |
| E1↔E4 | 20/30 = 66.67% | 24/30 = 80.00% |
| E2↔E3 | 20/30 = 66.67% | 24/30 = 80.00% |
| E2↔E4 | 30/30 = 100% | 30/30 = 100% |
| E3↔E4 | 20/30 = 66.67% | 24/30 = 80.00% |

Order-sensitive cases снизились с 10 для S1 до 6 для S2. S1: `case_001`, `case_002`, `case_012`, `case_014`, `case_017`, `case_018`, `case_020`, `case_026`, `case_029`, `case_030`. S2: `case_001`, `case_014`, `case_020`, `case_021`, `case_023`, `case_024`.

Сравнение S1→S2 на 120 condition-case парах:

| показатель | E1 | E2 | E3 | E4 | всего |
|---|---:|---:|---:|---:|---:|
| изменившихся решений | 7 | 7 | 7 | 7 | 28 |
| исправленных ошибок | 6 | 7 | 6 | 7 | 26 |
| сломанных правильных ответов | 1 | 0 | 1 | 0 | 2 |

Изменения затронули 12 уникальных cases: `case_002`, `case_012`, `case_017`, `case_018`, `case_021`, `case_022`, `case_023`, `case_024`, `case_026`, `case_028`, `case_029`, `case_030`. Две поломки — `case_023` в E1 и E3.

Задержка и расход токенов, средние значения на запрос:

| условие | S1 latency | S2 latency | S1 prompt / completion tokens | S2 prompt / completion tokens |
|---|---:|---:|---:|---:|
| E1 | 0.16293 s | 0.61303 s | 319.33 / 10.83 | 353.33 / 59.50 |
| E2 | 0.15952 s | 0.63637 s | 319.33 / 10.60 | 353.33 / 60.93 |
| E3 | 0.11561 s | 0.58109 s | 319.33 / 10.83 | 353.33 / 59.50 |
| E4 | 0.11478 s | 0.59408 s | 319.33 / 10.60 | 353.33 / 60.93 |

Главный ответ: в этом эксперименте reason+route улучшил одновременно accuracy и устойчивость к prompt order. Accuracy выросла до 90% во всех условиях против 66.67–73.33% у S1. Agreement при перестановке prompt вырос с 66.67% до 80%, а число order-sensitive cases снизилось с 10 до 6. Чувствительность не устранена полностью.

## 6. Important examples

| id | expected | prediction A/B | score CONTINUE | score NEW | margin |
|---|---|---|---:|---:|---:|
| case_017 | CONTINUE | CONTINUE / CONTINUE | 16.9063 | 5.5938 | 11.3125 |
| case_020 | NEW | CONTINUE / CONTINUE | 17.3281 | 10.3750 | 6.9531 |
| case_021 | NEW | CONTINUE / CONTINUE | 18.1563 | 12.4219 | 5.7344 |
| case_023 | CONTINUE | CONTINUE / CONTINUE | 18.4531 | 12.4688 | 5.9844 |
| case_024 | NEW | CONTINUE / CONTINUE | 18.9063 | 10.9062 | 8.0000 |
| case_029 | CONTINUE | CONTINUE / CONTINUE | 18.3750 | 2.2500 | 16.1250 |

В ошибочных `case_020`, `case_021`, `case_024` binary match probability выбранного CONTINUE была примерно 1.0; это не calibrated probability correctness.

Representation-sensitive `case_016`:

| condition | expected | predicted | CONTINUE score | NEW score | delta |
|---|---|---|---:|---:|---:|
| name_semantic | NEW | NEW | -2.688 | 5.281 | -7.969 |
| name_candidate | NEW | CONTINUE | -4.344 | -6.250 | 1.906 |
| name_route_x | NEW | CONTINUE | -6.906 | -7.656 | 0.750 |
| description_equivalent_v2 | NEW | CONTINUE | -12.625 | -14.656 | 2.031 |
| semantic_name_swapped | NEW | CONTINUE | -3.094 | -7.437 | 4.344 |

Обязательные случаи reranker, Run A:

| id | expected | predicted | score `CONTINUE` | score `NEW` | margin |
|---|---|---|---:|---:|---:|
| case_017 | CONTINUE | CONTINUE | -0.7051 | -5.9492 | 5.2441 |
| case_020 | NEW | CONTINUE | -3.9766 | -7.7266 | 3.7500 |
| case_021 | NEW | CONTINUE | -1.7090 | -4.9492 | 3.2402 |
| case_023 | CONTINUE | CONTINUE | -0.5425 | -5.8203 | 5.2778 |
| case_024 | NEW | CONTINUE | -1.6670 | -6.4492 | 4.7822 |
| case_029 | CONTINUE | CONTINUE | -1.7725 | -8.2266 | 6.4541 |

Обязательные случаи direct structured LLM routing:

| id | expected | Run A | Run B | изменение из-за порядка |
|---|---|---|---|---|
| case_017 | CONTINUE | CONTINUE ✓ | NEW ✗ | да |
| case_020 | NEW | CONTINUE ✗ | NEW ✓ | да |
| case_021 | NEW | CONTINUE ✗ | CONTINUE ✗ | нет |
| case_023 | CONTINUE | CONTINUE ✓ | CONTINUE ✓ | нет |
| case_024 | NEW | CONTINUE ✗ | CONTINUE ✗ | нет |
| case_029 | CONTINUE | CONTINUE ✓ | NEW ✗ | да |

Обязательные случаи S1-E1..E4 и S2:

| id | expected | S1 E1/E2/E3/E4 | S2 E1/E2/E3/E4 | reason S2 |
|---|---|---|---|---|
| case_017 | CONTINUE | C / N / C / N | error / error / error / error | недоступен: JSON обрезан до route |
| case_020 | NEW | C / N / C / N | error / error / error / error | недоступен: JSON обрезан до route |
| case_021 | NEW | C / C / C / C | error / error / error / error | недоступен: JSON обрезан до route |
| case_023 | CONTINUE | C / C / C / C | error / error / error / error | недоступен: JSON обрезан до route |
| case_024 | NEW | C / C / C / C | error / error / error / error | недоступен: JSON обрезан до route |
| case_029 | CONTINUE | C / N / C / N | error / error / error / error | недоступен: JSON обрезан до route |

Здесь `C` означает semantic route `CONTINUE`, `N` — semantic route `NEW`. Полные частичные тексты S2 находятся в raw JSON artifact и не использовались для изменения ground truth или вывода route.

Обязательные случаи в новом max128 условии:

| id | expected | S1 E1/E2/E3/E4 | S2 E1/E2/E3/E4 | основание S2 по двум prompt orders |
|---|---|---|---|---|
| case_017 | CONTINUE | C/N/C/N | C/C/C/C | E1/E3: смена представления данных продолжает задачу; E2/E4: таблица вместо графика остаётся той же задачей |
| case_020 | NEW | C/N/C/N | C/N/C/N | E1/E3: назначение роли трактуется как продолжение темы прав; E2/E4: назначение роли трактуется как новая операция |
| case_021 | NEW | C/C/C/C | C/N/C/N | E1/E3: новый отчёт трактуется как продолжение темы отчётов; E2/E4: формирование отчёта трактуется как отдельная операция |
| case_023 | CONTINUE | C/C/C/C | N/C/N/C | E1/E3: вопрос про VPN трактуется как новая задача; E2/E4: он трактуется как продолжение диагностики авторизации |
| case_024 | NEW | C/C/C/C | N/C/N/C | E1/E3: уведомления трактуются как отдельный запрос; E2/E4: они трактуются как часть настройки аккаунта |
| case_029 | CONTINUE | C/N/C/N | C/C/C/C | во всех условиях изменение городов правильно трактуется как изменение параметров текущей задачи |

Reason не использовался кодом для выбора или исправления route. Таблица только пересказывает сохранённые диагностические строки модели; полный текст каждого reason находится в `results/structured_reason_max128_20260907.csv` и JSON artifact.

## 7. Observations

- Independent runner не включает другие routes в candidate prompt.
- Алгоритм не содержит special-case логики для `CONTINUE` или `NEW`.
- Перестановка routes-файла не изменила ни одного prediction.
- Новый runner выбрал CONTINUE в 29/30 cases и NEW только в `case_016`.
- Высокая confidence сопровождала ошибки, включая required cases `case_020`, `case_021`, `case_024`.
- Исторические baseline artifacts сохранены.
- Score CONTINUE превышал score NEW в 29/30 случаях в обоих runs; единственное исключение — `case_016`, единственный случай с prediction NEW.
- Для expected=NEW средний score incorrect CONTINUE был 10.968 (A) против среднего score correct NEW 0.670 (A); это непосредственно согласуется с observed collapse.
- В 20/30 случаях оба независимых бинарных вопроса одновременно давали положительный score и binary match probability выше 0.9.
- OBSERVATION: neutral route names increased mean delta from 8.373 to 12.949 (`CANDIDATE`) and 11.815 (`ROUTE_X`).
- OBSERVATION: name changes affected NEW scores much more strongly on average than CONTINUE scores; only `case_016` changed prediction.
- OBSERVATION: equivalent description rewrite changed both score vectors; NEW correlation (0.641) was lower than CONTINUE correlation (0.860), while semantic predictions remained identical between the two neutral-name description conditions.
- OBSERVATION: assigning surface name NEW to semantic CONTINUE did not move its scores toward the low semantic-NEW distribution; its mean score increased slightly by 0.572.
- OBSERVATION: assigning surface name CONTINUE to semantic NEW did not carry the high CONTINUE score with the token; semantic NEW mean score decreased by 3.322.
- OBSERVATION: swap increased mean semantic delta by 3.894 and changed only `case_016`.
- НАБЛЮДЕНИЕ: BGE reranker выбрал `CONTINUE` во всех 30 случаях обоих запусков и получил accuracy 50%.
- НАБЛЮДЕНИЕ: перестановка маршрутов не изменила ни одного semantic prediction, хотя небольшие численные различия появились в 13/60 оценок.
- НАБЛЮДЕНИЕ: специализированный reranker в этом zero-shot условии не превзошёл semantic-name listwise baseline и показал более сильный collapse в `CONTINUE`, чем independent MATCH/NO_MATCH baseline.
- НАБЛЮДЕНИЕ: direct structured Run A получил 22/30, Run B — 20/30; ни один запрос не завершился технической ошибкой.
- НАБЛЮДЕНИЕ: перестановка массива маршрутов изменила 10/30 semantic predictions; agreement составил 66.67%.
- НАБЛЮДЕНИЕ: Run A выбрал `CONTINUE` 23 раза, тогда как Run B дал ровно 15 `CONTINUE` и 15 `NEW`.
- НАБЛЮДЕНИЕ: JSON Schema обеспечила допустимый route id, но не устранила чувствительность смыслового решения к порядку enum и строк маршрутов.
- НАБЛЮДЕНИЕ: Run A правильно классифицировал все 15 expected `CONTINUE`, но только 7/15 expected `NEW`; в Run B обе полноты классов равны 10/15.
- НАБЛЮДЕНИЕ: при фиксированном prompt order перестановка enum дала 30/30 одинаковых решений в обеих парах E1↔E3 и E2↔E4.
- НАБЛЮДЕНИЕ: при фиксированном enum order перестановка prompt descriptions дала только 20/30 одинаковых решений в обеих парах E1↔E2 и E3↔E4.
- НАБЛЮДЕНИЕ: набор из 10 изменившихся cases был одинаковым при обоих enum orders; описательная interaction difference-in-differences равна 0.
- НАБЛЮДЕНИЕ: все 120 S2-генераций достигли лимита 32 completion tokens и завершились до генерации поля route.
- НАБЛЮДЕНИЕ: ни одно S2-решение не было доступно для сравнения accuracy или устойчивости; количество исправленных и сломанных решений не вычислялось.
- НАБЛЮДЕНИЕ: при лимите 128 S2 вернул 120/120 допустимых structured responses и получил 27/30 во всех E1–E4.
- НАБЛЮДЕНИЕ: S2 повысил prompt-order agreement с 20/30 до 24/30 и сократил число order-sensitive cases с 10 до 6.
- НАБЛЮДЕНИЕ: из 28 изменений S1→S2 26 исправили ошибку и 2 сломали правильный ответ; обе поломки относятся к `case_023` при prompt order CONTINUE→NEW.
- НАБЛЮДЕНИЕ: перестановка только enum не изменила ни одного S2 prediction: E1↔E3 и E2↔E4 дали 30/30 agreement.
- НАБЛЮДЕНИЕ: S2 потребовал в среднем 59.5–60.9 completion tokens и имел среднюю задержку 0.58–0.64 s против 10.6–10.8 tokens и 0.11–0.16 s для S1.

## 8. Interpretation / hypotheses

Это результат одного малого 30-case routing experiment, а не доказательство общей надёжности метода.

Гипотеза: independent prompts устранили прямой order effect между двумя route entries в данном запуске, поскольку predictions совпали. Это не доказывает отсутствия других representation/model/dataset effects.

Гипотеза: наблюдается route-specific calibration/offset bias в пользу CONTINUE, а не только случайная ошибка отдельных cases: offset положителен в 29/30 случаях и составляет около 8.38 score units в среднем. На 30 примерах это диагностический сигнал, не доказательство универсальности.

Гипотеза: binary match probability плохо отражает межмаршрутную исключительность — 20/30 cases одновременно выглядят как MATCH для обоих candidates. Это объясняет, почему высокая per-candidate confidence не гарантирует правильный route ranking.

HYPOTHESIS: часть исходного offset связана с route-name representation, причём имя `NEW` повышает NEW scores относительно нейтральных names примерно на 3.5–4.9 score units; эффект на CONTINUE scores существенно меньше.

HYPOTHESIS: descriptions также вносят крупный representation effect, но текущие две формулировки не позволяют отделить lexical sensitivity от semantic framing. Поскольку prediction agreement между description variants осталось 100%, на этом dataset эффект проявился в score scale сильнее, чем в argmax decisions.

HYPOTHESIS: высокий score преимущественно следует за semantic description, а не переносится вместе с surface token/name. Если бы dominant effect принадлежал только surface token CONTINUE, semantic NEW с этим именем должен был бы получить заметное повышение, но наблюдалось снижение.

HYPOTHESIS: name-description mismatch создаёт асимметричный interaction effect, особенно ухудшающий scores semantic NEW. Этот diagnostic не отделяет inconsistency penalty от более сложного lexical-semantic interaction.

ГИПОТЕЗА: описание `CONTINUE` лексически ближе к query, содержащему историю текущего диалога, поэтому общий reranker релевантности систематически ставит его выше описания `NEW`. Текущий эксперимент не устанавливает причину и не проверяет другие candidate representations.

ГИПОТЕЗА: малые различия оценок между порядками могут быть следствием численной точности или особенностей пакетной обработки кандидатов. Поскольку решения не изменились, наблюдаемая semantic order robustness в этом наборе сохраняется.

ГИПОТЕЗА: в direct structured условии порядок влияет либо через последовательность route descriptions в prompt, либо через порядок значений JSON Schema enum, либо через их взаимодействие. Текущий совместный reversal не разделяет эти источники.

ГИПОТЕЗА: JSON Schema ограничивает синтаксис результата, но сама по себе не нейтрализует позиционное смещение генеративной модели. Это объяснение согласуется с 10 изменившимися решениями, но 30 случаев недостаточно для общего вывода.

ГИПОТЕЗА: наблюдавшаяся ранее чувствительность совместной перестановки возникает преимущественно в текстовом представлении маршрутов внутри prompt, а не в порядке enum. На текущих 30 случаях enum effect отсутствовал полностью, но это не доказывает его отсутствия для других моделей, схем или числа маршрутов.

ГИПОТЕЗА: отсутствие описательного взаимодействия означает, что enum order не усиливал и не ослаблял prompt-order effect в данном запуске. Статистически сильный общий вывод по 30 случаям делать нельзя.

ГИПОТЕЗА: S2 не завершился потому, что модель формулировала reason длиннее доступного бюджета до перехода к route. Это непосредственно подтверждается `finish_reason=length`, 32/32 использованными completion tokens и обрывом каждого ответа внутри reason; влияние самого рассуждения на качество маршрута этим экспериментом не измерено.

ГИПОТЕЗА: требование явно сформулировать основание до route помогает модели применять semantic distinction между продолжением и новой операцией, что согласуется с 26 исправлениями против 2 поломок. Это объяснение не доказано причинно и ограничено одной моделью и 30 cases.

ГИПОТЕЗА: оставшиеся шесть order-sensitive cases допускают для модели обе локально правдоподобные интерпретации; сохранённые reasons меняют интерпретацию вместе с prompt order. Reasons являются диагностикой поведения модели, а не независимым доказательством правильности.

## 9. Problems / limitations

- Предыдущая проблема запуска vLLM была временной; текущий API был доступен и оба runs завершились.
- Dataset содержит только 30 случаев; даже успешные результаты будут ограничены по статистической силе.
- Analysis использует два runs одного и того же малого dataset, а не независимую выборку.
- Положительный offset может смешивать свойства route descriptions, route names, prompt wording и модели; этот этап не разделяет эти причины.
- Controlled conditions используют только одну модель и 30 cases; correlations и offsets не следует обобщать на другие models/routes.
- Description experiment содержит две формулировки; этого недостаточно для оценки полного диапазона linguistic representation effects.
- Swap меняет оба name-description pairings одновременно; взаимодействия могут быть нелинейными, поэтому результат не является чистой оценкой независимого token prior.
- Reranker не обучался на текущей политике `CONTINUE`/`NEW`; его исходная оценка отражает общую релевантность, а не гарантированно политику разделения задач.
- Dataset содержит только 30 сбалансированных случаев; collapse в `CONTINUE` даёт ровно 50% accuracy и не позволяет делать сильные выводы о других наборах.
- Не проводились настройка prompt, изменение descriptions, обучение, threshold или calibration.
- В старом `main_sequence.py` остаётся известная ошибка labels для mean diagnostic; файл не изменялся, чтобы сохранить baseline. Если его mean-результаты будут использоваться, этот baseline нужно отдельно перезапустить после явного исправления.
- Direct structured эксперимент одновременно переставляет строки маршрутов в prompt и порядок значений `enum`; поэтому источник order effect внутри представления пока не локализован.
- Structured generation гарантирует допустимый формат, но не предоставляет score, margin или измеренную уверенность решения.
- `temperature=0` уменьшает sampling variability, однако отдельная проверка повторяемости идентичного запуска на этом этапе не выполнялась.
- В S2 прежний лимит `max_completion_tokens=32` оказался недостаточен для schema `reason + route`; 0/120 допустимых ответов делают сравнение качества S1/S2 невозможным.
- Изменение лимита после просмотра S2 считалось бы новым экспериментальным условием, поэтому в текущем этапе оно не выполнялось.
- Вывод о доминировании prompt order относится только к одной модели, двум маршрутам и 30 примерам.
- Сравнение max128 меняет одновременно требуемую структуру ответа и фактическую длину генерации между S1 и S2; более высокая стоимость S2 является частью метода reason+route.
- Выполнен один детерминированный прогон каждого condition; повторяемость результатов отдельными идентичными повторами не измерялась.
- Несмотря на улучшение, S2 остаётся чувствительным к prompt order в 6/30 cases и не должен считаться инвариантным.

## 10. Files produced

- `independent_candidate_scoring.py`
- `compare_independent_runs.py`
- `docs/superpowers/plans/2026-09-04-independent-candidate-scoring.md`
- `reports/CODEX_REPORT.md`
- `results/independent_A_20260904_1445.csv`
- `results/independent_A_20260904_raw.json`
- `results/independent_B_20260904_1445.csv`
- `results/independent_B_20260904_raw.json`
- `results/independent_comparison_20260904_1445.json`
- `analyze_independent_score_bias.py`
- `results/independent_score_bias_20260904.json`
- `results/independent_score_bias_20260904.csv`
- `representation_conditions.json`
- `representation_diagnostic.py`
- `test_representation_diagnostic.py`
- `results/representation_diagnostic_20260904_raw.csv`
- `results/representation_diagnostic_20260904_cases.csv`
- `results/representation_diagnostic_20260904_analysis.json`
- `semantic_name_swap_conditions.json`
- `results/semantic_name_swap_20260904_raw.csv`
- `results/semantic_name_swap_20260904_cases.csv`
- `results/semantic_name_swap_20260904_analysis.json`
- `reranker_routing.py`
- `compare_reranker_runs.py`
- `test_reranker_routing.py`
- `test_compare_reranker_runs.py`
- `results/reranker_A_20260905.csv`
- `results/reranker_A_20260905.json`
- `results/reranker_B_20260905.csv`
- `results/reranker_B_20260905.json`
- `results/reranker_comparison_20260905.json`
- `structured_routing.py`
- `compare_structured_runs.py`
- `test_structured_routing.py`
- `test_compare_structured_runs.py`
- `results/structured_A_20260907.csv`
- `results/structured_A_20260907.json`
- `results/structured_B_20260907.csv`
- `results/structured_B_20260907.json`
- `results/structured_comparison_20260907.json`
- `structured_factorial_routing.py`
- `analyze_structured_factorial.py`
- `test_structured_factorial_routing.py`
- `test_analyze_structured_factorial.py`
- `results/structured_factorial_20260907.csv`
- `results/structured_factorial_20260907.json`
- `results/structured_reason_20260907.csv`
- `results/structured_reason_20260907.json`
- `results/structured_factorial_comparison_20260907.json`
- `results/structured_reason_comparison_20260907.json`
- `results/structured_factorial_max128_20260907.csv`
- `results/structured_factorial_max128_20260907.json`
- `results/structured_reason_max128_20260907.csv`
- `results/structured_reason_max128_20260907.json`
- `results/structured_factorial_max128_comparison_20260907.json`
- `results/structured_reason_max128_comparison_20260907.json`

Все result artifacts созданы; исторические результаты не перезаписывались.

## 11. Exact commands

```bash
PYTHONPYCACHEPREFIX=/tmp/llm-routing-pycache python3 -m py_compile independent_candidate_scoring.py compare_independent_runs.py
python3 independent_candidate_scoring.py --routes routes_continue_first.json --dataset dataset_v2.jsonl --output results/independent_A_<timestamp>.csv --raw-output results/independent_A_raw.json
python3 independent_candidate_scoring.py --routes routes_new_first.json --dataset dataset_v2.jsonl --output results/independent_B_<timestamp>.csv --raw-output results/independent_B_raw.json
python3 compare_independent_runs.py results/independent_A_<timestamp>.csv results/independent_B_<timestamp>.csv --output results/independent_comparison_<timestamp>.json
```

Фактически выполнены обе команды runner и comparator:

```bash
python3 independent_candidate_scoring.py --routes routes_continue_first.json --dataset dataset_v2.jsonl --output results/independent_A_20260904_1445.csv --raw-output results/independent_A_20260904_raw.json
python3 independent_candidate_scoring.py --routes routes_new_first.json --dataset dataset_v2.jsonl --output results/independent_B_20260904_1445.csv --raw-output results/independent_B_20260904_raw.json
python3 compare_independent_runs.py results/independent_A_20260904_1445.csv results/independent_B_20260904_1445.csv --output results/independent_comparison_20260904_1445.json
python3 -m unittest -v test_representation_diagnostic.py
python3 representation_diagnostic.py --conditions representation_conditions.json --dataset dataset_v2.jsonl --baseline results/independent_A_20260904_1445.csv --raw-csv results/representation_diagnostic_20260904_raw.csv --cases-csv results/representation_diagnostic_20260904_cases.csv --analysis-json results/representation_diagnostic_20260904_analysis.json
python3 representation_diagnostic.py --conditions semantic_name_swap_conditions.json --dataset dataset_v2.jsonl --baseline results/independent_A_20260904_1445.csv --raw-csv results/semantic_name_swap_20260904_raw.csv --cases-csv results/semantic_name_swap_20260904_cases.csv --analysis-json results/semantic_name_swap_20260904_analysis.json
python3 -m unittest -v test_reranker_routing.py test_compare_reranker_runs.py
python3 reranker_routing.py --routes routes_continue_first.json --dataset dataset_v2.jsonl --case-ids case_001 case_020 case_024 case_029 --output /tmp/reranker_smoke_A.csv --json-output /tmp/reranker_smoke_A.json
python3 reranker_routing.py --routes routes_new_first.json --dataset dataset_v2.jsonl --case-ids case_001 case_020 case_024 case_029 --output /tmp/reranker_smoke_B.csv --json-output /tmp/reranker_smoke_B.json
python3 reranker_routing.py --routes routes_continue_first.json --dataset dataset_v2.jsonl --output results/reranker_A_20260905.csv --json-output results/reranker_A_20260905.json
python3 reranker_routing.py --routes routes_new_first.json --dataset dataset_v2.jsonl --output results/reranker_B_20260905.csv --json-output results/reranker_B_20260905.json
python3 compare_reranker_runs.py results/reranker_A_20260905.csv results/reranker_B_20260905.csv --output results/reranker_comparison_20260905.json
python3 -m unittest -v test_structured_routing.py test_compare_structured_runs.py
python3 structured_routing.py --routes routes_continue_first.json --dataset dataset_v2.jsonl --case-ids case_001 case_020 case_024 case_029 --output /tmp/structured_smoke_A_20260907.csv --json-output /tmp/structured_smoke_A_20260907.json
python3 structured_routing.py --routes routes_new_first.json --dataset dataset_v2.jsonl --case-ids case_001 case_020 case_024 case_029 --output /tmp/structured_smoke_B_20260907.csv --json-output /tmp/structured_smoke_B_20260907.json
python3 structured_routing.py --routes routes_continue_first.json --dataset dataset_v2.jsonl --output results/structured_A_20260907.csv --json-output results/structured_A_20260907.json
python3 structured_routing.py --routes routes_new_first.json --dataset dataset_v2.jsonl --output results/structured_B_20260907.csv --json-output results/structured_B_20260907.json
python3 compare_structured_runs.py results/structured_A_20260907.csv results/structured_B_20260907.csv --output results/structured_comparison_20260907.json
python3 -m unittest -v test_structured_factorial_routing.py test_analyze_structured_factorial.py
python3 structured_factorial_routing.py --mode route-only --routes routes_continue_first.json --dataset dataset_v2.jsonl --case-ids case_001 case_020 case_024 case_029 --output /tmp/structured_factorial_smoke_20260907.csv --json-output /tmp/structured_factorial_smoke_20260907.json
python3 structured_factorial_routing.py --mode reason-route --routes routes_continue_first.json --dataset dataset_v2.jsonl --case-ids case_001 case_020 case_024 case_029 --output /tmp/structured_reason_smoke_20260907.csv --json-output /tmp/structured_reason_smoke_20260907.json
python3 structured_factorial_routing.py --mode route-only --routes routes_continue_first.json --dataset dataset_v2.jsonl --output results/structured_factorial_20260907.csv --json-output results/structured_factorial_20260907.json
python3 structured_factorial_routing.py --mode reason-route --routes routes_continue_first.json --dataset dataset_v2.jsonl --output results/structured_reason_20260907.csv --json-output results/structured_reason_20260907.json
python3 analyze_structured_factorial.py --route-only results/structured_factorial_20260907.csv --reason-route results/structured_reason_20260907.csv --factorial-output results/structured_factorial_comparison_20260907.json --reason-output results/structured_reason_comparison_20260907.json
python3 structured_factorial_routing.py --mode reason-route --max-completion-tokens 128 --routes routes_continue_first.json --dataset dataset_v2.jsonl --case-ids case_001 case_020 case_024 case_029 --output /tmp/structured_reason_max128_smoke_20260907.csv --json-output /tmp/structured_reason_max128_smoke_20260907.json
python3 structured_factorial_routing.py --mode route-only --max-completion-tokens 128 --routes routes_continue_first.json --dataset dataset_v2.jsonl --output results/structured_factorial_max128_20260907.csv --json-output results/structured_factorial_max128_20260907.json
python3 structured_factorial_routing.py --mode reason-route --max-completion-tokens 128 --routes routes_continue_first.json --dataset dataset_v2.jsonl --output results/structured_reason_max128_20260907.csv --json-output results/structured_reason_max128_20260907.json
python3 analyze_structured_factorial.py --route-only results/structured_factorial_max128_20260907.csv --reason-route results/structured_reason_max128_20260907.csv --factorial-output results/structured_factorial_max128_comparison_20260907.json --reason-output results/structured_reason_max128_comparison_20260907.json
```

## 12. Recommended next experiment

Следующий минимальный эксперимент: повторить S1 и S2 max128 несколько раз при тех же E1–E4, чтобы проверить детерминированную повторяемость улучшения accuracy и сокращения prompt-order sensitivity. Prompt, schemas и descriptions не менять.

---

# Дополнение 2026-09-07: повторяемость S2 и условия с мышлением

## 1. Goal

Проверить повторяемость результата S2 тремя новыми полными прогонами и сравнить его с двумя заранее зафиксированными условиями:

1. `reason + route`, мышление выключено;
2. `reason + route`, мышление включено;
3. только `route`, мышление включено.

Каждая группа выполнена три раза. Каждый повтор содержит E1–E4 по всем 30 случаям, то есть 120 запросов. Цель — измерить повторяемость, точность и устойчивость к порядку, а не настраивать prompt.

## 2. Starting state

Исходной точкой был завершённый S2 max128: обязательные поля `reason` и `route`, `enable_thinking=false`, 27/30 во всех E1–E4, шесть order-sensitive cases. Исторический artifact `results/structured_reason_max128_20260907.json` сохранён и не засчитывался как один из трёх новых повторов.

## 3. Changes made

- `structured_factorial_routing.py` — добавлен флаг `--enable-thinking`. По умолчанию он выключен, поэтому прежнее поведение runner сохранено. Флаг меняет только `chat_template_kwargs.enable_thinking`; model, temperature, seed, limit, prompt order, enum order и schema выбираются прежним кодом.
- `analyze_structured_repeatability.py` — добавлен отдельный анализатор трёх повторов и сравнений между группами: accuracy, confusion matrix, agreement, нестабильные случаи, order sensitivity, исправленные и сломанные решения.
- `test_structured_factorial_routing.py` — добавлена проверка, что включение мышления не меняет остальные параметры декодирования.
- `test_analyze_structured_repeatability.py` — добавлены проверки согласия трёх повторов, разделения повторяемости и order sensitivity, а также исключения ошибочных ответов из знаменателя.

Изменение `enable_thinking` образует новое экспериментальное условие. Dataset, descriptions, annotation policy и существующие результаты не изменялись.

## 4. Experiment configuration

- Model: `qwen3-4b`, root `Qwen/Qwen3-4B-AWQ`.
- Backend: локальный vLLM, `http://localhost:8000/v1/chat/completions`.
- Dataset: `dataset_v2.jsonl`, SHA-256 `fe6357ef072442689848d72f942828f5d12da4f71611fd86b4f9ceeddad6088c`, 30 случаев.
- Routes: `routes_continue_first.json`, SHA-256 `cfdff764daf9e962e49fb05003148c1469462fc1e0558573cf08a02430da4561`.
- Temperature: `0`; seed: `0`; `max_completion_tokens=128`.
- E1: prompt C,N; enum C,N.
- E2: prompt N,C; enum C,N.
- E3: prompt C,N; enum N,C.
- E4: prompt N,C; enum N,C.
- Structured output: прежние динамические JSON Schema режимов `reason-route` и `route-only`.
- Logprobs и отдельная scoring formula не использовались.
- Новых полных запросов: 3 группы × 3 повтора × 4 условия × 30 случаев = 1080.
- Перед полными прогонами обе thinking-группы прошли smoke test на `case_001`: 8/8 завершённых JSON, ошибок разбора нет. Лимит 128 не менялся.

## 5. Results

### 5.1 Повторяемость

Во всех трёх группах результат каждого из 120 сочетаний `condition × case` полностью совпал между R1, R2 и R3:

| группа | E1 | E2 | E3 | E4 | нестабильные случаи | ошибки |
|---|---:|---:|---:|---:|---:|---:|
| reason+route, без мышления | 30/30 | 30/30 | 30/30 | 30/30 | 0 | 0/360 |
| reason+route, с мышлением | 30/30 | 30/30 | 30/30 | 30/30 | 0 | 0/360 |
| только route, с мышлением | 30/30 | 30/30 | 30/30 | 30/30 | 0 | 0/360 |

Для двух режимов с `reason` полностью совпали не только route-векторы, но и тексты `reason`, prompt/completion token counts. Три новых S2-прогона без мышления также полностью совпали с ранее сохранённым успешным S2 max128, включая reasons.

### 5.2 Точность и устойчивость к порядку

Поскольку три повтора каждой группы идентичны, таблица относится к каждому повтору:

| группа | E1 | E2 | E3 | E4 | среднее по 120 | agreement E1↔E2 | order-sensitive cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| reason+route, без мышления | 27/30 | 27/30 | 27/30 | 27/30 | 108/120 = 90.00% | 24/30 = 80.00% | 6 |
| reason+route, с мышлением | 28/30 | 25/30 | 28/30 | 25/30 | 106/120 = 88.33% | 23/30 = 76.67% | 7 |
| только route, с мышлением | 24/30 | 19/30 | 24/30 | 19/30 | 86/120 = 71.67% | 17/30 = 56.67% | 13 |

Перестановка только enum не изменила ни одного решения во всех группах: E1↔E3 и E2↔E4 дали 30/30 agreement. Наблюдаемая чувствительность полностью следовала порядку описаний в prompt.

Order-sensitive cases:

- reason+route, без мышления: `case_001`, `case_014`, `case_020`, `case_021`, `case_023`, `case_024`;
- reason+route, с мышлением: тот же набор плюс `case_029`;
- только route, с мышлением: `case_001`, `case_002`, `case_006`, `case_012`, `case_015`, `case_017`, `case_020`, `case_021`, `case_022`, `case_023`, `case_026`, `case_027`, `case_029`.

### 5.3 Confusion matrices

Строки — expected `CONTINUE`, `NEW`; столбцы — predicted `CONTINUE`, `NEW`.

| группа | условия | matrix |
|---|---|---|
| reason+route, без мышления | E1–E4 | `[[14,1],[2,13]]` |
| reason+route, с мышлением | E1/E3 | `[[15,0],[2,13]]` |
| reason+route, с мышлением | E2/E4 | `[[12,3],[2,13]]` |
| только route, с мышлением | E1/E3 | `[[15,0],[6,9]]` |
| только route, с мышлением | E2/E4 | `[[6,9],[2,13]]` |

### 5.4 Изменения между группами

Числа ниже суммируют три соответствующих повтора, поэтому знаменатель равен 360 condition-case решений:

| сравнение | изменилось | исправлено | сломано |
|---|---:|---:|---:|
| без мышления reason+route → с мышлением reason+route | 18/360 | 6 | 12 |
| без мышления reason+route → с мышлением только route | 90/360 | 12 | 78 |
| с мышлением reason+route → с мышлением только route | 72/360 | 6 | 66 |

Для одного повтора включение мышления в S2 меняло шесть решений: исправляло `case_023` в E1/E3, но ломало `case_023` в E2/E4 и `case_029` в E2/E4. Таким образом, оно перераспределяло ошибки между prompt orders и уменьшало среднюю accuracy на 1.67 процентного пункта.

### 5.5 Latency и tokens

Средние значения по 120 запросам каждого повтора:

| группа | mean latency R1/R2/R3, s | mean prompt tokens | mean completion tokens |
|---|---|---:|---:|
| reason+route, без мышления | 0.600 / 0.605 / 0.600 | 353.33 | 60.22 |
| reason+route, с мышлением | 0.607 / 0.595 / 0.596 | 349.33 | 60.80 |
| только route, с мышлением | 0.142 / 0.116 / 0.117 | 315.33 | 10.48 |

Raw timing зависит от текущего состояния локального сервера и не является аппаратно-независимым benchmark. Route-only был существенно короче по выходу и быстрее в этих запусках.

## 6. Important examples

`C` означает `CONTINUE`, `N` — `NEW`. Внутри каждой группы строка полностью повторилась в R1–R3.

| id | expected | reason+route без мышления E1/E2/E3/E4 | reason+route с мышлением | только route с мышлением |
|---|---|---|---|---|
| case_017 | C | C/C/C/C | C/C/C/C | C/N/C/N |
| case_020 | N | C/N/C/N | C/N/C/N | C/N/C/N |
| case_021 | N | C/N/C/N | C/N/C/N | C/N/C/N |
| case_023 | C | N/C/N/C | C/N/C/N | C/N/C/N |
| case_024 | N | N/C/N/C | N/C/N/C | C/C/C/C |
| case_029 | C | C/C/C/C | C/N/C/N | C/N/C/N |

`case_023` особенно показателен: мышление перевернуло решение во всех четырёх условиях, но правильность всё равно зависела от prompt order. `case_029` стал чувствительным к порядку только при включённом мышлении. Удаление явного `reason` при включённом мышлении дополнительно сделало чувствительным `case_017` и ухудшило `case_024` во всех порядках.

## 7. Observations

- Наблюдение: при фиксированных входах, `temperature=0`, `seed=0` и неизменном состоянии сервера все три метода дали 100% повторяемость semantic predictions в трёх повторах.
- Наблюдение: полное совпадение агрегатной accuracy сопровождалось полным совпадением каждого отдельного route, а не взаимной компенсацией разных ошибок.
- Наблюдение: включение мышления в S2 не улучшило среднюю accuracy или устойчивость к prompt order: 90.00% → 88.33%, agreement 80.00% → 76.67%, order-sensitive cases 6 → 7.
- Наблюдение: thinking + route-only был быстрее и короче, но заметно хуже по accuracy и prompt-order robustness: 71.67%, agreement 56.67%, 13 order-sensitive cases.
- Наблюдение: enum order не повлиял на predictions ни в одном прогоне; изменения следовали prompt order.
- Наблюдение: ни один из 1080 полных запросов не завершился ошибкой или обрезанным JSON.

## 8. Interpretation / hypotheses

- ГИПОТЕЗА: в этой конфигурации vLLM greedy decoding практически детерминирован, поэтому прежнее улучшение S2 max128 воспроизводимо на уровне всех отдельных решений. Три повтора и один сервер недостаточны для общего утверждения о детерминизме на другом оборудовании или версии backend.
- ГИПОТЕЗА: включение внутреннего мышления не нейтрализует позиционное смещение и может закреплять интерпретацию, подсказанную первым route description. Это согласуется с ростом order-sensitive cases, но причинный механизм из сохранённых ответов не доказан.
- ГИПОТЕЗА: явное поле `reason` действует не только как дополнительный вывод, но и как инструкция сформулировать семантическое основание до route. При thinking enabled его удаление ухудшило 22 решения и улучшило только 2 на один повтор; однако одновременно меняются schema и системная инструкция, поэтому вклад каждого компонента отдельно не идентифицирован.
- ГИПОТЕЗА: короткий route-only ответ быстрее главным образом из-за меньшего числа completion tokens, а не из-за лучшего процесса выбора.

## 9. Problems / limitations

- Dataset содержит 30 сбалансированных случаев и одну модель; проценты имеют низкую статистическую силу.
- Три повтора выполнены последовательно на одном процессе vLLM. Они проверяют локальную повторяемость, но не устойчивость после рестарта сервера, на другом оборудовании или при другой версии vLLM.
- `temperature=0` и `seed=0` не гарантируют универсальную битовую детерминированность; здесь фиксируется только наблюдаемый результат.
- Сравнение thinking reason+route и thinking route-only меняет одновременно требуемую JSON Schema и добавочную системную инструкцию про `reason`. Оно измеряет целый формат ответа, а не изолированный эффект одного поля.
- Внутреннее мышление не сохранялось как отдельный проверяемый trace; анализ использует structured response, route, tokens и latency.
- Latency первых запросов может включать прогрев/состояние кэша, поэтому сравнение времени описательное.
- Prompt tuning, calibration, изменение descriptions, dataset или annotations не выполнялись.

## 10. Files produced

- `analyze_structured_repeatability.py`
- `test_analyze_structured_repeatability.py`
- `results/s2_no_thinking_repeat1_20260907.csv`
- `results/s2_no_thinking_repeat1_20260907.json`
- `results/s2_no_thinking_repeat2_20260907.csv`
- `results/s2_no_thinking_repeat2_20260907.json`
- `results/s2_no_thinking_repeat3_20260907.csv`
- `results/s2_no_thinking_repeat3_20260907.json`
- `results/s2_thinking_repeat1_20260907.csv`
- `results/s2_thinking_repeat1_20260907.json`
- `results/s2_thinking_repeat2_20260907.csv`
- `results/s2_thinking_repeat2_20260907.json`
- `results/s2_thinking_repeat3_20260907.csv`
- `results/s2_thinking_repeat3_20260907.json`
- `results/route_only_thinking_repeat1_20260907.csv`
- `results/route_only_thinking_repeat1_20260907.json`
- `results/route_only_thinking_repeat2_20260907.csv`
- `results/route_only_thinking_repeat2_20260907.json`
- `results/route_only_thinking_repeat3_20260907.csv`
- `results/route_only_thinking_repeat3_20260907.json`
- `results/structured_repeatability_thinking_comparison_20260907.json`
- `reports/CODEX_REPORT.md`

## 11. Exact commands

```bash
python3 structured_factorial_routing.py --mode reason-route --max-completion-tokens 128 --output results/s2_no_thinking_repeatN_20260907.csv --json-output results/s2_no_thinking_repeatN_20260907.json
python3 structured_factorial_routing.py --mode reason-route --enable-thinking --max-completion-tokens 128 --output results/s2_thinking_repeatN_20260907.csv --json-output results/s2_thinking_repeatN_20260907.json
python3 structured_factorial_routing.py --mode route-only --enable-thinking --max-completion-tokens 128 --output results/route_only_thinking_repeatN_20260907.csv --json-output results/route_only_thinking_repeatN_20260907.json
python3 analyze_structured_repeatability.py --group no_thinking_reason=results/s2_no_thinking_repeat1_20260907.csv,results/s2_no_thinking_repeat2_20260907.csv,results/s2_no_thinking_repeat3_20260907.csv --group thinking_reason=results/s2_thinking_repeat1_20260907.csv,results/s2_thinking_repeat2_20260907.csv,results/s2_thinking_repeat3_20260907.csv --group thinking_route_only=results/route_only_thinking_repeat1_20260907.csv,results/route_only_thinking_repeat2_20260907.csv,results/route_only_thinking_repeat3_20260907.csv --output results/structured_repeatability_thinking_comparison_20260907.json
```

Для первых трёх команд `N` последовательно принимал значения 1, 2 и 3. Smoke tests использовали те же команды с `--case-ids case_001` и путями в `/tmp`.

## 12. Recommended next experiment

Минимальный следующий эксперимент — повторить только один полный S2 max128 после контролируемого рестарта vLLM и сравнить каждый route и reason с текущими четырьмя идентичными S2 результатами. Это проверит, сохраняется ли повторяемость между сессиями сервера, не меняя метод маршрутизации.

---

# Дополнение 2026-09-07: проверка окружения перед model-size experiment

> Последующая проверка с расширенным доступом показала, что GPU и Docker на хосте исправны. Зафиксированный ниже сбой относился только к первоначальной песочнице Codex; artifact сохранён как хронология проверки и не является состоянием завершённого эксперимента.

## 1. Goal

Подготовить контролируемое сравнение `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-8B-AWQ` и `Qwen/Qwen3-14B-AWQ` для метода reason+route при фиксированных P1/P2 и сначала проверить обязательный доступ к Docker и NVIDIA GPU.

## 2. Starting state

Предыдущий S2 max128 использовал 4B с `max_model_len=4096`. Согласно новому протоколу этот результат нельзя использовать как прямую точку model-size comparison: 4B требуется перезапустить с `max_model_len=2048` и `max_num_seqs=1`.

## 3. Environment check

Проверка остановила этап до запуска моделей:

| проверка | результат |
|---|---|
| `command -v nvidia-smi` | `/usr/bin/nvidia-smi` |
| `nvidia-smi` | exit 9; невозможно связаться с NVIDIA driver |
| `command -v docker` | `/usr/bin/docker` |
| Docker client | 29.6.1, API 1.55 |
| Docker daemon | недоступен: permission denied для `/var/run/docker.sock` |

В соответствии с протоколом:

- существующий model server не останавливался;
- Docker containers не запускались и не изменялись;
- M4, M8 и M14 не загружались;
- smoke test не выполнялся;
- запросы к dataset не выполнялись;
- старый результат 4B/4096 не использовался как новый comparator;
- выводы о влиянии размера модели не делались.

## 4. Exact commands required from the user environment

Сначала выполнить в обычном терминале хоста:

```bash
nvidia-smi
sudo docker version
sudo docker ps -a --filter publish=8000 --format 'table {{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}'
```

Если `nvidia-smi` на хосте тоже завершается ошибкой, сначала необходимо восстановить NVIDIA driver средствами операционной системы. Конкретную команду установки драйвера нельзя безопасно определить без дистрибутива и модели GPU.

Если Docker работает только через `sudo`, для предоставления текущему пользователю доступа к daemon выполнить:

```bash
sudo usermod -aG docker "$(id -un)"
```

После этого полностью завершить пользовательский сеанс и войти снова либо перезапустить компьютер, затем проверить:

```bash
id
docker version
nvidia-smi
```

Для проверки Docker-доступа к GPU тем же уже имеющимся vLLM image выполнить:

```bash
VLLM_IMAGE="$(docker ps -a --filter publish=8000 --format '{{.Image}}' | head -n 1)"
test -n "$VLLM_IMAGE"
docker image inspect "$VLLM_IMAGE" --format '{{index .RepoDigests 0}}'
docker run --rm --gpus all --entrypoint nvidia-smi "$VLLM_IMAGE"
```

Если контейнера, публиковавшего порт 8000, нет, вместо попытки угадать image tag нужно сообщить точный ранее использованный vLLM image либо выбрать и зафиксировать один новый image tag/digest для всех трёх моделей.

После успешного выполнения последних трёх проверок необходимо полностью перезапустить Codex и повторить этот этап. Codex должен видеть успешные `nvidia-smi`, `docker version` и Docker GPU smoke test до остановки текущего сервера.

## 5. Results

Model-size results отсутствуют. Главная таблица accuracy/order sensitivity не создавалась, поскольку ни одна новая модельная конфигурация не была запущена.

## 6. Important examples

`case_001`, `case_014`, `case_020`, `case_021`, `case_023`, `case_024`, `case_029` не оценивались в этом этапе.

## 7. Observations

- Наблюдение: бинарные файлы Docker и `nvidia-smi` установлены.
- Наблюдение: текущий процесс Codex не имеет доступа к Docker daemon socket.
- Наблюдение: текущий процесс Codex не получает рабочий ответ NVIDIA driver через `nvidia-smi`.

## 8. Interpretation / hypotheses

- ГИПОТЕЗА: Docker daemon работает, но пользователь Codex не входит в группу `docker`; это согласуется с permission denied, но не доказано без `id`, `getent group docker` и успешного подключения.
- ГИПОТЕЗА: ошибка `nvidia-smi` может быть связана либо с состоянием драйвера на хосте, либо с изоляцией окружения Codex. Разделить причины можно сравнением с запуском той же команды в обычном терминале хоста.

## 9. Problems / limitations

- Нет доступа к обязательной вычислительной среде.
- Exact Docker image и vLLM version нельзя достоверно получить без доступа к daemon; они не угадывались.
- Никакие численные ответы на вопросы об accuracy и order sensitivity до реального запуска невозможны.

## 10. Files produced

- `results/model_size_environment_check_20260907.json`
- `reports/CODEX_REPORT.md`

## 11. Exact commands run by Codex

```bash
command -v nvidia-smi
command -v docker
nvidia-smi
docker version
```

## 12. Recommended next experiment

После восстановления доступа выполнить ровно исходный model-size protocol: новый M4/2048, затем M8 с единственным разрешённым offload fallback, затем M14 с 4 GB CPU offload; для каждой успешно поднятой модели только P1/P2 reason+route без мышления.

---

# Дополнение 2026-09-07: model-size comparison M4/M8/M14

## 1. Goal

Проверить, уменьшаются ли ошибка direct structured routing и чувствительность к порядку route descriptions с ростом размера Qwen3 при текущем лучшем методе reason+route.

## 2. Starting state

Исторический S2 max128 использовал 4B при `max_model_len=4096`. Он не применялся как прямой comparator. Для этого experiment family все три модели были заново запущены с `max_model_len=2048`, `max_num_seqs=1` и одним закреплённым Docker image.

Расширенная проверка окружения перед запуском:

- GPU: NVIDIA GeForce RTX 3060 Ti, 8192 MiB;
- driver: 595.84; CUDA reported by driver: 13.2;
- Docker client/server: 29.6.1;
- vLLM image: `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`;
- vLLM: `0.28.0`.

## 3. Changes made

- `model_size_routing.py` — новый generic runner P1/P2 для одной уже поднятой модели. Он переиспользует неизменённые prompt/schema функции reason+route; P2 разворачивает только prompt order, enum order остаётся фиксированным.
- `model_size_orchestrator.py` — новый последовательный Docker orchestrator: сохраняет `nvidia-smi`, останавливает сервер на порту 8000, запускает ровно одну модель, ожидает `/v1/models`, выполняет smoke/full run, сохраняет конфигурацию и логи, затем останавливает экспериментальный контейнер.
- `analyze_model_size.py` — новый comparator accuracy, confusion matrices, P1/P2 agreement, order-sensitive cases и переходов ошибок между моделями.
- `test_model_size_routing.py`, `test_model_size_orchestrator.py`, `test_analyze_model_size.py` — проверки P1/P2, фиксированных Docker flags, единственного M8 fallback, OOM detection и model comparison.
- `reports/CODEX_REPORT.md` — добавлен этот раздел; прежняя запись о sandbox environment дополнена уточнением.

Предыдущие baseline implementations и result artifacts не перезаписывались.

## 4. Experiment configuration

Общее для всех моделей:

- dataset: `dataset_v2.jsonl`, 30 cases, SHA-256 `fe6357ef072442689848d72f942828f5d12da4f71611fd86b4f9ceeddad6088c`;
- routes/descriptions: `routes_continue_first.json`, SHA-256 `cfdff764daf9e962e49fb05003148c1469462fc1e0558573cf08a02430da4561`;
- method: structured `reason + route`;
- prompt version: `structured-factorial-v1`, без изменений;
- JSON Schema: обязательные `reason` и `route`, dynamic enum, без дополнительных полей;
- temperature `0`, seed `0`, `enable_thinking=false`, `max_completion_tokens=128`;
- `max_model_len=2048`, `max_num_seqs=1`, AWQ;
- P1 prompt order: `CONTINUE`, затем `NEW`;
- P2 prompt order: `NEW`, затем `CONTINUE`;
- enum order в P1/P2: `CONTINUE`, затем `NEW`;
- 7-case smoke: `case_001`, `case_014`, `case_020`, `case_021`, `case_023`, `case_024`, `case_029`;
- full run: 30 cases × 2 orders = 60 requests на модель;
- token-level logprobs, reranker, thinking, few-shot и tuning не использовались.

Фактически использованные server configurations:

| model | exact id | gpu memory | CPU offload | eager | max len | max seqs | quantization |
|---|---|---:|---:|---|---:|---:|---|
| M4 | `Qwen/Qwen3-4B-AWQ` | 0.90 | 0 GB | false | 2048 | 1 | AWQ |
| M8 | `Qwen/Qwen3-8B-AWQ` | 0.85 | 0 GB | true | 2048 | 1 | AWQ |
| M14 | `Qwen/Qwen3-14B-AWQ` | 0.85 | 4 GB | true | 2048 | 1 | AWQ |

M8 успешно запустился с offload 0; разрешённый fallback offload 1 не применялся. M14 использовал заранее заданный offload 4 GB; vLLM log подтвердил `Total CPU offloaded parameters: 4.01`.

Состояние GPU после загрузки:

| model | общий расход GPU | VLLM EngineCore |
|---|---:|---:|
| M4 | 7044 MiB | 6524 MiB |
| M8 | 7147 MiB | 6628 MiB |
| M14 | 7084 MiB | 6564 MiB |

## 5. Results

### 5.1 Главная таблица

| model | params | P1 accuracy | P2 accuracy | mean accuracy | P1/P2 agreement | order-sensitive cases |
|---|---:|---:|---:|---:|---:|---:|
| M4 | 4B | 27/30 = 90.00% | 27/30 = 90.00% | 90.00% | 24/30 = 80.00% | 6 |
| M8 | 8B | 29/30 = 96.67% | 27/30 = 90.00% | 93.33% | 26/30 = 86.67% | 4 |
| M14, offload 4 GB | 14B | 27/30 = 90.00% | 27/30 = 90.00% | 90.00% | 28/30 = 93.33% | 2 |

Все 180 full responses завершились с `finish_reason=stop`; ошибок HTTP, schema parsing и обрезанных JSON не было.

### 5.2 Confusion matrices и ошибки

Строки — expected `CONTINUE`, `NEW`; столбцы — predicted `CONTINUE`, `NEW`.

| model/order | confusion matrix | incorrect cases |
|---|---|---|
| M4 P1 | `[[14,1],[2,13]]` | `case_020`, `case_021`, `case_023` |
| M4 P2 | `[[14,1],[2,13]]` | `case_001`, `case_014`, `case_024` |
| M8 P1 | `[[14,1],[0,15]]` | `case_002` |
| M8 P2 | `[[14,1],[2,13]]` | `case_026`, `case_028`, `case_029` |
| M14 P1 | `[[13,2],[1,14]]` | `case_013`, `case_020`, `case_027` |
| M14 P2 | `[[13,2],[1,14]]` | `case_010`, `case_013`, `case_020` |

Order-sensitive cases:

- M4: `case_001`, `case_014`, `case_020`, `case_021`, `case_023`, `case_024`;
- M8: `case_002`, `case_026`, `case_028`, `case_029`;
- M14: `case_010`, `case_027`.

### 5.3 Переходы ошибок

- M4→M8: исчезли ошибки на `case_001`, `case_014`, `case_020`, `case_021`, `case_023`, `case_024`; появились `case_002`, `case_026`, `case_028`, `case_029`; общих ошибочных cases нет.
- M4→M14: исчезли `case_001`, `case_014`, `case_021`, `case_023`, `case_024`; `case_020` остался ошибочным; появились `case_010`, `case_013`, `case_027`.
- M8→M14: исчезли `case_002`, `case_026`, `case_028`, `case_029`; появились `case_010`, `case_013`, `case_020`, `case_027`; общих ошибочных cases нет.
- Нет ни одного case, ошибочного хотя бы в одном порядке у всех трёх моделей.
- У M14 устойчивые к порядку ошибки: `case_013` и `case_020`; его `case_010` и `case_027` зависят от порядка.

### 5.4 Latency и tokens

| model/order | mean latency | median latency | mean prompt tokens | mean completion tokens |
|---|---:|---:|---:|---:|
| M4 P1 | 0.620 s | 0.608 s | 353.33 | 59.57 |
| M4 P2 | 0.622 s | 0.596 s | 353.33 | 60.57 |
| M8 P1 | 1.427 s | 1.386 s | 353.33 | 66.17 |
| M8 P2 | 1.394 s | 1.404 s | 353.33 | 64.37 |
| M14 P1 | 11.314 s | 11.017 s | 353.33 | 59.83 |
| M14 P2 | 11.208 s | 11.020 s | 353.33 | 59.07 |

Latency M14 нельзя трактовать как чистый эффект размера: он использовал 4 GB CPU offload, тогда как M4/M8 — 0 GB. M4 также работал без `enforce_eager`, а M8/M14 — с ним, согласно заранее заданной memory-конфигурации.

## 6. Important examples

| id | expected | M4 P1/P2 | M8 P1/P2 | M14 P1/P2 |
|---|---|---|---|---|
| case_001 | CONTINUE | C/N | C/C | C/C |
| case_014 | NEW | N/C | N/N | N/N |
| case_020 | NEW | C/N | N/N | C/C |
| case_021 | NEW | C/N | N/N | N/N |
| case_023 | CONTINUE | N/C | C/C | C/C |
| case_024 | NEW | N/C | N/N | N/N |
| case_029 | CONTINUE | C/C | C/N | C/C |

`C` означает `CONTINUE`, `N` — `NEW`. M8 исправил все шесть order-sensitive cases M4, но создал четыре других. M14 сохранил исправления `case_001`, `case_014`, `case_021`, `case_023`, `case_024`, однако снова ошибся на `case_020` в обоих порядках.

## 7. Observations

- Наблюдение: mean accuracy не росла монотонно: 90.00% → 93.33% → 90.00%.
- Наблюдение: P1/P2 agreement росло монотонно: 80.00% → 86.67% → 93.33%; число order-sensitive cases уменьшилось 6 → 4 → 2.
- Наблюдение: M8 показал максимальную accuracy, включая 96.67% в P1, но сохранил четыре новые order-sensitive cases.
- Наблюдение: M14 был устойчивее к порядку, но две его ошибки (`case_013`, `case_020`) одинаковы в P1/P2, поэтому order robustness не равна correctness.
- Наблюдение: prompt token count одинаков для моделей и порядков, что подтверждает сохранение входного представления.
- Наблюдение: M8 не потребовал fallback; M14 успешно завершил запуск и experiment с заранее заданным offload.

## 8. Interpretation / hypotheses

- ГИПОТЕЗА: рост размера модели в этом диапазоне повышает устойчивость к перестановке route descriptions, но не гарантирует монотонного улучшения соответствия текущей annotation policy.
- ГИПОТЕЗА: M14 формирует более стабильную собственную интерпретацию спорных случаев; на `case_020` эта интерпретация стабильна, но расходится с ground truth.
- ГИПОТЕЗА: различия M4/M8/M14 отражают одновременно model size и неизбежно разные memory execution settings (`enforce_eager`, offload). Accuracy обычно не должна зависеть от них методологически, но малые численные различия при greedy decoding нельзя полностью исключить.
- ГИПОТЕЗА: непересекающиеся error sets M8 и M14 указывают, что увеличение размера меняет границу решений, а не просто исправляет подмножество ошибок меньшей модели.

## 9. Problems / limitations

- Dataset содержит только 30 cases; одна ошибка меняет accuracy на 3.33 процентного пункта для одного порядка.
- Выполнен один full run на модель. Предыдущая проверка повторяемости относилась к M4/4096 в другом server condition и не заменяет повторы M4/M8/M14/2048.
- M14 использует CPU offload 4 GB; его latency нельзя напрямую сравнивать с M4/M8.
- M4 использует `gpu_memory_utilization=0.90` без eager; M8/M14 используют 0.85 с eager. Это заранее заданные условия вместимости, но они остаются инфраструктурным различием.
- Только одна model family, одна quantization и один GPU.
- Меньшее число order-sensitive cases не означает меньше semantic errors: M14 стабильно ошибается на двух случаях.
- Prompt tuning, few-shot и обучение после просмотра ошибок не выполнялись.
- Прежний контейнер `qwen3-4b-vllm` был остановлен по протоколу и автоматически не восстановлен; экспериментальные контейнеры после сохранения логов удалены.

## 10. Files produced

- `model_size_routing.py`
- `model_size_orchestrator.py`
- `analyze_model_size.py`
- `test_model_size_routing.py`
- `test_model_size_orchestrator.py`
- `test_analyze_model_size.py`
- `results/model_size_20260907/manifest.json`
- `results/model_size_20260907/comparison.json`
- `results/model_size_20260907/nvidia_initial.txt`
- `results/model_size_20260907/nvidia_final.txt`
- `results/model_size_20260907/m4/m4_smoke.csv`
- `results/model_size_20260907/m4/m4_smoke.json`
- `results/model_size_20260907/m4/m4_full.csv`
- `results/model_size_20260907/m4/m4_full.json`
- `results/model_size_20260907/m4/server_metadata.json`
- `results/model_size_20260907/m4/attempt1_server.log`
- `results/model_size_20260907/m8/m8_smoke.csv`
- `results/model_size_20260907/m8/m8_smoke.json`
- `results/model_size_20260907/m8/m8_full.csv`
- `results/model_size_20260907/m8/m8_full.json`
- `results/model_size_20260907/m8/server_metadata.json`
- `results/model_size_20260907/m8/attempt1_server.log`
- `results/model_size_20260907/m14/m14_smoke.csv`
- `results/model_size_20260907/m14/m14_smoke.json`
- `results/model_size_20260907/m14/m14_full.csv`
- `results/model_size_20260907/m14/m14_full.json`
- `results/model_size_20260907/m14/server_metadata.json`
- `results/model_size_20260907/m14/attempt1_server.log`
- отдельные `nvidia_before`, `nvidia_after_load`, `nvidia_after_experiment`, `smoke_runner.log`, `full_runner.log` в каждой model directory;
- `reports/CODEX_REPORT.md`.

## 11. Exact commands

Основной запуск:

```bash
python3 model_size_orchestrator.py --output-dir results/model_size_20260907
```

Анализ:

```bash
python3 analyze_model_size.py \
  --model M4=results/model_size_20260907/m4/m4_full.csv \
  --model M8=results/model_size_20260907/m8/m8_full.csv \
  --model M14=results/model_size_20260907/m14/m14_full.csv \
  --manifest results/model_size_20260907/manifest.json \
  --output results/model_size_20260907/comparison.json
```

Exact Docker commands сохранены как arrays в `results/model_size_20260907/manifest.json`.

## 12. Recommended next experiment

Следующий этап — новый существенно больший заранее размеченный held-out dataset и однократная оценка M8/M14 тем же неизменным reason+route P1/P2 protocol. Few-shot и обучение пока преждевременны: сначала нужно проверить, сохраняются ли преимущество M8 по accuracy и преимущество M14 по order robustness вне текущих 30 случаев.

---

# Дополнение 2026-09-09: проектирование Dataset V3

Создана только спецификация будущего held-out benchmark; сами случаи не генерировались. `dataset_v2.jsonl`, route descriptions, prompts, модели и существующие результаты не изменялись, модели не запускались.

- `docs/DATASET_V3_SPEC.md` формализует идентичность задачи, policy `CONTINUE`/`NEW`, taxonomy, защиту от утечки, разметку, закрытый test и будущие показатели.
- `docs/DATASET_V3_PLAN.json` фиксирует машинно-читаемые квоты: 300 случаев, классы 150/150, разделение 120/60/120, сложности 84/132/84, 15 областей по 20 случаев, 75 контрастивных семейств по три случая и 75 самостоятельных случаев.
- Текущие 30 случаев закреплены как отдельный диагностический набор и не входят в Dataset V3.
- Зафиксированы вопросы policy о границе новой операции над той же сущностью, сообщениях без задачи и внешних действиях над готовым артефактом.

Следующий этап допускается только после ручного утверждения спецификации: генерация кандидатов с независимой разметкой. Автоматический переход к генерации не выполнялся.

---

# Дополнение 2026-09-09: Dataset V3 pre-test

Выполнены аудит и development/validation experiment без доступа к test. Frozen winner по заранее заданному правилу — M8 (`Qwen/Qwen3-8B-AWQ`): validation P1 81.67%, P2 76.67%, mean 79.17%, agreement 91.67%. M4 mean validation accuracy 73.33%, agreement 76.67%. M14 condition осталось неполным из-за семи инфраструктурно отсутствующих P2 ответов и последующего CUDA OOM при неизменной recovery-конфигурации.

Message-only TF-IDF baseline достиг 81.67%, поэтому перед финальным test рекомендована независимая ручная проверка синтетических меток и surface leakage. Test не открывался и не запускался. Полный отчёт: `reports/DATASET_V3_PRETEST_REPORT.md`; frozen protocol: `results/dataset_v3_pretest_20260909/FROZEN_TEST_PROTOCOL.json`.

По явному запросу пользователя выполнена ещё одна попытка восстановления M14 в строго прежней конфигурации. Сервер снова не поднялся: CUDA OOM при AWQ conversion, попытка выделения 680 MiB. Smoke и семь отсутствующих P2 requests не запускались, новых predictions нет, frozen winner и метрики не изменились. Артефакт: `results/dataset_v3_pretest_20260909/TECHNICAL_INCIDENT_M14_RECOVERY2_FAILURE.json`.

После отдельного разрешения пользователя создано supplemental M14 condition с `cpu_offload_gb=6` вместо 4. Модель загрузилась, smoke дал 12/12 валидных JSON, получены все семь отсутствовавших P2 ответов (6/7 correct). Supplemental merged M14: P1 85.00%, P2 81.67%, mean 83.33%, agreement 93.33%, 4 order-sensitive cases. Это численно выше M8, но результат смешивает 113 baseline-ответов offload 4 GB и 7 recovery-ответов offload 6 GB, поэтому формальный frozen winner M8 не изменён. Для чистой смены winner нужен полный M14 validation rerun с offload 6 GB. Подробности: `results/dataset_v3_pretest_20260909/M14_OFFLOAD6_RECOVERY_SUCCESS.json`.
