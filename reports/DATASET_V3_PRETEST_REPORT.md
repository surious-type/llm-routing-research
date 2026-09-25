# Dataset V3 pre-test: development + validation

Дата: 2026-09-09  
Статус test: **NOT RUN**  
Метки: предварительные синтетические, не human-validated gold.

## 1. Цель

Проверить на новых development/validation данных ранее выбранный метод `reason + route`, сравнить M4/M8/M14 при двух порядках описаний маршрутов и до test зафиксировать одного победителя. Дополнительная цель — аудит качества набора и поиск поверхностных подсказок.

## 2. Dataset audit

До первого LLM prediction создан и зафиксирован [протокол](../docs/DATASET_V3_EXPERIMENT_PROTOCOL.md). Его SHA-256 остался `75e3542f0c8fbc8d2e3c1aef3a5558802b62e86566aa15811d286289a71687da`.

Аудит пройден:

- development: 120 случаев, `CONTINUE/NEW = 60/60`;
- validation: 60 случаев, `CONTINUE/NEW = 30/30`;
- 180 уникальных id;
- пересечений `family_id` между splits нет;
- точных дубликатов полных случаев и router inputs нет;
- обязательные metadata присутствуют;
- JSONL валиден;
- prompt строится только из `history[].role`, `history[].content`, `message` и внешних route descriptions;
- regression test подтвердил, что замена evaluation metadata не меняет байты prompt.

В prompt не передавались `id`, `expected_route`, `split`, `category`, `difficulty`, `ambiguity`, `domain`, `family_id`, `policy_version`, `annotation_notes`.

## 3. Surface-leakage diagnostic

Обучение TF-IDF + logistic regression выполнялось на development, оценка — на validation.

| Вход | Accuracy | Balanced accuracy | Confusion matrix C/N |
|---|---:|---:|---|
| Только message | 81.67% | 81.67% | `[[26,4],[7,23]]` |
| Только history | 51.67% | 51.67% | `[[17,13],[16,14]]` |
| History + message | 65.00% | 65.00% | `[[19,11],[10,20]]` |

**НАБЛЮДЕНИЕ:** message-only baseline выше случайного уровня на 31.67 п.п. и даже выше каждого отдельного порядка M8 на validation.

**ГИПОТЕЗА:** в синтетических сообщениях присутствуют лексические или стилевые признаки метки. Неожиданное ухудшение при добавлении history также допускает несовпадение поверхностных распределений development и validation. Эти результаты не доказывают конкретный источник утечки, но требуют ручного аудита до финального test.

## 4. Exact protocol

- Метод: structured JSON `{"reason": "...", "route": "<route id>"}`.
- Prompt version: `structured-factorial-v1`.
- Route descriptions: без изменений, SHA-256 `cfdff764daf9e962e49fb05003148c1469462fc1e0558573cf08a02430da4561`.
- P1: `CONTINUE → NEW`; P2: `NEW → CONTINUE`.
- Enum order: `CONTINUE, NEW`.
- `temperature=0`, `seed=0`, `enable_thinking=false`, `max_completion_tokens=128`.
- Для всех моделей: AWQ, `max_model_len=2048`, `max_num_seqs=1`.
- Docker image: `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`.
- vLLM: `0.28.0`.
- M4: `Qwen/Qwen3-4B-AWQ`, GPU 0.90, offload 0, eager false.
- M8: `Qwen/Qwen3-8B-AWQ`, GPU 0.85, offload 0, eager true.
- M14: `Qwen/Qwen3-14B-AWQ`, GPU 0.85, offload 4 GB, eager true.

Smoke cases были детерминированно выбраны до просмотра outputs: `v3_development_0003`, `0006`, `0002`, `0001`, `0011`, `0005`. M4 и M8 прошли smoke и полные запуски. Первый M14 smoke также прошёл, но full validation потерял семь последних P2 ответов из-за инфраструктурного сбоя.

## 5. Development results

| Model | P1 accuracy | P2 accuracy | Mean | P1/P2 agreement | Agreed coverage | Order-sensitive | Agreed but wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| M4 | 81.67% | 75.00% | 78.33% | 68.33% | 68.33% | 38 | 7 |
| M8 | 82.50% | 75.00% | 78.75% | 84.87%* | 84.17% | 18 | 16 |

`*` Agreement M8 рассчитано среди 119 валидных P1/P2 пар. `v3_development_0090` P2 достиг лимита 128 токенов и вернул обрезанный JSON; строка сохранена как ошибка, не повторялась и входит в accuracy как неправильная.

Confusion matrices (`expected × predicted`, порядок C/N):

- M4 P1 `[[57,3],[19,41]]`, P2 `[[36,24],[6,54]]`;
- M8 P1 `[[58,2],[19,41]]`, P2 `[[57,3],[26,33]]` плюс одна unresolved NEW строка.

**НАБЛЮДЕНИЕ:** M8 практически не улучшил mean development accuracy относительно M4 (+0.42 п.п.), но заметно увеличил согласие порядков и уменьшил order-sensitive cases с 38 до 18.

## 6. Validation results и выбор модели

| Model | P1 accuracy | P2 accuracy | Mean | Agreement среди валидных пар | Agreed coverage | Order-sensitive | Agreed but wrong | Статус |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M4 | 80.00% | 66.67% | 73.33% | 76.67% | 76.67% | 14 | 9 | complete |
| M8 | 81.67% | 76.67% | **79.17%** | 91.67% | 91.67% | 5 | 10 | complete |
| M14 | 85.00% | 71.67%† | 78.33%† | 92.45%† | 81.67% | 4† | 7† | incomplete |

`†` У M14 отсутствуют семь P2 predictions. Они считаются неправильными в aggregate accuracy; agreement и order-sensitive count относятся только к 53 валидным парам. Эти цифры нельзя трактовать как чистое сравнение model quality.

Confusion matrices:

- M4 P1 `[[26,4],[8,22]]`, P2 `[[17,13],[7,23]]`;
- M8 P1 `[[28,2],[9,21]]`, P2 `[[26,4],[10,20]]`;
- M14 P1 `[[27,3],[6,24]]`, P2 `[[23,4],[6,20]]` и 7 unresolved.

M8 errors:

- P1 (11): `0002, 0004, 0005, 0007, 0008, 0009, 0012, 0029, 0045, 0046, 0059`;
- P2 (14): `0002, 0004, 0005, 0007, 0008, 0009, 0012, 0024, 0030, 0036, 0045, 0046, 0049, 0059`.

Во всех id выше подразумевается префикс `v3_validation_`.

Заранее заданное правило ранжирования выбрало **M8** по максимальной mean validation accuracy. Рейтинг численно: M8 79.17%, M14 78.33%, M4 73.33%; при этом M14 помечен как incomplete и не является чистым условием.

## 7. P1/P2 robustness и dual-order consistency

Для победителя M8 на validation:

- P1/P2 совпали для 55/60 случаев: coverage 91.67%;
- accuracy среди согласованных решений: 45/55 = 81.82%;
- согласованных, но неправильных: 10;
- disagreement subset: 5 случаев (`0024`, `0029`, `0030`, `0036`, `0049`);
- accuracy на disagreement subset: P1 80%, P2 20%.

**НАБЛЮДЕНИЕ:** совпадение P1/P2 полезно как фильтр устойчивости, но оно не гарантирует правильность: 18.18% согласованных M8 решений на validation ошибочны.

**ГИПОТЕЗА:** P1 в этом наборе систематически благоприятнее P2, но выбирать P1 после просмотра validation запрещено; frozen method сохраняет оба порядка и заранее заданную отчётность.

## 8. Breakdown: label, difficulty, ambiguity, category, domain

M8 validation recall:

- `CONTINUE`: P1 93.33%, P2 86.67%;
- `NEW`: P1 70.00%, P2 66.67%.

Ошибки чаще относятся к ожидаемому `NEW`, то есть сохраняется склонность выбирать `CONTINUE`.

Difficulty, mean двух порядков:

| Difficulty | Cases | Mean accuracy | Agreement |
|---|---:|---:|---:|
| easy | 16 | 71.88% | 93.75% |
| medium | 28 | 78.57% | 85.71% |
| hard | 16 | 87.50% | 100.00% |

**НАБЛЮДЕНИЕ:** заявленная difficulty не упорядочивает фактическую сложность для M8; hard оказался лучше easy. Это дополнительный сигнал для ручной проверки метаданных и генерационного шаблона.

Наиболее слабые категории M8 validation по mean accuracy:

- `mixed_continuation_and_independent`: 33.33% (6 случаев);
- `entity_substitution_same_operation`: 40.00% (5);
- `new_goal_same_workflow`: 50.00% (4);
- `completed_then_new_task`: 62.50% (4).

Наиболее слабые domains: `files_storage` 25%, `data_analytics_sql` 50%, `programming` 50%, `support_accounts` 50%. В каждом domain всего четыре случая, поэтому это описательные показатели, не устойчивые оценки домена.

По ambiguity: high 66.67% в обоих порядках, medium 93.75%/81.25%, low 78.95%/76.32%. High содержит только шесть случаев и при этом P1/P2 agreement 100%, включая две согласованные ошибки.

## 9. Contrastive family analysis

| Model/split | P1 family exact | P2 family exact | P1 mean family acc. | P2 mean family acc. | Families with order disagreement |
|---|---:|---:|---:|---:|---:|
| M4 development | 53.33% | 36.67% | 81.11% | 73.33% | 20/30 |
| M8 development | 53.33% | 40.00% | 83.33% | 75.56% | 12/30 |
| M4 validation | 60.00% | 26.67% | 84.44% | 64.44% | 10/15 |
| M8 validation | 60.00% | 53.33% | 82.22% | 80.00% | 2/15 |
| M14 validation† | 66.67% | 46.67% | 84.44% | 73.33% | 3/15 |

`†` M14 P2 включает инфраструктурно отсутствующие строки как ошибки; две семьи содержат invalid pair.

**НАБЛЮДЕНИЕ:** M8 значительно устойчивее M4 на contrastive families validation, особенно в P2. Однако только 8 из 15 семейств полностью правильны в обоих смыслах P2 family exact, поэтому отдельные строки нельзя считать независимыми доказательствами высокого качества.

## 10. Latency и tokens

Средняя latency на validation:

- M4: P1 0.616 s, P2 0.642 s;
- M8: P1 1.546 s, P2 1.577 s;
- M14: P1 12.694 s, P2 12.392 s среди ответивших запросов.

Средние prompt tokens одинаковы по моделям: 443.4. Средние completion tokens: M4 65.9/71.77, M8 75.87/77.27, M14 66.65/68.85. M14 latency с CPU offload нельзя интерпретировать как чистый эффект размера.

## 11. Infrastructure incidents

1. M8 development: один завершённый HTTP response содержал JSON, обрезанный на лимите 128 токенов. Лимит не увеличивался, ответ не повторялся.
2. M14 validation: после 113 ответов семь P2 запросов завершились разрывом соединения. В это время появился внешний GPU-heavy процесс; его память вместе с vLLM превышала VRAM.
3. После закрытия внешнего приложения была предпринята зафиксированная попытка поднять M14 с абсолютно теми же флагами. Она завершилась явным CUDA OOM при загрузке AWQ.
4. После завершения исходного отчёта пользователь явно запросил ещё одну попытку. При 563 MiB занятой GPU memory до старта та же конфигурация снова завершилась CUDA OOM во время AWQ conversion, пытаясь выделить 680 MiB. Endpoint не поднялся, smoke и семь recovery requests не выполнялись. Memory flags и методология не менялись.

Первичный M14 artifact сохранён, успешные predictions не перезапускались.

## 12. Сравнение с Dataset V2

Историческая справка, не объединяемая с V3:

| Model | Dataset V2 mean accuracy | Agreement |
|---|---:|---:|
| M4 | 90.00% | 80.00% |
| M8 | 93.33% | 86.67% |
| M14 | 90.00% | 93.33% |

**НАБЛЮДЕНИЕ:** качественный вывод «M8 лучше M4 по accuracy» сохранился на validation и едва сохранился на development. Вывод «более крупная модель менее чувствительна к order» сохранился для M8 против M4. Для M14 direction среди валидных пар сходный, но инфраструктурная неполнота не позволяет считать это полноценной репликацией.

## 13. Ограничения

- Метки синтетические и ещё не прошли независимую ручную валидацию.
- Message-only baseline 81.67% указывает на существенный риск surface leakage.
- Каждая category/domain группа мала; проценты нестабильны.
- Contrastive family members зависимы и не должны интерпретироваться как независимые наблюдения.
- Одна M8 строка невалидна; M14 validation не завершён полностью.
- Один запуск каждой модели не измеряет повторяемость.
- Разные memory execution settings были заранее заданы для вместимости моделей.
- Test не открывался и не запускался.

## 14. Frozen winner и test protocol

Frozen winner: **M8 / `Qwen/Qwen3-8B-AWQ` / reason + route / thinking off / 128 tokens / P1+P2**.

Конфигурация, hashes, evaluator file hashes, порядок маршрутов и правило отчётности сохранены в `results/dataset_v3_pretest_20260909/FROZEN_TEST_PROTOCOL.json`.

Несмотря на заморозку технического победителя, рекомендация — **не открывать test до независимой ручной проверки меток и расследования message-only shortcut**. Если remediation изменит данные или метод, потребуется новая версия dataset/protocol; нельзя молча переносить текущую validation-селекцию.

## 15. Файлы и команды

Основные artifacts:

- `results/dataset_v3_pretest_20260909/dataset_audit.json`;
- `results/dataset_v3_pretest_20260909/lexical_baselines.json`;
- `results/dataset_v3_pretest_20260909/m4_development.{csv,json}`;
- `results/dataset_v3_pretest_20260909/m8_development.{csv,json}`;
- `results/dataset_v3_pretest_20260909/m4_validation.{csv,json}`;
- `results/dataset_v3_pretest_20260909/m8_validation.{csv,json}`;
- `results/dataset_v3_pretest_20260909/m14_validation.{csv,json}`;
- `results/dataset_v3_pretest_20260909/comparison.json`;
- `results/dataset_v3_pretest_20260909/breakdown_by_category.json`;
- `results/dataset_v3_pretest_20260909/breakdown_by_difficulty.json`;
- `results/dataset_v3_pretest_20260909/breakdown_by_domain.json`;
- `results/dataset_v3_pretest_20260909/family_analysis.json`;
- `results/dataset_v3_pretest_20260909/FROZEN_TEST_PROTOCOL.json`;
- server logs, metadata, NVIDIA snapshots и incident files в той же директории.

Команды:

```bash
python3 dataset_v3_pretest_audit.py \
  --development dataset_v3_development.jsonl \
  --validation dataset_v3_validation.jsonl \
  --output-dir results/dataset_v3_pretest_20260909

python3 dataset_v3_pretest_orchestrator.py \
  --output-dir results/dataset_v3_pretest_20260909

python3 dataset_v3_pretest_analyze.py \
  --run M4:development=results/dataset_v3_pretest_20260909/m4_development.json \
  --run M8:development=results/dataset_v3_pretest_20260909/m8_development.json \
  --run M4:validation=results/dataset_v3_pretest_20260909/m4_validation.json \
  --run M8:validation=results/dataset_v3_pretest_20260909/m8_validation.json \
  --run M14:validation=results/dataset_v3_pretest_20260909/m14_validation.json \
  --output-dir results/dataset_v3_pretest_20260909
```

Exact Docker commands находятся в `run_manifest.json`, `server_metadata.json` и recovery `docker_inspect.json`.

## 16. Итог

**НАБЛЮДЕНИЕ:** M8 — победитель по заранее заданному validation rule; он превосходит M4 по accuracy и намного устойчивее к порядку. P1/P2 agreement полезно, но десять согласованных ошибок показывают, что это не confidence и не замена correctness.

**ГИПОТЕЗА:** значительная часть результата может поддерживаться поверхностными регулярностями синтетической генерации. До проверки людьми нельзя делать сильный вывод об обобщающей способности semantic router.

## 17. Supplemental M14 recovery с CPU offload 6 GB

После двух CUDA OOM при baseline-конфигурации пользователь явно разрешил добавить оперативную память. Создано новое инфраструктурное условие, в котором изменён только `cpu_offload_gb`: 4 → 6. vLLM сообщил `6.03 GB` offloaded parameters; модель успешно загрузилась.

- smoke: 12/12 валидных JSON, 0 ошибок;
- повторены только семь отсутствовавших P2 случаев `0054–0060`;
- получено 7/7 ответов, правильных 6/7;
- единственная ошибка: `v3_validation_0059`, предсказан `CONTINUE` при expected `NEW`;
- mean latency восстановленных запросов: 23.79 s.

После подстановки этих семи строк в отдельной supplemental-сводке M14 имеет:

| P1 | P2 | Mean | P1/P2 agreement | Order-sensitive | Accuracy when agree | Agreed but wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 85.00% | 81.67% | 83.33% | 93.33% | 4 | 85.71% | 8 |

**НАБЛЮДЕНИЕ:** завершённая supplemental-сводка M14 численно выше M8 по mean validation accuracy: 83.33% против 79.17%.

**ОГРАНИЧЕНИЕ:** это смешанный результат: 113 ответов получены с offload 4 GB, семь — с offload 6 GB. Поэтому исходный `FROZEN_TEST_PROTOCOL.json` и формально выбранный M8 не переписывались. Для чистой смены победителя требуется полный повтор всех 120 M14 validation requests в единой заранее зафиксированной конфигурации offload 6 GB.
