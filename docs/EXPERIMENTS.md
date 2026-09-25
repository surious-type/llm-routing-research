# Карта экспериментов

Этот файл отделяет действующие experiment entry points от исторических скриптов.

| Эксперимент | Что меняется | Основной запуск | Исторические источники |
|---|---|---|---|
| Prompt conditions | семантика A/B и позиция A, условия E1-E4 | `experiments/prompt_conditions/run.py` | `archive/legacy_code/main.py`, ранние `run_*_E*.csv` |
| Independent scoring | отдельная LLM-оценка каждого кандидата | `experiments/independent_scoring/run.py` | `independent_candidate_scoring.py`, representation diagnostics |
| Reranker | LLM scoring заменяется cross-encoder reranker | `experiments/reranker/run.py` | `reranker_routing.py` |
| Structured routing | прямой выбор route через JSON Schema | `experiments/structured_routing/run.py` | `structured_routing.py`, structured factorial/repeatability |
| Model comparison | меняется обслуживаемая Qwen3-модель при фиксированном reason+route pipeline | `experiments/model_comparison/run.py` | `model_size_routing.py`, `model_size_orchestrator.py` |

## Важное обозначение E1-E4

В новой структуре обозначения E1-E4 закреплены за исходным A/B prompt experiment:

- E1: A=CONTINUE, B=NEW, A стоит первой;
- E2: A=NEW, B=CONTINUE, A стоит первой;
- E3: A=CONTINUE, B=NEW, A стоит второй;
- E4: A=NEW, B=CONTINUE, A стоит второй.

В старом structured factorial коде обозначения E1-E4 также использовались для другой факторной матрицы (prompt order × enum order). Эти файлы сохранены в archive и results/structured_routing/legacy, но в новых entry points это обозначение повторно не используется.

## Датасеты

`data/diagnostic/routing_v2.jsonl` — канонический исторический 30-case набор для воспроизведения старых сравнений.

`data/v3/development.jsonl` и `data/v3/validation.jsonl` — Dataset V3. Их структура богаче: кроме expected_route присутствуют category, difficulty, ambiguity, domain и family_id.

Backup и промежуточные версии лежат только в `archive/data/`.
