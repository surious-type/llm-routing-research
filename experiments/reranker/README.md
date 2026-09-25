# Reranker

Отдельный эксперимент с cross-encoder reranker. Здесь нет A/B и token-level LLM scoring.

Запуск:
~~~bash
python -m experiments.reranker.run --endpoint http://localhost:8011/rerank
~~~

Порядок кандидатов задаётся через --routes.
