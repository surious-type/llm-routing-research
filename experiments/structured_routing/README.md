# Structured LLM routing

Модель выбирает семантический route через JSON Schema. Режим задаётся через --mode route-only или --mode reason-route.

Пример:
~~~bash
python -m experiments.structured_routing.run   --model Qwen/Qwen3-4B-AWQ   --mode reason-route   --output results/structured_routing/qwen3-4b.json
~~~

E1-E4 здесь не используются: эти имена относятся к исходному A/B prompt experiment.
