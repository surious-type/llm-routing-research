# Independent candidate scoring

Каждый маршрут оценивается отдельно. Для кандидата вычисляется score = logP(1) - logP(0), затем выбирается максимальный score.

Запуск:
~~~bash
python -m experiments.independent_scoring.run --model Qwen/Qwen3-4B-AWQ
~~~

Чтобы проверить влияние порядка кандидатов, повторите запуск с --routes data/routes/new_first.json.
