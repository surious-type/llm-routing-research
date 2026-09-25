# Model comparison

Один запуск относится к одной уже поднятой модели. Pipeline фиксирован: reason + route, thinking выключен. Автоматически выполняются P1 и P2 — два порядка descriptions.

~~~bash
python -m experiments.model_comparison.run   --model Qwen/Qwen3-8B-AWQ   --output results/model_comparison/m8.json
~~~

Сравнение сохранённых запусков выполняет analyze.py.
