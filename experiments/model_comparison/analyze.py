"""Сводная таблица по JSON-результатам нескольких моделей."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run", action="append", required=True, help="LABEL=results.json")
parser.add_argument("--output", default="results/model_comparison/comparison.json")
args = parser.parse_args()

comparison = {}
for item in args.run:
    label, path = item.split("=", 1)
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    comparison[label] = {
        "model": run["metadata"]["model"],
        "P1": run["summary"]["P1"],
        "P2": run["summary"]["P2"],
        "order_agreement": run["summary"]["order_agreement"],
    }

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(comparison, ensure_ascii=False, indent=2))
