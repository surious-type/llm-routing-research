import json
from pathlib import Path

SOURCE = Path("dataset.jsonl")
TARGET = Path("dataset_v2.jsonl")

mapping = {
    "A": "CONTINUE",
    "B": "NEW",
}

counts = {}

with SOURCE.open(encoding="utf-8") as src, \
     TARGET.open("w", encoding="utf-8") as dst:

    for line in src:
        if not line.strip():
            continue

        row = json.loads(line)

        old = row.pop("expected")
        row["expected_route"] = mapping[old]

        counts[row["expected_route"]] = (
            counts.get(row["expected_route"], 0) + 1
        )

        dst.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

print("Created:", TARGET)
print("Counts :", counts)
