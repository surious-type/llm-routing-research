import csv
import sys

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python3 compare_runs.py RUN1.csv RUN2.csv"
    )

def load(path):
    with open(path, encoding="utf-8") as f:
        return {
            row["id"]: row
            for row in csv.DictReader(f)
        }

a = load(sys.argv[1])
b = load(sys.argv[2])

ids = sorted(set(a) & set(b))

def is_true(x):
    return x.lower() == "true"

acc_a = sum(is_true(a[i]["correct"]) for i in ids)
acc_b = sum(is_true(b[i]["correct"]) for i in ids)

agreement = sum(
    a[i]["predicted_route"] == b[i]["predicted_route"]
    for i in ids
)

print(f"Cases: {len(ids)}")
print()
print(
    f"Run 1 accuracy: {acc_a}/{len(ids)} "
    f"= {acc_a / len(ids):.3%}"
)
print(
    f"Run 2 accuracy: {acc_b}/{len(ids)} "
    f"= {acc_b / len(ids):.3%}"
)
print(
    f"Semantic agreement: {agreement}/{len(ids)} "
    f"= {agreement / len(ids):.3%}"
)

print()
print("RUN 1 ERRORS")
for i in ids:
    r = a[i]
    if not is_true(r["correct"]):
        print(
            f"{i}: expected={r['expected_route']}, "
            f"predicted={r['predicted_route']}, "
            f"confidence={float(r['confidence']):.6f}"
        )

print()
print("RUN 2 ERRORS")
for i in ids:
    r = b[i]
    if not is_true(r["correct"]):
        print(
            f"{i}: expected={r['expected_route']}, "
            f"predicted={r['predicted_route']}, "
            f"confidence={float(r['confidence']):.6f}"
        )

print()
print("ORDER-SENSITIVE CASES")
for i in ids:
    x = a[i]
    y = b[i]

    if x["predicted_route"] != y["predicted_route"]:
        print()
        print(i)
        print(f"  expected : {x['expected_route']}")
        print(
            f"  run 1    : {x['predicted_route']} "
            f"(conf={float(x['confidence']):.6f})"
        )
        print(
            f"  run 2    : {y['predicted_route']} "
            f"(conf={float(y['confidence']):.6f})"
        )
        print(f"  message  : {x['message']}")
