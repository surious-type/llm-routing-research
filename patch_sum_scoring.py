from pathlib import Path
import ast
import difflib
import json
import re
import shutil
import sys


MAIN = Path("main_sequence.py")
BACKUP = Path("main_sequence.py.before_sum_scoring")
ROUTES_SOURCE = Path("routes_continue_first.json")
ROUTES_REVERSED = Path("routes_new_first.json")


if not MAIN.exists():
    raise SystemExit(f"Не найден {MAIN}")


# ---------------------------------------------------------------------
# 1. Backup
# ---------------------------------------------------------------------

if not BACKUP.exists():
    shutil.copy2(MAIN, BACKUP)
    print(f"Backup: {BACKUP}")
else:
    print(f"Backup уже существует: {BACKUP}")


source = MAIN.read_text(encoding="utf-8")
original = source


# ---------------------------------------------------------------------
# 2. Находим функции, где normalized_score_probability
#    считается через mean_logprob, и меняем ТОЛЬКО там mean -> sum.
#
#    Это позволяет оставить mean_logprob в CSV/diagnostics.
# ---------------------------------------------------------------------

tree = ast.parse(source)

lines = source.splitlines(keepends=True)

replacements = []

for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue

    if not hasattr(node, "end_lineno"):
        continue

    start = node.lineno - 1
    end = node.end_lineno

    block = "".join(lines[start:end])

    if (
        "normalized_score_probability" in block
        and "mean_logprob" in block
    ):
        new_block = block.replace(
            '["mean_logprob"]',
            '["sum_logprob"]',
        ).replace(
            "['mean_logprob']",
            "['sum_logprob']",
        )

        if new_block != block:
            replacements.append(
                (start, end, new_block, node.name)
            )


# Применяем снизу вверх, чтобы line indexes не съехали.
for start, end, new_block, name in sorted(
    replacements,
    reverse=True,
):
    print(
        f"Нормализация в функции {name}: "
        "mean_logprob -> sum_logprob"
    )

    lines[start:end] = [new_block]


source = "".join(lines)


# ---------------------------------------------------------------------
# 3. Основным prediction делаем predicted_sum.
#
#    Поддерживаем несколько распространённых вариантов записи.
# ---------------------------------------------------------------------

patterns = [
    # predicted_route = predicted_mean
    (
        r"(\bpredicted_route\s*=\s*)predicted_mean\b",
        r"\1predicted_sum",
    ),

    # "predicted_route": predicted_mean
    (
        r'(["\']predicted_route["\']\s*:\s*)predicted_mean\b',
        r"\1predicted_sum",
    ),

    # correct = predicted_mean == expected_route
    (
        r"(\bcorrect\s*=\s*)predicted_mean(\s*==\s*expected_route\b)",
        r"\1predicted_sum\2",
    ),

    # "correct": predicted_mean == ...
    (
        r'(["\']correct["\']\s*:\s*)predicted_mean(\s*==)',
        r"\1predicted_sum\2",
    ),
]

for pattern, replacement in patterns:
    source, count = re.subn(
        pattern,
        replacement,
        source,
    )

    if count:
        print(
            f"Patch prediction: {count} replacement(s)"
        )


# ---------------------------------------------------------------------
# 4. Если используется correct_sum, проще и надёжнее сделать
#    primary correct равным ему.
# ---------------------------------------------------------------------

source, count = re.subn(
    r'(["\']correct["\']\s*:\s*)'
    r'predicted_route\s*==\s*expected_route',
    r'\1predicted_sum == expected_route',
    source,
)

if count:
    print(
        f"Primary correct -> predicted_sum: {count}"
    )


# ---------------------------------------------------------------------
# 5. Меняем текст основного вывода, если есть старый
#    "Predicted mean".
#
#    Сам predicted_mean оставляем в диагностике.
# ---------------------------------------------------------------------

source = source.replace(
    'print(f"Predicted mean : {predicted_route}")',
    'print(f"Predicted      : {predicted_route}  [SUM]")',
)

source = source.replace(
    'print(f"Result(mean)   :',
    'print(f"Result         :',
)


# ---------------------------------------------------------------------
# 6. Проверяем синтаксис до записи.
# ---------------------------------------------------------------------

try:
    ast.parse(source)
except SyntaxError as exc:
    print()
    print("ОШИБКА: после patch файл невалиден.")
    print(exc)
    print("main_sequence.py НЕ изменён.")
    sys.exit(1)


# ---------------------------------------------------------------------
# 7. Записываем main_sequence.py
# ---------------------------------------------------------------------

MAIN.write_text(
    source,
    encoding="utf-8",
)


# ---------------------------------------------------------------------
# 8. Создаём второй routes-файл:
#
#       routes_continue_first.json
#       CONTINUE
#       NEW
#
#             ↓ reverse
#
#       routes_new_first.json
#       NEW
#       CONTINUE
#
#    Меняется ТОЛЬКО порядок объектов.
# ---------------------------------------------------------------------

if ROUTES_SOURCE.exists():
    routes = json.loads(
        ROUTES_SOURCE.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(routes, list):
        raise RuntimeError(
            f"{ROUTES_SOURCE} должен содержать JSON-массив"
        )

    reversed_routes = list(reversed(routes))

    ROUTES_REVERSED.write_text(
        json.dumps(
            reversed_routes,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Created: {ROUTES_REVERSED}")

    print("Order:")

    for i, route in enumerate(
        reversed_routes,
        start=1,
    ):
        name = (
            route.get("name")
            or route.get("id")
            or "???"
        )

        print(f"  {i}. {name}")

else:
    print()
    print(
        f"WARNING: {ROUTES_SOURCE} не найден. "
        f"{ROUTES_REVERSED} не создан."
    )


# ---------------------------------------------------------------------
# 9. Показываем diff
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("DIFF main_sequence.py")
print("=" * 70)

diff = difflib.unified_diff(
    original.splitlines(),
    source.splitlines(),
    fromfile=str(BACKUP),
    tofile=str(MAIN),
    lineterm="",
)

diff_text = "\n".join(diff)

if diff_text:
    print(diff_text)
else:
    print(
        "Изменений в main_sequence.py не найдено. "
        "Возможно, scoring уже использует sum_logprob."
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)
print()
print("Primary sequence score: SUM log-probability")
print("mean_logprob сохранён только для диагностики.")
