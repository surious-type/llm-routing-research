# Dataset V3 Pre-test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Провести защищённый development/validation предэксперимент Dataset V3 без доступа к закрытому test.

**Architecture:** Новые небольшие скрипты отдельно отвечают за аудит и лексические baseline, LLM-запуск одной модели, последовательный Docker orchestration и пост-анализ. Модельный prompt переиспользует неизменённые функции `structured_factorial_routing.py`; анализ присоединяет metadata только после получения ответа.

**Tech Stack:** Python 3.14, requests, scikit-learn 1.7.2, Docker, vLLM 0.28.0, JSONL/CSV.

**Spec:** `docs/DATASET_V3_EXPERIMENT_PROTOCOL.md`

## Global Constraints

- Не читать и не открывать test inputs, test labels, trusted full test или secret archive.
- Не выполнять LLM prediction до установки неизменяемого experiment protocol в проект.
- Не изменять baseline prompts, descriptions, schema или model flags.
- Новые outputs создаются эксклюзивно и не перезаписывают старые artifacts.
- Все метаданные исключены из prompt белым списком.

---

### Task 1: Dataset audit and lexical diagnostics

**Files:**
- Create: `dataset_v3_pretest_audit.py`
- Test: `test_dataset_v3_pretest_audit.py`

**Interfaces:**
- Consumes: development/validation JSONL и routes config.
- Produces: `audit_datasets(...)`, `build_lexical_text(...)`, `run_lexical_baselines(...)`, `dataset_audit.json`, `lexical_baselines.json`.

- [ ] Написать тесты валидных квот, пересечения family, exact duplicate и трёх лексических представлений.
- [ ] Запустить тесты и подтвердить ожидаемый FAIL из-за отсутствующего модуля.
- [ ] Реализовать минимальный аудит и три фиксированных TF-IDF/logistic regression pipeline.
- [ ] Запустить тесты и подтвердить PASS.

### Task 2: Metadata-safe model runner

**Files:**
- Create: `dataset_v3_pretest_routing.py`
- Test: `test_dataset_v3_pretest_routing.py`

**Interfaces:**
- Consumes: один разрешённый split, модельный endpoint, server metadata.
- Produces: P1/P2 CSV/JSON с raw responses и evaluation metadata.

- [ ] Написать регрессионный тест побайтовой идентичности prompt после замены всех evaluation metadata.
- [ ] Написать тесты фиксированных P1/P2, enum, параметров и сохранения metadata только в output.
- [ ] Запустить тесты и подтвердить ожидаемый FAIL.
- [ ] Реализовать runner через `build_payload`/`route_case` существующего reason+route метода.
- [ ] Запустить тесты и подтвердить PASS.

### Task 3: Metrics and family analysis

**Files:**
- Create: `dataset_v3_pretest_analyze.py`
- Test: `test_dataset_v3_pretest_analyze.py`

**Interfaces:**
- Consumes: CSV/JSON M4/M8/M14 и исходные metadata development/validation.
- Produces: `comparison.json`, breakdown-файлы, `family_analysis.json`, frozen winner record.

- [ ] Написать тесты confusion/recall, dual-order metrics, breakdown, family exact accuracy и заранее заданного tie-break ranking.
- [ ] Запустить тесты и подтвердить ожидаемый FAIL.
- [ ] Реализовать чистые функции анализа.
- [ ] Запустить тесты и подтвердить PASS.

### Task 4: Sequential Docker orchestration

**Files:**
- Create: `dataset_v3_pretest_orchestrator.py`
- Test: `test_dataset_v3_pretest_orchestrator.py`

**Interfaces:**
- Consumes: фиксированные model specs, image digest, smoke ids и список split на модель.
- Produces: model CSV/JSON, server logs, nvidia snapshots и run manifest.

- [ ] Написать тесты точных Docker flags, отсутствия fallback, M14 validation-only и шести smoke ids.
- [ ] Запустить тесты и подтвердить ожидаемый FAIL.
- [ ] Реализовать последовательный запуск одной модели и остановку при infrastructure/smoke failure.
- [ ] Запустить тесты и подтвердить PASS.

### Task 5: Freeze, execute and report

**Files:**
- Create: `results/dataset_v3_pretest_20260909/*`
- Create: `reports/DATASET_V3_PRETEST_REPORT.md`
- Create: `results/dataset_v3_pretest_20260909/FROZEN_TEST_PROTOCOL.json`
- Modify: `reports/CODEX_REPORT.md`

**Interfaces:**
- Consumes: все проверенные скрипты и protocol.
- Produces: полный development/validation handoff без test run.

- [ ] Установить protocol в проект и сохранить его SHA-256 до первого prediction.
- [ ] Выполнить dataset audit и три лексических baseline.
- [ ] Выполнить последовательные smoke/full model runs по матрице протокола.
- [ ] Сформировать comparison, breakdown и family analysis.
- [ ] Применить заранее заданное validation ranking и записать frozen winner.
- [ ] Зафиксировать evaluator hashes и отчётность в `FROZEN_TEST_PROTOCOL.json`.
- [ ] Создать отчёт с раздельными НАБЛЮДЕНИЯМИ и ГИПОТЕЗАМИ.
- [ ] Запустить полную верификацию outputs и подтвердить, что test status равен `NOT RUN`.
