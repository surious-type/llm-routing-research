# Independent Candidate Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run generic independent candidate scoring for both route-order configurations on `dataset_v2.jsonl`, preserving listwise baselines and producing reproducible raw results and a self-contained report.

**Architecture:** Add a separate experiment script that loads route definitions from JSON, builds one candidate-only prompt per request, scores single-token `1`/`0` with vLLM token-level logprobs, and selects the maximum log-odds score. Add a comparison script for the two runs and record metadata, raw candidate metrics, and analysis in the report.

**Tech Stack:** Python 3, `requests`, vLLM OpenAI-compatible API, CSV/JSON.

**Spec:** User request and repository `AGENTS.md`.

## Global Constraints

- Preserve all existing baseline scripts and historical result files.
- Do not change `dataset_v2.jsonl`, route descriptions, or the baseline prompt.
- Route names and count must be configuration-driven.
- Primary route decision is `argmax(logP(1) - logP(0))`.
- Ranking softmax must be labeled relative, not calibrated correctness probability.
- Do not report an experiment as complete if vLLM requests fail or only part of the dataset is processed.

### Task 1: Implement independent scoring runner

**Files:**
- Create: `independent_candidate_scoring.py`

- [ ] Implement CLI arguments for routes config, dataset, endpoint, model, seed, temperature, prompt version, limit/start, timeout, and output path.
- [ ] Implement generic route loading and validation, single-token `1`/`0` validation, candidate-only prompt construction, robust token-id logprob extraction, binary probability, score, and per-case argmax.
- [ ] Persist one CSV row per case with nested candidate JSON, raw logprobs, latency, token usage, and reproducibility metadata; fail the run if any case cannot be scored.

### Task 2: Implement run comparison

**Files:**
- Create: `compare_independent_runs.py`

- [ ] Compare two CSV files by case id, calculate accuracy for each run and semantic agreement, list disagreements/errors, and print score margins and selected candidate metrics for required cases.
- [ ] Write a JSON comparison artifact containing the same analysis for report consumption.

### Task 3: Validate and run experiments

**Files:**
- Create: `reports/CODEX_REPORT.md`
- Create: `results/independent_A_<timestamp>.csv`
- Create: `results/independent_B_<timestamp>.csv`
- Create: `results/independent_comparison_<timestamp>.json`

- [ ] Run syntax validation and a local single-case smoke test if vLLM is reachable.
- [ ] Run the full 30-case experiment with both route configurations without changing the prompt.
- [ ] Compare runs and inspect `case_017`, `case_020`, `case_021`, `case_023`, `case_024`, and `case_029`.
- [ ] Update the self-contained report with exact commands, numerical results, observations, hypotheses, limitations, and files produced.
