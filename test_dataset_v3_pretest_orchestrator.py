import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dataset_v3_pretest_orchestrator import (
    IMAGE,
    MODEL_SPECS,
    SMOKE_IDS,
    build_docker_command,
    classify_runner_result,
)


class OrchestratorConfigTests(unittest.TestCase):
    def test_model_matrix_and_smoke_ids_are_frozen(self):
        self.assertEqual([spec["label"] for spec in MODEL_SPECS], ["M4", "M8", "M14"])
        self.assertEqual(MODEL_SPECS[0]["splits"], ["development", "validation"])
        self.assertEqual(MODEL_SPECS[1]["splits"], ["development", "validation"])
        self.assertEqual(MODEL_SPECS[2]["splits"], ["validation"])
        self.assertEqual(
            SMOKE_IDS,
            [
                "v3_development_0003", "v3_development_0006",
                "v3_development_0002", "v3_development_0001",
                "v3_development_0011", "v3_development_0005",
            ],
        )

    def test_docker_commands_match_predeclared_memory_configs_without_fallback(self):
        expected = {
            "M4": ("0.9", "0", False),
            "M8": ("0.85", "0", True),
            "M14": ("0.85", "4", True),
        }
        for spec in MODEL_SPECS:
            command = build_docker_command(spec, image=IMAGE, cache_dir="/cache", container_name="server")
            gpu, offload, eager = expected[spec["label"]]
            self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], gpu)
            self.assertEqual(command[command.index("--cpu-offload-gb") + 1], offload)
            self.assertEqual(command[command.index("--max-model-len") + 1], "2048")
            self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
            self.assertEqual("--enforce-eager" in command, eager)
            self.assertEqual(command[command.index("--generation-config") + 1], "vllm")
            self.assertEqual(command[command.index("--default-chat-template-kwargs") + 1], '{"enable_thinking": false}')

    def test_complete_response_level_parse_error_does_not_become_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps({
                "rows": [
                    {"condition": "P1", "id": "a", "error": ""},
                    {"condition": "P2", "id": "a", "error": "ValueError: truncated"},
                ],
                "raw_responses": [
                    {"condition": "P2", "id": "a", "response": {"choices": [{"finish_reason": "length"}]}}
                ],
            }), encoding="utf-8")
            status = classify_runner_result(SimpleNamespace(returncode=1), path, expected_rows=2)
        self.assertEqual(status, {"acceptable": True, "status": "complete_with_response_errors", "response_error_count": 1})

    def test_missing_raw_response_remains_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps({
                "rows": [
                    {"condition": "P1", "id": "a", "error": "ConnectionError"},
                    {"condition": "P2", "id": "a", "error": ""},
                ],
                "raw_responses": [],
            }), encoding="utf-8")
            status = classify_runner_result(SimpleNamespace(returncode=1), path, expected_rows=2)
        self.assertEqual(status["acceptable"], False)

    def test_saved_complete_output_is_reusable_without_original_process_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps({
                "rows": [
                    {"condition": "P1", "id": "a", "error": ""},
                    {"condition": "P2", "id": "a", "error": ""},
                ],
                "raw_responses": [],
            }), encoding="utf-8")
            status = classify_runner_result(SimpleNamespace(returncode=1), path, expected_rows=2)
        self.assertEqual(status, {"acceptable": True, "status": "complete", "response_error_count": 0})


if __name__ == "__main__":
    unittest.main()
