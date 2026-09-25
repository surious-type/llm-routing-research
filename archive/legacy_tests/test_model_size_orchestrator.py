import unittest

import model_size_orchestrator as orchestrator


BASE = {
    "label": "M8",
    "model_id": "Qwen/Qwen3-8B-AWQ",
    "served_model_name": "qwen3-8b",
    "gpu_memory_utilization": 0.85,
    "cpu_offload_gb": 0,
    "max_model_len": 2048,
    "max_num_seqs": 1,
    "enforce_eager": True,
    "quantization": "awq",
}


class ModelSizeOrchestratorTests(unittest.TestCase):
    def test_docker_command_contains_fixed_model_configuration(self):
        command = orchestrator.build_docker_command(
            BASE,
            image="vllm/vllm-openai@sha256:abc",
            cache_dir="/cache/hf",
            container_name="routing-m8",
        )

        self.assertEqual(command[:3], ["docker", "run", "--detach"])
        self.assertIn("Qwen/Qwen3-8B-AWQ", command)
        self.assertEqual(command[command.index("--max-model-len") + 1], "2048")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
        self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], "0.85")
        self.assertEqual(command[command.index("--cpu-offload-gb") + 1], "0")
        self.assertIn("--enforce-eager", command)
        self.assertEqual(command[command.index("--quantization") + 1], "awq")

    def test_m8_fallback_changes_only_cpu_offload(self):
        fallback = orchestrator.with_cpu_offload(BASE, 1)

        self.assertEqual(fallback["cpu_offload_gb"], 1)
        self.assertEqual(
            {key: value for key, value in fallback.items() if key != "cpu_offload_gb"},
            {key: value for key, value in BASE.items() if key != "cpu_offload_gb"},
        )
        self.assertEqual(BASE["cpu_offload_gb"], 0)

    def test_oom_detection_does_not_treat_other_startup_errors_as_oom(self):
        self.assertTrue(orchestrator.is_gpu_oom("torch.OutOfMemoryError: CUDA out of memory"))
        self.assertFalse(orchestrator.is_gpu_oom("Repository not found"))


if __name__ == "__main__":
    unittest.main()
