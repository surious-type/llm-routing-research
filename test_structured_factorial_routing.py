import unittest

import structured_factorial_routing as factorial


ROUTES = [
    {"name": "CONTINUE", "description": "Описание продолжения."},
    {"name": "NEW", "description": "Описание новой задачи."},
]

ROW = {
    "id": "case_test",
    "history": [{"role": "user", "content": "Старое сообщение."}],
    "message": "Новое сообщение.",
    "expected_route": "CONTINUE",
}


class StructuredFactorialRoutingTests(unittest.TestCase):
    def test_conditions_change_prompt_order_independently_from_enum_order(self):
        conditions = factorial.build_conditions(["CONTINUE", "NEW"])

        self.assertEqual(conditions["E1"]["prompt_order"], ["CONTINUE", "NEW"])
        self.assertEqual(conditions["E2"]["prompt_order"], ["NEW", "CONTINUE"])
        self.assertEqual(conditions["E1"]["enum_order"], conditions["E2"]["enum_order"])

    def test_conditions_change_enum_order_independently_from_prompt_order(self):
        conditions = factorial.build_conditions(["CONTINUE", "NEW"])

        self.assertEqual(conditions["E1"]["enum_order"], ["CONTINUE", "NEW"])
        self.assertEqual(conditions["E3"]["enum_order"], ["NEW", "CONTINUE"])
        self.assertEqual(conditions["E1"]["prompt_order"], conditions["E3"]["prompt_order"])

    def test_prompt_keeps_each_description_with_its_semantic_route(self):
        prompt = factorial.build_user_prompt(ROW, ROUTES, ["NEW", "CONTINUE"])

        self.assertLess(prompt.index("NEW: Описание новой задачи."), prompt.index("CONTINUE: Описание продолжения."))
        self.assertNotIn("NEW: Описание продолжения.", prompt)
        self.assertNotIn("CONTINUE: Описание новой задачи.", prompt)

    def test_route_only_schema_contains_only_route(self):
        schema = factorial.build_response_format(["CONTINUE", "NEW"], "route-only")
        body = schema["json_schema"]["schema"]

        self.assertEqual(list(body["properties"]), ["route"])
        self.assertEqual(body["required"], ["route"])
        self.assertFalse(body["additionalProperties"])

    def test_reason_route_schema_requires_reason_and_route(self):
        schema = factorial.build_response_format(["NEW", "CONTINUE"], "reason-route")
        body = schema["json_schema"]["schema"]

        self.assertEqual(list(body["properties"]), ["reason", "route"])
        self.assertEqual(body["required"], ["reason", "route"])
        self.assertFalse(body["additionalProperties"])

    def test_enum_is_built_from_dynamic_order(self):
        schema = factorial.build_response_format(["THIRD", "FIRST", "SECOND"], "route-only")

        self.assertEqual(
            schema["json_schema"]["schema"]["properties"]["route"]["enum"],
            ["THIRD", "FIRST", "SECOND"],
        )

    def test_payload_does_not_request_logprobs(self):
        condition = {"prompt_order": ["CONTINUE", "NEW"], "enum_order": ["NEW", "CONTINUE"]}
        payload = factorial.build_payload("qwen3-4b", ROW, ROUTES, condition, "reason-route")

        self.assertNotIn("logprobs", payload)
        self.assertNotIn("top_logprobs", payload)
        self.assertNotIn("logprob_token_ids", payload)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_payload_keeps_previous_completion_limit_by_default(self):
        condition = {"prompt_order": ["CONTINUE", "NEW"], "enum_order": ["CONTINUE", "NEW"]}

        payload = factorial.build_payload(
            "qwen3-4b", ROW, ROUTES, condition, "route-only"
        )

        self.assertEqual(payload["max_completion_tokens"], 32)

    def test_payload_uses_preregistered_completion_limit_128(self):
        condition = {"prompt_order": ["CONTINUE", "NEW"], "enum_order": ["CONTINUE", "NEW"]}

        payload = factorial.build_payload(
            "qwen3-4b", ROW, ROUTES, condition, "reason-route",
            max_completion_tokens=128,
        )

        self.assertEqual(payload["max_completion_tokens"], 128)

    def test_payload_can_enable_thinking_without_changing_other_decoding_settings(self):
        condition = {"prompt_order": ["CONTINUE", "NEW"], "enum_order": ["NEW", "CONTINUE"]}

        payload = factorial.build_payload(
            "qwen3-4b", ROW, ROUTES, condition, "reason-route",
            max_completion_tokens=128, enable_thinking=True,
        )

        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(payload["max_completion_tokens"], 128)
        self.assertNotIn("logprobs", payload)

    def test_route_is_taken_from_structured_route_not_reason(self):
        response = {
            "choices": [{"message": {"content": '{"reason":"Нужно выбрать NEW","route":"CONTINUE"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }

        parsed = factorial.extract_structured_response(
            response, ["CONTINUE", "NEW"], "reason-route"
        )

        self.assertEqual(parsed["route"], "CONTINUE")
        self.assertEqual(parsed["reason"], "Нужно выбрать NEW")

    def test_route_case_preserves_raw_truncated_response(self):
        raw_response = {
            "choices": [{
                "message": {"content": '{"reason":"обрезано'},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 32},
        }

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return raw_response

        class Session:
            def post(self, *args, **kwargs):
                return Response()

        condition = {
            "prompt_order": ["CONTINUE", "NEW"],
            "enum_order": ["CONTINUE", "NEW"],
        }
        result, raw = factorial.route_case(
            Session(), "http://localhost:8000", "qwen3-4b", ROW, ROUTES,
            "E1", condition, "reason-route", 120,
        )

        self.assertIn("не является JSON", result["error"])
        self.assertEqual(result["structured_response"], '{"reason":"обрезано')
        self.assertEqual(result["completion_tokens"], 32)
        self.assertEqual(raw["response"], raw_response)
        self.assertIn("не является JSON", raw["parse_error"])


if __name__ == "__main__":
    unittest.main()
