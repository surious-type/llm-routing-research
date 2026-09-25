import json
import unittest

import structured_routing as routing


ROUTES = [
    {"name": "CONTINUE", "description": "Продолжение текущей задачи."},
    {"name": "NEW", "description": "Новая самостоятельная задача."},
]

ROW = {
    "id": "case_test",
    "history": [
        {"role": "user", "content": "Помоги настроить вход."},
        {"role": "assistant", "content": "Какая ошибка?"},
    ],
    "message": "Ошибка 403.",
    "expected_route": "CONTINUE",
}


class StructuredRoutingTests(unittest.TestCase):
    def test_schema_uses_route_names_in_configuration_order(self):
        response_format = routing.build_response_format(ROUTES)

        self.assertEqual(
            response_format,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "route_selection",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "route": {
                                "type": "string",
                                "enum": ["CONTINUE", "NEW"],
                            }
                        },
                        "required": ["route"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def test_user_prompt_contains_history_message_and_ordered_routes(self):
        prompt = routing.build_user_prompt(ROW, ROUTES)

        self.assertEqual(
            prompt,
            "ТЕКУЩИЙ ДИАЛОГ:\n"
            "Пользователь: Помоги настроить вход.\n"
            "Ассистент: Какая ошибка?\n\n"
            "НОВОЕ СООБЩЕНИЕ:\n"
            "Ошибка 403.\n\n"
            "ДОСТУПНЫЕ МАРШРУТЫ:\n"
            "CONTINUE: Продолжение текущей задачи.\n"
            "NEW: Новая самостоятельная задача.",
        )

    def test_payload_requests_structured_output_without_logprobs(self):
        payload = routing.build_payload("qwen3-4b", ROW, ROUTES)

        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertNotIn("logprobs", payload)
        self.assertNotIn("top_logprobs", payload)
        self.assertNotIn("logprob_token_ids", payload)

    def test_extract_response_validates_route_enum(self):
        response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"route": "NEW"}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 7,
                "total_tokens": 130,
            },
        }

        parsed = routing.extract_structured_response(response, ROUTES)

        self.assertEqual(parsed["route"], "NEW")
        self.assertEqual(parsed["structured_response"], {"route": "NEW"})
        self.assertEqual(parsed["prompt_tokens"], 123)
        self.assertEqual(parsed["completion_tokens"], 7)

        response["choices"][0]["message"]["content"] = '{"route":"OTHER"}'
        with self.assertRaisesRegex(ValueError, "неизвестный маршрут"):
            routing.extract_structured_response(response, ROUTES)


if __name__ == "__main__":
    unittest.main()
