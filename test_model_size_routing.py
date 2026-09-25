import unittest

import model_size_routing as routing


class ModelSizeRoutingTests(unittest.TestCase):
    def test_conditions_reverse_only_prompt_order(self):
        conditions = routing.build_conditions(["CONTINUE", "NEW"])

        self.assertEqual(conditions["P1"]["prompt_order"], ["CONTINUE", "NEW"])
        self.assertEqual(conditions["P2"]["prompt_order"], ["NEW", "CONTINUE"])
        self.assertEqual(conditions["P1"]["enum_order"], ["CONTINUE", "NEW"])
        self.assertEqual(conditions["P2"]["enum_order"], ["CONTINUE", "NEW"])

    def test_conditions_are_generic_for_more_than_two_routes(self):
        conditions = routing.build_conditions(["A", "B", "C"])

        self.assertEqual(conditions["P1"]["prompt_order"], ["A", "B", "C"])
        self.assertEqual(conditions["P2"]["prompt_order"], ["C", "B", "A"])
        self.assertEqual(conditions["P2"]["enum_order"], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
