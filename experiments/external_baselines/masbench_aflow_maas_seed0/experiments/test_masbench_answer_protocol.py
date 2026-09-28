import unittest

from masbench_answer_protocol import extract_masbench_answer, native_objective_score, score_masbench


class MasBenchAnswerProtocolTests(unittest.TestCase):
    def test_explicit_scalar_answer(self):
        self.assertEqual(extract_masbench_answer("Reasoning has 17.\nFINAL ANSWER: 2", 1), "2")

    def test_explicit_horizon_answer(self):
        text = "FINAL ANSWER: 02 <<horizon>> 17 <<horizon>> 0"
        self.assertEqual(extract_masbench_answer(text, 3), "2<<horizon>>17<<horizon>>0")

    def test_native_final_answers_boxed(self):
        text = """Some earlier calculation is \\boxed{99}.
### Final Answers
Problem 1: \\boxed{2}
Problem 2: \\boxed{17}
"""
        self.assertEqual(extract_masbench_answer(text, 2), "2<<horizon>>17")

    def test_partial_credit_is_shared_012(self):
        result = score_masbench(
            "### Final Answers\nProblem 1: \\boxed{2}\nProblem 2: \\boxed{9}",
            "2<<horizon>>17",
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["score_0_1_2"], 1)
        self.assertEqual(result["component_accuracy"], 0.5)

    def test_exact_match_gets_two(self):
        result = score_masbench("FINAL ANSWER: 2<<horizon>>17", "02<<horizon>>17")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["score_0_1_2"], 2)
        self.assertTrue(result["correct"])

    def test_native_search_objective_uses_same_012_rubric(self):
        gold = "2<<horizon>>17"
        self.assertEqual(native_objective_score("FINAL ANSWER: 2<<horizon>>17", gold)[0], 2)
        self.assertEqual(native_objective_score("FINAL ANSWER: 2<<horizon>>8", gold)[0], 1)
        self.assertEqual(native_objective_score("FINAL ANSWER: 8<<horizon>>9", gold)[0], 0)

    def test_incomplete_numbered_sequence_rejected(self):
        text = "### Final Answers\nProblem 1: \\boxed{2}\nProblem 3: \\boxed{17}"
        self.assertEqual(extract_masbench_answer(text, 2), "")

    def test_does_not_guess_intermediate_number(self):
        self.assertEqual(extract_masbench_answer("The intermediate value is 12.", 1), "")


if __name__ == "__main__":
    unittest.main()
