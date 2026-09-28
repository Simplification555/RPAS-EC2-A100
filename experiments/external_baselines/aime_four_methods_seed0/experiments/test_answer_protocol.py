import unittest

from answer_protocol import VERSION, extract_aime_answer, score_aime


class AimeAnswerProtocolTests(unittest.TestCase):
    def test_version_is_frozen(self):
        self.assertEqual(VERSION, "answer_protocol_v4")

    def test_bold_answer_label(self):
        self.assertEqual(extract_aime_answer("**Answer:** 70"), "70")

    def test_plain_answer_label(self):
        self.assertEqual(extract_aime_answer("Answer: 279"), "279")

    def test_final_answer_sentence(self):
        self.assertEqual(extract_aime_answer("Final Answer seems to be 62."), "62")

    def test_hash_answer(self):
        self.assertEqual(extract_aime_answer("#### 16"), "16")

    def test_step_heading_does_not_hide_boxed_answer(self):
        text = "### Step 3: calculate\nThe result is below.\n\\boxed{4}"
        self.assertEqual(extract_aime_answer(text), "4")

    def test_final_standalone_scalar(self):
        self.assertEqual(extract_aime_answer("Reasoning mentions 3.\n\n247"), "247")

    def test_does_not_guess_intermediate_number(self):
        self.assertEqual(extract_aime_answer("The intermediate value is 17, conclusion unclear."), "")

    def test_integer_normalization_and_exact_match(self):
        scored = score_aime("**Answer:** 007", "7")
        self.assertTrue(scored["correct"])
        self.assertEqual(scored["prediction"], "7")
        self.assertEqual(scored["score_0_1_2"], 2)

    def test_shared_ordinal_score_distinguishes_parseable_wrong_from_invalid(self):
        wrong = score_aime("FINAL ANSWER: 8", "7")
        invalid = score_aime("The answer may be 7", "7")
        self.assertEqual(wrong["score_0_1_2"], 1)
        self.assertEqual(invalid["score_0_1_2"], 0)


if __name__ == "__main__":
    unittest.main()
