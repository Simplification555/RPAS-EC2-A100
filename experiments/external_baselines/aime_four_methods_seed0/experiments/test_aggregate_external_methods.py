import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aggregate_external_methods import _check_rows, _verify_aime_freeze, summarize


class ExternalAggregateTests(unittest.TestCase):
    def test_aime_aggregate_fails_closed_without_dtest_freeze_ledger(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "missing D_test blind-freeze/access evidence"):
                _verify_aime_freeze(Path(temporary), {"method": "adas"}, {})

    def test_row_audit_enforces_common_score_and_exact_accuracy(self):
        point = {"rows": [
            {"id": "a", "correct": True, "parser_valid": True, "score_0_1_2": 2},
            {"id": "b", "correct": False, "parser_valid": True, "score_0_1_2": 1},
        ], "correct": 1, "accuracy": 0.5, "failure_fraction": 0.0,
            "mean_score_0_1_2": 1.5, "total_tokens": 200, "calls": 4,
            "failed_calls": 0, "completion_truncated": 0, "tokens_per_example": 100.0}
        _check_rows(point, ["a", "b"], 2, "unit")
        point["rows"][1]["score_0_1_2"] = 0
        with self.assertRaisesRegex(ValueError, "0/1/2"):
            _check_rows(point, ["a", "b"], 2, "unit")

    def test_seed_aggregation_reports_sample_standard_deviation(self):
        rows = []
        for dataset, axes, splits in (("aime", ("-",), ("aime_2025", "aime_2026")),
                                      ("masbench", ("breadth", "depth", "horizon", "parallel", "robustness"), ("test",))):
            for axis in axes:
                for split in splits:
                    for point in ("Q", "E"):
                        for seed, accuracy in enumerate((0.4, 0.5, 0.6)):
                            rows.append({"dataset": dataset, "axis": axis, "test_split": split,
                                "operating_point": point, "seed": seed, "accuracy": accuracy,
                                "tokens_per_example": 10.0 + seed, "calls": 2, "failure_fraction": 0.0})
        report = summarize(rows, "adas")
        self.assertEqual(report["verified_runs"], 18)
        self.assertAlmostEqual(report["tables"][0]["accuracy_mean"], 0.5)
        self.assertAlmostEqual(report["tables"][0]["accuracy_sample_std"], 0.1)


if __name__ == "__main__":
    unittest.main()
