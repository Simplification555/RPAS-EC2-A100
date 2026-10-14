from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aggregate_masbench_seed0 import AXES, aggregate_seed0, render_csv


class Seed0AggregationTests(unittest.TestCase):
    def test_aggregates_exactly_two_methods_five_axes_and_no_seed_sd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in ("aflow", "maas"):
                for axis_index, axis in enumerate(AXES):
                    run_dir = root / method / axis / "seed_0"
                    run_dir.mkdir(parents=True)
                    (run_dir / "quality_gate.json").write_text(json.dumps({
                        "passed": True, "method": method, "axis": axis, "seed": 0,
                        "run_id": f"{method}:{axis}:0",
                    }))
                    points = {point: {} for point in ("Q", "E")}
                    result = {"method": method, "test": points if method == "aflow" else {axis: points}}
                    (run_dir / "native_result.json").write_text(json.dumps(result))

            def fake_validate(run_dir: Path) -> dict:
                axis = run_dir.parent.name
                value = AXES.index(axis) + 1
                return {
                    "passed": True,
                    "run_id": run_dir.name + ":" + axis,
                    "point_summary": {
                        point: {
                            "accuracy": value / 10,
                            "tokens_per_example": value * 100,
                            "component_accuracy": value / 10,
                            "mean_score_0_1_2": value / 5,
                        }
                        for point in ("Q", "E")
                    },
                }

            with patch("aggregate_masbench_seed0.validate_run", side_effect=fake_validate):
                summary, rows = aggregate_seed0(root)
            self.assertEqual(summary["axes"], list(AXES))
            self.assertEqual(len(summary["run_ids"]), 10)
            self.assertEqual(len([r for r in rows if "metric" not in r]), 20)
            self.assertAlmostEqual(summary["methods"]["aflow"]["macro_avg"]["Q"]["accuracy"]["mean_across_five_axes"], 0.3)
            self.assertIn("no across-seed standard deviation", summary["uncertainty_note"])

    def test_missing_gate_fails_instead_of_partial_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                aggregate_seed0(Path(temporary))

    def test_csv_rendering_is_deterministic_and_has_fixed_header(self):
        rows = [
            {"method": "aflow", "axis": "breadth", "seed": 0, "point": "Q", "accuracy": 0.5},
            {"method": "maas", "axis": "breadth", "seed": 0, "point": "E", "accuracy": 0.4},
        ]
        first = render_csv(rows)
        self.assertEqual(first, render_csv(rows))
        self.assertTrue(first.startswith("method,axis,seed,point,accuracy,tokens_per_example"))
        self.assertEqual(len(first.splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
