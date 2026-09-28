from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_aime_formal as formal
import native_external_methods as external


class DTestBlindFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        validation = [
            {"id": f"v:{i}", "problem": f"validation problem {i}", "answer": str(i)}
            for i in range(90)
        ]
        (self.root / "aimo-validation-aime.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in validation), encoding="utf-8"
        )
        for year in (2025, 2026):
            rows = [
                {"id": f"{year}:{i}", "problem": f"AIME {year} problem {i}", "answer": str(i)}
                for i in range(30)
            ]
            (self.root / f"aime_{year}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

    def tearDown(self):
        self.temp.cleanup()

    def test_test_loader_fails_before_reading_any_test_file(self):
        target = self.root / "aime_2025.jsonl"
        original = Path.read_bytes
        reads = []

        def tracked(path):
            if path == target:
                reads.append(str(path))
            return original(path)

        with patch.object(Path, "read_bytes", tracked):
            with self.assertRaisesRegex(RuntimeError, "before selection_frozen"):
                formal.load_aime_test_rows_after_freeze(
                    self.root, self.run_dir, "aflow", target.name
                )
        self.assertEqual(reads, [])
        self.assertFalse((self.run_dir / "dtest_access_manifest.json").exists())

    def test_external_aime_partition_loader_reads_validation_only(self):
        original = formal.read_jsonl
        opened = []

        def tracked(path):
            opened.append(Path(path).name)
            return original(path)

        with patch.object(formal, "read_jsonl", tracked):
            rows = external.read_task_rows("aime", self.root, seed=0, axis=None)
        self.assertEqual(set(rows), {"search", "select"})
        self.assertEqual(opened, ["aimo-validation-aime.jsonl"])

    def test_selection_lock_is_exclusive_and_precedes_audited_test_open(self):
        search, select = formal.freeze_split(self.root, 2026, 60, 30)
        search = formal.namespace_aime_rows(search, "validation")
        select = formal.namespace_aime_rows(select, "validation")
        lock = formal.freeze_aime_selection(
            self.run_dir, "aflow", search, select,
            {"Q": {"round": 3}, "E": {"round": 2}},
        )
        self.assertFalse(lock["dtest_loaded_before_lock"])
        with self.assertRaises(FileExistsError):
            formal.freeze_aime_selection(self.run_dir, "aflow", search, select, {})
        rows = formal.load_aime_test_rows_after_freeze(
            self.root, self.run_dir, "aflow", "aime_2025.jsonl"
        )
        self.assertEqual(len(rows), 30)
        audit = json.loads((self.run_dir / "dtest_access_manifest.json").read_text())
        self.assertGreaterEqual(
            audit["test_splits"]["aime_2025.jsonl"]["opened_at_epoch"],
            audit["selection_frozen_at_epoch"],
        )

    def test_loader_rejects_test_question_overlapping_validation_after_freeze(self):
        search, select = formal.freeze_split(self.root, 2026, 60, 30)
        search = formal.namespace_aime_rows(search, "validation")
        select = formal.namespace_aime_rows(select, "validation")
        formal.freeze_aime_selection(self.run_dir, "maas", search, select, {"controller": "abc"})
        rows = [json.loads(line) for line in (self.root / "aime_2025.jsonl").read_text().splitlines()]
        rows[0]["problem"] = search[0]["problem"]
        (self.root / "aime_2025.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "overlapping D_search/D_select"):
            formal.load_aime_test_rows_after_freeze(
                self.root, self.run_dir, "maas", "aime_2025.jsonl"
            )
        self.assertFalse((self.run_dir / "dtest_access_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
