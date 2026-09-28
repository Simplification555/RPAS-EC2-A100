from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_aime_formal as formal
import native_external_methods as external


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class DTestBlindFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        validation = [
            {"id": i, "problem": f"validation problem {i}", "answer": str(i)}
            for i in range(90)
        ]
        source_hash = _write_jsonl(self.root / "aimo-validation-aime.jsonl", validation)
        # Deliberately choose a canonical split unlike random.Random(2026), so
        # this test catches accidental regeneration of D_search/D_select.
        search = [{**row, "id": f"canonical-search-{row['id']}"} for row in validation[30:]]
        select = [{**row, "id": f"canonical-select-{row['id']}"} for row in validation[:30]]
        search_hash = _write_jsonl(self.root / "aimo_validation" / "search_60.jsonl", search)
        select_hash = _write_jsonl(self.root / "aimo_validation" / "select_30.jsonl", select)
        self.test_rows = {}
        test_specs = {}
        for year in (2025, 2026):
            raw_rows = [
                {"id": i, "problem": f"AIME {year} problem {i}", "answer": str(i)}
                for i in range(30)
            ]
            raw_hash = _write_jsonl(self.root / f"aime_{year}.jsonl", raw_rows)
            canonical_rows = [
                {"id": f"aime:aime{year}:{i}", "problem": row["problem"], "answer": row["answer"]}
                for i, row in enumerate(raw_rows)
            ]
            split_path = f"aime{year}/test_30.jsonl"
            split_hash = _write_jsonl(self.root / split_path, canonical_rows)
            test_specs[f"aime_{year}.jsonl"] = {
                "source_sha256_git_content": raw_hash,
                "source_rows": 30,
                "split_path": split_path,
                "split_sha256_git_content": split_hash,
                "rows": 30,
            }
            self.test_rows[year] = raw_rows
        manifest = {
            "schema": "rpas_frozen_aime_external_v1",
            "source_repository": "JiangyueAnn/RPAS",
            "source_revision": "e12f58823be5f91a32f05f9af4d36e54838ffe59",
            "data_seed": 2026,
            "validation": {
                "source": {"path": "aimo-validation-aime.jsonl", "sha256_git_content": source_hash, "rows": 90},
                "search": {"path": "aimo_validation/search_60.jsonl", "sha256_git_content": search_hash, "rows": 60},
                "select": {"path": "aimo_validation/select_30.jsonl", "sha256_git_content": select_hash, "rows": 30},
            },
            "test": test_specs,
        }
        self.manifest_path = self.root / "frozen_aime_manifest.json"
        self._write_manifest(manifest)
        self.manifest_hash_patch = patch.object(
            formal, "FROZEN_AIME_MANIFEST_SHA256", formal.sha256_file(self.manifest_path)
        )
        self.manifest_hash_patch.start()

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def tearDown(self):
        self.manifest_hash_patch.stop()
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

    def test_loader_uses_pinned_canonical_search_select_files(self):
        search, select = formal.freeze_split(self.root, 2026, 60, 30)
        self.assertEqual([row["id"] for row in search], [f"canonical-search-{i}" for i in range(30, 90)])
        self.assertEqual([row["id"] for row in select], [f"canonical-select-{i}" for i in range(30)])
        self.assertEqual(
            (self.root / "frozen_aime_manifest.json").is_file(), True
        )

    def test_external_aime_partition_loader_reads_only_canonical_validation_files(self):
        original = formal.read_verified_jsonl
        opened = []

        def tracked(data_dir, relative_path, expected_sha256, expected_rows):
            opened.append(relative_path)
            return original(data_dir, relative_path, expected_sha256, expected_rows)

        with patch.object(formal, "read_verified_jsonl", tracked):
            rows = external.read_task_rows("aime", self.root, seed=0, axis=None)
        self.assertEqual(set(rows), {"search", "select"})
        self.assertEqual(opened, [
            "aimo-validation-aime.jsonl",
            "aimo_validation/search_60.jsonl",
            "aimo_validation/select_30.jsonl",
        ])

    def test_selection_lock_is_exclusive_and_precedes_audited_test_open(self):
        search, select = formal.freeze_split(self.root, 2026, 60, 30)
        search = formal.namespace_aime_rows(search, "validation")
        select = formal.namespace_aime_rows(select, "validation")
        lock = formal.freeze_aime_selection(
            self.run_dir, "aflow", search, select,
            {"Q": {"round": 3}, "E": {"round": 2}},
        )
        self.assertFalse(lock["dtest_loaded_before_lock"])
        self.assertEqual(lock["frozen_data_manifest_sha256"], formal.FROZEN_AIME_MANIFEST_SHA256)
        with self.assertRaises(FileExistsError):
            formal.freeze_aime_selection(self.run_dir, "aflow", search, select, {})
        rows = formal.load_aime_test_rows_after_freeze(
            self.root, self.run_dir, "aflow", "aime_2025.jsonl"
        )
        self.assertEqual(len(rows), 30)
        audit = json.loads((self.run_dir / "dtest_access_manifest.json").read_text())
        record = audit["test_splits"]["aime_2025.jsonl"]
        self.assertGreaterEqual(record["opened_at_epoch"], audit["selection_frozen_at_epoch"])
        self.assertEqual(record["frozen_split_path"], "aime2025/test_30.jsonl")

    def test_external_method_lock_uses_external_protocol_version(self):
        other_run = self.root / "external_run"
        other_run.mkdir()
        search, select = formal.freeze_split(self.root, 2026, 60, 30)
        search = formal.namespace_aime_rows(search, "validation")
        select = formal.namespace_aime_rows(select, "validation")
        lock = formal.freeze_aime_selection(other_run, "adas", search, select, {"Q": {"id": "frozen"}})
        self.assertEqual(lock["protocol_version"], formal.EXTERNAL_AIME_PROTOCOL_VERSION)
        loaded = formal.load_aime_test_rows_after_freeze(
            self.root, other_run, "adas", "aime_2026.jsonl"
        )
        self.assertEqual(len(loaded), 30)

    def test_loader_rejects_source_that_differs_from_pinned_test_split(self):
        search, select = formal.freeze_split(self.root, 2026, 60, 30)
        search = formal.namespace_aime_rows(search, "validation")
        select = formal.namespace_aime_rows(select, "validation")
        formal.freeze_aime_selection(self.run_dir, "maas", search, select, {"controller": "abc"})
        source = self.root / "aime_2025.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows[0]["problem"] = search[0]["problem"]
        _write_jsonl(source, rows)
        with self.assertRaisesRegex(ValueError, "exact frozen AIME source"):
            formal.load_aime_test_rows_after_freeze(
                self.root, self.run_dir, "maas", "aime_2025.jsonl"
            )
        self.assertFalse((self.run_dir / "dtest_access_manifest.json").exists())

    def test_loader_rejects_manifest_or_split_corruption(self):
        search_path = self.root / "aimo_validation" / "search_60.jsonl"
        search_path.write_text(search_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "data hash mismatch"):
            formal.freeze_split(self.root, 2026, 60, 30)
        with self.assertRaisesRegex(ValueError, "requires data_seed=2026"):
            formal.freeze_split(self.root, 0, 60, 30)


if __name__ == "__main__":
    unittest.main()
