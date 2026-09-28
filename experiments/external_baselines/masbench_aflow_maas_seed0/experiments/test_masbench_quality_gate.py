import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from masbench_quality_gate import BASELINE_COMMITS, BASELINE_REQUIRED, sha256_file, validate_run


class MasBenchQualityGateTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        repo = root / "repo"
        exp = repo / "experiments"
        exp.mkdir(parents=True)
        (exp / "native_masbench_external.py").write_text("runner", encoding="utf-8")
        (exp / "masbench_answer_protocol.py").write_text("parser", encoding="utf-8")
        baseline_root = repo / "baseline"
        baseline_hashes = {}
        for relative in BASELINE_REQUIRED["aflow"]:
            path = baseline_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
            baseline_hashes[relative] = sha256_file(path)
        data_root = repo / "data" / "masbench"
        data_root.mkdir(parents=True)
        safe_manifest = data_root / "manifest_search_select.json"
        safe_manifest.write_text('{"schema_version":"rpas_ec1_masbench_aime_v1"}', encoding="utf-8")
        full_manifest = data_root / "breadth" / "manifest.json"
        full_manifest.parent.mkdir(parents=True)
        full_manifest.write_text('{"test_metadata":"opened only after lock"}', encoding="utf-8")
        test_file = data_root / "breadth" / "test_60.jsonl"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text('{"id":"test-row"}\n', encoding="utf-8")
        run_dir = root / "outputs" / "aflow" / "breadth" / "seed_0"
        run_dir.mkdir(parents=True)
        point = {
            "num_examples": 60,
            "correct": 30,
            "accuracy": 0.5,
            "avg_cost": 100.0,
            "total_tokens": 6000,
            "calls": 60,
            "resource_usage": {"calls": 60, "total_tokens": 6000},
            "output_audit": {"csv_rows": 60, "failure_fraction": 0.0},
            "valid_answer_rate": 1.0,
            "component_accuracy": 0.5,
            "mean_score_0_1_2": 1.0,
        }
        controls = {
            "model": "Qwen/Qwen3.5-9B", "temperature": 0.0, "top_p": 1.0,
            "thinking": False, "max_tokens": 6144, "max_model_len": 8192,
            "max_num_seqs": 24, "tensor_parallel_size": 1, "concurrency": 8,
            "data_seed": 2026, "search_seed": 0, "search_size": 24,
            "selection_size": 24, "test_size": 60,
            "selection_split": "D_select", "test_split": "D_test",
        }
        result = {
            "run_id": "test-run", "method": "aflow", "controls": controls,
            "native_method_config": {
                "dataset": "MATH", "question_type": "math",
                "operators": ["Custom", "ScEnsemble", "Programmer"],
                "sample": 4, "check_convergence": False, "initial_round": 1,
                "max_rounds": 8, "validation_rounds": 1,
                "mutation_generation_retry_cap": 8,
            },
            "telemetry": {"context_guard": {"context_limit": 8192, "calls_seen": 100}},
            "selection_rows": [{"status": "evaluated", "round": 1,
                                "output_audit": {"failure_fraction": 0.0}}],
            "selected_round": {"round": 1}, "selected_efficiency_round": {"round": 1},
            "test": {"Q": point, "E": dict(point)},
        }
        result["test"]["Q"]["selected_round"] = 1
        result["test"]["E"]["selected_round"] = 1
        selection_lock = {
            "status": "frozen_before_test_access", "method": "aflow", "axis": "breadth", "seed": 0,
            "d_test_opened": False, "frozen_at_utc": "2026-01-01T00:00:00+00:00",
        }
        selection_path = run_dir / "selection_frozen.json"
        selection_path.write_text(json.dumps(selection_lock), encoding="utf-8")
        selection_hash = sha256_file(selection_path)
        result["selection_lock"] = {"path": selection_path.name, "sha256": selection_hash}
        result_path = run_dir / "native_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        manifest = {
            "protocol_version": "masbench_external_protocol_v1", "status": "complete",
            "method": "aflow", "axis": "breadth", "seed": 0, "run_id": "test-run",
            "pairwise_disjoint": True, "split_counts": {"search": 24, "select": 24, "test": 60},
            "native_result_sha256": sha256_file(result_path), "rpas_root": str(repo),
            "data_sha256": {
                "manifest_search_select.json": sha256_file(safe_manifest),
                "breadth/manifest.json": sha256_file(full_manifest),
                "breadth/test_60.jsonl": sha256_file(test_file),
            },
            "d_test_opened": True,
            "selection_lock_sha256": selection_hash,
            "d_test_access": {
                "opened_at_utc": "2026-01-01T00:00:01+00:00",
                "selection_lock_sha256": selection_hash,
                "test_file_sha256": sha256_file(test_file),
                "examples": 60,
            },
            "official_baseline_source": str(baseline_root),
            "baseline_source_sha256": baseline_hashes,
            "runner_sha256": sha256_file(exp / "native_masbench_external.py"),
            "answer_parser_sha256": sha256_file(exp / "masbench_answer_protocol.py"),
            "baseline_git_commit": BASELINE_COMMITS["aflow"],
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir

    def test_complete_consistent_run_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("masbench_quality_gate.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=BASELINE_COMMITS["aflow"] + "\n"
            )):
                report = validate_run(self.make_fixture(Path(directory)))
        self.assertTrue(report["passed"], report["errors"])

    def test_failure_fraction_above_five_percent_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_fixture(Path(directory))
            result_path = run_dir / "native_result.json"
            result = json.loads(result_path.read_text())
            result["test"]["Q"]["output_audit"]["failure_fraction"] = 0.1
            result_path.write_text(json.dumps(result), encoding="utf-8")
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["native_result_sha256"] = sha256_file(result_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("masbench_quality_gate.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=BASELINE_COMMITS["aflow"] + "\n"
            )):
                report = validate_run(run_dir)
        self.assertFalse(report["passed"])
        self.assertTrue(any("failure fraction" in item for item in report["errors"]))

    def test_maas_native_controller_and_24_24_60_protocol_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_fixture(Path(directory))
            result_path = run_dir / "native_result.json"
            result = json.loads(result_path.read_text())
            result["method"] = "maas"
            result["native_method_config"] = {
                "dataset": "MATH", "question_type": "math", "operators": ["native-controller"],
                "controller_sample": 1, "round": 1, "batch_size": 8,
                "learning_rate": 0.01, "is_textgrad": False, "training_repetitions": 4,
            }
            result["test"] = {"breadth": result["test"]}
            result["selection"] = {"num_examples": 24,
                                    "output_audit": {"csv_rows": 24, "failure_fraction": 0.0}}
            checkpoint = run_dir / "controller.ckpt"
            checkpoint.write_bytes(b"test controller")
            result["controller_checkpoint"] = str(checkpoint)
            result["controller_checkpoint_sha256"] = sha256_file(checkpoint)
            lock_path = run_dir / "selection_frozen.json"
            lock = json.loads(lock_path.read_text())
            lock["method"] = "maas"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            lock_hash = sha256_file(lock_path)
            result["selection_lock"]["sha256"] = lock_hash
            result_path.write_text(json.dumps(result), encoding="utf-8")

            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            baseline_root = Path(manifest["official_baseline_source"])
            maas_hashes = {}
            for relative in BASELINE_REQUIRED["maas"]:
                path = baseline_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
                maas_hashes[relative] = sha256_file(path)
            manifest.update({
                "method": "maas", "selection_lock_sha256": lock_hash,
                "native_result_sha256": sha256_file(result_path),
                "baseline_source_sha256": maas_hashes,
                "baseline_git_commit": BASELINE_COMMITS["maas"],
            })
            manifest["d_test_access"]["selection_lock_sha256"] = lock_hash
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("masbench_quality_gate.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=BASELINE_COMMITS["maas"] + "\n"
            )):
                report = validate_run(run_dir)
        self.assertTrue(report["passed"], report["errors"])


if __name__ == "__main__":
    unittest.main()
