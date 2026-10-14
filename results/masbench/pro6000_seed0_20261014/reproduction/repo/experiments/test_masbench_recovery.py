import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from masbench_recovery import answer_width, canonical_output, patch_output_repair, resume_search
from masbench_quality_gate import BASELINE_COMMITS, sha256_file, validate_run
import test_masbench_quality_gate as gate_tests


class RecoveryTests(unittest.TestCase):
    def test_explicit_formats_and_width(self):
        self.assertEqual(canonical_output("reasoning\n\\boxed{7}", 1), "FINAL ANSWER: 7")
        self.assertEqual(canonical_output("3\\n\\n5", 2), "FINAL ANSWER: 3<<horizon>>5")
        self.assertEqual(canonical_output("FINAL ANSWER: 3<<horizon>>5", 2), "FINAL ANSWER: 3<<horizon>>5")
        self.assertEqual(canonical_output("### Final Answers\nProblem 1: \\boxed{3}\nProblem 2: \\boxed{5}", 2), "FINAL ANSWER: 3<<horizon>>5")
        for raw in ("reasoning mentions 7 and 8", "### Final Answers\nProblem 2: \\boxed{3}\nProblem 1: \\boxed{5}", "FINAL ANSWER: 3"):
            self.assertEqual(canonical_output(raw, 2), "")
        problem = "### Final Answers\nProblem 1: \\boxed{?}\nProblem 2: \\boxed{?}"
        self.assertEqual(answer_width(problem), 2)
        self.assertEqual(answer_width("number of apples in distractor. From the passage, return the number of cats and the magic number for dogs"), 2)

    def test_one_gold_blind_request_and_audited_failure(self):
        class Benchmark:
            async def _generate_output(self, graph, input_text):
                return ("unfinished analysis", 12)
        create = AsyncMock(return_value=SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4, total_tokens=24, model_dump=lambda: {"total_tokens": 24}),
            choices=[SimpleNamespace(message=SimpleNamespace(content="FINAL ANSWER: 7"), finish_reason="stop")]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock())
        with patch("openai.AsyncOpenAI", return_value=client), patch("native_aime_formal.begin_call", return_value="select"), patch("native_aime_formal.record_usage") as usage, patch("native_aime_formal.failed_call") as failed, patch("masbench_recovery.append_repair") as audit:
            patch_output_repair(Benchmark, "http://local/v1", {}, "aflow")
            self.assertEqual(asyncio.run(Benchmark()._generate_output(None, "Solve the original question")), ("FINAL ANSWER: 7", 12))
            create.assert_awaited_once()
            self.assertNotIn("gold", json.dumps(create.call_args.kwargs).lower())
            usage.assert_called_once_with({}, "select", 20, 4, 24)
            self.assertTrue(audit.call_args.args[0]["repaired_valid"])
            create.reset_mock()
            create.side_effect = RuntimeError("offline")
            self.assertEqual(asyncio.run(Benchmark()._generate_output(None, "question")), ("unfinished analysis", 12))
            create.assert_awaited_once()
            failed.assert_called_once()
            self.assertFalse(audit.call_args.args[0]["repaired_valid"])

    def test_relaxed_gate_retains_quality_failure_but_rejects_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = gate_tests.MasBenchQualityGateTests().make_fixture(Path(directory))
            rp = run / "native_result.json"
            result = json.loads(rp.read_text())
            result["test"]["Q"]["protocol_error_rate"] = .1
            rp.write_text(json.dumps(result))
            mp = run / "run_manifest.json"
            manifest = json.loads(mp.read_text())
            manifest.update(completion_policy="relaxed", native_result_sha256=sha256_file(rp))
            mp.write_text(json.dumps(manifest))
            with patch("masbench_quality_gate.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=BASELINE_COMMITS["aflow"])):
                report = validate_run(run)
                self.assertFalse(report["passed"])
                self.assertTrue(report["accepted_for_completion"], report["errors"])
                self.assertTrue(report["quality_warnings"])
                result["test"]["Q"]["num_examples"] = 59
                rp.write_text(json.dumps(result))
                manifest["native_result_sha256"] = sha256_file(rp)
                mp.write_text(json.dumps(manifest))
                self.assertFalse(validate_run(run)["accepted_for_completion"])

    def test_search_resume_keeps_cost_and_exposure_without_reusing_test_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            relative = Path("workspace/rpas_masbench_native/MATH")
            artifact = source / "aflow_repo" / relative / "workflows/round_1"
            artifact.mkdir(parents=True)
            (artifact / "graph.py").write_text("import workspace.MATH.workflows.round_1.prompt")
            manifest = dict(method="aflow", axis="depth", seed=0, data_seed=2026, model="model",
                            context_limit=16384, output_limit=6144, data_sha256={"search": "frozen"},
                            baseline_git_commit="commit")
            (source / "run_manifest.json").write_text(json.dumps({**manifest, "d_test_opened": True}))
            (source / "native_result.json").write_text('{"test":"must not copy"}')
            (source / "request_ledger.jsonl").write_text("\n".join(json.dumps(x) for x in [
                {"phase": "search", "status": "completed", "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
                {"phase": "select", "status": "completed", "usage": {"total_tokens": 100}},
                {"phase": "test:depth", "status": "completed", "usage": {"total_tokens": 200}},
            ]) + "\n")
            args = SimpleNamespace(resume_search_from=source, run_dir=root / "new", method="aflow")
            with patch("native_aime_formal.merge_phase_usage") as merge:
                self.assertTrue(resume_search(args, relative, {}, manifest))
                self.assertEqual(merge.call_args.args[2]["total_tokens"], 30)
            self.assertTrue(manifest["search_resume"]["source_d_test_opened"])
            self.assertFalse(manifest["search_resume"]["selection_and_test_scores_reused"])
            self.assertFalse((args.run_dir / "native_result.json").exists())
            self.assertIn("workspace.rpas_masbench_native.MATH", (args.run_dir / "aflow_repo" / relative / "workflows/round_1/graph.py").read_text())
            manifest["axis"] = "breadth"
            with self.assertRaisesRegex(ValueError, "axis"):
                resume_search(args, relative, {}, manifest)


if __name__ == "__main__":
    unittest.main()
