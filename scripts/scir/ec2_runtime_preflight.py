"""Exercise pinned native GCN training and artifact output with simulated LLM calls."""

import argparse
import asyncio
import contextlib
import io
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from external_comparison.runners.ec2_v2 import OfficialGDesignerRuntime, _run_gdesigner
from external_comparison.runners.mmlu import MMLUExample


async def check(root: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="ec2_runtime_preflight_") as temporary:
        destination = Path(temporary)
        runtime = OfficialGDesignerRuntime(root, seed=0)
        runtime.configure_artifacts(destination / "gdesigner" / "seed_0")
        requests = []
        async def simulated_completion(**kwargs):
            assert kwargs["max_tokens"] == 256
            assert kwargs["temperature"] == 0.0
            requests.append((runtime.example_id.get(), json.dumps(kwargs, sort_keys=True)))
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
                                   choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="A"))])
        runtime.client.chat.completions.create = simulated_completion
        rows = {
            "search": [MMLUExample(f"synthetic-dev:{i}", "s", f"Select A. Synthetic case {i}.", ("yes", "no", "no", "no"), "A") for i in range(8)],
            "select": [MMLUExample("synthetic-val:0", "s", "Select A.", ("yes", "no", "no", "no"), "A")],
            "test": [MMLUExample("synthetic-test:0", "s", "Select A.", ("yes", "no", "no", "no"), "A")],
            "manifest": {"split_manifest_sha256": "synthetic-preflight", "search": {"count": 8}, "select": {"count": 1}, "test": {"count": 1}},
            "output_dir": str(destination),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            await _run_gdesigner(runtime, rows, 0)
        result_dir = destination / "gdesigner" / "seed_0"
        result = json.loads((result_dir / "result.json").read_text())
        calls = [json.loads(line) for line in (result_dir / "calls.jsonl").read_text().splitlines()]
        training = [call for call in calls if call["split"] == "search"]
        assert {call["training_iteration"] for call in training} == set(range(10))
        assert all(call["example_id"].startswith("synthetic-dev:") for call in training)
        assert all((result_dir / name).stat().st_size > 100 for name in result["checkpoint_files"])
        assert len(result["checkpoint_files"]) == 2
        fixed_rows = rows["search"][:4]
        fixed_graph = runtime.make_graph(topology="chain", optimized_spatial=False)
        requests.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            serial_rows, serial_communication = await runtime.evaluate(
                fixed_graph, fixed_rows, split="test", candidate_id="fixed_preflight", concurrency=1)
        serial_requests = sorted(requests)
        requests.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            parallel_rows, parallel_communication = await runtime.evaluate(
                fixed_graph, fixed_rows, split="test", candidate_id="fixed_preflight", concurrency=4)
        assert sorted(requests) == serial_requests
        assert len(requests) == 28
        assert parallel_communication == serial_communication
        assert [{k: v for k, v in row.items() if k != "latency_ms"} for row in parallel_rows] == [
            {k: v for k, v in row.items() if k != "latency_ms"} for row in serial_rows]
        return {"formal_result": False, "simulated_llm": True, "native_training_iterations": 10,
                "checkpoint_files": list(result["checkpoint_files"]), "training_calls": len(training),
                "fixed_serial_vs_parallel_requests_equal": True, "fixed_fixture_calls": len(requests),
                "status": "PASS_RUNTIME_ONLY"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdesigner-root", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, help="Validate staged code before deploying to pending jobs")
    args = parser.parse_args()
    if args.runtime_source:
        spec = importlib.util.spec_from_file_location("staged_ec2_runtime", args.runtime_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        OfficialGDesignerRuntime, _run_gdesigner = module.OfficialGDesignerRuntime, module._run_gdesigner
    print(json.dumps(asyncio.run(check(args.gdesigner_root))))
