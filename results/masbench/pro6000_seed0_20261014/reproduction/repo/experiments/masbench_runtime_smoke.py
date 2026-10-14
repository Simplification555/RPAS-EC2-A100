#!/usr/bin/env python3
"""Cheap integration checks for runtime monkey-patches; no model/GPU calls."""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

from native_masbench_external import patch_masbench_graph_namespace, patch_native_math_scorer


def check_native_scorer_adapter() -> None:
    module_name = "_rpas_masbench_scoring_smoke"
    module = types.ModuleType(module_name)

    class MATHBenchmark:
        pass

    module.MATHBenchmark = MATHBenchmark
    sys.modules[module_name] = module
    try:
        patch_native_math_scorer(module_name)
        benchmark = object.__new__(MATHBenchmark)
        expected = "FINAL ANSWER: 2<<horizon>>17"
        observed = [
            benchmark.calculate_score(expected, "FINAL ANSWER: 2<<horizon>>17")[0],
            benchmark.calculate_score(expected, "FINAL ANSWER: 2<<horizon>>9")[0],
            benchmark.calculate_score(expected, "FINAL ANSWER: 8<<horizon>>9")[0],
        ]
        if observed != [2, 1, 0]:
            raise AssertionError(f"native search scorer returned {observed}, expected [2, 1, 0]")
    finally:
        sys.modules.pop(module_name, None)


def check_graph_namespace_isolation() -> None:
    class FakeGraphUtils:
        def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str) -> None:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            (root / "graph.py").write_text("import workspace.rpas_aime_native.MATH\n", encoding="utf-8")
            (root / "prompt.py").write_text("import workspace.rpas_aime_native.MATH\n", encoding="utf-8")

    patch_masbench_graph_namespace(FakeGraphUtils)
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary)
        FakeGraphUtils().write_graph_files(str(target), {}, 1, "MATH")
        for name in ("graph.py", "prompt.py"):
            content = (target / name).read_text(encoding="utf-8")
            if "workspace.rpas_masbench_native.MATH" not in content:
                raise AssertionError(f"{name} leaked the AIME namespace: {content!r}")
            if "workspace.rpas_aime_native.MATH" in content:
                raise AssertionError(f"{name} retained the AIME namespace")


if __name__ == "__main__":
    check_native_scorer_adapter()
    check_graph_namespace_isolation()
    print("MASBENCH_RUNTIME_SMOKE_PASS")
