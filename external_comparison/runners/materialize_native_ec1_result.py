"""Materialize a completed native EC-1 driver artifact without re-execution.

The native upstream drivers may complete a held-out evaluator in a separate
process (notably an AFlow test-only recovery).  This tool turns that immutable
driver result plus its telemetry into the common harness artifacts.  It never
imports an upstream optimizer or calls an LLM endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from external_comparison.adapters.native_common import call_record, write_native_result


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a completed native EC-1 driver result")
    parser.add_argument("--method", choices=("aflow", "maas"), required=True)
    parser.add_argument("--driver-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-kind", choices=("pilot", "formal"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--gpu", choices=("4", "5"), required=True)
    args = parser.parse_args()

    result = json.loads(Path(args.driver_result).read_text(encoding="utf-8"))
    rows = list(result.get("test_rows", []))
    if len(rows) != 131 or len({str(row.get("task_id", "")) for row in rows}) != 131:
        raise ValueError("native EC-1 materialization requires exactly 131 unique held-out rows")
    telemetry_path = Path(result["telemetry_path"])
    raw_calls = _read_jsonl(telemetry_path)
    if not raw_calls or not any(row.get("phase") == "test" for row in raw_calls):
        raise ValueError("native EC-1 materialization requires preserved held-out telemetry")
    calls = [
        call_record(
            f"humaneval-{args.method}-seed-{args.seed}", args.method, "humaneval", str(row["phase"]),
            f"{row['phase']}-{index}", index, row,
        )
        for index, row in enumerate(raw_calls)
    ]
    manifest = {
        "run_id": f"humaneval-{args.method}-seed-{args.seed}",
        "method": args.method,
        "dataset": "humaneval",
        "seed": args.seed,
        "formal_result": args.run_kind == "formal",
        "run_kind": args.run_kind,
        "model": args.model,
        "api_base": args.api_base,
        "gpu": args.gpu,
        **result["manifest"],
        "search_calls": sum(row.get("phase") == "search" for row in raw_calls),
        "search_tokens": sum(int(row.get("total_tokens", 0)) for row in raw_calls if row.get("phase") == "search"),
    }
    selected = {"round": result["manifest"].get("selected_round")} if args.method == "aflow" else {"checkpoint": result["manifest"].get("checkpoint")}
    write_native_result(Path(args.output_dir), manifest, rows, calls, selected=selected, search_rows=result.get("search_rows", []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
