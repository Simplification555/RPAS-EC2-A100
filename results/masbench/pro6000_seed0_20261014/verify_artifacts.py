"""Verify lossless exports and recompute all 20 test operating points offline."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
COMPLETE = ROOT / "outputs/aflow_maas_completion_20261014"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def main() -> None:
    inventory = json.loads((ROOT / "artifact_manifest.json").read_text())
    records = inventory["files"]
    by_source = {r["source_relative_path"]: r for r in records}
    require(len(by_source) == len(records), "duplicate source paths")
    require(len({r["export_relative_path"] for r in records}) == len(records), "duplicate export paths")
    for record in records:
        file = ROOT / record["export_relative_path"]
        require(file.resolve().is_relative_to(ROOT), "export path escapes bundle")
        encoded = file.read_bytes()
        require(len(encoded) == record["export_bytes"], f"export size mismatch: {file}")
        require(hashlib.sha256(encoded).hexdigest() == record["export_sha256"], f"export hash mismatch: {file}")
        payload = gzip.decompress(encoded) if record["encoding"] == "gzip" else encoded
        require(len(payload) == record["source_bytes"], f"decoded size mismatch: {file}")
        require(hashlib.sha256(payload).hexdigest() == record["source_sha256"], f"decoded hash mismatch: {file}")
    require(len(records) == inventory["file_count"], "inventory count mismatch")
    require(sum(r["source_bytes"] for r in records) == inventory["source_bytes"], "source byte total mismatch")
    require(sum(r["export_bytes"] for r in records) == inventory["export_bytes"], "export byte total mismatch")

    sys.path.insert(0, str(ROOT / "reproduction/repo/experiments"))
    from masbench_answer_protocol import extract_masbench_answer, score_masbench

    def original_csv(path: str) -> str:
        parts = Path(path).parts
        index = parts.index(inventory["source_package"])
        relative = str(Path(*parts[index + 1:]))
        record = by_source[relative]
        file = ROOT / record["export_relative_path"]
        payload = gzip.decompress(file.read_bytes()) if record["encoding"] == "gzip" else file.read_bytes()
        return payload.decode("utf-8-sig")

    summary = json.loads((COMPLETE / "seed0_summary/aggregate.json").read_text())
    batch = json.loads((COMPLETE / "batch_status.json").read_text())
    require(batch["status"] == "completed", "batch did not complete")
    require(len(batch["units"]) == 10, "batch does not contain 10 units")
    require(all(u["status"] == "completed" for u in batch["units"]), "incomplete unit")
    point_checks = []
    integrity_gate_checks = 0
    strict_quality_passes = 0
    for method in ("aflow", "maas"):
        for axis in AXES:
            run = COMPLETE / method / axis / "seed_0"
            manifest = json.loads((run / "run_manifest.json").read_text())
            result = json.loads((run / "native_result.json").read_text())
            gate = json.loads((run / "quality_gate.json").read_text())
            lock = json.loads((run / "selection_frozen.json").read_text())
            require((manifest["method"], manifest["axis"], manifest["seed"]) == (method, axis, 0), "identity mismatch")
            require(result["run_id"] == manifest["run_id"], "run ID mismatch")
            require(manifest["status"] == "complete" and manifest["d_test_opened"], "run/test incomplete")
            require(manifest["split_counts"] == {"search": 24, "select": 24, "test": 60}, "split size mismatch")
            require(hashlib.sha256((run / "native_result.json").read_bytes()).hexdigest() == manifest["native_result_sha256"], "native result hash mismatch")
            require(hashlib.sha256((run / "selection_frozen.json").read_bytes()).hexdigest() == manifest["selection_lock_sha256"], "selection lock hash mismatch")
            require(lock["d_test_opened"] is False and lock["status"] == "frozen_before_test_access", "invalid freeze lock")
            require(gate["accepted_for_completion"] and not gate["integrity_errors"], "integrity gate failed")
            integrity_gate_checks += 1
            strict_quality_passes += int(gate["passed"])
            for filename, key in (("native_masbench_external.py", "runner_sha256"), ("native_aime_formal.py", "base_runner_sha256"),
                                  ("masbench_recovery.py", "recovery_sha256"), ("shared_scoring.py", "shared_scoring_sha256"),
                                  ("alignment_runtime.py", "alignment_runtime_sha256"), ("masbench_answer_protocol.py", "answer_parser_sha256")):
                require(hashlib.sha256((ROOT / "reproduction/repo/experiments" / filename).read_bytes()).hexdigest() == manifest[key], f"snapshot hash mismatch: {filename}")
            points = result["test"] if method == "aflow" else result["test"][axis]
            for label in ("Q", "E"):
                point = points[label]
                outputs = list(csv.DictReader(io.StringIO(original_csv(point["output_audit"]["csv"]))))
                require(len(outputs) == point["num_examples"] == 60, f"wrong CSV size: {method}/{axis}/{label}")
                scores = [score_masbench(o.get("prediction", ""), extract_masbench_answer(o["expected_output"])) for o in outputs]
                correct = sum(s["correct"] for s in scores)
                accuracy = correct / 60
                component = sum(s["component_correct"] for s in scores) / sum(s["component_total"] for s in scores)
                ordinal = sum(s["score_0_1_2"] for s in scores) / 60
                require(correct == point["correct"], f"exact count mismatch: {method}/{axis}/{label}")
                expected = {"accuracy": accuracy, "component_accuracy": component, "mean_score_0_1_2": ordinal, "tokens_per_example": point["total_tokens"] / 60}
                for metric, value in expected.items():
                    require(math.isclose(value, summary["methods"][method]["axes"][axis][label][metric], abs_tol=1e-9), f"aggregate mismatch: {method}/{axis}/{label}/{metric}")
                point_checks.append({"method": method, "axis": axis, "point": label, "csv_rows": 60, "correct": correct, **expected})
        for label in ("Q", "E"):
            for metric in ("accuracy", "component_accuracy", "mean_score_0_1_2", "tokens_per_example"):
                calculated = statistics.mean(p[metric] for p in point_checks if p["method"] == method and p["point"] == label)
                require(math.isclose(calculated, summary["methods"][method]["macro_avg"][label][metric]["mean_across_five_axes"], abs_tol=1e-9), "macro-average mismatch")
    report = {"verification_passed": True, "export_files_checked": len(records), "source_bytes_verified": inventory["source_bytes"],
              "completed_units_checked": integrity_gate_checks, "test_operating_points_checked": len(point_checks),
              "strict_quality_passes": strict_quality_passes, "point_checks": point_checks,
              "scope": "Offline byte-integrity and score verification. Preserved historical quality gate warnings remain in force; this does not rerun inference."}
    validation = ROOT / "validation"
    validation.mkdir(exist_ok=True)
    (validation / "artifact_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key:value for key,value in report.items() if key != "point_checks"}, indent=2))


if __name__ == "__main__":
    main()
