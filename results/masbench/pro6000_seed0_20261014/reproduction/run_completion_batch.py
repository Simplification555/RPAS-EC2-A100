"""Complete all ten exploratory units with audited native search reuse."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo"
OLD = ROOT / "outputs/aflow_maas_9b_ctx16k_seed0_pro6000_20261013"
OUT = ROOT / "outputs/aflow_maas_completion_20261014"
PY = ROOT / ".venv-method/bin/python"
lock = (ROOT / "completion_batch.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
env = os.environ.copy()
env.update(
    PYTHONPATH=f"{ROOT}/runtime_deps:{REPO}:{REPO}/experiments:{REPO}/external_comparison",
    RPAS_AFLOW_SOURCE=str(REPO / "external_baselines/AFlow"),
    RPAS_MAAS_SOURCE=str(REPO / "external_baselines/MaAS"),
    RPAS_MAAS_EMBEDDING_PATH=str(ROOT / "models/all-MiniLM-L6-v2"),
    RPAS_TOKENIZER_PATH=str(ROOT / "models/Qwen3.5-9B"),
    RPAS_MAX_MODEL_LEN="16384", RPAS_MAX_NUM_SEQS="16", RPAS_TP_SIZE="1",
    RPAS_GPU_MEMORY_UTILIZATION="0.85", RPAS_AFLOW_REQUEST_TIMEOUT_S="300",
    RPAS_AFLOW_SAMPLE_TIMEOUT_S="600", RPAS_MAAS_REQUEST_TIMEOUT_S="300",
    RPAS_MAAS_SAMPLE_TIMEOUT_S="600", RPAS_MAAS_SAMPLE_RETRIES="1",
    RPAS_MAAS_CODE_TIMEOUT_S="60", CUDA_VISIBLE_DEVICES="",
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENAI_API_KEY="EMPTY",
    NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
)
OUT.mkdir(parents=True, exist_ok=True)
status_path = OUT / "batch_status.json"
state = json.loads(status_path.read_text()) if status_path.exists() else {
    "status": "starting", "started_at": time.time(), "units": [],
    "completion_policy": "relaxed", "answer_repair": True,
    "scope": "exploratory_completion; reuse original searches, rerun selection and test",
    "sample_timeout_seconds": 600, "native_sample_retries": 1,
}


def save():
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, status_path)


def run(command, log):
    with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=stream)
        state["child_pid"] = process.pid
        save()
        print(json.dumps({"current_unit": state.get("current_unit"), "pid": process.pid, "log": str(log)}), flush=True)
        while process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        state.pop("child_pid", None)
        return process.returncode


try:
    # Prioritize the previously blocked depth unit, then complete all axes.
    for method in ("maas", "aflow"):
        for axis in ("depth", "breadth", "horizon", "parallel", "robustness"):
            unit = OUT / method / axis / "seed_0"
            state.update(status="running", current_unit=f"{method}/{axis}/seed_0")
            save()
            if any(r["method"] == method and r["axis"] == axis and r["status"] == "completed" for r in state["units"]):
                continue
            record = {"method": method, "axis": axis, "started_at": time.time()}
            if unit.exists():
                manifest = json.loads((unit / "run_manifest.json").read_text())
                if manifest.get("status") != "complete":
                    record.update(status="partial_preserved", failure=manifest.get("failure"))
                    state["units"].append(record)
                    save()
                    continue
                rc = 0
            else:
                command = [str(PY), "-u", str(REPO / "experiments/native_masbench_external.py"),
                           "--method", method, "--axis", axis, "--seed", "0",
                           "--rpas-root", str(REPO), "--run-dir", str(unit),
                           "--endpoint", "http://127.0.0.1:18001/v1",
                           "--completion-policy", "relaxed", "--repair-answers",
                           "--resume-search-from", str(OLD / method / axis / "seed_0")]
                rc = run(command, OUT / f"run_{method}_{axis}.log")
            record["runner_exit_code"] = rc
            if rc == 0:
                rc = run([str(PY), str(REPO / "experiments/masbench_quality_gate.py"),
                          "--run-dir", str(unit), "--rpas-root", str(REPO)], OUT / f"gate_{method}_{axis}.log")
                record.update(status="completed" if rc == 0 else "integrity_gate_failed", gate_exit_code=rc)
                gate = json.loads((unit / "quality_gate.json").read_text())
                record.update(strict_quality_passed=gate["passed"], quality_warnings=gate.get("quality_warnings", []))
            else:
                record.update(status="failed", diagnostic_tail=(OUT / f"run_{method}_{axis}.log").read_text()[-2500:])
            record["finished_at"] = time.time()
            state["units"] = [r for r in state["units"] if (r["method"], r["axis"]) != (method, axis)] + [record]
            save()
            print(json.dumps(record), flush=True)
    if len(state["units"]) == 10 and all(r["status"] == "completed" for r in state["units"]):
        state["current_unit"] = "aggregate"
        command = [str(PY), str(REPO / "experiments/aggregate_masbench_seed0.py"),
                   "--root", str(OUT), "--allow-quality-warnings"]
        if (OUT / "seed0_summary").exists():
            command.append("--verify-existing")
        rc = run(command, OUT / "aggregate.log")
        state["status"] = "completed" if rc == 0 else "aggregation_failed"
    else:
        state["status"] = "completed_with_failed_units"
    state["finished_at"] = time.time()
    state.pop("current_unit", None)
    save()
    print(json.dumps(state, indent=2), flush=True)
except BaseException as exc:
    state.update(status="batch_interrupted", failure=f"{type(exc).__name__}: {exc}")
    save()
    raise
