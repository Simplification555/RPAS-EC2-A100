"""Read-only readiness audit; passing does not certify experimental results."""

import argparse
import json
import time
from pathlib import Path

from external_comparison.adapters.native_common import git_commit, sha256_file
from external_comparison.runners.ec2_v2 import GDESIGNER_COMMIT
from external_comparison.runners.mmlu import build_mmlu_manifest


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty required file: {path}")


def check_model_files(model: Path) -> int:
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        require_file(model / name)
        json.loads((model / name).read_text(encoding="utf-8"))
    index_path = model / "model.safetensors.index.json"
    require_file(index_path)
    weights = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Model weight_map must be a nonempty object")
    shards = set(weights.values())
    for shard in shards:
        path = (model / shard).resolve()
        if not path.is_relative_to(model.resolve()):
            raise ValueError(f"Model shard escapes model directory: {shard}")
        require_file(path)
    return len(shards)


def check_destination(directory: Path) -> None:
    # Shell wrappers may have created operational logs, but never reuse an
    # execution identity, partial journal, checkpoint or completed result.
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Output directory is not a directory: {directory}")
    allowed = {"logs", "environment.txt", "health.json", "models.json", "smoke_completion.json"}
    for path in directory.iterdir() if directory.exists() else ():
        if path.name not in allowed or (path.name == "logs" and not path.is_dir()):
            raise ValueError(f"Existing experiment artifact requires explicit archival: {path}")


def audit(args: argparse.Namespace) -> dict:
    manifest = build_mmlu_manifest(args.data_dir, data_seed=2026,
                                   search_per_subject=1, select_per_subject=1, test_per_subject=10)
    frozen_path = args.output_root / "split_manifest.json"
    require_file(frozen_path)
    if json.loads(frozen_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Actual MMLU sources or selected IDs differ from the frozen split")
    checked = []
    for method in ("gdesigner", "rpas_comm"):
        for seed in args.seeds:
            check_destination(args.output_root / method / f"seed_{seed}")
            checked.append({"method": method, "seed": seed})
    commit = git_commit(args.gdesigner_root)
    if not commit.startswith(GDESIGNER_COMMIT):
        raise ValueError(f"Unexpected G-Designer revision: {commit}")
    native_sources = ("experiments/train_mmlu.py", "GDesigner/graph/graph.py", "GDesigner/llm/gpt_chat.py")
    for name in native_sources:
        require_file(args.gdesigner_root / name)
    for name in ("config.json", "modules.json", "model.safetensors", "1_Pooling/config.json", "tokenizer.json"):
        require_file(args.embedding_model / name)
    config_path = args.repo_root / "experiments/phase2_mmlu_qwen35_9b.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["models"][config["defaults"]["local_model"]]
    if model_config["completion_kwargs"] != {"temperature": 0.0, "max_tokens": 256}:
        raise ValueError("Worker decoding configuration differs from frozen EC2 parameters")
    if config["reflection"].get("allow_rule_fallback") is not False:
        raise ValueError("Rule reflection fallback must remain disabled")
    return {"status": "PASS_INPUT_READINESS_ONLY", "formal_result": False,
            "checked_at_epoch": time.time(), "model_calls": 0, "checked_tasks": checked,
            "split_counts": {phase: manifest[phase]["count"] for phase in ("search", "select", "test")},
            "source_file_count": len(manifest["source_files"]),
            "split_manifest_sha256": manifest["split_manifest_sha256"],
            "model_shards_present": check_model_files(args.model),
            "model_weight_content_hash_verified": False,
            "gdesigner_commit": commit,
            "native_source_sha256": {name: sha256_file(args.gdesigner_root / name) for name in native_sources},
            "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(args.repo_root / "external_comparison/runners/ec2_v2.py"),
            "limitations": ["Readiness snapshot, not a lock or a result/publication gate",
                            "Model shard existence checked; weights not rehashed",
                            "Native execution requires separate runtime preflight"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "data-dir", "output-root", "gdesigner-root", "model", "embedding-model", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report must be new; never overwrite earlier evidence")
    report = audit(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report))
