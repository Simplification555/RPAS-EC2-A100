"""Audit and aggregate EC-3 HotpotQA held-out test summaries."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

METHODS = ("rpas", "aflow")
SEEDS = (0, 1, 2)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    runs = []
    for method in METHODS:
        for seed in SEEDS:
            d = a.root / method / f"seed_{seed}"
            summary = d / "test_summary.json"
            outputs = d / "test_outputs.jsonl"
            calls = d / "test_calls.jsonl"
            if not summary.is_file() or not outputs.is_file() or not calls.is_file():
                raise FileNotFoundError(d)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            if payload.get("d_test_accessed") is not True or not payload.get("split_manifest_sha256"):
                raise ValueError(f"invalid EC-3 held-out summary: {summary}")
            rows = [json.loads(x) for x in outputs.read_text(encoding="utf-8").splitlines() if x.strip()]
            if len(rows) != 800 or len({str(x.get("id", x.get("task_id", ""))) for x in rows}) != 800:
                raise ValueError(f"EC-3 requires 800 unique held-out rows: {outputs}")
            runs.append({"method": method, "seed": seed, "answer_f1": float(payload.get("answer_f1", payload.get("score", 0.0))), "answer_em": float(payload.get("answer_em", 0.0)), "test_calls": int(payload.get("test_calls", 0)), "test_tokens": int(payload.get("test_tokens", 0)), "split_manifest_sha256": payload["split_manifest_sha256"]})
    if len({x["split_manifest_sha256"] for x in runs}) != 1:
        raise ValueError("EC-3 held-out runs do not share one split manifest")
    table = []
    for method in METHODS:
        vals = [x for x in runs if x["method"] == method]
        scores = [x["answer_f1"] for x in vals]
        table.append({"method": method, "seeds": [x["seed"] for x in vals], "answer_f1_mean": statistics.fmean(scores), "answer_f1_std": statistics.stdev(scores), "answer_em_mean": statistics.fmean(x["answer_em"] for x in vals), "test_calls_mean": statistics.fmean(x["test_calls"] for x in vals), "test_tokens_mean": statistics.fmean(x["test_tokens"] for x in vals)})
    out = {"protocol": "EC3_HOTPOTQA_V3", "formal_result": True, "runs": runs, "summary": table}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
