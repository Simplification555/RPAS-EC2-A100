"""One-shot process boundary for the pinned LiveCodeBench evaluator."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile

from external_comparison.runners.livecodebench_ec1 import _runner


def _prepare_code_for_runner(sample: dict, code: str) -> str:
    """Adapt standalone stdin programs to the pinned runner's callable ABI."""
    in_outs = json.loads(sample["input_output"])
    if in_outs.get("fn_name") is not None:
        return code
    run_test = _runner()
    module = sys.modules[run_test.__module__]
    return module.make_function(module.clean_if_name(code))


def main() -> int:
    payload = json.load(sys.stdin)
    sample = payload["sample"]
    code = _prepare_code_for_runner(sample, str(payload["code"]))
    timeout_seconds = int(payload["timeout_seconds"])
    # The pinned evaluator enables faulthandler and therefore requires stderr
    # to expose a real file descriptor. StringIO would fail with ``fileno``.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as captured_stdout, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8"
    ) as captured_stderr, contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        results, metadata = _runner()(sample, test=code, debug=False, timeout=timeout_seconds)
    normalized = [bool(value.item() if hasattr(value, "item") else value == 1) for value in results]
    sys.stdout.write(
        json.dumps(
            {
                "protocol": "rpas_lcb_isolated_eval_v1",
                "results": normalized,
                "metadata": metadata,
            },
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
