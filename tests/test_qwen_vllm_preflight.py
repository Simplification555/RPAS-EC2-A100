import io
import json
from copy import deepcopy

import pytest

from scripts.scir.qwen_vllm_preflight import baseline_call, check, compare_batches


def test_backend_probe_refuses_unallocated_execution(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="Slurm GPU"):
        check(tmp_path, "http://127.0.0.1:1234/v1")


def test_baseline_probe_preserves_synthetic_request_contract(monkeypatch):
    def response(request, timeout):
        payload = json.loads(request.data)
        assert request.full_url == "http://127.0.0.1:1234/v1/chat/completions"
        assert timeout == 600
        assert payload["max_tokens"] == 128
        assert payload["temperature"] == 0.0
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["messages"] == [{"role": "user", "content": "synthetic only"}]
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "code"}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}).encode())
    monkeypatch.setattr("urllib.request.urlopen", response)
    result = baseline_call("http://127.0.0.1:1234/v1", "synthetic only")
    assert result["content"] == "code"
    assert result["usage"]["total_tokens"] == 5
    assert result["latency_s"] >= 0


def synthetic_batches():
    baseline = [{"content": "1", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                 "finish_reason": "stop"}]
    output = [{"content": "1", "prompt_tokens": 4, "completion_tokens": 2, "finish_reason": "stop"}]
    return baseline, [deepcopy(output), deepcopy(output)]


def test_complete_synthetic_contract_passes():
    baseline, outputs = synthetic_batches()
    assert compare_batches(baseline, outputs, 1)["status"] == "PASS_SYNTHETIC_PARITY_ONLY"


@pytest.mark.parametrize("field,value", [("finish_reason", "length"), ("completion_tokens", 3),
                                        ("prompt_tokens", 5), ("content", "2")])
def test_same_text_alone_cannot_certify_token_or_stop_contract(field, value):
    baseline, outputs = synthetic_batches()
    for batch in outputs:
        batch[0][field] = value
    assert compare_batches(baseline, outputs, 1)["status"] == "BACKEND_DIFFERENCE"


def test_repetition_compares_metadata_not_only_text():
    baseline, outputs = synthetic_batches()
    outputs[0][0]["completion_tokens"] = 3
    assert compare_batches(baseline, outputs, 1)["status"] == "BACKEND_DIFFERENCE"


@pytest.mark.parametrize("baseline,outputs,count", [([], [], 0), ([], [], 1), ([{}], [[]], 1)])
def test_missing_batches_fail_closed(baseline, outputs, count):
    with pytest.raises(ValueError, match="synthetic"):
        compare_batches(baseline, outputs, count)


def test_invalid_baseline_accounting_fails():
    baseline, outputs = synthetic_batches()
    baseline[0]["usage"]["total_tokens"] = 100
    with pytest.raises(ValueError, match="accounting"):
        compare_batches(baseline, outputs, 1)
