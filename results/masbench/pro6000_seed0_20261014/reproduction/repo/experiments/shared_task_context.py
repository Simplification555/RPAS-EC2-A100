"""Opt-in task context policy, counted with the serving vLLM chat template."""

import json
import re
from urllib.request import Request, urlopen

POLICY = {"version": "task_context_v1", "limit": 8192, "margin": 64, "min_output": 512}
MASBENCH_POLICY = {"version": "task_context_masbench_v3", "limit": 16384, "margin": 64, "min_output": 512}


def policy_for_dataset(dataset):
    """Keep AIME frozen; use the same larger window for every MASBench method/axis."""
    return dict(MASBENCH_POLICY if dataset == "masbench" else POLICY)


class ContextBudgetError(ValueError):
    def __init__(self, message, budget):
        super().__init__(message)
        self.context_budget = budget

MARKERS = re.compile(
    r"\n\n(?:Solver output:|Plan:|Two agents proposed answers:|Subtask results:|"
    r"Dependency results:|A planner supplied the following possibly incomplete plan:|"
    r"A solver proposed the following answer/reasoning\. It may be wrong or truncated:)\n"
)


EVIDENCE_SPANS = re.compile(r"<intermediate_evidence>(.*?)</intermediate_evidence>", re.S)

def count_messages(messages, model, kwargs):
    base = kwargs["api_base"].rstrip("/").removesuffix("/v1")
    payload = {
        "model": model.removeprefix("openai/"), "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": kwargs.get("extra_body", {}).get("chat_template_kwargs", {}),
    }
    request = Request(
        base + "/tokenize", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + kwargs.get("api_key", "EMPTY")},
    )
    with urlopen(request, timeout=30) as response:
        count = json.load(response)["count"]
    if not isinstance(count, int) or count < 0:
        raise ValueError("Invalid serving tokenizer count")
    return count


def fit_context(messages, requested, policy, counter):
    """Preserve system/problem text; compact only identified downstream evidence."""
    messages = [dict(message) for message in messages]
    limit = policy["limit"] - policy["margin"]
    minimum = min(requested, policy["min_output"])
    original = counter(messages)
    count = original
    rounds = 0
    def fail(message):
        raise ContextBudgetError(message, {
            "policy": dict(policy), "original_input_tokens": original, "input_tokens": count,
            "configured_max_tokens": requested, "minimum_output_tokens": minimum,
            "available_output_tokens": limit - count, "compression_rounds": rounds,
            "count_source": "serving_tokenizer",
        })
    while count + minimum > limit:
        options = []
        for index, message in enumerate(messages):
            if message["role"] != "user":
                continue
            spans = list(EVIDENCE_SPANS.finditer(message["content"]))
            for span in spans:
                options.append((span.end(1) - span.start(1), index, span.start(1), span.end(1)))
            if not spans:
                match = MARKERS.search(message["content"])
                if match:
                    options.append((len(message["content"]) - match.end(), index, match.end(), len(message["content"])))
        if not options:
            fail("Task context cannot fit without removing protected problem/system text")
        length, index, start, end = max(options)
        if length < 256 or rounds >= 32:
            fail("Task context cannot fit after evidence compression")
        text = messages[index]["content"]
        evidence = text[start:end]
        keep = length // 4
        messages[index]["content"] = text[:start] + evidence[:keep] + "\n[context omitted]\n" + evidence[-keep:] + text[end:]
        count = counter(messages)
        rounds += 1
    effective = min(requested, limit - count)
    return messages, effective, {
        "policy": policy, "original_input_tokens": original, "input_tokens": count,
        "configured_max_tokens": requested, "effective_max_tokens": effective, "compression_rounds": rounds,
    }
