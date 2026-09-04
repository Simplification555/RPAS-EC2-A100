from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - local docs/help can still work without runtime deps.
    litellm = None


DEFAULT_AIME_DATA_DIR = "data/aime"
DEFAULT_AIME_TRAIN_FILE = "aimo-validation-aime.jsonl"
DEFAULT_AIME_TEST_FILE = "aime_2025.jsonl"
DEFAULT_GAIA_DATA_DIR = "data/gaia"
DEFAULT_GAIA_SPLIT = "validation"
DEFAULT_MASBENCH_DATA_DIR = "data/masbench"
MASBENCH_AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
MASBENCH_ANSWER_SEPARATOR = "<<horizon>>"
MASBENCH_PROTOCOL_VERSION = "masbench_igsm_mod23_v1"
PROMPT_PROTOCOL_VERSION = "rpas_roles_reflection_v4_explicit_terminal_marker"
EVALUATION_CACHE_VERSION = "rpas_shared_eval_v2"
EXPERIMENT_PROTOCOL_VERSION = "rpas_experiment_protocol_v1.0"
MIN_VALID_EXECUTION_RATE = 0.99
MAX_ERROR_EXAMPLE_RATE = 0.01
MAX_PROTOCOL_ERROR_RATE = 0.01
MAX_TRUNCATED_UNEXTRACTABLE_RATE = 0.05
EFFICIENCY_QUALITY_DELTA = 0.05

ANSWER_FIRST_INSTRUCTION = (
    "/no_think\n"
    "Solve the problem. Your first line must be exactly `### ANSWER`, where ANSWER is only the final "
    "numeric answer or simplified fraction. Do not write hidden reasoning. Keep any explanation brief. "
    "After that answer line, write one final line exactly `<<RPAS_END>>` and nothing else.\n\n"
)

DELIBERATE_INSTRUCTION = (
    "/no_think\n"
    "Solve the problem carefully. You may write concise reasoning, but avoid hidden chain-of-thought tags. "
    "End your response with a separate final line exactly in the format `### ANSWER`, where ANSWER is only "
    "the final numeric answer or simplified fraction. After that answer line, write one final line exactly "
    "`<<RPAS_END>>` and nothing else.\n\n"
)

THINKING_DELIBERATE_INSTRUCTION = (
    "Solve the problem carefully. You may use the model's normal mathematical reasoning mode. "
    "End your response with a separate final line exactly in the format `### ANSWER`, where ANSWER is only "
    "the final numeric answer or simplified fraction.\n\n"
)

THINKING_ANSWER_FIRST_INSTRUCTION = (
    "Solve the problem. Put the final answer on its own final line exactly in the format `### ANSWER`, "
    "where ANSWER is only the final numeric answer or simplified fraction.\n\n"
)

GAIA_INSTRUCTION = (
    "Answer the GAIA task using concise, tool-aware reasoning. If an attachment path is provided, use it only "
    "when the current runner exposes file/tool access; otherwise state the best answer from the visible prompt. "
    "End your response with a separate final line exactly in the format `FINAL ANSWER: ANSWER`. The final answer "
    "should be a number, a few words, or a comma-separated list as requested by the question.\n\n"
)

MASBENCH_INSTRUCTION = (
    "/no_think\n"
    "Solve every subproblem in the MASBench task carefully and preserve the requested answer order. Follow only "
    "the dependency paths needed for the requested quantities; do not restate every fact or repeatedly speculate "
    "about missing definitions. For iGSM arithmetic, every quantity is in Z_23: reduce additions, subtractions, "
    "multiplications, and intermediate results modulo 23 to an integer from 0 through 22. A plural category or "
    "abstract total denotes the modulo-23 sum of its direct listed members. "
    f"If there are multiple answers, join them using exactly `{MASBENCH_ANSWER_SEPARATOR}` with no omitted "
    "items or reordering. End with one separate line `FINAL ANSWER: ANSWER`; ANSWER must contain only the "
    "single answer or the ordered separator-delimited answer sequence.\n\n"
)

HUMANEVAL_INSTRUCTION = (
    "/no_think\n"
    "Complete the Python programming task below. Return only executable Python code, including the requested "
    "function definition. Do not include Markdown fences, explanations, tests, or a final-answer marker. Preserve "
    "the function name and signature from the prompt.\n\n"
)

MMLU_INSTRUCTION = (
    "/no_think\n"
    "Answer the multiple-choice question. Put the decision FIRST and return exactly one line, with no reasoning, "
    "in the format `FINAL ANSWER: X`, where X is one of A, B, C, or D. Do not output any other text or answer "
    "letter.\n\n"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    litellm_model: str
    api_base: str | None = None
    api_key_env: str | None = None
    api_key_value: str | None = None
    completion_kwargs: dict[str, Any] = field(default_factory=dict)
    input_cost_per_million_tokens_usd: float = 0.0
    output_cost_per_million_tokens_usd: float = 0.0
    pricing_source: str = ""


@dataclass(frozen=True)
class SiteSpec:
    name: str
    kind: str = "local"
    compute_latency_ms: float = 0.0
    notes: str = ""


@dataclass(frozen=True)
class NetworkEdge:
    src: str
    dst: str
    rtt_ms: float
    bandwidth_mbps: float
    failure_rate: float = 0.0


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    default_rtt_ms: float
    default_bandwidth_mbps: float
    default_failure_rate: float = 0.0
    retry_backoff_ms: float = 1000.0
    max_expected_retries: float = 1.0
    edges: list[NetworkEdge] = field(default_factory=list)


@dataclass
class CallTrace:
    agent: str
    model: str
    site: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    observed_latency_ms: float
    requested_max_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    output_excerpt: str = ""
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    inference_cost_usd: float = 0.0


@dataclass
class MessageTrace:
    src_agent: str
    dst_agent: str
    src_site: str
    dst_site: str
    compression: str
    message_tokens: int
    cross_center: bool
    message_latency_ms: float
    expected_retry_latency_ms: float
    emulated_latency_ms: float
    failure_probability: float


@dataclass
class RolloutTrace:
    calls: list[CallTrace] = field(default_factory=list)
    messages: list[MessageTrace] = field(default_factory=list)
    observed_model_wall_latency_ms: float | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def summary(self, profile: NetworkProfile | None = None) -> dict[str, Any]:
        call_tokens = sum(call.total_tokens for call in self.calls)
        requested_max_tokens = sum(call.requested_max_tokens or 0 for call in self.calls)
        prompt_tokens = sum(call.prompt_tokens for call in self.calls)
        completion_tokens = sum(call.completion_tokens for call in self.calls)
        input_cost_usd = sum(call.input_cost_usd for call in self.calls)
        output_cost_usd = sum(call.output_cost_usd for call in self.calls)
        inference_cost_usd = sum(call.inference_cost_usd for call in self.calls)
        maxed_calls = sum(
            1
            for call in self.calls
            if call.finish_reason == "length"
            or (
                call.requested_max_tokens is not None
                and call.completion_tokens >= call.requested_max_tokens
            )
        )
        observed_model_latency_ms = sum(call.observed_latency_ms for call in self.calls)
        observed_model_wall_latency_ms = (
            self.observed_model_wall_latency_ms
            if self.observed_model_wall_latency_ms is not None
            else observed_model_latency_ms
        )
        cross_message_tokens = sum(msg.message_tokens for msg in self.messages if msg.cross_center)
        cross_model_tokens = sum(call.total_tokens for call in self.calls if call.site != ORCHESTRATOR_SITE)
        local_tokens = sum(msg.message_tokens for msg in self.messages if not msg.cross_center)
        message_latency_ms = sum(msg.message_latency_ms for msg in self.messages)
        message_expected_retry_latency_ms = sum(msg.expected_retry_latency_ms for msg in self.messages)
        model_rpc_latency_ms = 0.0
        model_rpc_expected_retry_latency_ms = 0.0
        model_rpc_failure_probability = 0.0
        if profile is not None:
            model_rpc_estimates = [
                estimate_network_transfer(profile, ORCHESTRATOR_SITE, call.site, call.total_tokens)
                for call in self.calls
                if call.site != ORCHESTRATOR_SITE
            ]
            model_rpc_latency_ms = sum(estimate["base_latency_ms"] for estimate in model_rpc_estimates)
            model_rpc_expected_retry_latency_ms = sum(
                estimate["expected_retry_latency_ms"] for estimate in model_rpc_estimates
            )
            model_rpc_failure_probability = sum(estimate["failure_probability"] for estimate in model_rpc_estimates)
        site_compute_latency_ms = sum(call_site_compute_penalty(call.site) for call in self.calls)
        network_latency_ms = (
            message_latency_ms
            + message_expected_retry_latency_ms
            + model_rpc_latency_ms
            + model_rpc_expected_retry_latency_ms
        )
        emulated_latency_ms = (
            observed_model_latency_ms
            + network_latency_ms
            + site_compute_latency_ms
        )
        emulated_wall_latency_ms = (
            observed_model_wall_latency_ms
            + network_latency_ms
            + site_compute_latency_ms
        )
        return {
            "calls": len(self.calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_cost_usd": input_cost_usd,
            "output_cost_usd": output_cost_usd,
            "inference_cost_usd": inference_cost_usd,
            "maxed_calls": maxed_calls,
            "total_tokens": call_tokens,
            "requested_max_tokens": requested_max_tokens,
            "cross_center_tokens": cross_message_tokens + cross_model_tokens,
            "cross_center_message_tokens": cross_message_tokens,
            "cross_center_model_tokens": cross_model_tokens,
            "local_message_tokens": local_tokens,
            "message_count": len(self.messages),
            "cross_center_message_count": sum(1 for msg in self.messages if msg.cross_center),
            "observed_model_latency_ms": observed_model_latency_ms,
            "observed_model_wall_latency_ms": observed_model_wall_latency_ms,
            "parallel_model_latency_savings_ms": max(
                0.0,
                observed_model_latency_ms - observed_model_wall_latency_ms,
            ),
            "message_latency_ms": message_latency_ms,
            "message_expected_retry_latency_ms": message_expected_retry_latency_ms,
            "model_rpc_latency_ms": model_rpc_latency_ms,
            "model_rpc_expected_retry_latency_ms": model_rpc_expected_retry_latency_ms,
            "expected_retry_latency_ms": message_expected_retry_latency_ms + model_rpc_expected_retry_latency_ms,
            "network_latency_ms": network_latency_ms,
            "site_compute_latency_ms": site_compute_latency_ms,
            "emulated_latency_ms": emulated_latency_ms,
            "emulated_wall_latency_ms": emulated_wall_latency_ms,
            "expected_network_failures": sum(msg.failure_probability for msg in self.messages)
            + model_rpc_failure_probability,
            "errors": sum(1 for call in self.calls if call.error),
            "artifacts": self.artifacts,
        }


SITE_COMPUTE_LATENCY_MS: dict[str, float] = {}
ORCHESTRATOR_SITE = "center_a"


def call_site_compute_penalty(site: str) -> float:
    return SITE_COMPUTE_LATENCY_MS.get(site, 0.0)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as fin:
        return json.load(fin)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    excluded_keys = {"api_base", "api_key_env", "api_key_value"}

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): sanitize(item)
                for key, item in value.items()
                if str(key) not in excluded_keys
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    return sanitize(config)


def normalized_split_manifest(examples: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [
        {
            "id": str(example.get("id", "")),
            "input": str(example.get("input", "")),
            "answer": str(example.get("answer", "")),
            "dataset": str(example.get("dataset", "")),
            "axis": str(example.get("axis", "")),
            "axis_value": str(example.get("axis_value", "")),
            "attachment": str(example.get("attachment", "")),
        }
        for example in examples
    ]
    return {
        "count": len(normalized),
        "ids": [example["id"] for example in normalized],
        "normalized_content_sha256": sha256_json(normalized),
    }


def model_manifest_payload(models: dict[str, ModelSpec]) -> dict[str, Any]:
    return {
        name: {
            "litellm_model": resolve_litellm_model(spec),
            "completion_kwargs": spec.completion_kwargs,
            "input_cost_per_million_tokens_usd": spec.input_cost_per_million_tokens_usd,
            "output_cost_per_million_tokens_usd": spec.output_cost_per_million_tokens_usd,
            "pricing_source": spec.pricing_source,
        }
        for name, spec in sorted(models.items())
    }


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def resolve_api_key(spec: ModelSpec) -> str | None:
    if spec.api_key_value is not None:
        return spec.api_key_value
    if spec.api_key_env is not None:
        return os.environ.get(spec.api_key_env)
    return None


def resolve_api_base(spec: ModelSpec) -> str | None:
    env_key = f"GEPA_{spec.name.upper()}_API_BASE"
    return os.environ.get(env_key, spec.api_base)


def merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def should_disable_qwen_thinking(spec: ModelSpec) -> bool:
    disabled = os.environ.get("GEPA_QWEN_DISABLE_THINKING", "1").lower() not in {"0", "false", "no"}
    return disabled and "qwen" in spec.litellm_model.lower()


def make_completion_kwargs(
    spec: ModelSpec,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    kwargs = dict(spec.completion_kwargs)
    api_base = resolve_api_base(spec)
    if api_base:
        kwargs["api_base"] = api_base
    api_key = resolve_api_key(spec)
    if api_key is not None:
        kwargs["api_key"] = api_key
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if should_disable_qwen_thinking(spec):
        kwargs = merge_dicts(
            kwargs,
            {
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                }
            },
        )
    return kwargs


def resolve_litellm_model(spec: ModelSpec) -> str:
    env_key = f"GEPA_{spec.name.upper()}_LITELLM_MODEL"
    return os.environ.get(env_key, spec.litellm_model)


def model_call_costs(spec: ModelSpec, prompt_tokens: int, completion_tokens: int) -> tuple[float, float, float]:
    input_cost = max(0, prompt_tokens) * max(0.0, spec.input_cost_per_million_tokens_usd) / 1_000_000
    output_cost = max(0, completion_tokens) * max(0.0, spec.output_cost_per_million_tokens_usd) / 1_000_000
    return input_cost, output_cost, input_cost + output_cost


def litellm_call(
    *,
    model_spec: ModelSpec,
    agent_name: str,
    site: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, CallTrace]:
    completion_kwargs = make_completion_kwargs(model_spec, temperature=temperature, max_tokens=max_tokens)
    requested_max_tokens = completion_kwargs.get("max_tokens")
    if requested_max_tokens is not None:
        requested_max_tokens = int(requested_max_tokens)
    if os.environ.get("GEPA_PHASE2_MOCK_MODEL", "").lower() in {"1", "true", "yes"}:
        content = "### 0\nMock model output for pipeline validation only."
        prompt_text = "\n".join(message.get("content", "") for message in messages)
        prompt_tokens = approx_tokens(prompt_text)
        completion_tokens = approx_tokens(content)
        input_cost, output_cost, inference_cost = model_call_costs(
            model_spec,
            prompt_tokens,
            completion_tokens,
        )
        return content, CallTrace(
            agent=agent_name,
            model=model_spec.name,
            site=site,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            observed_latency_ms=1.0,
            requested_max_tokens=requested_max_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            inference_cost_usd=inference_cost,
        )
    if litellm is None:
        raise RuntimeError(
            "litellm is required for model calls. Install project dependencies or run inside the gepa conda env."
        )
    prompt_text = "\n".join(message.get("content", "") for message in messages)
    prompt_tokens_estimate = approx_tokens(prompt_text)
    start = time.time()
    try:
        response = litellm.completion(
            model=resolve_litellm_model(model_spec),
            messages=messages,
            **completion_kwargs,
        )
        observed_latency_ms = (time.time() - start) * 1000
        content = (response.choices[0].message.content or "").strip()
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or prompt_tokens_estimate)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or approx_tokens(content))
        total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
        input_cost, output_cost, inference_cost = model_call_costs(
            model_spec,
            prompt_tokens,
            completion_tokens,
        )
        return content, CallTrace(
            agent=agent_name,
            model=model_spec.name,
            site=site,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            observed_latency_ms=observed_latency_ms,
            requested_max_tokens=requested_max_tokens,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            inference_cost_usd=inference_cost,
        )
    except Exception as exc:
        observed_latency_ms = (time.time() - start) * 1000
        error = f"{type(exc).__name__}: {exc}"
        return "", CallTrace(
            agent=agent_name,
            model=model_spec.name,
            site=site,
            prompt_tokens=prompt_tokens_estimate,
            completion_tokens=0,
            total_tokens=prompt_tokens_estimate,
            observed_latency_ms=observed_latency_ms,
            requested_max_tokens=requested_max_tokens,
            error=error,
        )


def normalize_numeric_token(token: str) -> str:
    token = token.strip()
    if re.fullmatch(r"[-+]?\d+", token):
        return str(int(token))
    return token


def extract_final_answer(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).replace("\r\n", "\n")
    gaia_matches = list(re.finditer(r"(?im)^FINAL\s+ANSWER\s*:\s*(.+?)\s*$", cleaned))
    if gaia_matches:
        return gaia_matches[-1].group(1).strip()
    matches = list(re.finditer(r"(?m)^###\s*(.+?)\s*$", cleaned))
    if matches:
        payload = matches[-1].group(1).strip()
        if payload.upper() == "ANSWER":
            following = cleaned[matches[-1].end() :].strip()
            numeric_tokens = re.findall(r"[-+]?\d+(?:/\d+)?", following)
            if numeric_tokens:
                return normalize_numeric_token(numeric_tokens[0])
        if "=" in payload:
            rhs = payload.rsplit("=", 1)[-1].strip()
            if re.fullmatch(r"[-+]?\d+(?:/\d+)?", rhs):
                payload = rhs
        numeric_tokens = re.findall(r"[-+]?\d+(?:/\d+)?", payload)
        if numeric_tokens:
            return normalize_numeric_token(numeric_tokens[-1])
        return payload.strip()
    numeric_tokens = re.findall(r"[-+]?\d+(?:/\d+)?", cleaned)
    return normalize_numeric_token(numeric_tokens[-1]) if numeric_tokens else ""


def task_instruction(dataset: str | None = None) -> str:
    if dataset == "humaneval":
        return HUMANEVAL_INSTRUCTION
    if dataset == "mmlu":
        return MMLU_INSTRUCTION
    if dataset == "gaia":
        return GAIA_INSTRUCTION
    if dataset == "masbench":
        return MASBENCH_INSTRUCTION
    mode = os.environ.get("GEPA_PHASE2_PROMPT_MODE", "deliberate").lower()
    if mode in {"think", "thinking", "think_deliberate", "cot"}:
        return THINKING_DELIBERATE_INSTRUCTION
    if mode in {"think_answer_first", "thinking_answer_first"}:
        return THINKING_ANSWER_FIRST_INSTRUCTION
    if mode in {"answer_first", "fast", "short"}:
        return ANSWER_FIRST_INSTRUCTION
    return DELIBERATE_INSTRUCTION


def score_exact_answer(output: str, answer: str) -> float:
    expected = extract_final_answer(str(answer))
    if not expected:
        expected = normalize_numeric_token(str(answer).replace("###", "").strip())
    actual = extract_final_answer(output)
    return 1.0 if actual and expected and actual == expected else 0.0


def normalize_gaia_answer(text: str) -> str:
    normalized = extract_final_answer(text) or str(text)
    normalized = re.sub(r"<think>.*?</think>", "", normalized, flags=re.DOTALL)
    normalized = normalized.strip()
    normalized = re.sub(r"^(final\s+answer\s*:|###)\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip().strip("`").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    return normalized.strip().rstrip(".").lower()


def normalize_masbench_answer(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    has_sequence = bool(re.search(r"<<\s*horizon\s*>>", cleaned, flags=re.IGNORECASE))
    normalized = extract_final_answer(cleaned) if not has_sequence else cleaned
    final_matches = list(re.finditer(r"(?im)^FINAL\s+ANSWER\s*:\s*(.+?)\s*$", cleaned))
    if final_matches:
        normalized = final_matches[-1].group(1).strip()
    normalized = re.sub(r"^(final\s+answer\s*:|###)\s*", "", normalized.strip(), flags=re.IGNORECASE)
    normalized = normalized.strip().strip("`").strip()
    parts = re.split(r"\s*<<\s*horizon\s*>>\s*", normalized, flags=re.IGNORECASE)
    normalized_parts = []
    for part in parts:
        compact = re.sub(r"\s+", " ", part).strip().rstrip(".")
        normalized_parts.append(normalize_numeric_token(compact))
    return MASBENCH_ANSWER_SEPARATOR.join(normalized_parts)


def extract_masbench_final_answer(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    matches = list(re.finditer(r"(?im)^FINAL\s+ANSWER\s*:\s*(.+?)\s*$", cleaned))
    return matches[-1].group(1).strip() if matches else ""


def score_example_answer(output: str, answer: str, dataset: str | None = None) -> float:
    if dataset == "gaia":
        expected = normalize_gaia_answer(answer)
        actual = normalize_gaia_answer(output)
        return 1.0 if actual and expected and actual == expected else 0.0
    if dataset == "masbench":
        explicit_answer = extract_masbench_final_answer(output)
        if not explicit_answer:
            return 0.0
        expected = normalize_masbench_answer(answer)
        actual = normalize_masbench_answer(explicit_answer)
        return 1.0 if actual and expected and actual == expected else 0.0
    return score_exact_answer(output, answer)


def score_answer_components(output: str, answer: str, dataset: str | None = None) -> float:
    if dataset != "masbench" or MASBENCH_ANSWER_SEPARATOR not in str(answer):
        return score_example_answer(output, answer, dataset)
    explicit_answer = extract_masbench_final_answer(output)
    if not explicit_answer:
        return 0.0
    expected_parts = normalize_masbench_answer(answer).split(MASBENCH_ANSWER_SEPARATOR)
    actual_parts = normalize_masbench_answer(explicit_answer).split(MASBENCH_ANSWER_SEPARATOR)
    correct = sum(
        expected == actual
        for expected, actual in zip(expected_parts, actual_parts, strict=False)
    )
    return correct / len(expected_parts) if expected_parts else 0.0


def extract_prediction_for_dataset(output: str, dataset: str | None = None) -> str:
    if dataset == "gaia":
        return normalize_gaia_answer(output)
    if dataset == "masbench":
        explicit_answer = extract_masbench_final_answer(output)
        return normalize_masbench_answer(explicit_answer) if explicit_answer else ""
    return extract_final_answer(output)


def load_aime_dataset(
    *,
    data_dir: Path,
    train_file: str,
    test_file: str,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = read_jsonl(data_dir / train_file)
    test_rows = read_jsonl(data_dir / test_file)
    rng = random.Random(seed)
    train_examples = [
        {"id": f"train:{row.get('id', idx)}", "input": str(row["problem"]), "answer": str(row["answer"])}
        for idx, row in enumerate(train_rows)
    ]
    test_examples = [
        {"id": f"test:{row.get('problem_idx', idx)}", "input": str(row["problem"]), "answer": str(row["answer"])}
        for idx, row in enumerate(test_rows)
    ]
    rng.shuffle(train_examples)
    trainset = train_examples[:train_size]
    valset = train_examples[train_size : train_size + val_size]
    testset = test_examples[:test_size]
    return trainset, valset, testset


def load_generic_jsonl_dataset(
    *,
    path: Path,
    input_keys: tuple[str, ...] = ("problem", "question", "prompt", "input"),
    answer_keys: tuple[str, ...] = ("answer", "target", "output", "label"),
    limit: int | None = None,
    id_prefix: str = "",
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        input_value = next((row[key] for key in input_keys if key in row), None)
        answer_value = next((row[key] for key in answer_keys if key in row), None)
        if input_value is None or answer_value is None:
            continue
        raw_id = str(row.get("id", idx))
        example_id = f"{id_prefix}:{raw_id}" if id_prefix else raw_id
        examples.append({"id": example_id, "input": str(input_value), "answer": str(answer_value)})
        if limit is not None and len(examples) >= limit:
            break
    return examples


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local runtime.
        raise RuntimeError("Reading benchmark parquet files requires pyarrow.") from exc
    table = pq.read_table(path)
    return [dict(row) for row in table.to_pylist()]


def find_gaia_split_files(data_dir: Path, split: str) -> list[Path]:
    candidates: list[Path] = []
    for suffix in ("jsonl", "json", "parquet"):
        candidates.extend(data_dir.glob(f"**/{split}*.{suffix}"))
        candidates.extend(data_dir.glob(f"**/*{split}*.{suffix}"))
    for split_dir in data_dir.glob(f"**/{split}"):
        if not split_dir.is_dir():
            continue
        metadata_files = [
            split_dir / f"metadata.{suffix}"
            for suffix in ("jsonl", "json", "parquet")
            if (split_dir / f"metadata.{suffix}").is_file()
        ]
        if metadata_files:
            candidates.extend(metadata_files)
            continue
        for suffix in ("jsonl", "json", "parquet"):
            candidates.extend(split_dir.glob(f"*.{suffix}"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique)


def read_gaia_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    files = find_gaia_split_files(data_dir, split)
    if not files:
        raise FileNotFoundError(
            f"Could not find GAIA split '{split}' under {data_dir}. "
            "Download gaia-benchmark/GAIA with huggingface_hub snapshot_download first."
        )
    rows: list[dict[str, Any]] = []
    for path in files:
        file_rows: list[dict[str, Any]] = []
        if path.suffix == ".jsonl":
            file_rows = read_jsonl(path)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                file_rows = payload
            elif isinstance(payload, dict):
                file_rows = payload.get("data", payload.get(split, []))
        elif path.suffix == ".parquet":
            file_rows = read_parquet(path)
        for row in file_rows:
            normalized_row = dict(row)
            normalized_row["__source_dir"] = str(path.parent.resolve())
            rows.append(normalized_row)
    return rows


def gaia_attachment_path(data_dir: Path, row: dict[str, Any]) -> str:
    raw_path = row.get("file_path") or row.get("file_name") or ""
    if raw_path is None:
        return ""
    raw_path = str(raw_path).strip()
    if not raw_path:
        return ""
    path = Path(raw_path)
    source_dir = Path(str(row.get("__source_dir") or data_dir))
    relative_path = Path(*path.parts[1:]) if path.is_absolute() else path
    candidates = [
        path,
        source_dir / relative_path,
        data_dir / relative_path,
        source_dir / path.name,
        data_dir / path.name,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return str(resolved)
    return str((source_dir / path.name).resolve())


def format_gaia_input(question: str, attachment: str, level: str) -> str:
    parts = [question.strip()]
    if level:
        parts.append(f"GAIA level: {level}")
    if attachment:
        parts.append(f"Attachment path: {attachment}")
    return "\n\n".join(part for part in parts if part)


def load_gaia_dataset(
    *,
    data_dir: Path,
    split: str,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    include_attachments: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_gaia_split(data_dir, split)
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = row.get("Question") or row.get("question") or row.get("input") or row.get("prompt")
        answer = row.get("Final answer") or row.get("final_answer") or row.get("answer")
        if question is None or answer is None:
            continue
        attachment = gaia_attachment_path(data_dir, row)
        if attachment and not include_attachments:
            continue
        level = str(row.get("Level") or row.get("level") or "")
        task_id = str(row.get("task_id") or row.get("id") or idx)
        examples.append(
            {
                "id": f"gaia:{split}:{task_id}",
                "input": format_gaia_input(str(question), attachment, level),
                "answer": str(answer),
                "dataset": "gaia",
                "level": level,
                "attachment": attachment,
            }
        )
    rng = random.Random(seed)
    rng.shuffle(examples)
    trainset = examples[:train_size]
    valset = examples[train_size : train_size + val_size]
    testset = examples[train_size + val_size : train_size + val_size + test_size]
    return trainset, valset, testset


def parse_json_field(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, dict | list):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def find_masbench_split_files(data_dir: Path, axis: str, split: str) -> list[Path]:
    if axis not in MASBENCH_AXES:
        raise ValueError(f"Unknown MASBench axis '{axis}'. Expected one of {MASBENCH_AXES}.")
    candidates = [
        path
        for path in data_dir.glob("**/*.parquet")
        if axis.lower() in {part.lower() for part in path.parts}
        and (
            path.stem.lower() == split.lower()
            or path.stem.lower().startswith(f"{split.lower()}-")
            or f"-{split.lower()}-" in path.stem.lower()
        )
    ]
    if not candidates:
        candidates = [
            path
            for path in data_dir.glob(f"**/*{split}*.parquet")
            if axis.lower() in path.as_posix().lower()
        ]
    return sorted({path.resolve() for path in candidates if path.is_file()})


def read_masbench_split(data_dir: Path, axis: str, split: str) -> list[dict[str, Any]]:
    files = find_masbench_split_files(data_dir, axis, split)
    if not files:
        raise FileNotFoundError(
            f"Could not find MASBench axis='{axis}' split='{split}' under {data_dir}. "
            "Download Salesforce/MASBench with huggingface_hub snapshot_download first."
        )
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(read_parquet(path))
    return rows


def format_masbench_prompt(prompt_payload: Any) -> str:
    messages = parse_json_field(prompt_payload, default=[])
    if isinstance(messages, dict):
        messages = messages.get("messages", [messages])
    if not isinstance(messages, list):
        return str(prompt_payload).strip()
    contents = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is not None:
            contents.append(str(content).strip())
    return "\n\n".join(content for content in contents if content)


def masbench_row_to_example(row: dict[str, Any], axis: str, split: str, index: int) -> dict[str, Any] | None:
    reward = parse_json_field(row.get("reward_model_json"), default={})
    extra = parse_json_field(row.get("extra_info_json"), default={})
    prompt = format_masbench_prompt(row.get("prompt_json"))
    answer = reward.get("ground_truth") if isinstance(reward, dict) else None
    if not prompt or answer is None:
        return None
    example_id = extra.get("example_id", extra.get("index", index)) if isinstance(extra, dict) else index
    return {
        "id": f"masbench:{axis}:{split}:{example_id}",
        "input": prompt,
        "answer": str(answer),
        "dataset": "masbench",
        "axis": axis,
        "axis_value": str(row.get("value", "")),
        "official_split": split,
    }


def load_masbench_dataset(
    *,
    data_dir: Path,
    axis: str,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    test_values: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = read_masbench_split(data_dir, axis, "train")
    test_rows = read_masbench_split(data_dir, axis, "test")
    search_pool = [
        example
        for idx, row in enumerate(train_rows)
        if (example := masbench_row_to_example(row, axis, "train", idx)) is not None
    ]
    test_pool = [
        example
        for idx, row in enumerate(test_rows)
        if (example := masbench_row_to_example(row, axis, "test", idx)) is not None
    ]
    if test_values:
        ordered_values = tuple(dict.fromkeys(test_values))
        allowed_values = set(ordered_values)
        test_pool = [example for example in test_pool if example["axis_value"] in allowed_values]
    rng = random.Random(seed)
    rng.shuffle(search_pool)
    required_search = train_size + val_size
    if required_search > len(search_pool):
        raise ValueError(
            f"MASBench {axis} train split has {len(search_pool)} usable rows, "
            f"but train_size + val_size = {required_search}."
        )
    if test_size > len(test_pool):
        raise ValueError(
            f"MASBench {axis} test split has {len(test_pool)} usable rows, but test_size = {test_size}."
        )
    trainset = search_pool[:train_size]
    valset = search_pool[train_size:required_search]
    if test_values:
        groups = {
            value: [example for example in test_pool if example["axis_value"] == value]
            for value in ordered_values
        }
        per_value, remainder = divmod(test_size, len(ordered_values))
        testset = []
        for idx, value in enumerate(ordered_values):
            rng.shuffle(groups[value])
            requested = per_value + int(idx < remainder)
            if requested > len(groups[value]):
                raise ValueError(
                    f"MASBench {axis} test value={value} has {len(groups[value])} rows, "
                    f"but balanced sampling requested {requested}."
                )
            testset.extend(groups[value][:requested])
        rng.shuffle(testset)
    else:
        rng.shuffle(test_pool)
        testset = test_pool[:test_size]
    return trainset, valset, testset


def load_models(raw_models: dict[str, dict[str, Any]]) -> dict[str, ModelSpec]:
    models = {}
    for name, raw_payload in raw_models.items():
        payload = dict(raw_payload)
        api_base_override = os.environ.get(f"GEPA_{name.upper()}_API_BASE")
        if api_base_override:
            payload["api_base"] = api_base_override
        models[name] = ModelSpec(name=name, **payload)
    return models


def load_sites(raw_sites: dict[str, dict[str, Any]]) -> dict[str, SiteSpec]:
    return {name: SiteSpec(name=name, **payload) for name, payload in raw_sites.items()}


def load_network_profiles(raw_profiles: dict[str, dict[str, Any]]) -> dict[str, NetworkProfile]:
    profiles: dict[str, NetworkProfile] = {}
    for name, payload in raw_profiles.items():
        edge_payloads = payload.get("edges", [])
        edges = [NetworkEdge(**edge) for edge in edge_payloads]
        rest = {key: value for key, value in payload.items() if key != "edges"}
        profiles[name] = NetworkProfile(name=name, edges=edges, **rest)
    return profiles


def lookup_network_edge(profile: NetworkProfile, src: str, dst: str) -> tuple[float, float, float]:
    if src == dst:
        return 0.0, float("inf"), 0.0
    for edge in profile.edges:
        if (edge.src == src and edge.dst == dst) or (edge.src == dst and edge.dst == src):
            return edge.rtt_ms, edge.bandwidth_mbps, edge.failure_rate
    return profile.default_rtt_ms, profile.default_bandwidth_mbps, profile.default_failure_rate


def estimate_expected_retry_latency_ms(
    profile: NetworkProfile,
    *,
    base_latency_ms: float,
    failure_rate: float,
) -> float:
    if failure_rate <= 0.0 or base_latency_ms <= 0.0:
        return 0.0
    expected_retries = failure_rate / max(1e-9, 1.0 - failure_rate)
    expected_retries = min(max(expected_retries, 0.0), max(0.0, profile.max_expected_retries))
    return expected_retries * (base_latency_ms + max(0.0, profile.retry_backoff_ms))


def estimate_network_transfer(
    profile: NetworkProfile,
    src_site: str,
    dst_site: str,
    tokens: int,
) -> dict[str, float]:
    rtt_ms, bandwidth_mbps, failure_rate = lookup_network_edge(profile, src_site, dst_site)
    if src_site == dst_site:
        return {
            "base_latency_ms": 0.0,
            "expected_retry_latency_ms": 0.0,
            "emulated_latency_ms": 0.0,
            "failure_probability": 0.0,
        }
    # Rough estimate: 1 token ~= 4 bytes; include protocol overhead with a 2x multiplier.
    message_megabits = max(tokens, 1) * 4 * 8 * 2 / 1_000_000
    transfer_ms = (message_megabits / max(bandwidth_mbps, 0.001)) * 1000
    base_latency_ms = rtt_ms + transfer_ms
    expected_retry_latency_ms = estimate_expected_retry_latency_ms(
        profile,
        base_latency_ms=base_latency_ms,
        failure_rate=failure_rate,
    )
    return {
        "base_latency_ms": base_latency_ms,
        "expected_retry_latency_ms": expected_retry_latency_ms,
        "emulated_latency_ms": base_latency_ms + expected_retry_latency_ms,
        "failure_probability": failure_rate,
    }


def estimate_message_latency_ms(
    profile: NetworkProfile,
    src_site: str,
    dst_site: str,
    tokens: int,
) -> tuple[float, float]:
    estimate = estimate_network_transfer(profile, src_site, dst_site, tokens)
    return estimate["emulated_latency_ms"], estimate["failure_probability"]


def compress_message(text: str, compression: str) -> str:
    text = text.strip()
    final_answer = extract_final_answer(text)
    compact = re.sub(r"\s+", " ", text)
    if compression == "none" or compression == "full":
        return text
    if compression == "final_only":
        return f"Final answer: {final_answer}" if final_answer else text[:160]
    if compression == "summary":
        if len(compact) <= 700:
            excerpt = compact
        else:
            excerpt = f"{compact[:350]} ... [middle omitted] ... {compact[-350:]}"
        if final_answer:
            return f"Claimed final answer: {final_answer}\nExtractive reasoning summary: {excerpt}"
        return f"Extractive reasoning summary: {excerpt}"
    if compression == "critic_brief":
        if len(compact) <= 360:
            excerpt = compact
        else:
            excerpt = f"{compact[:180]} ... [middle omitted] ... {compact[-180:]}"
        return f"Brief note for downstream agent: {excerpt}"
    return text


def add_message_trace(
    trace: RolloutTrace,
    *,
    profile: NetworkProfile,
    src_agent: str,
    dst_agent: str,
    src_site: str,
    dst_site: str,
    content: str,
    compression: str,
) -> str:
    payload = compress_message(content, compression)
    tokens = approx_tokens(payload)
    network_estimate = estimate_network_transfer(profile, src_site, dst_site, tokens)
    trace.messages.append(
        MessageTrace(
            src_agent=src_agent,
            dst_agent=dst_agent,
            src_site=src_site,
            dst_site=dst_site,
            compression=compression,
            message_tokens=tokens,
            cross_center=src_site != dst_site,
            message_latency_ms=network_estimate["base_latency_ms"],
            expected_retry_latency_ms=network_estimate["expected_retry_latency_ms"],
            emulated_latency_ms=network_estimate["emulated_latency_ms"],
            failure_probability=network_estimate["failure_probability"],
        )
    )
    return payload


def agent_by_name(candidate: dict[str, Any], name: str) -> dict[str, Any]:
    for agent in candidate.get("agents", []):
        if agent.get("name") == name:
            return agent
    raise KeyError(f"Candidate {candidate.get('name')} has no agent named {name}")


def agent_system_prompt(agent: dict[str, Any]) -> str:
    role = agent.get("role", agent.get("name", "assistant"))
    prompt = agent.get("system_prompt")
    if prompt:
        return str(prompt)
    role_prompts = {
        "planner": (
            "You are a planning agent. Produce a compact, executable plan for another solver. Identify the "
            "mathematical structure, key constraints, likely failure modes, and at least one concrete check. "
            "Do not spend tokens carrying out routine algebra and do not guess a final answer."
        ),
        "solver": (
            "You are the primary problem-solving agent. Solve the problem yourself and treat plans or peer notes "
            "as suggestions, not facts. Track constraints, test edge cases, and perform a short independent check "
            "before answering. Avoid repetitive exploration and reserve enough budget to finish. End with exactly "
            "one final-answer line in the format required by the user prompt."
        ),
        "critic": (
            "You are a critical reviewer. Identify mistakes, missing cases, and answer-format problems. "
            "Be concise and focus on verifiable issues."
        ),
        "verifier": (
            "You are an independent verifier and final decision maker. First derive the answer independently or "
            "construct decisive checks without trusting the proposed answer. Then audit the proposal for invalid "
            "assumptions, arithmetic errors, missing cases, truncation, and format errors. If evidence is "
            "insufficient, solve the problem yourself. Return the corrected answer, not a critique-only response, "
            "and end with exactly one final-answer line in the format required by the user prompt."
        ),
        "summarizer": "You are a summarizer. Compress useful reasoning into a short message for another agent.",
        "decomposer": (
            "You are a task-DAG designer. Return only JSON with `subtasks` and `aggregation_instruction`. Each "
            "subtask must have a short unique `id`, an executable `instruction`, and a `dependencies` list of "
            "earlier subtask ids. You only segment and route: never solve, calculate, guess values, declare a task "
            "unsolvable, or add facts. Copy the minimum necessary original facts verbatim. When the prompt contains "
            "numbered problems, create exactly one concise subtask per requested answer and preserve their order. "
            "Independent subtasks must have no dependencies; never invent dependencies merely to use more agents."
        ),
        "worker": (
            "You are a subtask worker. Solve only the assigned subtask using the original problem and supplied "
            "dependency results. Return a concise result with the decisive calculation or evidence. Do not answer "
            "unassigned subtasks and do not fabricate missing dependency results."
        ),
        "aggregator": (
            "You are the final aggregator. Solve any remaining gap, reconcile worker results against the original "
            "problem, preserve requested answer ordering, and return exactly one final-answer line in the format "
            "required by the user prompt. Never concatenate worker answers without checking dependencies."
        ),
    }
    return role_prompts.get(role, role_prompts["solver"])


def call_agent(
    *,
    candidate: dict[str, Any],
    agent_name: str,
    models: dict[str, ModelSpec],
    trace: RolloutTrace,
    user_content: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    agent = agent_by_name(candidate, agent_name)
    if max_tokens is None and agent.get("max_tokens") is not None:
        max_tokens = int(agent["max_tokens"])
    model_name = agent["model"]
    model_spec = models[model_name]
    messages = [
        {"role": "system", "content": agent_system_prompt(agent)},
        {"role": "user", "content": user_content},
    ]
    content, call_trace = litellm_call(
        model_spec=model_spec,
        agent_name=agent_name,
        site=agent["site"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    call_trace.output_excerpt = content[:1000]
    trace.calls.append(call_trace)
    return content, agent


def find_compression(candidate: dict[str, Any], src: str, dst: str, default: str = "summary") -> str:
    for edge in candidate.get("edges", []):
        if edge.get("src") == src and edge.get("dst") == dst:
            return str(edge.get("compression", default))
    return default


def parse_task_dag(text: str, *, max_subtasks: int = 16) -> dict[str, Any]:
    payload = parse_json_object_from_text(text) or {}
    raw_subtasks = payload.get("subtasks", [])
    subtasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(raw_subtasks, list):
        for raw in raw_subtasks[:max_subtasks]:
            if not isinstance(raw, dict):
                continue
            task_id = str(raw.get("id", "")).strip()
            instruction = str(raw.get("instruction", "")).strip()
            dependencies = raw.get("dependencies", [])
            if not task_id or not instruction or task_id in seen or not isinstance(dependencies, list):
                continue
            dependency_ids = [str(item).strip() for item in dependencies if str(item).strip() in seen]
            subtasks.append(
                {
                    "id": task_id,
                    "instruction": instruction,
                    "dependencies": dependency_ids,
                }
            )
            seen.add(task_id)
    return {
        "subtasks": subtasks,
        "aggregation_instruction": str(payload.get("aggregation_instruction", "")).strip(),
    }


def numbered_problem_dag(problem: str, expected_count: int | None = None) -> dict[str, Any] | None:
    # MASBench repeats "Problem N:" in its answer-format footer. Parse only the
    # numbered task body so those placeholders cannot become duplicate subtasks.
    body_end = len(problem)
    for marker in (r"(?im)^\s*Note:\s*In this problem set:", r"(?im)^\s*Solve all problems\b"):
        footer = re.search(marker, problem)
        if footer is not None:
            body_end = min(body_end, footer.start())
    problem_body = problem[:body_end]
    matches = list(re.finditer(r"(?im)(?:^|\n)\s*Problem\s+(\d+)\s*[:.]", problem_body))
    if not matches or (expected_count is not None and len(matches) != expected_count):
        return None
    subtasks = []
    for idx, match in enumerate(matches):
        problem_number = int(match.group(1))
        if problem_number != idx + 1:
            return None
        subtasks.append(
            {
                "id": f"p{problem_number}",
                "instruction": (
                    f"Solve only the answer requested by numbered Problem {problem_number}. "
                    "Use any relevant facts from the complete shared task below because definitions may be "
                    "stated in other numbered blocks. Return exactly one scalar modulo-23 answer for your assigned "
                    "problem; do not answer the other numbered problems.\n\n"
                    "Complete shared task:\n"
                    + problem_body.strip()
                ),
                "dependencies": [],
            }
        )
    return {
        "subtasks": subtasks,
        "aggregation_instruction": (
            "Return the subtask answers in increasing Problem number, joined by <<horizon>>."
        ),
    }


def run_dag_decompose_architecture(
    *,
    candidate: dict[str, Any],
    problem: str,
    dataset_name: str,
    axis: str,
    axis_value: str,
    models: dict[str, ModelSpec],
    profile: NetworkProfile,
    trace: RolloutTrace,
) -> str:
    base_prompt = task_instruction(dataset_name) + "Problem:\n" + problem
    decomposer = agent_by_name(candidate, "decomposer")
    plan = None
    decomposition_mode = "llm"
    if dataset_name == "masbench" and axis == "parallel":
        try:
            expected_count = int(axis_value)
        except ValueError:
            expected_count = None
        plan = numbered_problem_dag(problem, expected_count)
        if plan is not None:
            decomposition_mode = "protocol"
    critical_path_latency_ms = 0.0
    if plan is None:
        decomposer_output, decomposer = call_agent(
            candidate=candidate,
            agent_name="decomposer",
            models=models,
            trace=trace,
            user_content=(
                base_prompt
                + "\n\nDesign the smallest useful execution DAG. Return JSON only:\n"
                '{"subtasks":[{"id":"s1","instruction":"...","dependencies":[]}],'
                '"aggregation_instruction":"..."}'
            ),
        )
        critical_path_latency_ms = trace.calls[-1].observed_latency_ms
        plan = parse_task_dag(decomposer_output)
    decomposition_fallback = not plan["subtasks"]
    if decomposition_fallback:
        plan["subtasks"] = [
            {
                "id": "s1",
                "instruction": "Solve the complete original problem and return the decisive result.",
                "dependencies": [],
            }
        ]
    trace.artifacts["dag"] = {
        "axis": axis,
        "mode": decomposition_mode,
        "fallback": decomposition_fallback,
        "subtask_count": len(plan["subtasks"]),
        "plan": plan,
    }
    worker_names = [
        str(agent["name"])
        for agent in candidate.get("agents", [])
        if agent.get("role") == "worker"
    ]
    if not worker_names:
        raise ValueError("dag_decompose requires at least one worker agent")

    pending = {task["id"]: task for task in plan["subtasks"]}
    results: dict[str, str] = {}
    task_workers: dict[str, str] = {}
    while pending:
        ready = [
            task
            for task in pending.values()
            if all(dependency in results for dependency in task["dependencies"])
        ]
        if not ready:
            # Invalid or cyclic plans degrade to independently executable tasks.
            ready = [next(iter(pending.values()))]
            ready[0]["dependencies"] = []

        jobs: list[tuple[str, str, str, RolloutTrace]] = []
        for offset, task in enumerate(ready):
            worker_name = worker_names[(len(results) + offset) % len(worker_names)]
            worker = agent_by_name(candidate, worker_name)
            plan_message = add_message_trace(
                trace,
                profile=profile,
                src_agent="decomposer",
                dst_agent=worker_name,
                src_site=decomposer["site"],
                dst_site=worker["site"],
                content=task["instruction"],
                compression=find_compression(candidate, "decomposer", worker_name, default="full"),
            )
            dependency_sections = []
            for dependency in task["dependencies"]:
                source_worker_name = task_workers[dependency]
                source_worker = agent_by_name(candidate, source_worker_name)
                dependency_message = add_message_trace(
                    trace,
                    profile=profile,
                    src_agent=source_worker_name,
                    dst_agent=worker_name,
                    src_site=source_worker["site"],
                    dst_site=worker["site"],
                    content=results[dependency],
                    compression=find_compression(candidate, source_worker_name, worker_name, default="summary"),
                )
                dependency_sections.append(f"{dependency}: {dependency_message}")
            worker_prompt = (
                task_instruction(dataset_name)
                + "Original problem:\n"
                + problem
                + "\n\nAssigned subtask:\n"
                + plan_message
            )
            if dependency_sections:
                worker_prompt += "\n\nDependency results:\n" + "\n".join(dependency_sections)
            jobs.append((task["id"], worker_name, worker_prompt, RolloutTrace()))

        def execute_job(job: tuple[str, str, str, RolloutTrace]) -> tuple[str, str, str, RolloutTrace]:
            task_id, worker_name, worker_prompt, worker_trace = job
            output, _ = call_agent(
                candidate=candidate,
                agent_name=worker_name,
                models=models,
                trace=worker_trace,
                user_content=worker_prompt,
            )
            return task_id, worker_name, output, worker_trace

        round_start = time.time()
        with ThreadPoolExecutor(max_workers=min(len(jobs), len(worker_names))) as executor:
            completed = list(executor.map(execute_job, jobs))
        critical_path_latency_ms += (time.time() - round_start) * 1000
        for task_id, worker_name, output, worker_trace in completed:
            trace.calls.extend(worker_trace.calls)
            trace.messages.extend(worker_trace.messages)
            results[task_id] = output
            task_workers[task_id] = worker_name
            pending.pop(task_id, None)

    aggregator = agent_by_name(candidate, "aggregator")
    result_sections = []
    for task in plan["subtasks"]:
        task_id = task["id"]
        source_worker_name = task_workers[task_id]
        source_worker = agent_by_name(candidate, source_worker_name)
        result_message = add_message_trace(
            trace,
            profile=profile,
            src_agent=source_worker_name,
            dst_agent="aggregator",
            src_site=source_worker["site"],
            dst_site=aggregator["site"],
            content=results[task_id],
            compression=find_compression(candidate, source_worker_name, "aggregator", default="summary"),
        )
        result_sections.append(f"{task_id} ({task['instruction']}):\n{result_message}")
    aggregator_prompt = (
        base_prompt
        + "\n\nAggregation instruction:\n"
        + (plan["aggregation_instruction"] or "Use the subtask results to produce the requested final answer.")
        + "\n\nSubtask results:\n"
        + "\n\n".join(result_sections)
    )
    aggregator_output, _ = call_agent(
        candidate=candidate,
        agent_name="aggregator",
        models=models,
        trace=trace,
        user_content=aggregator_prompt,
    )
    critical_path_latency_ms += trace.calls[-1].observed_latency_ms
    trace.observed_model_wall_latency_ms = critical_path_latency_ms
    return aggregator_output


def run_single_architecture(
    *,
    candidate: dict[str, Any],
    example: dict[str, Any],
    models: dict[str, ModelSpec],
    profile: NetworkProfile,
) -> tuple[str, RolloutTrace]:
    topology = candidate.get("topology", "single")
    trace = RolloutTrace()
    problem = example["input"].strip()
    dataset_name = str(example.get("dataset", "aime"))
    base_prompt = task_instruction(dataset_name) + "Problem:\n" + problem

    if topology == "single":
        output, _ = call_agent(
            candidate=candidate,
            agent_name="solver",
            models=models,
            trace=trace,
            user_content=base_prompt,
        )
        return output, trace

    if topology == "self_consistency":
        outputs: list[str] = []
        solver_agent = agent_by_name(candidate, "solver")
        samples = int(candidate.get("samples", 3))
        for idx in range(samples):
            strategies = [
                "Use a direct constructive or algebraic route.",
                "Use an independent counting, invariant, or complementary route and check boundary cases.",
                "Try to falsify likely answers with a small-case or reverse-substitution check before solving.",
            ]
            strategy = strategies[idx % len(strategies)]
            suffix = f"\nIndependent attempt {idx + 1}/{samples}: {strategy}"
            output, _ = call_agent(
                candidate=candidate,
                agent_name="solver",
                models=models,
                trace=trace,
                user_content=base_prompt + suffix,
                temperature=float(candidate.get("temperature", 0.3)),
            )
            outputs.append(output)
            if idx > 0:
                add_message_trace(
                    trace,
                    profile=profile,
                    src_agent="solver",
                    dst_agent="solver",
                    src_site=solver_agent["site"],
                    dst_site=solver_agent["site"],
                    content=output,
                    compression="final_only",
                )
        answer_counts: dict[str, int] = {}
        for output in outputs:
            answer = extract_final_answer(output)
            if answer:
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
        if answer_counts:
            best_answer = sorted(answer_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if dataset_name == "masbench":
                return f"Selected by majority over {samples} attempts.\nFINAL ANSWER: {best_answer}", trace
            if dataset_name == "gaia":
                return f"Selected by majority over {samples} attempts.\nFINAL ANSWER: {best_answer}", trace
            return f"### {best_answer}\nSelected by majority over {samples} attempts.", trace
        return outputs[-1] if outputs else "", trace

    if topology == "solver_verifier":
        solver_output, solver_agent = call_agent(
            candidate=candidate,
            agent_name="solver",
            models=models,
            trace=trace,
            user_content=base_prompt,
        )
        verifier_agent = agent_by_name(candidate, "verifier")
        compression = find_compression(candidate, "solver", "verifier", default="summary")
        solver_msg = add_message_trace(
            trace,
            profile=profile,
            src_agent="solver",
            dst_agent="verifier",
            src_site=solver_agent["site"],
            dst_site=verifier_agent["site"],
            content=solver_output,
            compression=compression,
        )
        verifier_prompt = (
            task_instruction(dataset_name)
            + "Problem:\n"
            + problem
            + "\n\nA solver proposed the following answer/reasoning. It may be wrong or truncated:\n"
            + solver_msg
            + "\n\nIndependently establish the result, identify any decisive discrepancy, and output the corrected "
            "final answer. Do not preserve the solver's answer merely because it is presented."
        )
        verifier_output, _ = call_agent(
            candidate=candidate,
            agent_name="verifier",
            models=models,
            trace=trace,
            user_content=verifier_prompt,
        )
        return verifier_output or solver_output, trace

    if topology == "planner_solver_verifier":
        planner_output, planner_agent = call_agent(
            candidate=candidate,
            agent_name="planner",
            models=models,
            trace=trace,
            user_content=(
                "Create a compact solution plan for this task. State the main method, critical constraints, "
                "likely traps, and a concrete verification step. Do not guess the final answer.\n\nProblem:\n"
                + problem
            ),
            max_tokens=int(candidate["planner_max_tokens"]) if candidate.get("planner_max_tokens") else None,
        )
        solver_agent = agent_by_name(candidate, "solver")
        planner_msg = add_message_trace(
            trace,
            profile=profile,
            src_agent="planner",
            dst_agent="solver",
            src_site=planner_agent["site"],
            dst_site=solver_agent["site"],
            content=planner_output,
            compression=find_compression(candidate, "planner", "solver", default="summary"),
        )
        solver_prompt = (
            base_prompt
            + "\n\nA planner supplied the following possibly incomplete plan:\n"
            + planner_msg
            + "\n\nValidate the plan before using it, repair any gap, solve the problem, "
            "and perform its proposed check."
        )
        solver_output, solver_agent = call_agent(
            candidate=candidate,
            agent_name="solver",
            models=models,
            trace=trace,
            user_content=solver_prompt,
        )
        verifier_agent = agent_by_name(candidate, "verifier")
        solver_msg = add_message_trace(
            trace,
            profile=profile,
            src_agent="solver",
            dst_agent="verifier",
            src_site=solver_agent["site"],
            dst_site=verifier_agent["site"],
            content=solver_output,
            compression=find_compression(candidate, "solver", "verifier", default="summary"),
        )
        verifier_prompt = (
            task_instruction(dataset_name)
            + "Problem:\n"
            + problem
            + "\n\nPlan:\n"
            + planner_msg
            + "\n\nSolver output:\n"
            + solver_msg
            + "\n\nIndependently verify the result. Check whether the solver followed a valid plan, repair any "
            "specific error, and produce the corrected final answer."
        )
        verifier_output, _ = call_agent(
            candidate=candidate,
            agent_name="verifier",
            models=models,
            trace=trace,
            user_content=verifier_prompt,
        )
        return verifier_output or solver_output, trace

    if topology == "debate":
        solver_a_output, solver_a = call_agent(
            candidate=candidate,
            agent_name="solver_a",
            models=models,
            trace=trace,
            user_content=(
                base_prompt
                + "\nYou are Solver A. Use a direct constructive/algebraic route and include one numerical or "
                "boundary-case check before the final answer."
            ),
        )
        solver_b_output, solver_b = call_agent(
            candidate=candidate,
            agent_name="solver_b",
            models=models,
            trace=trace,
            user_content=(
                base_prompt
                + "\nYou are Solver B. Work independently using a different representation, complementary count, "
                "or adversarial error search. Do not imitate an unseen peer."
            ),
        )
        verifier_agent = agent_by_name(candidate, "verifier")
        msg_a = add_message_trace(
            trace,
            profile=profile,
            src_agent="solver_a",
            dst_agent="verifier",
            src_site=solver_a["site"],
            dst_site=verifier_agent["site"],
            content=solver_a_output,
            compression=find_compression(candidate, "solver_a", "verifier", default="summary"),
        )
        msg_b = add_message_trace(
            trace,
            profile=profile,
            src_agent="solver_b",
            dst_agent="verifier",
            src_site=solver_b["site"],
            dst_site=verifier_agent["site"],
            content=solver_b_output,
            compression=find_compression(candidate, "solver_b", "verifier", default="summary"),
        )
        verifier_prompt = (
            task_instruction(dataset_name)
            + "Problem:\n"
            + problem
            + "\n\nTwo agents proposed answers:\nAgent A:\n"
            + msg_a
            + "\n\nAgent B:\n"
            + msg_b
            + "\n\nDo not decide by majority or confidence. Independently check the disputed steps and return the "
            "answer supported by valid reasoning; if both are wrong, solve it yourself."
        )
        verifier_output, _ = call_agent(
            candidate=candidate,
            agent_name="verifier",
            models=models,
            trace=trace,
            user_content=verifier_prompt,
        )
        return verifier_output or solver_a_output or solver_b_output, trace

    if topology == "dag_decompose":
        output = run_dag_decompose_architecture(
            candidate=candidate,
            problem=problem,
            dataset_name=dataset_name,
            axis=str(example.get("axis", "")),
            axis_value=str(example.get("axis_value", "")),
            models=models,
            profile=profile,
            trace=trace,
        )
        return output, trace

    raise ValueError(f"Unsupported topology: {topology}")


def evaluate_candidate(
    *,
    candidate: dict[str, Any],
    dataset: list[dict[str, Any]],
    models: dict[str, ModelSpec],
    profile: NetworkProfile,
    max_examples: int | None = None,
    capture_outputs: bool = False,
    eval_concurrency: int = 1,
    reflection_example_limit: int = 3,
) -> dict[str, Any]:
    examples = dataset[: max_examples or len(dataset)]
    scores: list[float] = []
    component_scores: list[float] = []
    valid_answer_protocol: list[float] = []
    valid_executions: list[float] = []
    unextractable_answers: list[float] = []
    truncated_unextractable_answers: list[float] = []
    trace_summaries: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []

    def evaluate_example(example: dict[str, Any]) -> tuple[str, RolloutTrace]:
        return run_single_architecture(candidate=candidate, example=example, models=models, profile=profile)

    worker_count = max(1, min(int(eval_concurrency), len(examples))) if examples else 1
    if worker_count == 1:
        evaluated_examples = [evaluate_example(example) for example in examples]
    else:
        # vLLM continuously batches concurrent HTTP requests; map preserves dataset order for reproducible outputs.
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="wan-gepa-eval") as executor:
            evaluated_examples = list(executor.map(evaluate_example, examples))

    for example, (output, trace) in zip(examples, evaluated_examples, strict=True):
        dataset_name = str(example.get("dataset", "aime"))
        prediction = extract_prediction_for_dataset(output, dataset_name)
        protocol_valid = bool(prediction)
        execution_valid = not any(call.error for call in trace.calls)
        truncated = any(call.finish_reason == "length" for call in trace.calls)
        score = score_example_answer(output, str(example["answer"]), dataset_name)
        component_score = score_answer_components(output, str(example["answer"]), dataset_name)
        scores.append(score)
        component_scores.append(component_score)
        valid_answer_protocol.append(float(protocol_valid))
        valid_executions.append(float(execution_valid))
        unextractable_answers.append(float(not protocol_valid))
        truncated_unextractable_answers.append(float(truncated and not protocol_valid))
        summary = trace.summary(profile)
        trace_summaries.append(summary)
        if score <= 0.0 and len(failure_examples) < max(0, reflection_example_limit):
            agent_roles = {
                str(agent.get("name", "")): str(agent.get("role", agent.get("name", "assistant")))
                for agent in candidate.get("agents", [])
            }
            failure_examples.append(
                {
                    "id": example.get("id"),
                    "input": str(example.get("input", ""))[:2400],
                    "gold_answer": str(example["answer"]),
                    "prediction": prediction,
                    "component_score": component_score,
                    "axis": example.get("axis", ""),
                    "axis_value": example.get("axis_value", ""),
                    "official_split": example.get("official_split", ""),
                    "answer_protocol_valid": protocol_valid,
                    "final_output_excerpt": output[-1200:],
                    "agent_outputs": [
                        {
                            "agent": call.agent,
                            "role": agent_roles.get(call.agent, call.agent),
                            "model": call.model,
                            "site": call.site,
                            "output_excerpt": call.output_excerpt,
                            "finish_reason": call.finish_reason,
                            "error": call.error,
                        }
                        for call in trace.calls
                    ],
                    "trace": summary,
                }
            )
        if capture_outputs:
            outputs.append(
                {
                    "id": example.get("id"),
                    "answer": str(example["answer"]),
                    "prediction": prediction,
                    "dataset": dataset_name,
                    "axis": example.get("axis", ""),
                    "axis_value": example.get("axis_value", ""),
                    "official_split": example.get("official_split", ""),
                    "answer_protocol_valid": protocol_valid,
                    "attachment": example.get("attachment", ""),
                    "score": score,
                    "component_score": component_score,
                    "output": output,
                    "trace": summary,
                }
            )
    aggregate = aggregate_trace_summaries(trace_summaries)
    dag_artifacts = [
        summary.get("artifacts", {}).get("dag")
        for summary in trace_summaries
        if summary.get("artifacts", {}).get("dag")
    ]
    aggregate.update(
        {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "num_examples": len(scores),
            "scores": scores,
            "component_scores": component_scores,
            "component_score": (
                sum(component_scores) / len(component_scores) if component_scores else 0.0
            ),
            "correct": int(sum(scores)),
            "valid_answer_rate": (
                sum(valid_answer_protocol) / len(valid_answer_protocol) if valid_answer_protocol else 0.0
            ),
            "valid_execution_rate": (
                sum(valid_executions) / len(valid_executions) if valid_executions else 0.0
            ),
            "error_example_rate": (
                1.0 - sum(valid_executions) / len(valid_executions) if valid_executions else 0.0
            ),
            "unextractable_answer_rate": (
                sum(unextractable_answers) / len(unextractable_answers) if unextractable_answers else 0.0
            ),
            "truncated_unextractable_rate": (
                sum(truncated_unextractable_answers) / len(truncated_unextractable_answers)
                if truncated_unextractable_answers
                else 0.0
            ),
            "avg_dag_subtasks": (
                sum(float(item.get("subtask_count", 0)) for item in dag_artifacts) / len(dag_artifacts)
                if dag_artifacts
                else 0.0
            ),
            "dag_fallback_rate": (
                sum(float(bool(item.get("fallback"))) for item in dag_artifacts) / len(dag_artifacts)
                if dag_artifacts
                else 0.0
            ),
            "failure_examples": failure_examples,
        }
    )
    if capture_outputs:
        aggregate["outputs"] = outputs
    return aggregate


def model_cache_fingerprint(models: dict[str, ModelSpec]) -> dict[str, Any]:
    return {
        name: {
            "litellm_model": resolve_litellm_model(spec),
            "completion_kwargs": spec.completion_kwargs,
            "disable_qwen_thinking": should_disable_qwen_thinking(spec),
            "input_cost_per_million_tokens_usd": spec.input_cost_per_million_tokens_usd,
            "output_cost_per_million_tokens_usd": spec.output_cost_per_million_tokens_usd,
            "pricing_source": spec.pricing_source,
        }
        for name, spec in sorted(models.items())
    }


def evaluation_cache_key(
    *,
    candidate: dict[str, Any],
    examples: list[dict[str, Any]],
    models: dict[str, ModelSpec],
    profile: NetworkProfile,
    max_examples: int | None,
    reflection_example_limit: int,
    capture_outputs: bool,
) -> str:
    selected_examples = examples[: max_examples or len(examples)]
    payload = {
        "version": EVALUATION_CACHE_VERSION,
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "prompt_mode": os.environ.get("GEPA_PHASE2_PROMPT_MODE", "deliberate"),
        "mock_model": os.environ.get("GEPA_PHASE2_MOCK_MODEL", "").lower() in {"1", "true", "yes"},
        "candidate": candidate_structure(candidate),
        "examples": [
            {
                "id": example.get("id"),
                "input": example.get("input"),
                "answer": example.get("answer"),
                "dataset": example.get("dataset"),
                "attachment": example.get("attachment", ""),
            }
            for example in selected_examples
        ],
        "models": model_cache_fingerprint(models),
        "network_profile": asdict(profile),
        "reflection_example_limit": reflection_example_limit,
        "capture_outputs": capture_outputs,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_cached_evaluation(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = read_json(cache_path)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("cache_version") != EVALUATION_CACHE_VERSION:
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def local_process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cache_lock_is_stale(lock_path: Path, *, stale_seconds: float) -> bool:
    try:
        if time.time() - lock_path.stat().st_mtime > stale_seconds:
            return True
        payload = read_json(lock_path)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return True
    owner_pid = payload.get("pid")
    owner_host = str(payload.get("hostname", socket.gethostname()))
    if owner_host == socket.gethostname() and isinstance(owner_pid, int):
        return not local_process_is_alive(owner_pid)
    return False


def acquire_cache_lock(lock_path: Path, *, stale_seconds: float = 21600.0) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if cache_lock_is_stale(lock_path, stale_seconds=stale_seconds):
                lock_path.unlink()
                return acquire_cache_lock(lock_path, stale_seconds=stale_seconds)
        except FileNotFoundError:
            return acquire_cache_lock(lock_path, stale_seconds=stale_seconds)
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as fout:
        fout.write(
            json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": time.time()})
        )
    return True


def evaluate_candidate_cached(
    *,
    candidate: dict[str, Any],
    dataset: list[dict[str, Any]],
    models: dict[str, ModelSpec],
    profile: NetworkProfile,
    cache_dir: Path | None,
    max_examples: int | None = None,
    capture_outputs: bool = False,
    eval_concurrency: int = 1,
    reflection_example_limit: int = 3,
    lock_poll_seconds: float = 1.0,
) -> tuple[dict[str, Any], str]:
    if cache_dir is None:
        result = evaluate_candidate(
            candidate=candidate,
            dataset=dataset,
            models=models,
            profile=profile,
            max_examples=max_examples,
            capture_outputs=capture_outputs,
            eval_concurrency=eval_concurrency,
            reflection_example_limit=reflection_example_limit,
        )
        return result, "disabled"

    cache_key = evaluation_cache_key(
        candidate=candidate,
        examples=dataset,
        models=models,
        profile=profile,
        max_examples=max_examples,
        reflection_example_limit=reflection_example_limit,
        capture_outputs=capture_outputs,
    )
    cache_path = cache_dir / f"{cache_key}.json"
    lock_path = cache_dir / f"{cache_key}.lock"
    while True:
        cached = read_cached_evaluation(cache_path)
        if cached is not None:
            return cached, "hit"
        if acquire_cache_lock(lock_path):
            break
        time.sleep(max(0.05, lock_poll_seconds))

    try:
        cached = read_cached_evaluation(cache_path)
        if cached is not None:
            return cached, "hit_after_wait"
        result = evaluate_candidate(
            candidate=candidate,
            dataset=dataset,
            models=models,
            profile=profile,
            max_examples=max_examples,
            capture_outputs=capture_outputs,
            eval_concurrency=eval_concurrency,
            reflection_example_limit=reflection_example_limit,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        write_json(
            temporary_path,
            {
                "cache_version": EVALUATION_CACHE_VERSION,
                "cache_key": cache_key,
                "candidate_id": candidate.get("id"),
                "candidate_name": candidate.get("name"),
                "result": result,
            },
        )
        temporary_path.replace(cache_path)
        return result, "miss"
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def aggregate_trace_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {
            "avg_calls": 0.0,
            "avg_total_tokens": 0.0,
            "avg_inference_cost_usd": 0.0,
            "avg_cross_center_tokens": 0.0,
            "avg_network_latency_ms": 0.0,
            "avg_emulated_latency_ms": 0.0,
            "avg_errors": 0.0,
        }
    keys = [
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "input_cost_usd",
        "output_cost_usd",
        "inference_cost_usd",
        "maxed_calls",
        "total_tokens",
        "requested_max_tokens",
        "cross_center_tokens",
        "cross_center_message_tokens",
        "cross_center_model_tokens",
        "local_message_tokens",
        "message_count",
        "cross_center_message_count",
        "observed_model_latency_ms",
        "observed_model_wall_latency_ms",
        "parallel_model_latency_savings_ms",
        "message_latency_ms",
        "message_expected_retry_latency_ms",
        "model_rpc_latency_ms",
        "model_rpc_expected_retry_latency_ms",
        "expected_retry_latency_ms",
        "network_latency_ms",
        "site_compute_latency_ms",
        "emulated_latency_ms",
        "emulated_wall_latency_ms",
        "expected_network_failures",
        "errors",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        values = [float(summary.get(key, 0.0)) for summary in summaries]
        result[f"avg_{key}"] = sum(values) / len(values)
        result[f"sum_{key}"] = sum(values)
    return result


def candidate_validity(
    row: dict[str, Any],
    *,
    contract_errors: list[str] | None = None,
) -> dict[str, Any]:
    reasons = list(contract_errors or row.get("candidate_contract_errors", []))
    avg_errors = max(0.0, float(row.get("avg_errors", 0.0)))
    valid_execution_rate = float(row.get("valid_execution_rate", max(0.0, 1.0 - min(1.0, avg_errors))))
    error_example_rate = float(row.get("error_example_rate", min(1.0, avg_errors)))
    valid_answer_rate = float(row.get("valid_answer_rate", 1.0))
    protocol_error_rate = max(0.0, 1.0 - valid_answer_rate)
    truncated_unextractable_rate = float(row.get("truncated_unextractable_rate", 0.0))
    if valid_execution_rate < MIN_VALID_EXECUTION_RATE:
        reasons.append(f"valid_execution_rate<{MIN_VALID_EXECUTION_RATE:g}:{valid_execution_rate:.6f}")
    if error_example_rate > MAX_ERROR_EXAMPLE_RATE:
        reasons.append(f"error_example_rate>{MAX_ERROR_EXAMPLE_RATE:g}:{error_example_rate:.6f}")
    if protocol_error_rate > MAX_PROTOCOL_ERROR_RATE:
        reasons.append(f"protocol_error_rate>{MAX_PROTOCOL_ERROR_RATE:g}:{protocol_error_rate:.6f}")
    if truncated_unextractable_rate > MAX_TRUNCATED_UNEXTRACTABLE_RATE:
        reasons.append(
            "truncated_unextractable_rate>"
            f"{MAX_TRUNCATED_UNEXTRACTABLE_RATE:g}:{truncated_unextractable_rate:.6f}"
        )
    unique_reasons = sorted({str(reason) for reason in reasons if reason})
    return {
        "is_valid_candidate": not unique_reasons,
        "invalid_reasons": unique_reasons,
        "candidate_contract_errors": sorted(set(contract_errors or row.get("candidate_contract_errors", []))),
        "valid_execution_rate": valid_execution_rate,
        "error_example_rate": error_example_rate,
        "protocol_error_rate": protocol_error_rate,
        "truncated_unextractable_rate": truncated_unextractable_rate,
    }


def is_candidate_eligible(row: dict[str, Any]) -> bool:
    if "is_valid_candidate" in row:
        return bool(row["is_valid_candidate"])
    return bool(candidate_validity(row)["is_valid_candidate"])


def eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if is_candidate_eligible(row)]


def normalize_objectives(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(row.get("score", 0.0)),
        -float(row.get("avg_errors", 0.0)),
        -float(row.get("avg_total_tokens", 0.0)),
        -float(row.get("avg_inference_cost_usd", 0.0)),
        -float(row.get("avg_cross_center_tokens", 0.0)),
        -float(row.get("avg_network_latency_ms", row.get("avg_emulated_latency_ms", 0.0))),
        -float(row.get("avg_calls", 0.0)),
    )


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_obj = normalize_objectives(left)
    right_obj = normalize_objectives(right)
    return all(a >= b for a, b in zip(left_obj, right_obj, strict=True)) and any(
        a > b for a, b in zip(left_obj, right_obj, strict=True)
    )


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = eligible_rows(rows)
    front: list[dict[str, Any]] = []
    for row in rows:
        if not any(dominates(other, row) for other in rows if other is not row):
            front.append(row)
    return front


def quality_operating_point_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        -float(row.get("score", 0.0)),
        float(row.get("avg_total_tokens", 0.0)),
        float(row.get("avg_calls", 0.0)),
        float(row.get("avg_inference_cost_usd", 0.0)),
        float(row.get("avg_cross_center_tokens", 0.0)),
        str(row.get("candidate_id", "")),
    )


def efficiency_operating_point_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(row.get("avg_total_tokens", 0.0)),
        float(row.get("avg_calls", 0.0)),
        float(row.get("avg_inference_cost_usd", 0.0)),
        float(row.get("avg_cross_center_tokens", 0.0)),
        str(row.get("candidate_id", "")),
    )


def select_operating_points(
    selection_rows: list[dict[str, Any]],
    *,
    efficiency_quality_delta: float = EFFICIENCY_QUALITY_DELTA,
) -> dict[str, dict[str, Any]]:
    valid_rows = eligible_rows(selection_rows)
    if not valid_rows:
        raise RuntimeError("No valid D_select candidate remains after applying the protocol validity gates")
    quality = min(valid_rows, key=quality_operating_point_key)
    quality_threshold = float(quality.get("score", 0.0)) - max(0.0, efficiency_quality_delta)
    efficient_pool = [
        row
        for row in pareto_front(valid_rows)
        if float(row.get("score", 0.0)) >= quality_threshold
    ]
    if not efficient_pool:
        efficient_pool = [quality]
    efficiency = min(efficient_pool, key=efficiency_operating_point_key)
    return {"quality": quality, "efficiency": efficiency}


def utility_score(row: dict[str, Any], mode: str) -> float:
    score = float(row.get("score", 0.0))
    if mode in {"quality_only", "aflow_style", "adas_style"}:
        return score
    total_tokens = float(row.get("avg_total_tokens", 0.0))
    cross_tokens = float(row.get("avg_cross_center_tokens", 0.0))
    network_latency = float(row.get("avg_network_latency_ms", row.get("avg_emulated_latency_ms", 0.0)))
    emulated_latency = float(row.get("avg_emulated_latency_ms", 0.0))
    calls = float(row.get("avg_calls", 0.0))
    inference_cost = float(row.get("avg_inference_cost_usd", 0.0))
    return (
        score
        - 0.00002 * total_tokens
        - inference_cost
        - 0.0002 * cross_tokens
        - 0.000005 * network_latency
        - 0.0000002 * emulated_latency
        - 0.01 * calls
    )


def quality_cost_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    return (
        float(row.get("avg_errors", 0.0)),
        float(row.get("avg_cross_center_tokens", 0.0)),
        float(row.get("avg_network_latency_ms", row.get("avg_emulated_latency_ms", 0.0))),
        float(row.get("avg_total_tokens", 0.0)),
        float(row.get("avg_inference_cost_usd", 0.0)),
        float(row.get("avg_emulated_latency_ms", 0.0)),
        float(row.get("avg_calls", 0.0)),
        -float(row.get("score", 0.0)),
    )


def rank_quality_band_cost(rows: list[dict[str, Any]], quality_band: float) -> list[dict[str, Any]]:
    rows = eligible_rows(rows)
    if not rows:
        return []
    best_score = max(float(row.get("score", 0.0)) for row in rows)
    threshold = best_score - max(0.0, quality_band)
    eligible = [row for row in rows if float(row.get("score", 0.0)) >= threshold]
    ineligible = [row for row in rows if float(row.get("score", 0.0)) < threshold]
    return sorted(eligible, key=quality_cost_key) + sorted(
        ineligible,
        key=lambda row: (-float(row.get("score", 0.0)), *quality_cost_key(row)),
    )


def select_rows_for_test(
    evaluated_rows: list[dict[str, Any]],
    *,
    mode: str,
    test_top_k: int,
    selection_strategy: str,
    quality_band: float,
) -> tuple[list[dict[str, Any]], str]:
    evaluated_rows = eligible_rows(evaluated_rows)
    if not evaluated_rows:
        raise RuntimeError("No valid candidate remains for selection")
    if mode == "quality_only":
        ranked = sorted(evaluated_rows, key=lambda row: utility_score(row, mode), reverse=True)
        return ranked[: max(1, test_top_k)], "validation_score_top_k"
    candidate_pool = pareto_front(evaluated_rows) if mode == "wan_pareto" else evaluated_rows
    if selection_strategy == "quality_band_cost":
        ranked = rank_quality_band_cost(candidate_pool, quality_band)
        policy_prefix = "pareto_front_" if mode == "wan_pareto" else "validation_"
        return (
            ranked[: max(1, test_top_k)],
            f"{policy_prefix}quality_band_cost_top_k.band={quality_band:g}",
        )
    ranked = sorted(candidate_pool, key=lambda row: utility_score(row, mode), reverse=True)
    policy = "pareto_front_utility_top_k" if mode == "wan_pareto" else "validation_utility_top_k"
    return ranked[: max(1, test_top_k)], policy


def shortlist_rows_for_selection(
    evaluated_rows: list[dict[str, Any]],
    *,
    mode: str,
    shortlist_size: int,
    selection_strategy: str,
    quality_band: float,
) -> tuple[list[dict[str, Any]], str]:
    """Select candidates on D_search that will be reevaluated on disjoint D_select."""
    shortlisted, policy = select_rows_for_test(
        evaluated_rows,
        mode=mode,
        test_top_k=max(1, shortlist_size),
        selection_strategy=selection_strategy,
        quality_band=quality_band,
    )
    return shortlisted, policy.replace("top_k", "selection_shortlist")


def default_agent_max_tokens(config: dict[str, Any], topology: str, role: str) -> int | None:
    defaults = config.get("defaults", {})
    role_budgets = defaults.get("agent_max_tokens", {})
    topology_budgets = defaults.get("topology_agent_max_tokens", {}).get(topology, {})
    value = topology_budgets.get(role, role_budgets.get(role))
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def max_tokens_pool_for_role(config: dict[str, Any], role: str, topology: str | None = None) -> list[int]:
    search = config.get("search", {})
    topology_pools = search.get("topology_max_tokens_pool", {})
    topology_pool = topology_pools.get(topology, {}) if topology else {}
    raw_pool = topology_pool if role in topology_pool else search.get("max_tokens_pool", {})
    values = raw_pool.get(role, []) if isinstance(raw_pool, dict) else raw_pool
    return sorted({int(value) for value in values if int(value) > 0})


def make_agent(
    name: str,
    role: str,
    model: str,
    site: str,
    *,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    agent = {"name": name, "role": role, "model": model, "site": site}
    if max_tokens is not None:
        agent["max_tokens"] = int(max_tokens)
    return agent


def make_configured_agent(
    config: dict[str, Any],
    topology: str,
    name: str,
    role: str,
    model: str,
    site: str,
) -> dict[str, Any]:
    return make_agent(
        name,
        role,
        model,
        site,
        max_tokens=default_agent_max_tokens(config, topology, role),
    )


def seed_architectures(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = config.get("defaults", {})
    local_model = defaults["local_model"]
    strong_model = defaults.get("strong_model", local_model)
    cheap_model = defaults.get("cheap_model", local_model)
    local_site = defaults.get("local_site", "center_a")
    remote_site = defaults.get("remote_site", "center_c")
    secondary_site = defaults.get("secondary_site", "center_b")
    dag_decomposer_model = defaults.get("dag_decomposer_model", cheap_model)
    dag_worker_models = list(defaults.get("dag_worker_models", [local_model, local_model, strong_model]))
    while len(dag_worker_models) < 3:
        dag_worker_models.append(dag_worker_models[-1] if dag_worker_models else local_model)
    dag_aggregator_model = defaults.get("dag_aggregator_model", strong_model)
    seeds = [
        {
            "name": "single_local",
            "topology": "single",
            "agents": [make_configured_agent(config, "single", "solver", "solver", local_model, local_site)],
            "edges": [],
        },
        {
            "name": "single_strong_remote",
            "topology": "single",
            "agents": [make_configured_agent(config, "single", "solver", "solver", strong_model, remote_site)],
            "edges": [],
        },
        {
            "name": "self_consistency_local",
            "topology": "self_consistency",
            "samples": 3,
            "temperature": 0.3,
            "agents": [
                make_configured_agent(
                    config,
                    "self_consistency",
                    "solver",
                    "solver",
                    local_model,
                    local_site,
                )
            ],
            "edges": [],
        },
        {
            "name": "solver_verifier_local",
            "topology": "solver_verifier",
            "agents": [
                make_configured_agent(
                    config,
                    "solver_verifier",
                    "solver",
                    "solver",
                    local_model,
                    local_site,
                ),
                make_configured_agent(
                    config,
                    "solver_verifier",
                    "verifier",
                    "verifier",
                    local_model,
                    local_site,
                ),
            ],
            "edges": [{"src": "solver", "dst": "verifier", "compression": "full"}],
        },
        {
            "name": "solver_local_verifier_remote_summary",
            "topology": "solver_verifier",
            "agents": [
                make_configured_agent(
                    config,
                    "solver_verifier",
                    "solver",
                    "solver",
                    local_model,
                    local_site,
                ),
                make_configured_agent(
                    config,
                    "solver_verifier",
                    "verifier",
                    "verifier",
                    strong_model,
                    remote_site,
                ),
            ],
            "edges": [{"src": "solver", "dst": "verifier", "compression": "summary"}],
        },
        {
            "name": "solver_local_verifier_remote_final_only",
            "topology": "solver_verifier",
            "agents": [
                make_configured_agent(
                    config,
                    "solver_verifier",
                    "solver",
                    "solver",
                    local_model,
                    local_site,
                ),
                make_configured_agent(
                    config,
                    "solver_verifier",
                    "verifier",
                    "verifier",
                    strong_model,
                    remote_site,
                ),
            ],
            "edges": [{"src": "solver", "dst": "verifier", "compression": "final_only"}],
        },
        {
            "name": "planner_solver_verifier_split",
            "topology": "planner_solver_verifier",
            "agents": [
                make_configured_agent(
                    config,
                    "planner_solver_verifier",
                    "planner",
                    "planner",
                    cheap_model,
                    secondary_site,
                ),
                make_configured_agent(
                    config,
                    "planner_solver_verifier",
                    "solver",
                    "solver",
                    local_model,
                    local_site,
                ),
                make_configured_agent(
                    config,
                    "planner_solver_verifier",
                    "verifier",
                    "verifier",
                    strong_model,
                    remote_site,
                ),
            ],
            "edges": [
                {"src": "planner", "dst": "solver", "compression": "summary"},
                {"src": "solver", "dst": "verifier", "compression": "summary"},
            ],
        },
        {
            "name": "debate_local_remote",
            "topology": "debate",
            "agents": [
                make_configured_agent(config, "debate", "solver_a", "solver", local_model, local_site),
                make_configured_agent(config, "debate", "solver_b", "solver", cheap_model, secondary_site),
                make_configured_agent(config, "debate", "verifier", "verifier", strong_model, remote_site),
            ],
            "edges": [
                {"src": "solver_a", "dst": "verifier", "compression": "summary"},
                {"src": "solver_b", "dst": "verifier", "compression": "summary"},
            ],
        },
        {
            "name": "dag_decompose_three_workers",
            "topology": "dag_decompose",
            "worker_count": 3,
            "agents": [
                make_configured_agent(
                    config,
                    "dag_decompose",
                    "decomposer",
                    "decomposer",
                    dag_decomposer_model,
                    local_site,
                ),
                make_configured_agent(
                    config,
                    "dag_decompose",
                    "worker_0",
                    "worker",
                    dag_worker_models[0],
                    local_site,
                ),
                make_configured_agent(
                    config,
                    "dag_decompose",
                    "worker_1",
                    "worker",
                    dag_worker_models[1],
                    secondary_site,
                ),
                make_configured_agent(
                    config,
                    "dag_decompose",
                    "worker_2",
                    "worker",
                    dag_worker_models[2],
                    remote_site,
                ),
                make_configured_agent(
                    config,
                    "dag_decompose",
                    "aggregator",
                    "aggregator",
                    dag_aggregator_model,
                    local_site,
                ),
            ],
            "edges": [
                {"src": "decomposer", "dst": "worker_0", "compression": "full"},
                {"src": "decomposer", "dst": "worker_1", "compression": "full"},
                {"src": "decomposer", "dst": "worker_2", "compression": "full"},
                {"src": "worker_0", "dst": "aggregator", "compression": "summary"},
                {"src": "worker_1", "dst": "aggregator", "compression": "summary"},
                {"src": "worker_2", "dst": "aggregator", "compression": "summary"},
            ],
        },
    ]
    allowed = set(config.get("allowed_topologies", []))
    if allowed:
        seeds = [seed for seed in seeds if seed["topology"] in allowed]
    return [with_candidate_id(seed) for seed in seeds]


CANDIDATE_METADATA_KEYS = {
    "applied_mutation",
    "id",
    "mutation_observation",
    "name",
    "mutation",
    "parent_id",
    "parent_reflection",
    "parent_source",
}


def candidate_structure(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(candidate))
    for key in CANDIDATE_METADATA_KEYS:
        payload.pop(key, None)
    return payload


def with_candidate_id(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate = json.loads(json.dumps(candidate))
    candidate["id"] = stable_hash(candidate_structure(candidate))
    return candidate


def _graph_has_cycle(agent_names: set[str], edges: list[dict[str, Any]]) -> bool:
    adjacency = {name: [] for name in agent_names}
    indegree = dict.fromkeys(agent_names, 0)
    for edge in edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src not in agent_names or dst not in agent_names:
            continue
        adjacency[src].append(dst)
        indegree[dst] += 1
    queue = [name for name, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for neighbor in adjacency[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    return visited != len(agent_names)


def validate_candidate_contract(
    candidate: dict[str, Any],
    config: dict[str, Any],
    models: dict[str, ModelSpec] | None = None,
) -> list[str]:
    errors: list[str] = []
    topology = str(candidate.get("topology", ""))
    allowed_topologies = set(config.get("allowed_topologies", []))
    if not topology or (allowed_topologies and topology not in allowed_topologies):
        errors.append(f"unsupported_topology:{topology or '<missing>'}")

    agents = candidate.get("agents")
    if not isinstance(agents, list) or not agents:
        return [*errors, "agents_missing_or_empty"]
    agent_names = [str(agent.get("name", "")) for agent in agents if isinstance(agent, dict)]
    if len(agent_names) != len(agents) or any(not name for name in agent_names):
        errors.append("agent_name_missing")
    if len(set(agent_names)) != len(agent_names):
        errors.append("duplicate_agent_name")
    agent_name_set = set(agent_names)

    configured_sites = set(config.get("search", {}).get("site_pool", []))
    defaults = config.get("defaults", {})
    configured_sites.update(
        str(defaults.get(key))
        for key in ("local_site", "secondary_site", "remote_site", "orchestrator_site")
        if defaults.get(key)
    )
    configured_models = set(models or {})
    for agent in agents:
        if not isinstance(agent, dict):
            errors.append("agent_not_object")
            continue
        name = str(agent.get("name", ""))
        model_name = str(agent.get("model", ""))
        site = str(agent.get("site", ""))
        if configured_models and model_name not in configured_models:
            errors.append(f"unknown_model:{name}:{model_name or '<missing>'}")
        if configured_sites and site not in configured_sites:
            errors.append(f"unknown_site:{name}:{site or '<missing>'}")
        max_tokens = agent.get("max_tokens")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            errors.append(f"invalid_max_tokens:{name}:{max_tokens}")

    edges = candidate.get("edges", [])
    if not isinstance(edges, list):
        return [*errors, "edges_not_list"]
    allowed_compressions = {"full", "summary", "final_only", "critic_brief"}
    allowed_compressions.update(config.get("search", {}).get("compression_pool", []))
    edge_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("edge_not_object")
            continue
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        compression = str(edge.get("compression", ""))
        if src not in agent_name_set or dst not in agent_name_set:
            errors.append(f"edge_unknown_endpoint:{src}->{dst}")
        if src == dst:
            errors.append(f"self_loop:{src}")
        if (src, dst) in edge_pairs:
            errors.append(f"duplicate_edge:{src}->{dst}")
        edge_pairs.add((src, dst))
        if allowed_compressions and compression not in allowed_compressions:
            errors.append(f"invalid_compression:{src}->{dst}:{compression or '<missing>'}")
    if _graph_has_cycle(agent_name_set, edges):
        errors.append("cyclic_topology")

    required_agents = {
        "single": {"solver"},
        "self_consistency": {"solver"},
        "solver_verifier": {"solver", "verifier"},
        "planner_solver_verifier": {"planner", "solver", "verifier"},
        "debate": {"solver_a", "solver_b", "verifier"},
        "dag_decompose": {"decomposer", "aggregator"},
    }.get(topology, set())
    missing_agents = required_agents - agent_name_set
    if missing_agents:
        errors.append(f"missing_required_agents:{','.join(sorted(missing_agents))}")
    if topology in {"single", "self_consistency"} and edges:
        errors.append(f"unexpected_edges_for_{topology}")
    if topology == "self_consistency":
        samples = candidate.get("samples", 0)
        if not isinstance(samples, int) or samples < 2:
            errors.append("self_consistency_requires_at_least_two_samples")
    if topology == "solver_verifier" and ("solver", "verifier") not in edge_pairs:
        errors.append("missing_required_edge:solver->verifier")
    if topology == "planner_solver_verifier":
        for required_edge in (("planner", "solver"), ("solver", "verifier")):
            if required_edge not in edge_pairs:
                errors.append(f"missing_required_edge:{required_edge[0]}->{required_edge[1]}")
    if topology == "debate":
        for required_edge in (("solver_a", "verifier"), ("solver_b", "verifier")):
            if required_edge not in edge_pairs:
                errors.append(f"missing_required_edge:{required_edge[0]}->{required_edge[1]}")
    if topology == "dag_decompose":
        workers = sorted(name for name in agent_name_set if name.startswith("worker_"))
        if not workers:
            errors.append("dag_requires_worker")
        worker_count = candidate.get("worker_count", len(workers))
        if not isinstance(worker_count, int) or worker_count != len(workers):
            errors.append("dag_worker_count_mismatch")
        for worker in workers:
            if ("decomposer", worker) not in edge_pairs:
                errors.append(f"missing_required_edge:decomposer->{worker}")
            if (worker, "aggregator") not in edge_pairs:
                errors.append(f"missing_required_edge:{worker}->aggregator")
    return sorted(set(errors))


VALID_MUTATION_TYPES = {
    "model",
    "site",
    "compression",
    "topology",
    "samples",
    "max_tokens",
    "worker_count",
}


def parse_json_object_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def reflection_notes(plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return []
    notes: list[str] = []
    for key in ("diagnosis", "preserve", "risks"):
        value = plan.get(key)
        if isinstance(value, list):
            notes.extend(str(item) for item in value)
        elif value:
            notes.append(str(value))
    if not notes:
        rule_notes = plan.get("rule_diagnosis", plan.get("rule_fallback", []))
        if isinstance(rule_notes, list):
            notes.extend(str(item) for item in rule_notes)
    return notes


def candidate_reflection_payload(row: dict[str, Any]) -> dict[str, Any]:
    metric_keys = [
        "score",
        "correct",
        "num_examples",
        "avg_calls",
        "avg_total_tokens",
        "avg_requested_max_tokens",
        "avg_cross_center_tokens",
        "avg_cross_center_message_tokens",
        "avg_cross_center_model_tokens",
        "avg_observed_model_latency_ms",
        "avg_network_latency_ms",
        "avg_emulated_latency_ms",
        "avg_expected_network_failures",
        "avg_errors",
    ]
    compact_failures = []
    for failure in row.get("failure_examples", [])[:3]:
        trace = failure.get("trace", {})
        compact_failures.append(
            {
                "id": failure.get("id"),
                "input": str(failure.get("input", ""))[:1200],
                "gold_answer": failure.get("gold_answer"),
                "prediction": failure.get("prediction"),
                "final_output_excerpt": str(failure.get("final_output_excerpt", ""))[-600:],
                "agent_outputs": [
                    {
                        "agent": agent.get("agent"),
                        "role": agent.get("role"),
                        "model": agent.get("model"),
                        "site": agent.get("site"),
                        "output_excerpt": str(agent.get("output_excerpt", ""))[-500:],
                        "error": str(agent.get("error", ""))[:500] or None,
                    }
                    for agent in failure.get("agent_outputs", [])[:4]
                ],
                "trace": {
                    key: trace.get(key)
                    for key in (
                        "calls",
                        "total_tokens",
                        "requested_max_tokens",
                        "maxed_calls",
                        "cross_center_tokens",
                        "network_latency_ms",
                        "emulated_latency_ms",
                        "errors",
                    )
                    if key in trace
                },
            }
        )
    return {
        "candidate_id": row.get("candidate_id"),
        "candidate_name": row.get("candidate_name"),
        "topology": row.get("topology"),
        "candidate": candidate_structure(row.get("candidate", {})),
        "metrics": {key: row.get(key) for key in metric_keys if key in row},
        "failure_examples": compact_failures,
        "lineage": {
            "parent_id": row.get("candidate", {}).get("parent_id"),
            "mutation": row.get("candidate", {}).get("mutation"),
            "applied_mutation": row.get("candidate", {}).get("applied_mutation"),
            "mutation_observation": row.get("candidate", {}).get("mutation_observation"),
        },
        "rule_reflection": reflect_on_candidate(row),
    }


def build_rule_reflection_plan(row: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "mode": "rule",
        "diagnosis": reflect_on_candidate(row),
        "preserve": [],
        "mutations": [],
        "risks": [],
    }


def resolve_reflection_model_name(config: dict[str, Any], reflection_model: str | None) -> str | None:
    defaults = config.get("defaults", {})
    return (
        reflection_model
        or config.get("reflection", {}).get("model")
        or defaults.get("reflection_model")
        or defaults.get("strong_model")
        or defaults.get("local_model")
    )


def build_llm_reflection_plan(
    *,
    row: dict[str, Any],
    config: dict[str, Any],
    models: dict[str, ModelSpec],
    profile: NetworkProfile,
    reflection_model: str | None,
    reflection_max_tokens: int,
    max_proposals: int,
) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    model_name = resolve_reflection_model_name(config, reflection_model)
    if not model_name or model_name not in models:
        plan = build_rule_reflection_plan(row)
        plan["mode"] = "rule_fallback"
        plan["fallback_reason"] = f"Reflection model {model_name!r} is not configured."
        return plan
    parent_candidate = row.get("candidate", {})
    agent_names = [str(agent.get("name")) for agent in parent_candidate.get("agents", [])]
    edge_names = [
        f"{edge.get('src')}->{edge.get('dst')}"
        for edge in parent_candidate.get("edges", [])
    ]
    search_space = config.get("search", {})
    system_prompt = (
        "You are an architecture-search reflector for a deployment-conditioned multi-agent LLM system. "
        "Given one evaluated parent architecture, diagnose concrete task failures and quality/cost tradeoffs, "
        "then propose diverse, single-edit typed mutations ranked by expected value. Return JSON only. "
        "Each mutation must include type, target, value, rationale, and expected_effect. Never encode a topology "
        "change as a model edit. In particular, self-consistency is "
        '`{"type":"topology","target":"candidate","value":"self_consistency"}`, never '
        '`{"type":"model","value":"samples"}`. A samples edit is legal only when the parent topology is already '
        "self_consistency. "
        f"Return at most {max(1, int(max_proposals))} independent mutation proposals. "
        "A site edit changes logical placement and emulated network/site cost, but does not change measured model "
        "generation latency or the selected model endpoint. A model edit changes the model service. Compression "
        "applies only to an existing directed edge and its target must be `src->dst`; it cannot reduce a single "
        "agent's own generation tokens. A max_tokens edit must target an existing agent or role. "
        "Do not propose no-ops or unavailable targets/values. Use the actual failed outputs and truncation evidence; "
        "do not claim a token limit caused a failure unless the trace reached that limit. Avoid unsupported numeric "
        "promises such as an exact future accuracy. Every proposal must have a plausible net benefit for this parent; "
        "do not fill the quota with an edit that your own rationale predicts will be neutral or harmful. Return fewer "
        "proposals when fewer beneficial legal edits exist. Prefer edits that preserve quality while reducing "
        "unnecessary calls, token use, remote/cross-site cost, or deployment latency."
    )
    mutation_contract = {
        "topology": {
            "target": "candidate",
            "legal_values": config.get("allowed_topologies", []),
        },
        "model": {
            "target": f"one existing agent: {agent_names}",
            "legal_values": search_space.get("model_pool", []),
        },
        "site": {
            "target": f"one existing agent: {agent_names}",
            "legal_values": search_space.get("site_pool", []),
        },
        "max_tokens": {
            "target": f"one existing agent: {agent_names}",
            "legal_values_by_agent": {
                str(agent.get("name")): max_tokens_pool_for_role(
                    config,
                    str(agent.get("role", "solver")),
                    str(parent_candidate.get("topology", "single")),
                )
                for agent in parent_candidate.get("agents", [])
            },
        },
        "compression": {
            "target": f"one existing edge: {edge_names}",
            "legal_values": search_space.get("compression_pool", []),
        },
        "worker_count": {
            "target": "candidate",
            "legal_values": [2, 3, 4] if parent_candidate.get("topology") == "dag_decompose" else [],
        },
        "samples": {
            "target": "candidate",
            "legal_values": [2, 3, 5],
            "condition": "parent topology must already be self_consistency",
        },
    }
    user_payload = {
        "network_profile": asdict(profile),
        "allowed_topologies": config.get("allowed_topologies", []),
        "search_space": search_space,
        "mutation_contract_for_this_parent": mutation_contract,
        "parent": candidate_reflection_payload(row),
        "required_schema": {
            "diagnosis": "short evidence-grounded diagnosis",
            "preserve": "short strengths to keep",
            "mutations": [
                {
                    "type": "compression|max_tokens|site|model|topology|samples",
                    "target": "candidate, an existing agent, or an existing edge src->dst as required",
                    "value": "one value explicitly listed in mutation_contract_for_this_parent",
                    "rationale": "why this edit is useful",
                    "expected_effect": "directional quality/cost/latency tradeoff without invented percentages",
                }
            ],
            "risks": ["possible failure modes"],
        },
    }
    reflection_config = config.get("reflection", {})
    max_attempts = max(1, int(reflection_config.get("max_attempts", 3)))
    retry_backoff_seconds = max(0.0, float(reflection_config.get("retry_backoff_seconds", 1.0)))
    call_traces: list[dict[str, Any]] = []
    content = ""
    parsed = None
    failed_attempts = 0
    attempt_max_tokens = max(128, int(reflection_max_tokens))
    for attempt in range(max_attempts):
        content, call_trace = litellm_call(
            model_spec=models[model_name],
            agent_name="reflector",
            site=defaults.get("orchestrator_site", ORCHESTRATOR_SITE),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            temperature=float(reflection_config.get("temperature", 0.2)),
            max_tokens=attempt_max_tokens,
        )
        call_traces.append(asdict(call_trace))
        if not call_trace.error:
            parsed = parse_json_object_from_text(content)
            if parsed is not None:
                break
        elif "ContextWindowExceeded" in call_trace.error:
            attempt_max_tokens = max(256, attempt_max_tokens // 2)
        failed_attempts += 1
        if attempt + 1 < max_attempts and retry_backoff_seconds > 0:
            time.sleep(retry_backoff_seconds * (2**attempt))
    if parsed is None:
        failure_reason = (
            "LLM reflector failed after "
            f"{max_attempts} attempts; last_error={call_traces[-1].get('error') or 'unparseable JSON'}"
        )
        if not bool(reflection_config.get("allow_rule_fallback", True)):
            raise RuntimeError(failure_reason)
        plan = build_rule_reflection_plan(row)
        plan["mode"] = "rule_fallback"
        plan["fallback_reason"] = failure_reason
        plan["raw_reflection"] = content[:2000]
    else:
        plan = {
            "mode": "llm",
            "model": model_name,
            "diagnosis": parsed.get("diagnosis", []),
            "preserve": parsed.get("preserve", []),
            "mutations": parsed.get("mutations", []),
            "risks": parsed.get("risks", parsed.get("risk", [])),
            "raw_reflection": content[:2000],
        }
    plan["call_trace"] = call_traces[-1]
    plan["call_traces"] = call_traces
    plan["reflection_attempts"] = len(call_traces)
    plan["reflection_failures"] = failed_attempts
    plan["rule_diagnosis"] = reflect_on_candidate(row)
    return plan


def build_reflection_plan(
    *,
    row: dict[str, Any],
    config: dict[str, Any],
    models: dict[str, ModelSpec],
    profile: NetworkProfile,
    reflection_mode: str,
    reflection_model: str | None,
    reflection_max_tokens: int,
    max_proposals: int,
) -> dict[str, Any]:
    if reflection_mode == "llm":
        return build_llm_reflection_plan(
            row=row,
            config=config,
            models=models,
            profile=profile,
            reflection_model=reflection_model,
            reflection_max_tokens=reflection_max_tokens,
            max_proposals=max_proposals,
        )
    return build_rule_reflection_plan(row)


def find_edge_for_mutation(candidate: dict[str, Any], target: Any) -> dict[str, Any] | None:
    if not target:
        return None
    target_text = str(target)
    if "->" not in target_text:
        return None
    src, dst = [part.strip() for part in target_text.split("->", 1)]
    for edge in candidate.get("edges", []):
        if edge.get("src") == src and edge.get("dst") == dst:
            return edge
    return None


def find_agent_for_mutation(candidate: dict[str, Any], target: Any) -> dict[str, Any] | None:
    if not target:
        return None
    target_text = str(target)
    return next(
        (
            agent
            for agent in candidate.get("agents", [])
            if agent.get("name") == target_text or agent.get("role") == target_text
        ),
        None,
    )


def planned_mutation_key(item: dict[str, Any]) -> str:
    return json.dumps(
        {
            "type": item.get("type"),
            "target": item.get("target"),
            "value": item.get("value"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def is_planned_mutation_applicable(
    item: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    mutation_type = str(item.get("type", "")).strip()
    if mutation_type not in VALID_MUTATION_TYPES:
        return False
    target = item.get("target")
    value = item.get("value")
    search = config.get("search", {})
    if mutation_type in {"model", "site", "max_tokens"}:
        agent = find_agent_for_mutation(candidate, target)
        if agent is None:
            return False
        if mutation_type == "model":
            return value in set(search.get("model_pool", [])) and value != agent.get("model")
        if mutation_type == "site":
            return value in set(search.get("site_pool", [])) and value != agent.get("site")
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return False
        pool = max_tokens_pool_for_role(
            config,
            str(agent.get("role", "solver")),
            str(candidate.get("topology", "single")),
        )
        if not pool:
            return False
        normalized_value = min(pool, key=lambda candidate_value: abs(candidate_value - value_int))
        current_value = int(
            agent.get("max_tokens")
            or default_agent_max_tokens(config, str(candidate.get("topology", "single")), str(agent.get("role")))
            or 0
        )
        return normalized_value != current_value
    if mutation_type == "compression":
        edge = find_edge_for_mutation(candidate, target)
        return (
            edge is not None
            and value in set(search.get("compression_pool", []))
            and value != edge.get("compression")
        )
    if mutation_type == "topology":
        return value in set(config.get("allowed_topologies", [])) and value != candidate.get("topology")
    if mutation_type == "samples":
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return False
        return (
            candidate.get("topology") == "self_consistency"
            and value_int in {2, 3, 5}
            and value_int != int(candidate.get("samples", 3))
        )
    if mutation_type == "worker_count":
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return False
        return (
            candidate.get("topology") == "dag_decompose"
            and value_int in {2, 3, 4}
            and value_int != int(candidate.get("worker_count", 3))
        )
    return False


def choose_planned_mutations(
    reflection_plan: dict[str, Any] | None,
    candidate: dict[str, Any],
    config: dict[str, Any],
    *,
    limit: int,
    excluded_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not reflection_plan:
        return []
    excluded = excluded_keys or set()
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in reflection_plan.get("mutations", []):
        if not isinstance(item, dict) or not is_planned_mutation_applicable(item, candidate, config):
            continue
        key = planned_mutation_key(item)
        if key in excluded or key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def choose_planned_mutation(
    reflection_plan: dict[str, Any] | None,
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    selected = choose_planned_mutations(
        reflection_plan,
        candidate,
        config,
        limit=1,
    )
    return selected[0] if selected else None


def apply_planned_max_tokens(
    candidate: dict[str, Any],
    config: dict[str, Any],
    planned_mutation: dict[str, Any] | None,
) -> bool:
    if not planned_mutation:
        return False
    target = planned_mutation.get("target")
    value = planned_mutation.get("value")
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return False
    for agent in candidate.get("agents", []):
        if target and agent.get("name") != target and agent.get("role") != target:
            continue
        pool = max_tokens_pool_for_role(
            config,
            str(agent.get("role", "solver")),
            str(candidate.get("topology", "single")),
        )
        if pool:
            value_int = min(pool, key=lambda candidate_value: abs(candidate_value - value_int))
        agent["max_tokens"] = value_int
        return True
    return False


def resize_dag_workers(candidate: dict[str, Any], worker_count: int) -> bool:
    if candidate.get("topology") != "dag_decompose" or worker_count not in {2, 3, 4}:
        return False
    existing_workers = [agent for agent in candidate.get("agents", []) if agent.get("role") == "worker"]
    if not existing_workers or len(existing_workers) == worker_count:
        return False
    workers = [json.loads(json.dumps(agent)) for agent in existing_workers[:worker_count]]
    while len(workers) < worker_count:
        template = json.loads(json.dumps(existing_workers[len(workers) % len(existing_workers)]))
        workers.append(template)
    for idx, worker in enumerate(workers):
        worker["name"] = f"worker_{idx}"
    non_workers = [agent for agent in candidate.get("agents", []) if agent.get("role") != "worker"]
    candidate["agents"] = [
        *[agent for agent in non_workers if agent.get("role") == "decomposer"],
        *workers,
        *[agent for agent in non_workers if agent.get("role") == "aggregator"],
    ]
    compression_by_kind = {
        "decomposer": next(
            (
                edge.get("compression", "full")
                for edge in candidate.get("edges", [])
                if edge.get("src") == "decomposer"
            ),
            "full",
        ),
        "aggregator": next(
            (
                edge.get("compression", "summary")
                for edge in candidate.get("edges", [])
                if edge.get("dst") == "aggregator"
            ),
            "summary",
        ),
    }
    candidate["edges"] = [
        *[
            {
                "src": "decomposer",
                "dst": worker["name"],
                "compression": compression_by_kind["decomposer"],
            }
            for worker in workers
        ],
        *[
            {
                "src": worker["name"],
                "dst": "aggregator",
                "compression": compression_by_kind["aggregator"],
            }
            for worker in workers
        ],
    ]
    candidate["worker_count"] = worker_count
    return True


def mutate_candidate(
    candidate: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
    parent_row: dict[str, Any] | None = None,
    mode: str = "wan_pareto",
    reflection_plan: dict[str, Any] | None = None,
    planned_mutation_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    child = json.loads(json.dumps(candidate))
    defaults = config.get("defaults", {})
    model_pool = list(config.get("search", {}).get("model_pool", [])) or [
        defaults["local_model"],
        defaults.get("cheap_model", defaults["local_model"]),
        defaults.get("strong_model", defaults["local_model"]),
    ]
    site_pool = list(config.get("search", {}).get("site_pool", [])) or [
        defaults.get("local_site", "center_a"),
        defaults.get("secondary_site", "center_b"),
        defaults.get("remote_site", "center_c"),
    ]
    compression_pool = list(config.get("search", {}).get("compression_pool", [])) or [
        "full",
        "summary",
        "final_only",
        "critic_brief",
    ]
    topology_pool = list(config.get("allowed_topologies", [])) or [
        "single",
        "self_consistency",
        "solver_verifier",
        "planner_solver_verifier",
        "debate",
    ]
    reflections = (
        reflection_notes(reflection_plan)
        if reflection_plan
        else (reflect_on_candidate(parent_row) if parent_row else [])
    )
    high_token_cost = any("token usage is high" in item for item in reflections)
    low_task_score = any("task score is low" in item for item in reflections)
    planned_mutation = planned_mutation_override or choose_planned_mutation(reflection_plan, child, config)
    if planned_mutation:
        mutation = planned_mutation["type"]
    elif mode == "wan_pareto" and high_token_cost:
        mutation = rng.choice(["max_tokens", "max_tokens", "compression", "topology"])
    elif mode == "wan_pareto" and any("cross-center communication is high" in item for item in reflections):
        mutation = rng.choice(["compression", "site", "compression"])
    elif mode == "wan_pareto" and any("latency is high" in item for item in reflections):
        mutation = rng.choice(["site", "compression", "topology"])
    elif low_task_score:
        mutation = rng.choice(["topology", "model", "samples", "max_tokens"])
    elif mode in {"quality_only", "adas_style"}:
        mutation = rng.choice(["topology", "model", "model", "samples", "max_tokens"])
    else:
        mutation = rng.choice(["model", "site", "compression", "topology", "samples", "max_tokens"])
    if mutation == "model" and child.get("agents"):
        planned_value = planned_mutation.get("value") if planned_mutation else None
        planned_target = planned_mutation.get("target") if planned_mutation else None
        agent = find_agent_for_mutation(child, planned_target) or rng.choice(child["agents"])
        agent["model"] = planned_value if planned_value in model_pool else rng.choice(model_pool)
    elif mutation == "site" and child.get("agents"):
        planned_value = planned_mutation.get("value") if planned_mutation else None
        planned_target = planned_mutation.get("target") if planned_mutation else None
        target_agent = find_agent_for_mutation(child, planned_target)
        if target_agent and planned_value in site_pool:
            target_agent["site"] = planned_value
        elif mode == "wan_pareto" and child.get("edges"):
            # Co-locate one communicating pair to reduce WAN traffic/latency.
            edge = rng.choice(child["edges"])
            src_agent = next((agent for agent in child["agents"] if agent["name"] == edge["src"]), None)
            dst_agent = next((agent for agent in child["agents"] if agent["name"] == edge["dst"]), None)
            if src_agent and dst_agent:
                dst_agent["site"] = src_agent["site"]
            else:
                rng.choice(child["agents"])["site"] = rng.choice(site_pool)
        else:
            agent = rng.choice(child["agents"])
            agent["site"] = rng.choice(site_pool)
    elif mutation == "compression" and child.get("edges"):
        edge = rng.choice(child["edges"])
        planned_value = planned_mutation.get("value") if planned_mutation else None
        planned_target = planned_mutation.get("target") if planned_mutation else None
        target_edge = find_edge_for_mutation(child, planned_target)
        if target_edge is not None:
            edge = target_edge
        edge["compression"] = planned_value if planned_value in compression_pool else rng.choice(compression_pool)
    elif mutation == "samples" and child.get("topology") == "self_consistency":
        planned_value = planned_mutation.get("value") if planned_mutation else None
        child["samples"] = int(planned_value) if planned_value in {2, 3, 5, "2", "3", "5"} else rng.choice([2, 3, 5])
    elif mutation == "worker_count" and child.get("topology") == "dag_decompose":
        planned_value = planned_mutation.get("value") if planned_mutation else None
        try:
            worker_count = int(planned_value)
        except (TypeError, ValueError):
            worker_count = rng.choice(
                [value for value in (2, 3, 4) if value != int(child.get("worker_count", 3))]
            )
        resize_dag_workers(child, worker_count)
    elif mutation == "max_tokens" and child.get("agents"):
        if not apply_planned_max_tokens(child, config, planned_mutation):
            mutate_agent_max_tokens(
                child,
                config,
                rng,
                prefer_lower=mode == "wan_pareto" and high_token_cost,
                prefer_higher=low_task_score,
            )
    elif mutation == "topology":
        planned_value = planned_mutation.get("value") if planned_mutation else None
        topology = planned_value if planned_value in topology_pool else rng.choice(topology_pool)
        child = make_random_architecture(config, rng, topology=topology)
    else:
        child = make_random_architecture(config, rng, topology=rng.choice(topology_pool))
    child["name"] = f"{candidate.get('name', 'candidate')}_mut_{mutation}_{rng.randrange(100000)}"
    child["parent_id"] = candidate.get("id")
    child["mutation"] = mutation
    child["applied_mutation"] = (
        json.loads(json.dumps(planned_mutation))
        if planned_mutation
        else {"type": mutation, "target": None, "value": None, "source": "fallback"}
    )
    if reflection_plan:
        child["parent_reflection"] = reflection_plan
    elif reflections:
        child["parent_reflection"] = {"mode": "rule", "diagnosis": reflections, "mutations": []}
    return with_candidate_id(child)


def reflect_on_candidate(row: dict[str, Any] | None) -> list[str]:
    if row is None:
        return []
    reflections: list[str] = []
    score = float(row.get("score", 0.0))
    avg_cross = float(row.get("avg_cross_center_tokens", 0.0))
    avg_total_tokens = float(row.get("avg_total_tokens", 0.0))
    avg_latency = float(row.get("avg_network_latency_ms", row.get("avg_emulated_latency_ms", 0.0)))
    avg_calls = float(row.get("avg_calls", 0.0))
    topology = row.get("topology", "")
    if score <= 0.0:
        reflections.append("The task score is low; add verification, stronger models, or a richer topology.")
    if avg_cross > 250:
        reflections.append(
            "The cross-center communication is high; compress messages or co-locate communicating agents."
        )
    if avg_total_tokens > 10000:
        reflections.append("The token usage is high; reduce per-agent max tokens or use fewer reasoning calls.")
    if avg_latency > 500:
        reflections.append(
            "The WAN network latency is high; reduce remote calls, compress messages, or co-locate verifier and solver."
        )
    if avg_calls > 3:
        reflections.append(
            "The model-call count is high; reduce debate/self-consistency rounds or use a cheaper verifier."
        )
    if topology == "single" and score <= 0.0:
        reflections.append("A single-agent workflow failed; try solver-verifier or planner-solver-verifier.")
    if not reflections:
        reflections.append(
            "This candidate is a useful parent; preserve its strengths while searching for lower WAN cost."
        )
    return reflections


def mutate_agent_max_tokens(
    candidate: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
    *,
    prefer_lower: bool,
    prefer_higher: bool,
) -> None:
    mutation_options: list[tuple[dict[str, Any], list[int]]] = []
    for agent in candidate.get("agents", []):
        role = str(agent.get("role", "solver"))
        pool = max_tokens_pool_for_role(config, role, str(candidate.get("topology", "single")))
        if not pool:
            continue
        current = int(agent.get("max_tokens") or default_agent_max_tokens(config, candidate["topology"], role) or 0)
        if prefer_lower:
            choices = [value for value in pool if value < current]
        elif prefer_higher:
            choices = [value for value in pool if value > current]
        else:
            choices = [value for value in pool if value != current]
        if choices:
            mutation_options.append((agent, choices))
    if not mutation_options:
        return
    agent, choices = rng.choice(mutation_options)
    agent["max_tokens"] = rng.choice(choices)


def make_random_architecture(config: dict[str, Any], rng: random.Random, topology: str | None = None) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    model_pool = list(config.get("search", {}).get("model_pool", [])) or [
        defaults["local_model"],
        defaults.get("cheap_model", defaults["local_model"]),
        defaults.get("strong_model", defaults["local_model"]),
    ]
    site_pool = list(config.get("search", {}).get("site_pool", [])) or [
        defaults.get("local_site", "center_a"),
        defaults.get("secondary_site", "center_b"),
        defaults.get("remote_site", "center_c"),
    ]
    compression_pool = list(config.get("search", {}).get("compression_pool", [])) or [
        "full",
        "summary",
        "final_only",
        "critic_brief",
    ]
    topology_pool = list(config.get("allowed_topologies", [])) or [
        "single",
        "self_consistency",
        "solver_verifier",
        "planner_solver_verifier",
        "debate",
    ]
    topology = topology or rng.choice(topology_pool)

    def rand_agent(name: str, role: str) -> dict[str, Any]:
        token_pool = max_tokens_pool_for_role(config, role, topology)
        max_tokens = (
            rng.choice(token_pool)
            if token_pool
            else default_agent_max_tokens(config, topology or "single", role)
        )
        return make_agent(
            name,
            role,
            rng.choice(model_pool),
            rng.choice(site_pool),
            max_tokens=max_tokens,
        )

    if topology == "single":
        candidate = {
            "name": f"random_single_{rng.randrange(100000)}",
            "topology": topology,
            "agents": [rand_agent("solver", "solver")],
            "edges": [],
        }
    elif topology == "self_consistency":
        candidate = {
            "name": f"random_self_consistency_{rng.randrange(100000)}",
            "topology": topology,
            "samples": rng.choice([2, 3, 5]),
            "temperature": rng.choice([0.2, 0.3, 0.5]),
            "agents": [rand_agent("solver", "solver")],
            "edges": [],
        }
    elif topology == "solver_verifier":
        candidate = {
            "name": f"random_solver_verifier_{rng.randrange(100000)}",
            "topology": topology,
            "agents": [rand_agent("solver", "solver"), rand_agent("verifier", "verifier")],
            "edges": [{"src": "solver", "dst": "verifier", "compression": rng.choice(compression_pool)}],
        }
    elif topology == "planner_solver_verifier":
        candidate = {
            "name": f"random_planner_solver_verifier_{rng.randrange(100000)}",
            "topology": topology,
            "agents": [
                rand_agent("planner", "planner"),
                rand_agent("solver", "solver"),
                rand_agent("verifier", "verifier"),
            ],
            "edges": [
                {"src": "planner", "dst": "solver", "compression": rng.choice(compression_pool)},
                {"src": "solver", "dst": "verifier", "compression": rng.choice(compression_pool)},
            ],
        }
    elif topology == "debate":
        candidate = {
            "name": f"random_debate_{rng.randrange(100000)}",
            "topology": topology,
            "agents": [
                rand_agent("solver_a", "solver"),
                rand_agent("solver_b", "solver"),
                rand_agent("verifier", "verifier"),
            ],
            "edges": [
                {"src": "solver_a", "dst": "verifier", "compression": rng.choice(compression_pool)},
                {"src": "solver_b", "dst": "verifier", "compression": rng.choice(compression_pool)},
            ],
        }
    elif topology == "dag_decompose":
        worker_count = rng.choice([2, 3, 4])
        workers = [rand_agent(f"worker_{idx}", "worker") for idx in range(worker_count)]
        candidate = {
            "name": f"random_dag_decompose_{worker_count}w_{rng.randrange(100000)}",
            "topology": topology,
            "worker_count": worker_count,
            "agents": [rand_agent("decomposer", "decomposer"), *workers, rand_agent("aggregator", "aggregator")],
            "edges": [
                *[
                    {
                        "src": "decomposer",
                        "dst": worker["name"],
                        "compression": rng.choice(compression_pool),
                    }
                    for worker in workers
                ],
                *[
                    {
                        "src": worker["name"],
                        "dst": "aggregator",
                        "compression": rng.choice(compression_pool),
                    }
                    for worker in workers
                ],
            ],
        }
    else:
        raise ValueError(f"Unsupported random topology: {topology}")
    return with_candidate_id(candidate)


def top_score_band_rows(evaluated: list[dict[str, Any]], score_band: float, top_k: int) -> list[dict[str, Any]]:
    evaluated = eligible_rows(evaluated)
    if not evaluated:
        return []
    best_score = max(float(row.get("score", 0.0)) for row in evaluated)
    threshold = best_score - max(0.0, score_band)
    ranked = sorted(
        [row for row in evaluated if float(row.get("score", 0.0)) >= threshold],
        key=lambda row: (-float(row.get("score", 0.0)), *quality_cost_key(row)),
    )
    return ranked[: max(1, top_k)]


def select_parent(
    evaluated: list[dict[str, Any]],
    rng: random.Random,
    mode: str,
    *,
    pareto_parent_prob: float,
    parent_score_band: float,
    parent_top_k: int,
) -> tuple[dict[str, Any], str]:
    evaluated = eligible_rows(evaluated)
    if not evaluated:
        raise ValueError("No valid evaluated candidates to select from")
    if mode == "wan_pareto":
        front = pareto_front(evaluated)
        score_band = top_score_band_rows(evaluated, parent_score_band, parent_top_k)
        use_pareto = rng.random() < max(0.0, min(1.0, pareto_parent_prob))
        if use_pareto and front:
            return rng.choice(front), "pareto_front"
        if score_band:
            return rng.choice(score_band), f"top_score_band.band={parent_score_band:g}.top_k={parent_top_k}"
        return rng.choice(front), "pareto_front_fallback"
    if mode == "aflow_style":
        # A shallow UCB tree policy: completed child rollouts are visits.
        total_children = max(
            1,
            sum(1 for row in evaluated if row.get("candidate", {}).get("parent_id")),
        )

        def ucb(row: dict[str, Any]) -> float:
            visits = sum(
                1
                for child in evaluated
                if child.get("candidate", {}).get("parent_id") == row.get("candidate_id")
            )
            return float(row.get("score", 0.0)) + 0.35 * math.sqrt(
                math.log(total_children + 1) / (visits + 1)
            )

        return max(evaluated, key=ucb), "mcts_ucb"
    if mode == "adas_style":
        ranked = sorted(
            evaluated,
            key=lambda row: (
                -float(row.get("score", 0.0)),
                float(row.get("avg_total_tokens", 0.0)),
                str(row.get("candidate_id", "")),
            ),
        )
        return rng.choice(ranked[: max(1, min(4, len(ranked)))]), "meta_agent_quality_band"
    ranked = sorted(evaluated, key=lambda row: utility_score(row, mode), reverse=True)
    top_n = max(1, min(4, len(ranked)))
    return rng.choice(ranked[:top_n]), f"utility_top_{top_n}"


def candidate_key(candidate: dict[str, Any]) -> str:
    return stable_hash(candidate_structure(candidate))


MUTATION_OBSERVATION_METRICS = [
    "score",
    "avg_calls",
    "avg_total_tokens",
    "avg_cross_center_tokens",
    "avg_network_latency_ms",
    "avg_emulated_latency_ms",
]


def summarize_search_overhead(records: list[dict[str, Any]]) -> dict[str, Any]:
    traces = [trace for record in records for trace in record.get("call_traces", [])]
    return {
        "controller_invocations": len(records),
        "controller_calls": len(traces),
        "controller_prompt_tokens": sum(int(trace.get("prompt_tokens", 0)) for trace in traces),
        "controller_completion_tokens": sum(int(trace.get("completion_tokens", 0)) for trace in traces),
        "controller_total_tokens": sum(int(trace.get("total_tokens", 0)) for trace in traces),
        "controller_input_cost_usd": sum(float(trace.get("input_cost_usd", 0.0)) for trace in traces),
        "controller_output_cost_usd": sum(float(trace.get("output_cost_usd", 0.0)) for trace in traces),
        "controller_cost_usd": sum(float(trace.get("inference_cost_usd", 0.0)) for trace in traces),
        "controller_observed_model_latency_ms": sum(
            float(trace.get("observed_latency_ms", 0.0)) for trace in traces
        ),
        "controller_wall_time_ms": sum(float(record.get("wall_time_ms", 0.0)) for record in records),
        "controller_failures": sum(bool(trace.get("error")) for trace in traces),
        "rule_fallbacks": sum(record.get("reflection_mode") == "rule_fallback" for record in records),
    }


def build_mutation_observation(
    parent_row: dict[str, Any],
    child_result: dict[str, Any],
    applied_mutation: dict[str, Any] | None,
) -> dict[str, Any]:
    before = {key: float(parent_row.get(key, 0.0)) for key in MUTATION_OBSERVATION_METRICS}
    after = {key: float(child_result.get(key, 0.0)) for key in MUTATION_OBSERVATION_METRICS}
    return {
        "expected_effect": (applied_mutation or {}).get("expected_effect", ""),
        "before": before,
        "after": after,
        "delta": {key: after[key] - before[key] for key in MUTATION_OBSERVATION_METRICS},
    }


def run_search(
    *,
    config: dict[str, Any],
    models: dict[str, ModelSpec],
    profile: NetworkProfile,
    searchset: list[dict[str, Any]],
    selectionset: list[dict[str, Any]],
    testset: list[dict[str, Any]],
    mode: str,
    seed_candidate_budget: int,
    new_candidate_budget: int,
    search_examples: int | None,
    selection_shortlist_size: int,
    test_top_k: int,
    eval_concurrency: int,
    output_dir: Path,
    seed: int,
    selection_strategy: str,
    quality_band: float,
    pareto_parent_prob: float,
    parent_score_band: float,
    parent_top_k: int,
    reflection_mode: str,
    reflection_model: str | None,
    reflection_max_tokens: int,
    reflection_children: int,
    reflection_example_limit: int,
    evaluation_cache_dir: Path | None = None,
    resume: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "search_rows.jsonl"
    selection_rows_path = output_dir / "selection_rows.jsonl"
    checkpoint_path = output_dir / "search_checkpoint.json"
    proposal_rows_path = output_dir / "proposal_rows.jsonl"
    search_overhead_rows_path = output_dir / "search_overhead_rows.jsonl"
    if seed_candidate_budget < 0:
        raise ValueError("seed_candidate_budget must be non-negative")
    if new_candidate_budget < 0:
        raise ValueError("new_candidate_budget must be non-negative")
    if not searchset:
        raise ValueError("D_search is empty; provide a non-empty disjoint search split")
    if not selectionset:
        raise ValueError("D_select is empty; provide a non-empty disjoint selection split")
    search_ids = {str(example.get("id", "")) for example in searchset}
    selection_ids = {str(example.get("id", "")) for example in selectionset}
    test_ids = {str(example.get("id", "")) for example in testset}
    overlap = {
        "search_selection": search_ids & selection_ids,
        "search_test": search_ids & test_ids,
        "selection_test": selection_ids & test_ids,
    }
    leaked = {name: sorted(ids)[:5] for name, ids in overlap.items() if ids}
    if leaked:
        raise ValueError(f"Dataset split leakage detected: {leaked}")
    available_seeds = seed_architectures(config)
    initial_queue = available_seeds[:seed_candidate_budget]
    if len(initial_queue) != seed_candidate_budget:
        raise ValueError(
            f"Requested {seed_candidate_budget} seed candidates, but the configured search space provides "
            f"{len(available_seeds)}."
        )
    invalid_seeds: dict[str, list[str]] = {}
    for candidate in initial_queue:
        contract_errors = validate_candidate_contract(candidate, config, models)
        if contract_errors:
            invalid_seeds[str(candidate.get("name", candidate.get("id", "<unnamed>")))] = contract_errors
    if invalid_seeds:
        raise ValueError(f"Seed architecture contract violations: {invalid_seeds}")
    seed_candidate_keys = {candidate_key(candidate) for candidate in initial_queue}
    target_candidates = len(initial_queue) + (0 if mode == "baselines" else new_candidate_budget)
    effective_search_examples = len(searchset) if search_examples is None else min(search_examples, len(searchset))
    resume_strategy = "fresh"
    if resume:
        evaluated_rows = load_partial_rows(rows_path)
        proposal_rows = load_partial_rows(proposal_rows_path)
        search_overhead_records = load_partial_rows(search_overhead_rows_path)
        checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {}
        if checkpoint and int(checkpoint.get("completed_rows", -1)) == len(evaluated_rows):
            candidate_queue = list(checkpoint.get("candidate_queue", []))
            seen = {str(key) for key in checkpoint.get("seen", [])}
            rng.setstate(_nested_tuple(checkpoint["rng_state"]))
            resume_strategy = "checkpoint"
        else:
            evaluated_keys = {candidate_key(row["candidate"]) for row in evaluated_rows}
            candidate_queue = [
                candidate for candidate in initial_queue if candidate_key(candidate) not in evaluated_keys
            ]
            seen = evaluated_keys | {candidate_key(candidate) for candidate in candidate_queue}
            continuation_key = stable_hash([row.get("candidate_id") for row in evaluated_rows])
            rng.seed(f"{seed}:resume:{continuation_key}")
            resume_strategy = "reconstructed_deterministic"
        print(
            "[WAN-GEPA] RESUME "
            f"strategy={resume_strategy} completed={len(evaluated_rows)} "
            f"queued={len(candidate_queue)} target={target_candidates}",
            flush=True,
        )
    else:
        candidate_queue = initial_queue
        seen = {candidate_key(candidate) for candidate in candidate_queue}
        evaluated_rows = []
        proposal_rows = []
        search_overhead_records = []
        for stale_path in (
            rows_path,
            checkpoint_path,
            selection_rows_path,
            proposal_rows_path,
            search_overhead_rows_path,
        ):
            if stale_path.exists():
                stale_path.unlink()

    for row in evaluated_rows:
        row.setdefault(
            "candidate_origin",
            "seed" if candidate_key(row["candidate"]) in seed_candidate_keys else "generated",
        )
        row.update(
            candidate_validity(
                row,
                contract_errors=validate_candidate_contract(row["candidate"], config, models),
            )
        )

    while len(evaluated_rows) < target_candidates:
        if not candidate_queue:
            if mode == "baselines":
                break
            if mode == "random" or not evaluated_rows:
                children = [make_random_architecture(config, rng)]
            elif mode == "aflow_style":
                parent_row, parent_source = select_parent(
                    evaluated_rows,
                    rng,
                    mode,
                    pareto_parent_prob=pareto_parent_prob,
                    parent_score_band=parent_score_band,
                    parent_top_k=parent_top_k,
                )
                child = mutate_candidate(
                    parent_row["candidate"],
                    config,
                    rng,
                    parent_row=parent_row,
                    mode=mode,
                )
                child["parent_source"] = parent_source
                children = [child]
            else:
                parent_row, parent_source = select_parent(
                    evaluated_rows,
                    rng,
                    mode,
                    pareto_parent_prob=pareto_parent_prob,
                    parent_score_band=parent_score_band,
                    parent_top_k=parent_top_k,
                )
                reflection_started = time.time()
                reflection_plan = build_reflection_plan(
                    row=parent_row,
                    config=config,
                    models=models,
                    profile=profile,
                    reflection_mode=reflection_mode,
                    reflection_model=reflection_model,
                    reflection_max_tokens=reflection_max_tokens,
                    max_proposals=reflection_children,
                )
                overhead_record = {
                    "index": len(search_overhead_records),
                    "kind": "reflection",
                    "mode": mode,
                    "parent_candidate_id": parent_row["candidate_id"],
                    "parent_candidate_name": parent_row["candidate_name"],
                    "reflection_mode": reflection_plan.get("mode", reflection_mode),
                    "wall_time_ms": (time.time() - reflection_started) * 1000,
                    "call_traces": reflection_plan.get("call_traces", []),
                }
                search_overhead_records.append(overhead_record)
                append_jsonl(search_overhead_rows_path, overhead_record)
                attempted_mutations = {
                    planned_mutation_key(row["candidate"]["applied_mutation"])
                    for row in evaluated_rows
                    if row.get("candidate", {}).get("parent_id") == parent_row["candidate_id"]
                    and isinstance(row.get("candidate", {}).get("applied_mutation"), dict)
                }
                proposals = choose_planned_mutations(
                    reflection_plan,
                    parent_row["candidate"],
                    config,
                    limit=reflection_children,
                    excluded_keys=attempted_mutations,
                )
                children = [
                    mutate_candidate(
                        parent_row["candidate"],
                        config,
                        rng,
                        parent_row=parent_row,
                        mode=mode,
                        reflection_plan=reflection_plan,
                        planned_mutation_override=proposal,
                    )
                    for proposal in proposals
                ]
                if not children:
                    fallback_child = mutate_candidate(
                        parent_row["candidate"],
                        config,
                        rng,
                        parent_row=parent_row,
                        mode=mode,
                        reflection_plan=None,
                    )
                    fallback_child["parent_reflection"] = reflection_plan
                    children = [fallback_child]
                for child in children:
                    child["parent_source"] = parent_source
            enqueued = False
            for child in children:
                key = candidate_key(child)
                if key in seen:
                    proposal_row = {
                        "index": len(proposal_rows),
                        "mode": mode,
                        "status": "duplicate",
                        "candidate_id": child.get("id", key),
                        "candidate_name": child.get("name", ""),
                        "candidate": child,
                        "reasons": ["duplicate_candidate_structure"],
                    }
                    proposal_rows.append(proposal_row)
                    append_jsonl(proposal_rows_path, proposal_row)
                    continue
                seen.add(key)
                contract_errors = validate_candidate_contract(child, config, models)
                if contract_errors:
                    proposal_row = {
                        "index": len(proposal_rows),
                        "mode": mode,
                        "status": "invalid",
                        "candidate_id": child.get("id", key),
                        "candidate_name": child.get("name", ""),
                        "candidate": child,
                        "reasons": contract_errors,
                    }
                    proposal_rows.append(proposal_row)
                    append_jsonl(proposal_rows_path, proposal_row)
                    continue
                candidate_queue.append(child)
                proposal_row = {
                    "index": len(proposal_rows),
                    "mode": mode,
                    "status": "accepted",
                    "candidate_id": child.get("id", key),
                    "candidate_name": child.get("name", ""),
                    "candidate": child,
                    "reasons": [],
                }
                proposal_rows.append(proposal_row)
                append_jsonl(proposal_rows_path, proposal_row)
                enqueued = True
            if not enqueued:
                continue

        candidate = candidate_queue.pop(0)
        eval_result, cache_status = evaluate_candidate_cached(
            candidate=candidate,
            dataset=searchset,
            models=models,
            profile=profile,
            cache_dir=evaluation_cache_dir,
            max_examples=effective_search_examples,
            eval_concurrency=eval_concurrency,
            reflection_example_limit=reflection_example_limit,
        )
        parent_id = candidate.get("parent_id")
        parent_row = next(
            (row for row in evaluated_rows if row.get("candidate_id") == parent_id),
            None,
        )
        if parent_row is not None:
            candidate["mutation_observation"] = build_mutation_observation(
                parent_row,
                eval_result,
                candidate.get("applied_mutation"),
            )
        row = {
            "index": len(evaluated_rows),
            "mode": mode,
            "candidate_origin": (
                "seed" if candidate_key(candidate) in seed_candidate_keys else "generated"
            ),
            "candidate": candidate,
            "candidate_id": candidate["id"],
            "candidate_name": candidate["name"],
            "topology": candidate["topology"],
            "evaluation_cache_status": cache_status,
            "reflection": reflect_on_candidate(eval_result),
            **eval_result,
        }
        row.update(
            candidate_validity(
                row,
                contract_errors=validate_candidate_contract(candidate, config, models),
            )
        )
        evaluated_rows.append(row)
        append_jsonl(rows_path, row)
        write_search_checkpoint(
            checkpoint_path,
            completed_rows=len(evaluated_rows),
            candidate_queue=candidate_queue,
            seen=seen,
            rng=rng,
        )
        print(
            "[WAN-GEPA] "
            f"{len(evaluated_rows)}/"
            f"{target_candidates} "
            f"{candidate['name']} "
            f"score={row['score']:.3f} calls={row['avg_calls']:.2f} "
            f"tok={row['avg_total_tokens']:.1f} cross={row['avg_cross_center_tokens']:.1f} "
            f"maxed={row.get('avg_maxed_calls', 0.0):.2f} "
            f"valid={int(row['is_valid_candidate'])}/answer={row.get('valid_answer_rate', 1.0):.3f} "
            f"dag={row.get('avg_dag_subtasks', 0.0):.1f}/fb={row.get('dag_fallback_rate', 0.0):.2f} "
            f"cache={cache_status} "
            f"wall={row.get('avg_emulated_wall_latency_ms', row['avg_emulated_latency_ms']):.1f}ms "
            f"net={row.get('avg_network_latency_ms', 0.0):.1f}ms "
            f"emu={row['avg_emulated_latency_ms']:.1f}ms",
            flush=True,
        )

    shortlisted_rows, shortlist_policy = shortlist_rows_for_selection(
        evaluated_rows,
        mode=mode,
        shortlist_size=selection_shortlist_size,
        selection_strategy=selection_strategy,
        quality_band=quality_band,
    )

    selection_rows: list[dict[str, Any]] = []
    if selection_rows_path.exists():
        selection_rows_path.unlink()
    for selection_index, search_row in enumerate(shortlisted_rows):
        candidate = search_row["candidate"]
        selection_result, selection_cache_status = evaluate_candidate_cached(
            candidate=candidate,
            dataset=selectionset,
            models=models,
            profile=profile,
            cache_dir=evaluation_cache_dir,
            eval_concurrency=eval_concurrency,
            reflection_example_limit=reflection_example_limit,
        )
        selection_row = {
            "index": selection_index,
            "mode": mode,
            "candidate": candidate,
            "candidate_id": candidate["id"],
            "candidate_name": candidate["name"],
            "topology": candidate["topology"],
            "candidate_origin": search_row.get("candidate_origin", ""),
            "evaluation_cache_status": selection_cache_status,
            "search": strip_outputs(search_row),
            **selection_result,
        }
        selection_row.update(
            candidate_validity(
                selection_row,
                contract_errors=validate_candidate_contract(candidate, config, models),
            )
        )
        selection_rows.append(selection_row)
        append_jsonl(selection_rows_path, selection_row)
        print(
            "[WAN-GEPA] SELECT "
            f"{selection_index + 1}/{len(shortlisted_rows)} {candidate['name']} "
            f"score={selection_row['score']:.3f} calls={selection_row['avg_calls']:.2f} "
            f"tok={selection_row['avg_total_tokens']:.1f} "
            f"valid={int(selection_row['is_valid_candidate'])} cache={selection_cache_status}",
            flush=True,
        )

    operating_point_rows = select_operating_points(
        selection_rows,
        efficiency_quality_delta=EFFICIENCY_QUALITY_DELTA,
    )
    quality_row = operating_point_rows["quality"]
    efficiency_row = operating_point_rows["efficiency"]
    selected_rows = [quality_row]
    if efficiency_row["candidate_id"] != quality_row["candidate_id"]:
        selected_rows.append(efficiency_row)
    operating_point_labels: dict[str, list[str]] = {
        quality_row["candidate_id"]: ["Q"],
    }
    operating_point_labels.setdefault(efficiency_row["candidate_id"], []).append("E")
    selection_policy = f"protocol_q_e.delta={EFFICIENCY_QUALITY_DELTA:g}"
    write_json(output_dir / "selected_quality_candidate.json", quality_row["candidate"])
    write_json(output_dir / "selected_efficiency_candidate.json", efficiency_row["candidate"])

    test_rows: list[dict[str, Any]] = []
    for selected_rank, row in enumerate(selected_rows):
        candidate = row["candidate"]
        test_result, test_cache_status = evaluate_candidate_cached(
            candidate=candidate,
            dataset=testset,
            models=models,
            profile=profile,
            cache_dir=evaluation_cache_dir,
            capture_outputs=True,
            eval_concurrency=eval_concurrency,
        )
        test_validity = candidate_validity(
            test_result,
            contract_errors=validate_candidate_contract(candidate, config, models),
        )
        test_result.update(test_validity)
        test_row = {
            "selected_rank": selected_rank,
            "operating_points": operating_point_labels[candidate["id"]],
            "candidate_id": candidate["id"],
            "candidate_name": candidate["name"],
            "topology": candidate["topology"],
            "candidate": candidate,
            "evaluation_cache_status": test_cache_status,
            "selection": strip_outputs(row),
            "val": strip_outputs(row),
            "test": test_result,
        }
        test_rows.append(test_row)
        write_json(output_dir / f"test_outputs_{candidate['id']}.json", test_result.get("outputs", []))
        print(
            "[WAN-GEPA] TEST "
            f"rank={selected_rank} {candidate['name']} "
            f"score={test_result['score']:.3f} calls={test_result['avg_calls']:.2f} "
            f"tok={test_result['avg_total_tokens']:.1f} cross={test_result['avg_cross_center_tokens']:.1f} "
            f"maxed={test_result.get('avg_maxed_calls', 0.0):.2f} "
            f"valid={int(test_result['is_valid_candidate'])}/answer={test_result.get('valid_answer_rate', 1.0):.3f} "
            f"cache={test_cache_status} "
            f"dag={test_result.get('avg_dag_subtasks', 0.0):.1f}/"
            f"fb={test_result.get('dag_fallback_rate', 0.0):.2f} "
            f"wall={test_result.get('avg_emulated_wall_latency_ms', test_result['avg_emulated_latency_ms']):.1f}ms "
            f"net={test_result.get('avg_network_latency_ms', 0.0):.1f}ms "
            f"emu={test_result['avg_emulated_latency_ms']:.1f}ms",
            flush=True,
        )

    search_overhead_summary = summarize_search_overhead(search_overhead_records)
    proposal_summary = {
        "attempts": len(proposal_rows),
        "accepted": sum(row.get("status") == "accepted" for row in proposal_rows),
        "duplicates": sum(row.get("status") == "duplicate" for row in proposal_rows),
        "invalid": sum(row.get("status") == "invalid" for row in proposal_rows),
    }
    candidate_evaluation_overhead = {
        "calls": sum(float(row.get("sum_calls", 0.0)) for row in evaluated_rows),
        "prompt_tokens": sum(float(row.get("sum_prompt_tokens", 0.0)) for row in evaluated_rows),
        "completion_tokens": sum(float(row.get("sum_completion_tokens", 0.0)) for row in evaluated_rows),
        "total_tokens": sum(float(row.get("sum_total_tokens", 0.0)) for row in evaluated_rows),
        "cost_usd": sum(float(row.get("sum_inference_cost_usd", 0.0)) for row in evaluated_rows),
    }
    operating_points = {
        "Q": {
            "candidate_id": quality_row["candidate_id"],
            "candidate_name": quality_row["candidate_name"],
            "selection": strip_outputs(quality_row),
        },
        "E": {
            "candidate_id": efficiency_row["candidate_id"],
            "candidate_name": efficiency_row["candidate_name"],
            "selection": strip_outputs(efficiency_row),
            "quality_delta": EFFICIENCY_QUALITY_DELTA,
        },
    }
    model_manifest = model_manifest_payload(models)
    run_manifest = {
        "experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "evaluation_cache_version": EVALUATION_CACHE_VERSION,
        "code_commit": os.environ.get("GEPA_CODE_COMMIT", ""),
        "runtime_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "formal_result": False,
        "formal_result_reason": "formal G1-G9 gates must be verified by the formal aggregator",
        "mode": mode,
        "search_seed": seed,
        "data_seed": (metadata or {}).get("data_seed", ""),
        "config_sha256": sha256_json(scientific_config_payload(config)),
        "model_manifest_sha256": sha256_json(model_manifest),
        "model_manifest": model_manifest,
        "network_profile_sha256": sha256_json(asdict(profile)),
        "network_profile": asdict(profile),
        "dataset_splits": {
            "search": normalized_split_manifest(searchset),
            "selection": normalized_split_manifest(selectionset),
            "test": normalized_split_manifest(testset),
        },
        "metadata": metadata or {},
    }
    result = {
        "metadata": metadata or {},
        "mode": mode,
        "seed": seed,
        "network_profile": asdict(profile),
        "num_candidates": len(evaluated_rows),
        "num_seed_candidates": sum(
            row.get("candidate_origin") == "seed" for row in evaluated_rows
        ),
        "num_new_candidates": sum(
            row.get("candidate_origin") == "generated" for row in evaluated_rows
        ),
        "seed_candidate_budget": seed_candidate_budget,
        "new_candidate_budget": new_candidate_budget,
        "target_candidates": target_candidates,
        "search_examples": effective_search_examples,
        "selection_examples": len(selectionset),
        "selection_shortlist_size": selection_shortlist_size,
        "test_top_k": test_top_k,
        "eval_concurrency": eval_concurrency,
        "selection_strategy": selection_strategy,
        "quality_band": quality_band,
        "pareto_parent_prob": pareto_parent_prob,
        "parent_score_band": parent_score_band,
        "parent_top_k": parent_top_k,
        "reflection_mode": reflection_mode,
        "reflection_model": resolve_reflection_model_name(config, reflection_model) or "",
        "reflection_max_tokens": reflection_max_tokens,
        "reflection_children": reflection_children,
        "reflection_example_limit": reflection_example_limit,
        "evaluation_cache": {
            "enabled": evaluation_cache_dir is not None,
            "directory": str(evaluation_cache_dir) if evaluation_cache_dir is not None else "",
            "version": EVALUATION_CACHE_VERSION,
            "scope": "search, selection, and test candidate evaluations shared across modes",
        },
        "resumed": resume,
        "resume_strategy": resume_strategy,
        "search_shortlist_policy": shortlist_policy,
        "selection_policy": selection_policy,
        "operating_points": operating_points,
        "validity_gate": {
            "min_valid_execution_rate": MIN_VALID_EXECUTION_RATE,
            "max_error_example_rate": MAX_ERROR_EXAMPLE_RATE,
            "max_protocol_error_rate": MAX_PROTOCOL_ERROR_RATE,
            "max_truncated_unextractable_rate": MAX_TRUNCATED_UNEXTRACTABLE_RATE,
            "invalid_search_candidates": sum(not is_candidate_eligible(row) for row in evaluated_rows),
            "invalid_selection_candidates": sum(not is_candidate_eligible(row) for row in selection_rows),
        },
        "proposal_summary": proposal_summary,
        "search_overhead": search_overhead_summary,
        "candidate_evaluation_overhead": candidate_evaluation_overhead,
        "run_manifest_hash": sha256_json(run_manifest),
        "wan_emulation": {
            "type": "trace_based_profile_estimate",
            "description": (
                "No wall-clock sleep or stochastic packet drops are injected. "
                "Each rollout trace is mapped to WAN-profile message latency, model-RPC latency, "
                "and expected retry penalty for reproducible architecture search."
            ),
            "network_latency_ms": (
                "message latency + remote model RPC latency + deterministic expected retry penalty"
            ),
            "emulated_latency_ms": "observed model latency + network latency + site compute latency",
        },
        "primary_selection": (
            "Q and E are frozen on disjoint D_select before D_test. selected_test_rows[0] is Q for compatibility."
        ),
        "duration_seconds": time.time() - start_time,
        "search_pareto_front_ids": [row["candidate_id"] for row in pareto_front(evaluated_rows)],
        "selection_pareto_front_ids": [row["candidate_id"] for row in pareto_front(selection_rows)],
        "pareto_front_ids": [row["candidate_id"] for row in pareto_front(selection_rows)],
        "search_rows": [strip_outputs(row) for row in evaluated_rows],
        "selection_rows": [strip_outputs(row) for row in selection_rows],
        "selected_test_rows": test_rows,
    }
    write_json(output_dir / "result.json", result)
    write_summary_csv(output_dir / "summary.csv", result)
    write_json(
        output_dir / "search_overhead.json",
        {"summary": search_overhead_summary, "records": search_overhead_records},
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "best_candidate.json", quality_row["candidate"])
    return result


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(strip_outputs(payload), ensure_ascii=False) + "\n")


def load_partial_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def write_search_checkpoint(
    path: Path,
    *,
    completed_rows: int,
    candidate_queue: list[dict[str, Any]],
    seen: set[str],
    rng: random.Random,
) -> None:
    payload = {
        "version": 1,
        "completed_rows": completed_rows,
        "candidate_queue": candidate_queue,
        "seen": sorted(seen),
        "rng_state": rng.getstate(),
    }
    temporary_path = path.with_suffix(".tmp")
    write_json(temporary_path, payload)
    temporary_path.replace(path)


def strip_outputs(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    cleaned.pop("outputs", None)
    if "test" in cleaned and isinstance(cleaned["test"], dict):
        cleaned["test"] = dict(cleaned["test"])
        cleaned["test"].pop("outputs", None)
    return cleaned


def write_summary_csv(path: Path, result: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    metadata = result.get("metadata", {})
    network_profile_name = result.get("network_profile", {}).get("name", "")
    common = {
        "dataset": metadata.get("dataset", ""),
        "test_name": metadata.get("test_name", ""),
        "network_profile": metadata.get("network_profile", network_profile_name),
        "mode": result["mode"],
        "seed": result.get("seed", metadata.get("seed", "")),
        "selection_strategy": result.get("selection_strategy", ""),
        "quality_band": result.get("quality_band", ""),
        "pareto_parent_prob": result.get("pareto_parent_prob", ""),
        "parent_score_band": result.get("parent_score_band", ""),
        "parent_top_k": result.get("parent_top_k", ""),
        "reflection_mode": result.get("reflection_mode", ""),
        "reflection_model": result.get("reflection_model", ""),
        "reflection_max_tokens": result.get("reflection_max_tokens", ""),
        "reflection_children": result.get("reflection_children", ""),
        "reflection_example_limit": result.get("reflection_example_limit", ""),
        "prompt_protocol_version": metadata.get("prompt_protocol_version", ""),
        "selection_policy": result.get("selection_policy", ""),
        "run_manifest_hash": result.get("run_manifest_hash", ""),
        "controller_calls": result.get("search_overhead", {}).get("controller_calls", 0),
        "controller_total_tokens": result.get("search_overhead", {}).get("controller_total_tokens", 0),
        "controller_cost_usd": result.get("search_overhead", {}).get("controller_cost_usd", 0.0),
    }

    def append_metric_row(
        *,
        split: str,
        row: dict[str, Any],
        selected_rank: int | str = "",
        operating_point: str = "",
    ) -> None:
        rows.append(
            {
                **common,
                "split": split,
                "selected_rank": selected_rank,
                "operating_point": operating_point,
                "candidate_id": row["candidate_id"],
                "candidate_name": row["candidate_name"],
                "topology": row["topology"],
                "score": row["score"],
                "correct": row["correct"],
                "num_examples": row["num_examples"],
                "avg_calls": row["avg_calls"],
                "avg_total_tokens": row["avg_total_tokens"],
                "avg_prompt_tokens": row.get("avg_prompt_tokens", 0.0),
                "avg_completion_tokens": row.get("avg_completion_tokens", 0.0),
                "avg_inference_cost_usd": row.get("avg_inference_cost_usd", 0.0),
                "avg_requested_max_tokens": row.get("avg_requested_max_tokens", 0.0),
                "avg_cross_center_tokens": row["avg_cross_center_tokens"],
                "avg_cross_center_message_tokens": row.get("avg_cross_center_message_tokens", 0.0),
                "avg_cross_center_model_tokens": row.get("avg_cross_center_model_tokens", 0.0),
                "avg_network_latency_ms": row.get("avg_network_latency_ms", 0.0),
                "avg_message_latency_ms": row.get("avg_message_latency_ms", 0.0),
                "avg_message_expected_retry_latency_ms": row.get("avg_message_expected_retry_latency_ms", 0.0),
                "avg_model_rpc_latency_ms": row.get("avg_model_rpc_latency_ms", 0.0),
                "avg_model_rpc_expected_retry_latency_ms": row.get("avg_model_rpc_expected_retry_latency_ms", 0.0),
                "avg_expected_retry_latency_ms": row.get("avg_expected_retry_latency_ms", 0.0),
                "avg_observed_model_latency_ms": row.get("avg_observed_model_latency_ms", 0.0),
                "avg_site_compute_latency_ms": row.get("avg_site_compute_latency_ms", 0.0),
                "avg_expected_network_failures": row.get("avg_expected_network_failures", 0.0),
                "avg_emulated_latency_ms": row["avg_emulated_latency_ms"],
                "is_valid_candidate": row.get("is_valid_candidate", True),
                "valid_execution_rate": row.get("valid_execution_rate", 1.0),
                "valid_answer_rate": row.get("valid_answer_rate", 1.0),
                "error_example_rate": row.get("error_example_rate", 0.0),
                "protocol_error_rate": row.get("protocol_error_rate", 0.0),
                "truncated_unextractable_rate": row.get("truncated_unextractable_rate", 0.0),
                "invalid_reasons": "|".join(row.get("invalid_reasons", [])),
            }
        )

    for row in result.get("search_rows", []):
        append_metric_row(split="search", row=row)
    for row in result.get("selection_rows", []):
        append_metric_row(split="selection", row=row)
    for row in result.get("selected_test_rows", []):
        test = row["test"]
        for operating_point in row.get("operating_points", [""]):
            append_metric_row(
                split="test",
                row={
                    "candidate_id": row["candidate_id"],
                    "candidate_name": row["candidate_name"],
                    "topology": row["topology"],
                    **test,
                },
                selected_rank=row.get("selected_rank", ""),
                operating_point=operating_point,
            )
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_site_penalties(sites: dict[str, SiteSpec], orchestrator_site: str = "center_a") -> None:
    global ORCHESTRATOR_SITE
    ORCHESTRATOR_SITE = orchestrator_site
    SITE_COMPUTE_LATENCY_MS.clear()
    for name, site in sites.items():
        SITE_COMPUTE_LATENCY_MS[name] = site.compute_latency_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAN-aware training-free multi-agent architecture search.")
    parser.add_argument("--config", default="experiments/phase2_wan_agent_config.json")
    parser.add_argument(
        "--mode",
        choices=["baselines", "random", "aflow_style", "adas_style", "quality_only", "wan_pareto"],
        default="baselines",
    )
    parser.add_argument("--network-profile", default="wan_normal")
    parser.add_argument(
        "--topologies",
        nargs="+",
        help="Optional subset of configured topology names for focused calibration or ablation runs.",
    )
    parser.add_argument("--dataset", choices=["aime", "generic_jsonl", "gaia", "masbench"], default="aime")
    parser.add_argument("--data-dir", default=os.environ.get("GEPA_AIME_DATA_DIR", DEFAULT_AIME_DATA_DIR))
    parser.add_argument("--aime-train-file", default=DEFAULT_AIME_TRAIN_FILE)
    parser.add_argument("--aime-test-file", default=DEFAULT_AIME_TEST_FILE)
    parser.add_argument("--gaia-data-dir", default=os.environ.get("GEPA_GAIA_DATA_DIR", DEFAULT_GAIA_DATA_DIR))
    parser.add_argument("--gaia-split", default=DEFAULT_GAIA_SPLIT)
    parser.add_argument(
        "--gaia-include-attachments",
        action="store_true",
        help="Include GAIA examples with attachments. Keep off until file/tool runtime is enabled.",
    )
    parser.add_argument(
        "--masbench-data-dir",
        default=os.environ.get("GEPA_MASBENCH_DATA_DIR", DEFAULT_MASBENCH_DATA_DIR),
    )
    parser.add_argument("--masbench-axis", choices=MASBENCH_AXES, default="depth")
    parser.add_argument(
        "--masbench-test-values",
        nargs="*",
        default=[],
        help="Optional official test difficulty values, for example: --masbench-test-values 2 4.",
    )
    parser.add_argument("--generic-train-file")
    parser.add_argument("--generic-val-file")
    parser.add_argument("--generic-test-file")
    parser.add_argument("--train-size", type=int, default=30)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument(
        "--search-size",
        type=int,
        help="Size of disjoint D_search. Must be paired with --selection-size for protocol-v1 runs.",
    )
    parser.add_argument(
        "--selection-size",
        type=int,
        help="Size of disjoint D_select. Must be paired with --search-size for protocol-v1 runs.",
    )
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument(
        "--dataset-manifest",
        help="Optional frozen dataset manifest. Its content hash is recorded without a local path.",
    )
    parser.add_argument(
        "--search-examples",
        type=int,
        help="Optional cap within D_search. Protocol-v1 formal runs evaluate the full D_search split.",
    )
    parser.add_argument("--seed-candidates", type=int, default=9)
    parser.add_argument("--new-candidate-budget", type=int, default=24)
    parser.add_argument("--selection-shortlist-size", type=int, default=8)
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Deprecated pilot-only total candidate cap. Use --seed-candidates and --new-candidate-budget.",
    )
    parser.add_argument("--test-top-k", type=int, default=1)
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=1,
        help="Concurrent examples evaluated against one vLLM endpoint. Use 4-8 to enable continuous batching.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=2026,
        help="Fixed dataset partition seed, independent of search randomness.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["quality_band_cost", "utility"],
        default="quality_band_cost",
        help="Primary validation-time selection rule. quality_band_cost is recommended for final WAN-GEPA tables.",
    )
    parser.add_argument(
        "--quality-band",
        type=float,
        default=0.0,
        help=(
            "Allowed validation-score drop when selecting lower-cost candidates. "
            "0 means tie on best validation score and is recommended for main tables."
        ),
    )
    parser.add_argument(
        "--pareto-parent-prob",
        type=float,
        default=0.5,
        help="For wan_pareto, probability of sampling mutation parents from the strict Pareto front.",
    )
    parser.add_argument(
        "--parent-score-band",
        type=float,
        default=0.05,
        help="For wan_pareto, validation-score band used to keep high-quality parents in the mutation pool.",
    )
    parser.add_argument(
        "--parent-top-k",
        type=int,
        default=6,
        help="For wan_pareto, maximum number of high-score-band parents used for non-Pareto parent sampling.",
    )
    parser.add_argument(
        "--reflection-mode",
        choices=["rule", "llm"],
        default="llm",
        help="Use rule-based reflection or an LLM reflector to propose typed architecture mutations.",
    )
    parser.add_argument(
        "--reflection-model",
        help="Configured model alias used for LLM reflection. Defaults to config reflection/defaults model.",
    )
    parser.add_argument("--reflection-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--reflection-children",
        type=int,
        default=3,
        help="Maximum number of distinct child architectures instantiated from one reflection.",
    )
    parser.add_argument(
        "--reflection-example-limit",
        type=int,
        default=3,
        help="Maximum number of validation failures included in architecture reflection.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from search_rows.jsonl and search_checkpoint.json without reevaluating completed candidates.",
    )
    parser.add_argument(
        "--no-shared-evaluation-cache",
        action="store_true",
        help="Disable the validation-evaluation cache shared by modes in the same experiment/profile/seed.",
    )
    parser.add_argument("--output-dir", default="outputs/phase2_wan_agent")
    parser.add_argument(
        "--flat-output-dir",
        action="store_true",
        help="Do not append seed_<N> under the mode directory. Use only for backwards-compatible one-off runs.",
    )
    return parser.parse_args()


def load_datasets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    explicit_search_size = getattr(args, "search_size", None)
    explicit_selection_size = getattr(args, "selection_size", None)
    if (explicit_search_size is None) != (explicit_selection_size is None):
        raise ValueError("--search-size and --selection-size must be provided together")
    search_size = explicit_search_size if explicit_search_size is not None else args.train_size
    selection_size = explicit_selection_size if explicit_selection_size is not None else args.val_size
    data_seed = getattr(args, "data_seed", args.seed)
    data_dir = Path(args.data_dir)
    if args.dataset == "aime":
        return load_aime_dataset(
            data_dir=data_dir,
            train_file=args.aime_train_file,
            test_file=args.aime_test_file,
            train_size=search_size,
            val_size=selection_size,
            test_size=args.test_size,
            seed=data_seed,
        )
    if args.dataset == "gaia":
        return load_gaia_dataset(
            data_dir=Path(args.gaia_data_dir),
            split=args.gaia_split,
            train_size=search_size,
            val_size=selection_size,
            test_size=args.test_size,
            seed=data_seed,
            include_attachments=args.gaia_include_attachments,
        )
    if args.dataset == "masbench":
        return load_masbench_dataset(
            data_dir=Path(args.masbench_data_dir),
            axis=args.masbench_axis,
            train_size=search_size,
            val_size=selection_size,
            test_size=args.test_size,
            seed=data_seed,
            test_values=tuple(args.masbench_test_values),
        )
    if not args.generic_train_file or not args.generic_val_file or not args.generic_test_file:
        raise ValueError("generic_jsonl requires --generic-train-file, --generic-val-file, and --generic-test-file")
    trainset = load_generic_jsonl_dataset(
        path=Path(args.generic_train_file),
        limit=search_size,
        id_prefix="search",
    )
    valset = load_generic_jsonl_dataset(
        path=Path(args.generic_val_file),
        limit=selection_size,
        id_prefix="selection",
    )
    testset = load_generic_jsonl_dataset(
        path=Path(args.generic_test_file),
        limit=args.test_size,
        id_prefix="test",
    )
    return trainset, valset, testset


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    raw_config = read_json(config_path)
    if args.topologies:
        configured_topologies = set(raw_config.get("allowed_topologies", []))
        unknown_topologies = set(args.topologies) - configured_topologies
        if unknown_topologies:
            raise ValueError(
                f"Unknown --topologies values {sorted(unknown_topologies)}. "
                f"Configured values: {sorted(configured_topologies)}"
            )
        raw_config["allowed_topologies"] = list(dict.fromkeys(args.topologies))
    models = load_models(raw_config["models"])
    sites = load_sites(raw_config["sites"])
    profiles = load_network_profiles(raw_config["network_profiles"])
    if args.network_profile not in profiles:
        raise KeyError(f"Unknown network profile {args.network_profile}. Available: {sorted(profiles)}")
    configure_site_penalties(sites, raw_config.get("defaults", {}).get("orchestrator_site", "center_a"))
    loaded_searchset, loaded_selectionset, testset = load_datasets(args)
    explicit_disjoint_split = args.search_size is not None and args.selection_size is not None
    if explicit_disjoint_split:
        searchset = loaded_searchset
        selectionset = loaded_selectionset
        split_protocol = "disjoint_search_selection_v1"
    elif loaded_searchset:
        searchset = loaded_searchset
        selectionset = loaded_selectionset
        split_protocol = "legacy_loader_disjoint"
    else:
        # Preserve old pilot commands such as --train-size 0 --val-size 90.
        search_cap = args.search_examples or len(loaded_selectionset)
        searchset = loaded_selectionset[:search_cap]
        selectionset = loaded_selectionset[search_cap:]
        split_protocol = "legacy_auto_disjoint_from_validation"
        print(
            "[WAN-GEPA] WARNING: inferred disjoint D_search/D_select from legacy validation flags. "
            "Use explicit --search-size and --selection-size for protocol-v1 formal runs.",
            flush=True,
        )
    seed_candidate_budget = args.seed_candidates
    new_candidate_budget = args.new_candidate_budget
    if args.max_candidates is not None:
        available_seed_count = min(seed_candidate_budget, len(seed_architectures(raw_config)))
        new_candidate_budget = max(0, args.max_candidates - available_seed_count)
        print(
            "[WAN-GEPA] WARNING: --max-candidates is deprecated; resolved to "
            f"seed_candidates={available_seed_count}, new_candidate_budget={new_candidate_budget}.",
            flush=True,
        )
    if args.dataset == "aime":
        test_name = args.aime_test_file.replace(".jsonl", "")
    elif args.dataset == "gaia":
        test_name = f"{args.gaia_split}_local_split"
    elif args.dataset == "masbench":
        value_suffix = (
            "_values_" + "-".join(args.masbench_test_values)
            if args.masbench_test_values
            else ""
        )
        test_name = f"{args.masbench_axis}_official_test{value_suffix}"
    else:
        test_name = Path(args.generic_test_file).stem if args.generic_test_file else "generic_test"
    output_dir = Path(args.output_dir) / args.dataset / test_name / args.network_profile / args.mode
    if not args.flat_output_dir:
        output_dir = output_dir / f"seed_{args.seed}"
    evaluation_cache_dir = None
    if not args.no_shared_evaluation_cache:
        profile_dir = (
            output_dir.parent.parent
            if not args.flat_output_dir
            else output_dir.parent
        )
        evaluation_cache_dir = profile_dir / "_shared_eval_cache" / f"seed_{args.seed}"
    dataset_manifest_sha256 = ""
    dataset_manifest_name = ""
    if args.dataset_manifest:
        dataset_manifest_path = Path(args.dataset_manifest)
        if not dataset_manifest_path.is_file():
            raise FileNotFoundError(f"dataset manifest is missing: {dataset_manifest_path}")
        dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
        dataset_manifest_name = dataset_manifest_path.name
    metadata = {
        "config_path": str(config_path),
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "aime_train_file": args.aime_train_file if args.dataset == "aime" else "",
        "aime_test_file": args.aime_test_file if args.dataset == "aime" else "",
        "gaia_data_dir": args.gaia_data_dir if args.dataset == "gaia" else "",
        "gaia_split": args.gaia_split if args.dataset == "gaia" else "",
        "gaia_include_attachments": args.gaia_include_attachments if args.dataset == "gaia" else "",
        "masbench_data_dir": args.masbench_data_dir if args.dataset == "masbench" else "",
        "masbench_axis": args.masbench_axis if args.dataset == "masbench" else "",
        "masbench_test_values": args.masbench_test_values if args.dataset == "masbench" else [],
        "masbench_protocol_version": MASBENCH_PROTOCOL_VERSION if args.dataset == "masbench" else "",
        "generic_train_file": args.generic_train_file or "",
        "generic_val_file": args.generic_val_file or "",
        "generic_test_file": args.generic_test_file or "",
        "test_name": test_name,
        "network_profile": args.network_profile,
        "topologies": raw_config.get("allowed_topologies", []),
        "mode": args.mode,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "dataset_manifest_name": dataset_manifest_name,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "search_size": len(searchset),
        "selection_size": len(selectionset),
        "split_protocol": split_protocol,
        "experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "test_size": args.test_size,
        "search_examples": args.search_examples or len(searchset),
        "seed_candidates": seed_candidate_budget,
        "new_candidate_budget": new_candidate_budget,
        "selection_shortlist_size": args.selection_shortlist_size,
        "deprecated_max_candidates": args.max_candidates,
        "test_top_k": args.test_top_k,
        "eval_concurrency": args.eval_concurrency,
        "selection_strategy": args.selection_strategy,
        "quality_band": args.quality_band,
        "pareto_parent_prob": args.pareto_parent_prob,
        "parent_score_band": args.parent_score_band,
        "parent_top_k": args.parent_top_k,
        "reflection_mode": args.reflection_mode,
        "reflection_model": resolve_reflection_model_name(raw_config, args.reflection_model) or "",
        "reflection_max_tokens": args.reflection_max_tokens,
        "reflection_children": args.reflection_children,
        "reflection_example_limit": args.reflection_example_limit,
        "resume": args.resume,
        "shared_evaluation_cache": not args.no_shared_evaluation_cache,
        "evaluation_cache_dir": str(evaluation_cache_dir) if evaluation_cache_dir is not None else "",
        "prompt_mode": os.environ.get("GEPA_PHASE2_PROMPT_MODE", "deliberate"),
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "output_dir": str(output_dir),
        "reporting_rule": "Main tables must report selected_rank=0, selected before test evaluation.",
    }
    print(
        "[WAN-GEPA] "
        f"mode={args.mode} profile={args.network_profile} "
        f"seed={args.seed} search={len(searchset)} select={len(selectionset)} "
        f"test={len(testset)} split={split_protocol} output={output_dir}",
        flush=True,
    )
    result = run_search(
        config=raw_config,
        models=models,
        profile=profiles[args.network_profile],
        searchset=searchset,
        selectionset=selectionset,
        testset=testset,
        mode=args.mode,
        seed_candidate_budget=seed_candidate_budget,
        new_candidate_budget=new_candidate_budget,
        search_examples=args.search_examples,
        selection_shortlist_size=args.selection_shortlist_size,
        test_top_k=args.test_top_k,
        eval_concurrency=args.eval_concurrency,
        output_dir=output_dir,
        seed=args.seed,
        selection_strategy=args.selection_strategy,
        quality_band=args.quality_band,
        pareto_parent_prob=args.pareto_parent_prob,
        parent_score_band=args.parent_score_band,
        parent_top_k=args.parent_top_k,
        reflection_mode=args.reflection_mode,
        reflection_model=args.reflection_model,
        reflection_max_tokens=args.reflection_max_tokens,
        reflection_children=args.reflection_children,
        reflection_example_limit=args.reflection_example_limit,
        evaluation_cache_dir=evaluation_cache_dir,
        resume=args.resume,
        metadata=metadata,
    )
    primary = result.get("selected_test_rows", [])
    if primary:
        best = primary[0]
        test = best["test"]
        print(
            "[WAN-GEPA] Primary validation-selected candidate: "
            f"{best['candidate_name']} score={test['score']:.3f} "
            f"cross={test['avg_cross_center_tokens']:.1f} "
            f"net={test.get('avg_network_latency_ms', 0.0):.1f}ms "
            f"emu={test['avg_emulated_latency_ms']:.1f}ms",
            flush=True,
        )
    print(f"[WAN-GEPA] Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
