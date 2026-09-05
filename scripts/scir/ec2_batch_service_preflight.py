"""Check real resident-service batching on non-benchmark, short synthetic prompts."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI


async def check(base_url: str) -> dict:
    prompts = ["Reply with only the digit 1.", "Compute 1+1. Reply with only the resulting digit.",
               "Compute 1+2. Reply with only the resulting digit, without explanation.",
               "For this simple addition question, give just a single digit and no other words: 2+2."]
    async with AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=600, max_retries=0) as client:
        async def call(prompt):
            started = time.perf_counter()
            response = await client.chat.completions.create(model="Qwen/Qwen3.5-9B", temperature=0.0,
                max_tokens=4, messages=[{"role": "user", "content": prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return {"content": response.choices[0].message.content, "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump(), "latency_ms": (time.perf_counter() - started) * 1000}
        serial = [await call(prompt) for prompt in prompts]
        parallel = await asyncio.gather(*(call(prompt) for prompt in prompts))
    assert [row["content"].strip() for row in serial] == ["1", "2", "3", "4"]
    assert [row["content"] for row in serial] == [row["content"] for row in parallel]
    for left, right in zip(serial, parallel, strict=True):
        assert left["usage"]["prompt_tokens"] == right["usage"]["prompt_tokens"]
        assert left["finish_reason"] == right["finish_reason"] == "stop"
    return {"formal_result": False, "synthetic_only": True, "status": "PASS_REAL_SERVICE_BATCH_CONSISTENCY",
            "base_url": base_url, "serial": serial, "parallel": parallel}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    payload = asyncio.run(check(args.base_url))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({key: value for key, value in payload.items() if key not in {"serial", "parallel"}}))
