#!/usr/bin/env python3
"""Minimal OpenAI chat-completions server for Qwen3.5 via Transformers.

vLLM 0.12 cannot execute Qwen3_5ForConditionalGeneration. This server keeps
the experimental clients unchanged while using the official implementation.
Requests are serialized because one model replica serves one selected GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def _safe_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """Fallback for causal-conv1d 1.2's incompatible 3-D update wrapper."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    joined = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(joined[:, :, -state_len:])
    out = F.conv1d(joined, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = getattr(F, activation)(out) if hasattr(F, activation) else torch.nn.functional.silu(out)
    return out.to(hidden_states.dtype)


import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35_impl
_qwen35_impl.causal_conv1d_update = _safe_causal_conv1d_update


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = Field(default=1024, ge=1)
    temperature: float | None = Field(default=0.0, ge=0.0)
    stream: bool = False


@dataclass
class PendingRequest:
    request: ChatRequest
    future: asyncio.Future[dict[str, Any]]


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                pieces.append(str(part.get("text", "")))
        return "\n".join(pieces)
    return str(content)


def normalized_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        role = str(message.get("role", "user"))
        normalized.append(
            {"role": role, "content": [{"type": "text", "text": message_text(message)}]}
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum compatible requests generated together on this one GPU.",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=25.0,
        help="Brief queueing interval used to form a dynamic inference batch.",
    )
    parser.add_argument(
        "--stop-string",
        help="Optional explicit terminal marker. It is stripped from returned content after stopping.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    # Decoder-only generation must left-pad batched prompts; right padding
    # changes the position of the final token and corrupts continuations.
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    request_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()

    def request_generation_key(request: ChatRequest) -> tuple[int, bool, float]:
        max_new_tokens = min(request.max_tokens or 1024, args.max_new_tokens)
        do_sample = bool(request.temperature and request.temperature > 0.0)
        return max_new_tokens, do_sample, float(request.temperature or 0.0)

    def generate_batch(items: list[PendingRequest]) -> list[dict[str, Any]]:
        requests = [item.request for item in items]
        max_new_tokens, do_sample, temperature = request_generation_key(requests[0])
        conversations = [normalized_messages(request.messages) for request in requests]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        if args.stop_string:
            generation_kwargs["stop_strings"] = [args.stop_string]
            generation_kwargs["tokenizer"] = processor.tokenizer
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)

        input_width = int(inputs.input_ids.shape[1])
        eos_token_ids = processor.tokenizer.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = {eos_token_ids}
        else:
            eos_token_ids = set(eos_token_ids or [])
        responses = []
        for index, request in enumerate(requests):
            raw_completion_ids = generated[index, input_width:]
            raw_tokens = raw_completion_ids.tolist()
            eos_index = next(
                (token_index for token_index, token_id in enumerate(raw_tokens) if token_id in eos_token_ids),
                None,
            )
            completion_length = (eos_index + 1) if eos_index is not None else len(raw_tokens)
            completion_ids = raw_completion_ids[:completion_length]
            content = processor.batch_decode(
                completion_ids.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            stopped_at_marker = bool(args.stop_string and args.stop_string in content)
            if stopped_at_marker:
                content = content.split(args.stop_string, 1)[0].rstrip()
            prompt_tokens = int(inputs.attention_mask[index].sum().item())
            completion_tokens = int(completion_length)
            finish_reason = "stop" if eos_index is not None or stopped_at_marker else "length"
            responses.append(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model or args.served_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
        return responses

    async def batch_worker() -> None:
        while True:
            first = await request_queue.get()
            # A client can time out while its request waits in the queue.
            # Do not spend a full generation on work whose response can no
            # longer be observed. This matters for upstream search workflows
            # that intentionally abandon over-budget candidates.
            if first.future.done() or first.future.cancelled():
                continue
            pending = [first]
            await asyncio.sleep(max(0.0, args.batch_wait_ms) / 1000.0)
            while len(pending) < args.max_batch_size:
                try:
                    candidate = request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not candidate.future.done() and not candidate.future.cancelled():
                    pending.append(candidate)
            groups: dict[tuple[int, bool, float], list[PendingRequest]] = {}
            for item in pending:
                groups.setdefault(request_generation_key(item.request), []).append(item)
            for items in groups.values():
                items = [item for item in items if not item.future.done() and not item.future.cancelled()]
                if not items:
                    continue
                try:
                    responses = await asyncio.to_thread(generate_batch, items)
                except Exception as exc:  # Surface model/runtime failures to every affected caller.
                    for item in items:
                        if not item.future.done():
                            item.future.set_exception(exc)
                else:
                    for item, response in zip(items, responses):
                        if not item.future.done():
                            item.future.set_result(response)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker = asyncio.create_task(batch_worker())
        try:
            yield
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
        del model
        torch.cuda.empty_cache()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": args.served_model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported")
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        await request_queue.put(PendingRequest(request=request, future=future))
        return await future

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
