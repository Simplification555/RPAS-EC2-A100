"""Exercise service CLI and teardown without importing/loading GPU dependencies."""

import argparse
import ast
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path("scripts/serve_transformers_qwen35_openai.py")


def service_function(name):
    return next(node for node in ast.walk(ast.parse(SOURCE.read_text(encoding="utf-8")))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


@pytest.mark.parametrize("dtype", ["auto", "float16", "bfloat16"])
def test_launch_dtype_is_accepted(monkeypatch, dtype):
    module = ast.Module(body=[service_function("parse_args")], type_ignores=[])
    namespace = {"argparse": argparse}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    monkeypatch.setattr("sys.argv", [str(SOURCE), "--model", "synthetic", "--dtype", dtype])
    assert namespace["parse_args"]().dtype == dtype


def test_lifespan_shutdown_does_not_delete_unbound_model():
    calls = []

    async def batch_worker():
        await asyncio.Future()

    namespace = {"asynccontextmanager": asynccontextmanager, "asyncio": asyncio,
                 "suppress": suppress, "FastAPI": object, "batch_worker": batch_worker,
                 "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append("cleanup")))}
    module = ast.Module(body=[service_function("lifespan")], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)

    async def exercise():
        async with namespace["lifespan"](None):
            await asyncio.sleep(0)

    asyncio.run(exercise())
    assert calls == ["cleanup"]
