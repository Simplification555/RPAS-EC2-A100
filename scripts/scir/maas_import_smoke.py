#!/usr/bin/env python3
"""Validate the EC-1 MaAS import path without starting a model service."""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from pathlib import Path

from external_comparison.runners.native_ec1_driver import (
    _install_maas_actions_compat,
    _install_maas_import_compat,
    _install_maas_optional_encoding_compat,
    _install_maas_provider_compat,
    _install_maas_embedding_compat,
)
from external_comparison.adapters.native_runtime import write_maas_config


def main() -> None:
    faulthandler.dump_traceback_later(20, repeat=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    if not (workspace / "maas").is_dir():
        raise SystemExit(f"not a staged MaAS workspace: {workspace}")
    sys.path.insert(0, str(workspace))
    write_maas_config(workspace, "Qwen/Qwen3.5-9B", "http://127.0.0.1:1/v1", "EMPTY", 1)
    os.environ["METAGPT_PROJECT_ROOT"] = str(workspace)
    _install_maas_import_compat(workspace)
    _install_maas_optional_encoding_compat()
    _install_maas_provider_compat(workspace)
    _install_maas_actions_compat(workspace)
    _install_maas_embedding_compat(workspace)
    from maas.configs.models_config import ModelsConfig
    from maas.ext.maas.scripts.optimizer import Optimizer
    from maas.provider.openai_api import OpenAILLM

    assert ModelsConfig is not None and Optimizer is not None and OpenAILLM is not None
    from maas.ext.maas.models.utils import SentenceEncoder, get_sentence_embedding
    import torch

    sentence = "Generate a Python function and verify its public tests."
    vector = get_sentence_embedding(sentence)
    encoder = SentenceEncoder()
    assert vector.shape == (384,) and torch.isfinite(vector).all()
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert torch.allclose(vector, encoder(sentence), atol=1e-6)
    faulthandler.cancel_dump_traceback_later()
    print("maas_import_smoke=PASS")


if __name__ == "__main__":
    main()
