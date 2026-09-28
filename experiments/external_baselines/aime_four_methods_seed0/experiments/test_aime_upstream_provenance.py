from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import native_aime_formal as formal
import native_external_methods as external


def _git_responses(*values):
    return [SimpleNamespace(stdout=value) for value in values]


@pytest.mark.parametrize(
    "module,method,origin,commit",
    [
        (formal, "aflow", "git@github.com:FoundationAgents/AFlow.git", formal.UPSTREAMS["aflow"]["commit"]),
        (formal, "maas", "https://github.com/bingreeky/MaAS", formal.UPSTREAMS["maas"]["commit"]),
        (external, "adas", "git@github.com:ShengranHu/ADAS.git", external.UPSTREAM_COMMITS["adas"]),
        (external, "gdesigner", "https://github.com/yanweiyue/GDesigner.git", external.UPSTREAM_COMMITS["gdesigner"]),
    ],
)
def test_verified_upstream_accepts_only_expected_clean_origin_and_commit(
    tmp_path: Path, module, method, origin, commit
):
    git = formal.verify_pinned_upstream if module is formal else external.verify_upstream_source
    with patch.object(
        module.subprocess,
        "run",
        side_effect=_git_responses(commit, origin, ""),
    ):
        provenance = git(method, tmp_path)
    assert provenance["commit"] == commit
    assert provenance["clean_worktree"] is True
    assert provenance["origin"] == origin


@pytest.mark.parametrize(
    "module,method,outputs",
    [
        (formal, "aflow", [formal.UPSTREAMS["aflow"]["commit"], "https://evil.invalid/AFlow.git", ""]),
        (external, "gdesigner", [external.UPSTREAM_COMMITS["gdesigner"], "https://github.com/yanweiyue/GDesigner.git", " M prompt.py"]),
        (external, "adas", ["not-the-pinned-commit", external.UPSTREAM_REPOSITORIES["adas"], ""]),
    ],
)
def test_upstream_verifier_fails_closed_on_origin_commit_or_dirty_tree(tmp_path: Path, module, method, outputs):
    git = formal.verify_pinned_upstream if module is formal else external.verify_upstream_source
    with patch.object(module.subprocess, "run", side_effect=_git_responses(*outputs)):
        with pytest.raises(RuntimeError):
            git(method, tmp_path)
