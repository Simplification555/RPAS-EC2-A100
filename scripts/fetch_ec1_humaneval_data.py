"""Materialize and hash the EC-1 HumanEval provenance bundle.

The files are deliberately downloaded outside the repository's tracked tree.
The script records hashes and source revisions but never adds benchmark data to
Git, because the upstream licenses and redistribution terms differ.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


OFFICIAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
OFFICIAL_REPOSITORY = "https://github.com/openai/human-eval.git"
AFLOW_ARCHIVE_URL = "https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e"
AFLOW_MIRROR_REPOSITORY = "https://github.com/CitrusYL/AgentSlimming.git"
OFFICIAL_GZ_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
AFLOW_COMMIT = "0bb1afc677e3751e09dc535e373f0316b0a8369f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def materialize_git_file(repository: str, revision: str, source_path: str, target: Path) -> str:
    """Copy one pinned Git artifact without weakening TLS verification.

    Some managed clusters intercept direct HTTPS downloads with an untrusted
    certificate.  GitHub access through the configured Git transport remains
    verified, so this is a provenance-preserving fallback rather than an
    insecure download workaround.
    """
    with tempfile.TemporaryDirectory(prefix="ec1_humaneval_") as directory:
        checkout = Path(directory) / "source"
        subprocess.run(
            ["git", "clone", "--depth", "1", repository, str(checkout)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        resolved = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if revision != "master" and resolved != revision:
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", revision], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", revision], check=True, stdout=subprocess.DEVNULL)
            resolved = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if revision != "master" and resolved != revision:
            raise RuntimeError(f"resolved unexpected revision for {repository}: {resolved}")
        source = checkout / source_path
        if not source.is_file():
            raise RuntimeError(f"missing pinned Git artifact: {repository}@{revision}:{source_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return resolved


def git_master_revision(repository: str) -> str:
    """Record the upstream master SHA when the direct HTTPS artifact succeeded."""
    output = subprocess.check_output(["git", "ls-remote", repository, "refs/heads/master"], text=True)
    revision = output.split()[0] if output.split() else ""
    if not revision:
        raise RuntimeError(f"could not resolve master revision for {repository}")
    return revision


def fetch(output_dir: Path, *, skip_archive: bool = False) -> dict:
    official = output_dir / "official"
    aflow = output_dir / "aflow"
    official.mkdir(parents=True, exist_ok=True)
    aflow.mkdir(parents=True, exist_ok=True)
    gz_path = official / "HumanEval.jsonl.gz"
    official_revision = None
    if not gz_path.exists():
        try:
            download(OFFICIAL_URL, gz_path)
        except Exception:
            gz_path.unlink(missing_ok=True)
            official_revision = materialize_git_file(OFFICIAL_REPOSITORY, "master", "data/HumanEval.jsonl.gz", gz_path)
    if official_revision is None:
        official_revision = git_master_revision(OFFICIAL_REPOSITORY)
    actual_gz_sha = sha256(gz_path)
    if actual_gz_sha != OFFICIAL_GZ_SHA256:
        raise RuntimeError(f"official HumanEval gzip hash mismatch: {actual_gz_sha}")
    jsonl_path = official / "humaneval.jsonl"
    if not jsonl_path.exists():
        with gzip.open(gz_path, "rb") as source, jsonl_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    archive = aflow / "aflow_data.tar.gz"
    if not skip_archive and not archive.exists():
        try:
            download(AFLOW_ARCHIVE_URL, archive)
        except Exception:
            # The immutable Git mirror below supplies the same AFlow-derived
            # fixtures when Drive is unavailable through a managed network.
            archive.unlink(missing_ok=True)
    extracted: dict[str, str] = {}
    if archive.exists():
        with tarfile.open(archive, "r:gz") as tar:
            names = {member.name.rsplit("/", 1)[-1]: member for member in tar.getmembers()}
            for filename in ("humaneval_validate.jsonl", "humaneval_test.jsonl", "humaneval_public_test.jsonl"):
                member = names.get(filename)
                if member is None:
                    continue
                target = aflow / filename
                with tar.extractfile(member) as source, target.open("wb") as handle:
                    assert source is not None
                    shutil.copyfileobj(source, handle)
                extracted[filename] = str(target)
    # This exact Git mirror is pinned to the AFlow-derived fixture revision.
    # It covers all three fixtures when the Drive archive is unavailable.
    mirror_revision = None
    for filename in ("humaneval_validate.jsonl", "humaneval_test.jsonl", "humaneval_public_test.jsonl"):
        target = aflow / filename
        if not target.exists():
            mirror_revision = materialize_git_file(
                AFLOW_MIRROR_REPOSITORY,
                AFLOW_COMMIT,
                f"data/datasets/{filename}",
                target,
            )
            extracted[filename] = str(target)
    manifest = {
        "benchmark": "HumanEval",
        "official": {"upstream": "openai/human-eval", "version": "1.0.0", "artifact": str(gz_path), "sha256": actual_gz_sha, "git_repository": OFFICIAL_REPOSITORY, "git_revision": official_revision, "derived_jsonl_sha256": sha256(jsonl_path), "tasks": sum(1 for _ in jsonl_path.open(encoding="utf-8"))},
        "aflow": {"archive_url": AFLOW_ARCHIVE_URL, "mirror_repository": AFLOW_MIRROR_REPOSITORY, "mirror_commit": mirror_revision or AFLOW_COMMIT, "files": {}},
    }
    for name in ("humaneval_validate.jsonl", "humaneval_test.jsonl", "humaneval_public_test.jsonl"):
        path = aflow / name
        if path.exists():
            manifest["aflow"]["files"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    (output_dir / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/ec1_humaneval")
    parser.add_argument("--skip-aflow-archive", action="store_true")
    args = parser.parse_args()
    print(json.dumps(fetch(Path(args.output_dir), skip_archive=args.skip_aflow_archive), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
