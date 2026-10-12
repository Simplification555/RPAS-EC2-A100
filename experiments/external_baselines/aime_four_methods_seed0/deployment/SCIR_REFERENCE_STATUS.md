# SCIR Reference Status: AIME Seed-0 Pilot

This branch packages the external-method AIME runner and tests from source PR head
37f1ac5662123bbd69f796c88d303363a1c96b01. It contains no RPAS method implementation.
The commit adds no benchmark data, model weights, API credentials, or formal prediction
bundle. The parent branch already carries the frozen AIME files and their manifest;
this branch leaves those bytes unchanged.

## What SCIR evidence actually establishes

- In the persisted SCIR A6000 serial-v2 output, AFlow seeds 0, 1, and 2 each have a
  quality_gate.json with status PASS. This is a protocol/operational gate, not a
  claim of scientific superiority or a complete paper result.
- The MaAS seed-0 directory has a native_result.json, but no quality_gate.json.
  Treat it as an incomplete attempt, not a verified pass.
- A complete four-method formal AIME aggregate was not established by those SCIR runs.
  The included four-method source has offline/runtime-safety and synthetic-diagnostic
  evidence; synthetic diagnostics are not benchmark results.
- The historical SCIR manifests refer to runner SHA-256
  c884fe8c9605e0cdd40e790c3b5ada6f3e2b34261b0b05bca70d7678ec4560f2. That exact
  source file was not present among the accessible SCIR source copies. Therefore this
  branch is not represented as byte-identical to the historical AFlow runner.
- The exact package in this branch passed local Python compilation and offline tests:
  69 passed, 2 skipped. The skipped tests require the pinned upstream source checkouts
  for real-template integration. Install those sources and run the included
  preflight/diagnostic steps before a long GPU run.
- The recorded real-model synthetic diagnostics were on a 48-GB RTX 5880 Ada host.
  RTX PRO 6000 compatibility has not been established; perform a local model-load and
  request smoke test on that hardware first.

## Frozen pilot protocol

Seed 0 only; data seed 2026; canonical D_search=60 and D_select=30; separate
30-question AIME2025 and AIME2026 D_test; task model Qwen/Qwen3.5-9B; context 8192;
method output cap 6144; concurrency 8; vLLM max-num-seqs 24. The shared score is
0 for unparseable, 1 for parseable but wrong, and 2 for exact match. Freeze the
selected candidate before any D_test access. This is a one-seed pilot, not a
three-seed mean +/- standard deviation result.

## Launch caution

Follow the package README. Use the exact pinned upstream commits, locally available
authorized frozen data, verified model/embedding files, separate Python environments
where required, and a fresh output directory per attempt. Do not tune from D_test.
The RTX PRO 6000 target still needs an on-device smoke test; this branch is code-ready
for that preflight, not a claim that a full run has completed there.
