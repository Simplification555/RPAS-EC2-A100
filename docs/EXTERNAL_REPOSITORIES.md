# External Baseline Sources

The EC-1 adapters use these upstream repositories. They are intentionally not
vendored into this public repository.

| Method | Official repository | Pin checked during adapter audit |
| --- | --- | --- |
| AFlow | https://github.com/FoundationAgents/AFlow | `3f457218fc716093fe53f6df8a5d5e6379d66346` |
| MaAS | https://github.com/bingreeky/MaAS | `987f3c1bc9a96e844fe090db3791446e3ef0f5c7` |
| G-Designer | https://github.com/yanweiyue/GDesigner | `a6efcfa` |

Clone them locally under `external_baselines/` before running a native adapter.
The adapter records the resolved source commit in each run manifest. Do not
commit model weights, benchmark data, API keys, generated outputs, or logs.

## EC-1 fidelity boundary

The native AFlow and MaAS adapters require a fresh per-seed search/training
workspace. An existing upstream `round_1` workflow or an untrained random
controller is not an EC-1 result. The public repository therefore ships the
adapter contracts and validation code, while execution artifacts remain local.
