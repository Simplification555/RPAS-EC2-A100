# RPAS 外部实验继续运行手册

本文面向接手另一台机器的执行者。目标是在不重复已完成 seed-0 pilot
计算的前提下，继续运行经过审计的 EC-1/EC-2/EC-3 基础设施。所有命令都
要求显式指定一张物理 GPU；本项目的实验脚本只允许 GPU `4` 或 `5`。

## 0. 当前交接状态

| 项目 | 状态 | 可否作为论文主表 |
| --- | --- | --- |
| EC-1 AFlow seed 0 | 原生 `Optimizer.optimize("Graph")` 搜索与 held-out pilot 已完成；工作流历史已交接 | 否，仍是 pilot |
| EC-1 MaAS seed 0 | 原生 fresh controller training 与 official held-out test 已完成；完整 pilot 结果已交接 | 否，仍是 pilot |
| EC-1 RPAS seed 0 | 原机器仍在执行；不要从公开仓库假定已有完整状态 | 否 |
| EC-2 legacy MMLU | 仅开发/排错结果 | 否 |
| EC-2 v2 | 代码、协议和门禁已准备；尚未开始正式运行 | 否 |
| EC-3 V3 | 数据/协议/preflight/formal-gate 已准备；尚未实现正式 AFlow/RPAS 搜索 runner | 否 |

不要将 AFlow validation 分数、任何 pilot、legacy EC-2，或单个 seed 写入论文
主表。只有在完成三 seed、冻结预算、通过所有 formal gate 后才可汇总。

## 1. 获取代码并准备固定上游源码

```bash
git clone https://github.com/ChiangYuhsin/RPAS-EC2-A100.git
cd RPAS-EC2-A100
git rev-parse HEAD
bash scripts/bootstrap_external_comparison.sh --install-python-deps
```

最后一条命令会在 `external_baselines/` 中拉取并 detached checkout 下列版本：

| 组件 | 固定版本 |
| --- | --- |
| AFlow | `FoundationAgents/AFlow@3f457218fc716093fe53f6df8a5d5e6379d66346` |
| MaAS | `bingreeky/MaAS@987f3c1bc9a96e844fe090db3791446e3ef0f5c7` |
| G-Designer | `yanweiyue/GDesigner@a6efcfa3b40bb4d9cbf46f883a95d62020bd8251` |

`--install-python-deps` 不会安装 PyTorch，因为 CUDA wheel 必须与接手机器的
driver/CUDA 版本对应。先按该机器的 CUDA 环境安装可用的 GPU PyTorch，再检查：

```bash
.rpas-run/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
.rpas-run/bin/python -m pytest \
  tests/test_hotpotqa_ec3_data.py \
  tests/test_ec3_preflight.py \
  tests/test_ec3_formal_gate.py \
  tests/test_ec1_native_runtime.py \
  tests/test_ec2_v2.py -q
```

预期是 `True` 和 `18 passed`。若 CUDA 不可用，禁止开始模型服务或实验。

## 2. 验证已交接数据和恢复状态

EC-1 和 EC-3 可公开再分发的固定夹具已包含在仓库中；运行前校验：

```bash
sha256sum data/ec1_humaneval/official/HumanEval.jsonl.gz
sha256sum data/ec1_humaneval/official/humaneval.jsonl
sha256sum data/ec1_humaneval/aflow/humaneval_{validate,test,public_test}.jsonl
sha256sum data/ec3_hotpotqa/source_aflow_mirror/hotpotqa_{validate,test}.jsonl
sha256sum handoff_progress/ec1_seed_0/maas/HumanEval_controller_sample4.pth
```

将输出与以下文件逐项核对：

- `data/ec1_humaneval/DATASET_MANIFEST.json`
- `data/ec3_hotpotqa/DATASET_MANIFEST.md`
- `handoff_progress/README.md`

EC-2 需要自行按上游许可准备 MMLU `dev`、`val`、`test`；不得拿 legacy MMLU
输出代替数据。EC-3 在 calibration 前还必须从官方 HotpotQA distractor `train`
准备 `D_calib=40`，与公开 fixture 的 1,000 个 ID 完全去重。

## 3. 模型服务：一张卡只启动一个常驻服务

模型权重未上传。设置本地 Qwen3.5-9B 路径：

```bash
export RPAS_MODEL_PATH=/absolute/path/to/Qwen3.5-9B
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

GPU 4 服务，端口 `29500`：

```bash
CUDA_VISIBLE_DEVICES=4 .rpas-run/bin/python scripts/serve_transformers_qwen35_openai.py \
  --model "$RPAS_MODEL_PATH" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 29500 --max-new-tokens 1024 \
  --max-batch-size 4 --batch-wait-ms 25 --stop-string '<<RPAS_END>>'
```

GPU 5 服务，端口 `29501`：

```bash
CUDA_VISIBLE_DEVICES=5 .rpas-run/bin/python scripts/serve_transformers_qwen35_openai.py \
  --model "$RPAS_MODEL_PATH" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 29501 --max-new-tokens 1024 \
  --max-batch-size 4 --batch-wait-ms 25 --stop-string '<<RPAS_END>>'
```

服务终端保持运行。任何 EC-1 worker 必须只连接到同卡服务：GPU 4 对应
`http://127.0.0.1:29500/v1`，GPU 5 对应 `http://127.0.0.1:29501/v1`。
不要设 `CUDA_VISIBLE_DEVICES=4,5`，不要启动第三张卡，也不要让一个 worker
跨端口访问另一张卡。

## 4. 恢复已完成的 EC-1 seed-0 工作

### AFlow

AFlow seed-0 的搜索、generated workflow history、telemetry 与结果已在
`handoff_progress/ec1_seed_0/aflow/`。它用于审计或研究 workflow，不必重新
搜索。该 pilot 已做过 held-out test，因此也不要为了“补跑”而反复测试同一
fixture。

### MaAS

MaAS seed-0 的官方训练和 held-out pilot 已完成，结果位于
`handoff_progress/ec1_seed_0/maas/completed_pilot/`，无需在新机器补跑。保留的
controller 仅用于未来发生中断时恢复 official held-out test，不重跑
`Optimizer.optimize("Graph")`：

```bash
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_EXTERNAL_API_BASE=http://127.0.0.1:29501/v1
export RPAS_EC1_SEED=0
export RPAS_MAAS_TEST_ONLY=1
export RPAS_MAAS_CONTROLLER_PATH="$PWD/handoff_progress/ec1_seed_0/maas/HumanEval_controller_sample4.pth"
bash scripts/run_ec1_native.sh maas 5 \
  data/ec1_humaneval/official/humaneval.jsonl \
  data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
  outputs/ec1_native_resume pilot
```

此命令会为工作目录创建全新 pinned MaAS checkout，复制已训练 controller，随后只
执行官方 test 分支。它是中断恢复，不能把这个 seed-0 continuation 宣称为独立的
fresh formal run。

## 5. EC-1 后续运行顺序

正式实验的最小结构是 AFlow/MaAS/RPAS 各三个独立 seed。统一条件必须保持：

- frozen `Qwen/Qwen3.5-9B` executor；temperature `0`；相同 max tokens；
- AFlow 33-item validation fixture 仅用于 search，131-item fixture 仅用于 test；
- 相同 HumanEval parser、公测工具、timeout/retry 和底层 telemetry；
- 每 seed 独立 workspace，不能读取另一个 seed 的 experience、workflow 或 controller；
- 预算只能根据 pilot 的 **search** 成本冻结，不能根据 held-out test 调整。

先运行无模型调用门禁：

```bash
CUDA_VISIBLE_DEVICES=4 .rpas-run/bin/python -m external_comparison.runners.ec1_preflight \
  --dataset-path data/ec1_humaneval/official/humaneval.jsonl \
  --public-test-path data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
  --aflow-validate-path data/ec1_humaneval/aflow/humaneval_validate.jsonl \
  --aflow-test-path data/ec1_humaneval/aflow/humaneval_test.jsonl
```

pilot 审核通过后，由负责人将 pilot 观察到的搜索预算显式冻结，例如：

```bash
export RPAS_AFLOW_MAX_ROUNDS=2
export RPAS_AFLOW_SAMPLE=4
export RPAS_MAAS_SAMPLE=4
```

再启动 formal seed。每个进程仅绑定一张卡，以下只是模板，不能同时在同一张卡
运行两个 worker：

```bash
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_EXTERNAL_API_BASE=http://127.0.0.1:29500/v1
export RPAS_EC1_SEED=1
bash scripts/run_ec1_native.sh aflow 4 \
  data/ec1_humaneval/official/humaneval.jsonl \
  data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
  outputs/ec1_native formal
```

将 `aflow` 改为 `maas` 或 `rpas`、改变 GPU/endpoint/seed 后运行其他作业。不要
在 seed 之间共享输出目录；runner 会按 `method/seed_N` 分隔。

## 6. EC-2 v2 门禁

EC-2 的目标是 fixed six-agent communication comparison，不是让 RPAS 退化为
single agent。开始前读 `docs/EC2_V2_PROTOCOL.md`，并确认：

1. G-Designer 使用固定 commit，6 roles，官方 10-iteration training，且训练前后
   GCN checksum 不同；
2. RPAS-Comm 有 LLM reflection、typed topology mutation、新 candidate 与 Pareto
   archive，不能 rule fallback；
3. `dev -> D_search`、`val -> D_select`、`test -> D_test`，三者不重叠；
4. 主表报告 Active Edges、Messages/Query、Comm Tokens/Query、Total Tokens/Query
   和 Search Tokens，而不只报告 accuracy；
5. pilot 审核通过后才开 formal 3 seeds。

只允许用对应卡启动：

```bash
RPAS_CUDA_VISIBLE_DEVICES=4 RPAS_MODEL_PATH="$RPAS_MODEL_PATH" \
  bash experiments/run_ec2_mmlu_v2_a100.sh pilot
```

运行前不要以旧 `run_ec2_mmlu_a100.sh` 的数字作任何主表结论。

## 7. EC-3 V3 门禁

EC-3 的具体协议是 `external_comparison/configs/hotpotqa.json`。目前仓库提供
data freezer、preflight 与 formal final-state hash gate；正式 AFlow/RPAS HotpotQA
search runner 仍需实现后才能开始计算。实现前不得假装可以运行。

其必经顺序是：

1. 固化且校验 official HotpotQA train 的 `D_calib=40`；
2. 用公开 AFlow validate/test fixture 构建 `D_search=120`、`D_select=80`、
   `D_test=800`，保持原 200/800 边界；
3. 只在 `D_calib` 冻结 `M_meta`、token cap 和 `B_opt`；
4. AFlow 和 RPAS 各完成三次 fresh search；
5. hash 所有六个 final state；
6. 仅在 formal gate 放行后第一次访问 `D_test`；
7. 汇总 seed mean/std 和 two-level paired bootstrap。

运行 preflight 的模板：

```bash
export RPAS_EC3_GPU=4
export CUDA_VISIBLE_DEVICES=4
export RPAS_EXTERNAL_API_BASE=http://127.0.0.1:29500/v1
export RPAS_AFLOW_ROOT="$PWD/external_baselines/AFlow"
export RPAS_EC3_MANIFEST=/absolute/path/to/frozen/hotpotqa_manifest.json
bash experiments/run_ec3_hotpotqa_v3.sh calibration
```

## 8. 常见失败与处理

| 现象 | 处理 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` 与 runner GPU 不一致 | 停止该 worker；将它改为唯一的 `4` 或 `5`，并使用同卡 endpoint 后重启。 |
| AFlow/MaAS checkout commit 不一致或 dirty | 删除该独立 workspace，重新运行 bootstrap；不要修改官方 checkout。 |
| AFlow public-test fixture 只有 159/164 | 这是已记录的 AFlow 派生 fixture 覆盖边界；不要伪造缺失 test。 |
| MaAS test-only 找不到 controller | 校验 `handoff_progress` SHA-256，设置绝对 `RPAS_MAAS_CONTROLLER_PATH`。 |
| EC-3 preflight 提示 calibration 缺失/overlap | 从 official HotpotQA distractor train 重新抽取，先做 ID 去重与 hash，再重新 freeze。 |
| 一个方法 valid-answer rate <99% 或产生截断 | 将该 run 判无效；修复后提升 protocol version 并重跑受影响的所有 formal seeds。 |

## 9. 每次运行后的归档

保留每个 run 的 `run_manifest.json`、`calls.jsonl`、search trace、candidate/workflow
artifact、final-state hash、环境摘要和 Git diff。不要把密钥写入这些文件。结果汇总前
检查：相同 split hash、相同 model/tokenizer hash、相同 endpoint contract、相同
temperature/token cap、独立 workspace、以及 `D_test` 未在 final-state freeze 前访问。
