# External Comparison Foundation

这是外部对比实验的基础设施层，不是实验结果。当前目标是先固定比较边界和统计口径，再接入各论文的原生搜索实现。

## HumanEval common-space runner

`runners/humaneval.py` 已提供可审计的 controlled-search 共同空间执行路径：HumanEval 数据按
`data_seed=2026` 固定切分为 `D_search/D_select/D_test`，候选由
`experiments/search_adapters/` 提议，代码由独立 Python 子进程执行并以 pass@1 评分。
runner 统一生成调用、通信、候选、选择和 provenance artifact，并支持共享 search/select cache
与 `--resume`。它默认写入 `formal_result: false`，在 G1-G9 门禁完成前不得作为正式结果。

示例（只做 split/manifest 检查，不调用模型）：

```bash
uv run python -m external_comparison.runners.humaneval \
  --dataset-path /path/to/HumanEval.json \
  --method random_as --dry-run
```

共同空间方法名是 `random_as`、`aflow_style`、`adas_style`、`rpas_quality` 和 `rpas`。
这些方法是受控共域比较策略，不是对应论文官方代码的完整复现，不能写成 AFlow、MaAS 或 G-Designer。

正式外部实验配置是 `configs/ec1_humaneval.json` 和 `configs/ec2_mmlu_v2.json`。

EC-1 启动前必须先运行无模型调用的门禁：

```bash
CUDA_VISIBLE_DEVICES=4 uv run python -m external_comparison.runners.ec1_preflight \
  --dataset-path /path/to/humaneval.jsonl \
  --public-test-path /path/to/humaneval_public_test.jsonl \
  --aflow-validate-path /path/to/humaneval_validate.jsonl \
  --aflow-test-path /path/to/humaneval_test.jsonl
```

推荐先在本地准备四文件 provenance bundle（数据不会被 Git 跟踪）：

```bash
uv run python scripts/fetch_ec1_humaneval_data.py --output-dir data/ec1_humaneval
```

然后将 `official/humaneval.jsonl`、`aflow/humaneval_validate.jsonl`、
`aflow/humaneval_test.jsonl` 和 `aflow/humaneval_public_test.jsonl` 的路径传给
preflight；脚本会记录每个实际文件的 SHA-256。

门禁要求 164 个 HumanEval 任务、固定 AFlow `33` 个 search/dev 与 `131` 个 held-out test fixture，且 `CUDA_VISIBLE_DEVICES` 只能包含 GPU 4/5。所有原生方法都会实际消费同一对 fixture；stage 前会检查 task ID 互斥、数量，以及 `prompt/test/entry_point` 与官方 HumanEval 逐字段一致。原生 adapter 会为每个 seed 复制一个独立、干净且 commit-pinned 的官方 checkout：AFlow 必须现场调用 `Optimizer.optimize("Graph")` 并以验证集选择 workflow；MaAS 必须现场 train、验证新的 controller checkpoint 后调用官方 test。既有 `round_1` 或随机 controller 都会被拒绝。AFlow 的 task-level 执行并发固定为 1，避免共享后端排队把上游不变的 60 秒 workflow timeout 误记为算法失败。

先用无模型调用的 staging smoke 检查官方源码和四文件路径：

```bash
python -m external_comparison.runners.ec1_native_smoke \
  --method aflow --source-root external_baselines/AFlow \
  --dataset-path data/ec1_humaneval/official/humaneval.jsonl \
  --public-test-path data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
  --aflow-validate-path data/ec1_humaneval/aflow/humaneval_validate.jsonl \
  --aflow-test-path data/ec1_humaneval/aflow/humaneval_test.jsonl \
  --output-dir outputs/ec1_smoke --seed 0
```

seed 0 pilot 使用显式的 GPU 4 或 GPU 5；`scripts/run_ec1_native.sh` 每次只允许一张卡，并默认传入仓库固定的 AFlow validate/test fixture。正式三 seed 运行要求显式冻结 `RPAS_AFLOW_MAX_ROUNDS` 与 `RPAS_MAAS_SAMPLE`，避免基于 test 结果调整搜索预算。
它们只接受 repository-local native adapter；`validate_protocol.py --require-native`
会在缺失时失败。

运行前需要有一个可达的 OpenAI-compatible inference endpoint。若 GPU 4 或 5 空闲，可用 `scripts/start_ec1_qwen_server.sh <4|5> <port>` 启动 EC-1 专用服务；该脚本强制 1024-token cap，并拒绝与已有 compute process 共享 GPU。该上限在 code-answer 和 workflow-edit pilot 中已足够，同时避免无终止生成独占服务。设置不含密钥的参数后可启动 pilot：

```bash
export RPAS_EXTERNAL_MODEL='Qwen/Qwen3.5-9B'
export RPAS_EXTERNAL_API_BASE='http://127.0.0.1:29500/v1'
export RPAS_EC1_SEED=0
bash scripts/run_ec1_native.sh aflow 4 \
  data/ec1_humaneval/official/humaneval.jsonl \
  data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
  outputs/ec1_native pilot
```

`RPAS_EXTERNAL_API_KEY` 只应在运行环境中设置，不能写入配置、日志或 Git。pilot 成功后，依据 search cost 冻结预算，再以 `formal` 运行 seed 0/1/2。AFlow 和 MaAS 已使用同源 public-test fixture；在 RPAS 的 native code-agent path 获得相同的可调用工具前，adapter 会拒绝将 RPAS 标记为 formal，防止不公平比较进入主表。

## 当前 EC-2 结果边界

仓库中已有的 EC-2 结果是 `MMLU-57x10 controlled subset`：57 个 subject、每个 subject 10 道测试题，搜索集每 subject 5 题。它们统一保留 `formal_result: false`，在 G1-G9 门禁完成前不能写成 formal result 或完整 MMLU。

RPAS 本次运行是 9 个预定义候选架构上的 controlled candidate selection，`RPAS_MMLU_NEW_CANDIDATES=0`；这不是完整 reflective mutation search。G-Designer 的 `search_calls=0` 表示没有单独 instrumented 的搜索阶段，不表示没有额外推理调用。论文表格必须同时报告 test inference calls/tokens、search calls/tokens 和 total calls/tokens。

这些 legacy outputs 仅用于开发和排错，不能进入论文主表。主表唯一允许的实现是
`runners/ec2_v2.py`，协议见 `../docs/EC2_V2_PROTOCOL.md`。v2 强制固定六个官方
MMLU roles、同一 FinalRefer、同一 Qwen3.5-9B endpoint、temperature 0、256-token cap、
一轮通信、无压缩，以及 `dev -> search`、`val -> select`、`test -> held-out` 三段切分。
它直接调用官方 G-Designer 的 10-iteration training loop，并拒绝未训练 GCN；RPAS-Comm
必须产生 LLM reflection、typed topology mutation 和新 candidate。v2 aggregator 会拒绝
legacy manifest，并将 worker communication 与 FinalRefer 输入 token 分开报告。

```bash
RPAS_CUDA_VISIBLE_DEVICES=4 bash experiments/run_ec2_mmlu_v2_a100.sh pilot
```

脚本只接受物理 GPU `4` 或 `5`。通过 pilot 的 fidelity review 后才可运行 `formal`；无三
seed、同一 split hash 与 formal gate 的结果仍不能称为论文正式结果。

可从每个 seed artifact 生成主表：

```bash
python -m external_comparison.runners.aggregate_mmlu_v2 \
  --root outputs/external_comparison/ec2_mmlu_v2 \
  --output-dir outputs/external_comparison/ec2_mmlu_v2/aggregate
```

## 实验主线

| 实验 | 主要回答的问题 | 首选方法 |
|---|---|---|
| EC-1 HumanEval | 不同架构搜索方法在统一执行器下的质量--成本 Pareto | RPAS / AFlow / MaAS |
| EC-2 MMLU | 通信拓扑是否带来可测的通信收益 | RPAS / G-Designer |
| EC-3 HotpotQA | workflow 搜索能否跨任务结构泛化 | RPAS / AFlow |
| EC-4 Transfer（可选） | 搜出的拓扑能否跨 backbone 复用 | RPAS / MaAS / AFlow |

## 当前已经准备好的内容

- `configs/`：三组实验协议配置，不包含 API key、绝对路径或结果数字。
- `common/schema.py`：统一的 call、candidate、run manifest 数据结构。
- `common/telemetry.py`：JSONL 轨迹和 tokens/calls/cost/network accounting。
- `common/manifest.py`：协议、数据、源码和配置的 SHA-256 追溯接口。
- `common/pareto.py`：统一的 Pareto、Quality operating point 和 Efficiency operating point 选择规则。
- `adapters/`：RPAS artifact reader 和外部方法的明确适配器占位。
- `runners/validate_protocol.py`：只做静态校验，不发起模型/API 调用。

## 重要边界

1. `EXPERIMENT_PROTOCOL.md` 是冻结实验协议的权威文件；附件研读笔记是外部扩展实验建议，若有冲突以仓库协议为准。
2. 不把原论文中的数字直接抄进表格；每个方法必须在同一 executor、数据划分、模型、解码、评测器和 telemetry 边界下重新跑，或明确标注为 original-paper-fidelity appendix。
3. 不强迫不同方法使用同样的迭代次数；比较 realized task-model calls/tokens 和累计预算曲线。
4. test split 不参与搜索、重排、停止、调参或选择。
5. EC-1 已接入 AFlow 和 MaAS 的原生 fresh-search/fresh-train adapter，但尚未产生正式实验结果；在 pilot、预算冻结和三 seed formal run 完成前，不得宣称已有外部对比结果。

## 下一步运行顺序

1. 验证 OpenAI-compatible endpoint 与冻结的 Qwen3.5-9B 模型版本。
2. 使用 AFlow、MaAS 与 RPAS 分别完成 seed 0 pilot。
3. 仅按 search cost 冻结预算，并记录在 run manifest。
4. 在 GPU 4/5 上完成三 seed formal runs，再通过 valid-rate 和 data-leakage gates 汇总主表。
