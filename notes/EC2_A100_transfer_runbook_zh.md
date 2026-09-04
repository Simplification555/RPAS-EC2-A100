# EC-2 个人 A100 迁移与执行手册

## 剩余任务

按照 `RPAS_external_experiment_plan.md`，当前只推进核心外部实验 EC-2：

- MMLU，57 个 subject；固定 `data_seed=2026`。
- `D_search=285`，`D_test=570`，不使用 test 做搜索或选择。
- G-Designer：`seed 0/1/2`。
- RPAS：`seed 0/1/2`。
- 同一份 Qwen3.5-9B vLLM 服务，单张 A100，严格 `temperature=0`、`max_tokens=256`、关闭 thinking。

EC-1 的已有 HumanEval 结果保留为 pilot/旧协议产物；EC-3 HotpotQA 暂缓，因为当前官方 adapter 尚未完成。旧的 `ec2_formal_v2` 结果全部作废，原因记录在会话和 `slurm-122678.out` 中。

## 传输原则

不要从 Windows 传 Qwen 权重。代码、实验配置和启动器走 SSH；MMLU 数据从 SCIR 直接同步或从已有数据盘挂载；目标机已有的 G-Designer 和 MiniLM 直接复用。只有目标机缺少对应目录时才传输这些大文件。

需要传输的代码范围：

```text
experiments/phase2_wan_agent_search.py
experiments/phase2_wan_agent_config_qwen35_9b_homogeneous.json
experiments/search_adapters/
experiments/run_ec2_mmlu_a100.sh
external_comparison/
src/
pyproject.toml
uv.lock
```

不传输：`.git/`、`.venv/`、`outputs/`、日志、缓存和任何 API key。

## 目标机准备

目标机需要提供以下本地目录：

```text
<repo>/
<mmlu-data>/
<Qwen3.5-9B>/
<GDesigner>/
<all-MiniLM-L6-v2>/
```

至少安装 Python 3.10+、`pyarrow`、`torch`、`vllm`、`openai`、`litellm`、`sentence-transformers` 及仓库依赖。建议目标机直接使用已有 conda 环境，不在实验期间重新解析依赖版本。

## Windows 端传输

将下面的 `A100_HOST`、`A100_PATH` 和 SSH key 换成个人 A100 的实际值。若目标机能访问 SCIR，MMLU 可以在目标机上从 SCIR 拉取；否则先从 SCIR 拉到本地再传。

```powershell
$repo = "<local-repo>\RPAS"
$bundle = "<local-bundle>\RPAS_ec2_a100_bundle"
New-Item -ItemType Directory -Force $bundle | Out-Null
robocopy "$repo\experiments" "$bundle\experiments" phase2_wan_agent_search.py phase2_wan_agent_config_qwen35_9b_homogeneous.json run_ec2_mmlu_a100.sh /E
robocopy "$repo\experiments\search_adapters" "$bundle\experiments\search_adapters" /E
robocopy "$repo\external_comparison" "$bundle\external_comparison" /E
robocopy "$repo\src" "$bundle\src" /E
Copy-Item "$repo\pyproject.toml","$repo\uv.lock" "$bundle\"
scp -r "$bundle\*" "A100_HOST:A100_PATH/"
```

`robocopy` 返回码 0--7 均表示复制完成或有普通差异；确认目录内容后再启动。不能把 `outputs`、`data` 或模型权重混进代码包。

## A100 端启动

```bash
cd A100_PATH/RPAS
export RPAS_REPO_ROOT="$PWD"
export RPAS_MODEL_PATH=/path/to/Qwen/Qwen3.5-9B
export RPAS_MMLU_DATA_DIR=/path/to/mmlu
export RPAS_GDESIGNER_ROOT=/path/to/GDesigner
export RPAS_MAAS_EMBEDDING_MODEL=/path/to/all-MiniLM-L6-v2
export RPAS_PYTHON_BIN=/path/to/conda/env/bin/python
export RPAS_GDESIGNER_PYTHON_BIN=/path/to/conda/env/bin/python
export RPAS_VLLM_BIN=/path/to/conda/env/bin/vllm
export RPAS_OUTPUT_DIR="$PWD/outputs/external_comparison/ec2_fixed_v5_a100"
bash experiments/run_ec2_mmlu_a100.sh 2>&1 | tee logs/ec2_fixed_v5_a100.log
```

启动器会先生成 split manifest，启动 vLLM，访问 `/v1/models`，对两个方法各跑 8 条 smoke；任一方法出现空答案、解析失败或模型错误都会立即停止，不进入 6 个正式 seed。

## 从 SCIR 迁移数据/外部仓库

目标机入口确定后，可以从 Windows 执行：

```powershell
scp -r "SCIR_HOST:/path/to/RPAS/data/mmlu" "A100_HOST:A100_PATH/data/"
scp -r "SCIR_HOST:/path/to/external_baselines/GDesigner" "A100_HOST:A100_PATH/external_baselines/"
scp -r "SCIR_HOST:/path/to/model/sentence-transformers/all-MiniLM-L6-v2" "A100_HOST:A100_PATH/model/sentence-transformers/"
```

Qwen3.5-9B 约 19G，优先使用目标机现有权重；没有权重时再单独传输，不要与代码包一起重复压缩。

## 完成判定

只有同时满足以下条件，EC-2 才能进入论文正式汇总：

- `_smoke/gdesigner` 和 `_smoke/rpas` 的 `valid_answer_rate >= 0.99`，无错误、无截断。
- 6 个 seed 均有 `result.json`、`test_outputs.jsonl`、`calls.jsonl` 和 manifest。
- manifest 记录相同 split SHA256、模型 ID、解码参数和代码版本。
- 最终统计从 6 个结果自动汇总，不手填表格。
