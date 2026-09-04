# RPAS 实验代码与结果交付包

本交付包对应 `RPAS-EC2-A100` 的受控实验实现与已有结果，重点覆盖：

- EC-2 MMLU-57x10：Vanilla、RPAS-no-selection、G-Designer graph-executor、RPAS controlled selection；
- EC-1 HumanEval 的 AFlow、MaAS、G-Designer、RPAS 原生适配器与运行器；
- AIME formal-track 脚本、配置与数据清单；
- 原始 MMLU 主表运行输出、聚合结果与交付文档；
- AFlow 与 MaAS 的本地基线源码快照。

## 包含与排除

包含源码、配置、测试、协议、文档、数据清单，以及可公开再分发并已固化校验和的 EC-1 HumanEval/AFlow 和 EC-3 HotpotQA AFlow-derived fixtures。

刻意排除：模型权重（本机 `models/` 约 19 GB）、Python 虚拟环境、缓存、Git 历史、编译缓存、临时测试目录，以及任何凭据、环境变量文件、运行日志、搜索 workspace、模型 checkpoint 和实验输出。MMLU 原始 CSV 及其他未明确授权再分发的数据不在仓库中；请依照项目文档下载并校验。

## 解包与环境

```bash
tar -xzf RPAS_EC2_experiment_code_and_results_20260904.tar.gz
cd RPAS-EC2-A100
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

另一台机器的完整交接和固定外部基线源码准备方式见 `docs/TRANSFER_TO_ANOTHER_MACHINE.md`。实验所需的模型权重不包含在此包中，须在目标机器设置模型路径和 OpenAI-compatible 推理服务。

## 核验与复现入口

```bash
python -m pytest -q
python -m external_comparison.runners.validate_protocol \
  --config-dir external_comparison/configs --require-native
bash experiments/run_ec2_mmlu_a100.sh
```

使用 GPU、模型路径和服务地址前，先阅读 `A100_QUICKSTART_zh.md`、`EXPERIMENT_PROTOCOL_zh.md`、`docs/MMLU_MAIN_EXPERIMENT_DELIVERY_20260904.md` 与 `notes/EC2_A100_transfer_runbook_zh.md`。不要将 API key 写入配置文件；通过环境变量传入。

## 结果边界

MMLU 结果是 57 个 subject、每个 10 个 held-out test item 的 MMLU-57x10 受控子集，不是 full MMLU。当前 RPAS 是在 9 个预定义候选架构上进行 controlled selection，不是完整 reflective mutation search；G-Designer 是 graph-executor setting，不是完整 topology-training/search。
