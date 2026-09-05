# EC 实验提交质量门

本清单用于 `RPAS_three_external_experiments_ICLR_V3_1.md` 规定的实验交付。任何缺少可追溯输入、运行清单或调用 telemetry 的目录都只能标记为 pilot/debug，不能进入论文主表。

## 运行前

- 固定 Qwen3.5-9B、tokenizer、temperature、completion cap 和数据 split。
- 记录 upstream commit、模型路径、环境包版本、节点和物理 GPU UUID。
- 每个 worker 独占一张物理卡；`CUDA_VISIBLE_DEVICES` 的作业内编号不能代替 UUID 记录。
- 先跑 preflight 和 smoke completion，再启动 search/test。

## 运行中

- 每个 LLM call 写入 `calls.jsonl`，至少包含 run、method、dataset、seed、phase、example、model、token、latency、finish_reason 和 error。
- Search、selection、held-out test 分目录保存，禁止测试后修改参数或读取测试答案。
- RPAS 必须产生 LLM reflection、合法新 candidate 和 mutation log；`rule_fallbacks` 必须为 0。
- G-Designer 必须保存 initial/trained GCN checksum，且两者不同。

## 提交前

```bash
python scripts/validate_experiment_artifact.py outputs/<experiment>/<job>
```

- `formal_result=true` 只允许在完整 manifest、split hash、calls、search trace、结果和 `SHA256SUMS` 都存在时使用。
- pilot 结果保留原始状态，不改名为 formal，不并入正式 aggregate。
- 公开前检查仓库中没有凭据、本机绝对路径、临时日志或未锁定的 hidden test 数据。
- 对失败作业保留错误日志，修复后使用新 job id 重跑，不覆盖旧目录。

当前 SCIR 采用 3 卡并行 pilot：EC-1、EC-2 G-Designer、EC-2 RPAS-Comm。EC-2 worker completion cap 为 256；RPAS topology reflector 为独立的 768-token JSON meta-call，以避免反射计划被截断。该差异必须在 manifest 和报告中披露。
