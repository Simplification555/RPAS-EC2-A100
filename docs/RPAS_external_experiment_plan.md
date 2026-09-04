# RPAS 外部性能对比实验方案

> **目标**：用尽可能少、但足够有说服力的外部实验，验证 RPAS 相比现有自动化 Agent / Multi-Agent System 设计方法的性能与效率优势。  
> **原则**：不重复主实验，不堆 baseline，不追求“实验数量多”，而是让每一组实验都回答一个明确的 reviewer question。  
> **核心配置**：2 个必做外部实验 + 1 个可选泛化实验。  
> **推荐硬件**：1 × NVIDIA A100（建议 80GB），单模型服务常驻，所有方法共享统一 inference backend。

---

## 1. 实验总览

最终建议只保留以下三组外部性能对比：

| 编号 | Benchmark | 方法 | 主要目的 | 优先级 |
|---|---|---|---|---|
| **EC-1** | **HumanEval** | **AFlow / MaAS / RPAS** | 自动 workflow / MAS architecture search 的总体性能与成本对比 | **P0 必做** |
| **EC-2** | **MMLU** | **G-Designer / RPAS** | 验证 RPAS 的收益是否只是来自 communication topology optimization | **P0 必做** |
| **EC-3** | **HotpotQA** | **AFlow / RPAS** | 验证 typed architecture search 的跨任务泛化能力 | **P1 可选** |

其中：

- **GPTSwarm 不建议正式跑数值实验**，放入 Related Work 中作为 automatic graph optimization 的经典代表即可；
- 现有 `random_as / aflow_style / adas_style / rpas_quality / rpas` 不删除，但重新定位为 **Controlled Search / Ablation Study**，不作为 External SOTA Comparison；
- 不再扩展更多 benchmark，除非前两组核心实验完成后仍有充足算力与时间。

---

# 2. 实验总体研究问题

整个外部实验只回答三个问题：

### RQ1：RPAS 相比现有自动 Agent / MAS 搜索方法，是否具有更好的质量–成本折中？

由 **EC-1 HumanEval** 回答。

重点不是单纯 Accuracy，而是：

\[
\text{Task Quality}
\quad+\quad
\text{Inference Cost}
\quad+\quad
\text{Search Cost}
\]

---

### RQ2：RPAS 的性能优势是否仅仅可以由“更好的通信拓扑”解释？

由 **EC-2 MMLU** 回答。

重点比较：

\[
\text{Accuracy}
\quad vs \quad
\text{Communication Tokens}
\]

如果 RPAS 只是在 communication topology 上做得好，那么 G-Designer 应已经足够；如果 RPAS 在相同通信预算下仍有更好的 performance / cost operating point，说明联合搜索 topology、budget、communication、deployment 等变量具有额外价值。

---

### RQ3：RPAS 是否只在代码/数学类任务上有效？

由 **EC-3 HotpotQA** 回答。

该实验只作为可选补强，不影响主实验闭环。

---

# 3. 实验公平性总原则

外部实验必须采用 **Controlled Comparison**，不能直接拿原论文报告数字进行横向拼表。

所有方法统一：

| 项目 | 统一要求 |
|---|---|
| Backbone | 完全相同的 frozen LLM |
| Model version | 固定版本并记录 |
| API / inference backend | 相同 |
| Dataset split | 相同 |
| Evaluator | 相同 |
| Temperature | 相同 |
| Top-p / Top-k | 尽量相同 |
| Max output tokens | 相同 |
| Answer parser | 相同 |
| Timeout / retry | 相同 |
| Tool access | 相同 |
| Search / validation set | 相同 |
| Held-out test | 相同 |
| Random seeds | 3 seeds |
| Token accounting | 统一底层统计 |
| Call accounting | 统一底层统计 |
| Wall-clock accounting | 统一底层统计 |

---

## 3.1 不强行统一的内容

不能为了“同参数”破坏 baseline 本身。

下列内容必须保留各自原生算法：

- AFlow：保留 workflow search / MCTS / optimizer；
- MaAS：保留 Agentic Supernet / controller / architecture selection；
- G-Designer：保留 topology generation / GNN-based communication design；
- RPAS：保留 reflective typed mutation、Pareto selection 和 deployment-aware architecture search。

公平比较的含义是：

> **Same environment + Same resource accounting + Native algorithm**

而不是：

> Same internal search mechanism。

---

# 4. 搜索预算对齐原则

不同方法的一个 iteration / candidate 计算量完全不同，因此：

**禁止仅使用“相同 iteration 数”作为公平预算。**

主预算优先采用：

\[
B_{\text{search}}
=
\text{Cumulative Search Tokens}
\]

并同时记录：

- Task-model calls
- Optimizer / reflection calls
- Prompt tokens
- Completion tokens
- Total search tokens
- Search wall-clock
- Candidate / workflow evaluations

Candidate count 只作为辅助信息。

---

## 4.1 多预算点

推荐在搜索过程中记录：

\[
25\%,\ 50\%,\ 75\%,\ 100\%
\]

四个 cumulative search budget checkpoint。

输出搜索曲线：

> **Best Validation Score vs Cumulative Search Tokens**

用于比较搜索效率，而不仅是最终结果。

---

# 5. EC-1：HumanEval 自动 Agent / MAS 搜索主对比

## 5.1 目标

这是最重要的一组外部实验。

核心问题：

> 在相同 frozen backbone、相同 HumanEval split、相同 evaluator 和匹配搜索预算下，RPAS 相比 AFlow 与 MaAS 是否得到更好的质量–成本 trade-off？

---

## 5.2 对比方法

只保留三个核心方法：

1. **AFlow**
   - 代表 automated workflow search；
   - 搜索可执行 workflow / prompt；
   - 强项是 workflow 表达能力。

2. **MaAS**
   - 代表 multi-agent architecture search；
   - 强项是 query-conditioned architecture / inference resource allocation；
   - 是与 RPAS 最接近的外部强 baseline。

3. **RPAS**
   - typed multi-agent architecture search；
   - 联合考虑 architecture、budget、communication 和 deployment-related cost。

可额外加入：

4. **Single Agent**
   - 不参与搜索；
   - 作为最低成本参考。

---

## 5.3 Benchmark

使用：

> **HumanEval**

推荐同时保留 HumanEval 官方测试与可选 HumanEval+ stricter execution，但主表以 HumanEval Pass@1 为主。

---

## 5.4 数据划分

建议固定一次 split 后永久冻结。

推荐：

- Development / Search set：约 20%
- Held-out Test set：约 80%

若使用 164 道 HumanEval：

- Search / validation：33
- Test：131

重要规则：

- Test label / test execution result 不能用于搜索；
- 不允许根据 test performance 重新选择 architecture；
- 三个方法必须使用完全相同的 split。

---

## 5.5 Seeds

正式实验：

```text
seed = 0
seed = 1
seed = 2
```

即：

```text
AFlow × 3
MaAS × 3
RPAS × 3
```

共：

```text
9 个正式搜索作业
```

Single Agent 无需搜索 seed，可直接重复运行或固定一次 deterministic decoding。

---

## 5.6 指标

### Task Performance

- HumanEval Pass@1

### Inference Cost

- Calls / query
- Prompt Tokens / query
- Completion Tokens / query
- Total Tokens / query
- E2E Latency / query

### Search Cost

- Search Task-model Calls
- Search Optimizer / Reflection Calls
- Search Prompt Tokens
- Search Completion Tokens
- Total Search Tokens
- Candidate / Workflow Evaluations
- Search Wall-clock

---

## 5.7 主表

建议最终论文表：

| Method | Pass@1 ↑ | Calls ↓ | Infer. Tokens ↓ | Search Calls ↓ | Search Tokens ↓ | Search Time ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Single Agent |  |  |  | – | – | – |
| AFlow |  |  |  |  |  |  |
| MaAS |  |  |  |  |  |  |
| **RPAS** |  |  |  |  |  |  |

所有搜索方法报告：

\[
\text{mean} \pm \text{std}
\]

over 3 seeds。

---

## 5.8 推荐附加指标

### Quality-at-Matched-Cost

固定 inference token / call budget：

\[
Q(B)
\]

比较不同方法在同等执行预算下的 Pass@1。

---

### Cost-at-Matched-Quality

固定目标质量：

\[
C_{\text{target}}(q)
=
\min_{A:Q(A)\ge q} C(A)
\]

例如：

- Cost @ 80% Pass@1
- Cost @ 85% Pass@1

若样本量不足以支持精确阈值，可仅画 quality–cost Pareto curve。

---

## 5.9 核心图

### Figure EC-1A

```text
X-axis: Cumulative Search Tokens
Y-axis: Best Validation Pass@1
```

比较：

- AFlow
- MaAS
- RPAS

---

### Figure EC-1B

```text
X-axis: Inference Tokens / Query
Y-axis: Held-out Pass@1
```

展示最终 quality–cost operating point。

---

## 5.10 EC-1 能支持的 Claim

只有当正式结果支持时才可以写：

- RPAS achieves a better quality–cost trade-off than AFlow / MaAS.
- RPAS reaches comparable quality using fewer inference tokens / calls.
- RPAS reaches a target validation quality with lower search cost.
- RPAS provides a competitive operating point under matched search compute.

不能提前写：

- RPAS universally outperforms MaAS.
- RPAS is strictly better than AFlow in all settings.
- RPAS provides stronger per-query adaptation than MaAS.

---

# 6. EC-2：MMLU Communication Topology 对比

## 6.1 目标

核心问题：

> RPAS 的收益是否仅仅来自更合理的 multi-agent communication topology？

这一实验只需要：

- **G-Designer**
- **RPAS**

可增加：

- Fully Connected
- Fixed Chain

作为低成本结构参考。

不需要再塞 AFlow / MaAS。

---

## 6.2 为什么选 MMLU

MMLU：

- 题目格式规整；
- 输出短；
- 多领域覆盖；
- 很适合控制 Agent 数量和 communication rounds；
- G-Designer 与 GPTSwarm 类工作都适合在 MMLU 上分析 topology。

因此它是 communication-specific comparison 的理想 benchmark。

---

## 6.3 数据规模

不建议一开始全量 14k+ 反复跑 3 seeds。

推荐：

### 正式主实验

构建固定：

> **MMLU-Stratified Subset**

建议：

- 每个 subject 固定抽取 10–20 道；
- 所有方法使用完全相同样本；
- 固定抽样 seed；
- 在论文中明确写为 stratified subset，而不是 Full MMLU。

推荐规模：

```text
约 600–1200 examples
```

如算力充足，可在最终 Appendix 再补一个 Full MMLU confirmatory run。

---

## 6.4 控制变量

必须统一：

- Agent 数量
- Agent role pool
- Backbone
- Decoding
- Max output tokens
- Communication rounds
- Final aggregator / answer parser
- Dataset subset
- Random seeds

这组实验的核心是隔离：

> communication topology

而不是让某个方法偷偷多用 Agent。

---

## 6.5 Seeds

正式：

```text
G-Designer × 3 seeds
RPAS × 3 seeds
```

共：

```text
6 个正式搜索 / 评测作业
```

---

## 6.6 指标

### Task

- MMLU Accuracy

### Communication

- Active Edges
- Messages / query
- Inter-Agent Tokens / query
- Average Message Length
- Communication Rounds

### Total Execution

- Calls / query
- Total Tokens / query
- E2E Latency / query

---

## 6.7 主表

| Method | Accuracy ↑ | Calls ↓ | Active Edges ↓ | Comm. Tokens ↓ | Total Tokens ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed Chain |  |  |  |  |  |
| Fully Connected |  |  |  |  |  |
| G-Designer |  |  |  |  |  |
| **RPAS** |  |  |  |  |  |

---

## 6.8 最重要的图

### Figure EC-2

```text
X-axis: Inter-Agent Communication Tokens
Y-axis: MMLU Accuracy
```

如果多个搜索 budget / selected architecture，可以画 Pareto frontier。

---

## 6.9 EC-2 能支持的 Claim

若正式结果支持：

- RPAS achieves a competitive or better communication–accuracy trade-off.
- Communication topology optimization alone does not fully explain RPAS's selected operating point.
- RPAS reduces communication overhead at comparable task performance.

不能提前写：

- RPAS always discovers better topology than G-Designer.
- RPAS is more query-adaptive than G-Designer.

---

# 7. EC-3：HotpotQA 跨任务泛化（可选）

## 7.1 目标

只回答：

> RPAS 的优势是否能从代码任务迁移到 multi-hop QA？

因此不需要完整大乱斗。

仅比较：

- AFlow
- RPAS

可加：

- Single Agent
- Fixed Workflow

---

## 7.2 Benchmark

使用：

> **HotpotQA**

建议固定一个约：

```text
1000-example subset
```

例如：

- 200 search / validation
- 800 held-out test

所有方法共享同一 split。

---

## 7.3 指标

- Exact Match
- F1
- Calls / query
- Inference Tokens / query
- Search Tokens
- Search Calls
- Search Wall-clock
- Invalid / Failed Candidate Rate

---

## 7.4 主表

| Method | EM ↑ | F1 ↑ | Calls ↓ | Infer. Tokens ↓ | Search Tokens ↓ |
|---|---:|---:|---:|---:|---:|
| Single |  |  |  |  | – |
| AFlow |  |  |  |  |  |
| **RPAS** |  |  |  |  |  |

---

## 7.5 是否必须做

不是。

如果：

- EC-1 HumanEval 结果完整；
- EC-2 MMLU 结果完整；
- 主论文原有 AIME / MASBench / GAIA 已经充分；

那么 HotpotQA 可以作为：

> Appendix / Future Extension

而不影响外部实验主闭环。

---

# 8. 现有 style baseline 怎么处理

当前已经准备好的：

```text
random_as
aflow_style
adas_style
rpas_quality
rpas
```

保留。

但重新命名实验定位：

> **Controlled Search Study**

而不是：

> External SOTA Comparison

---

## 8.1 建议使用方式

### Random-AS

回答：

> Reflection / guided mutation 是否优于随机架构搜索？

---

### AFlow-style

回答：

> 简化的 workflow mutation 与 RPAS typed mutation 有何差异？

注意论文中必须写：

> AFlow-style proxy

不能写成：

> AFlow

---

### ADAS-style

回答：

> 更自由的 meta-agent / agent-program modification 与 typed search 的差异。

同样不能写成：

> ADAS official baseline

---

### RPAS-Quality

回答：

> Deployment-aware Pareto objective 是否改变最终 architecture selection？

---

### RPAS

完整方法。

---

# 9. 统一工程实现建议

推荐建立统一接口：

```text
external_comparison/
├── configs/
│   ├── common.yaml
│   ├── humaneval.yaml
│   ├── mmlu.yaml
│   └── hotpotqa.yaml
│
├── adapters/
│   ├── aflow_adapter.py
│   ├── maas_adapter.py
│   ├── gdesigner_adapter.py
│   └── rpas_adapter.py
│
├── common/
│   ├── llm_client.py
│   ├── evaluator.py
│   ├── token_meter.py
│   ├── call_meter.py
│   ├── latency_meter.py
│   └── trace_schema.py
│
├── runners/
│   ├── run_search.py
│   ├── run_test.py
│   └── run_budget_curve.py
│
└── analysis/
    ├── aggregate.py
    ├── pareto.py
    ├── bootstrap.py
    └── plot_curves.py
```

---

# 10. 统一 Telemetry Schema

每一次 LLM 请求至少记录：

```json
{
  "method": "",
  "dataset": "",
  "seed": 0,
  "phase": "search|validation|test",
  "candidate_id": "",
  "query_id": "",
  "model": "",
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "total_tokens": 0,
  "latency_s": 0.0,
  "source_agent": "",
  "target_agent": "",
  "tool_calls": 0,
  "success": true,
  "exception": null
}
```

额外搜索级信息：

```json
{
  "search_step": 0,
  "parent_id": "",
  "architecture_id": "",
  "validation_score": 0.0,
  "cumulative_search_calls": 0,
  "cumulative_search_tokens": 0
}
```

---

# 11. A100 时间预算

以下为工程级粗估，不是理论峰值。

假设：

- 1 × A100 80GB
- 约 7B–10B backbone
- BF16
- vLLM / SGLang
- 单个模型服务常驻
- 所有方法通过同一个 OpenAI-compatible endpoint 请求
- 3 seeds
- 正常 continuous batching
- 不做 Full MMLU × 3 seeds

---

## 11.1 EC-1 HumanEval

```text
AFlow × 3
MaAS × 3
RPAS × 3
```

预计：

```text
12–24 A100 GPU-hours
```

主要取决于：

- 搜索预算；
- AFlow workflow evaluation 次数；
- MaAS architecture samples；
- RPAS candidate calls；
- 平均输出长度。

---

## 11.2 EC-2 MMLU

使用约 600–1200 条 stratified subset：

```text
G-Designer × 3
RPAS × 3
```

预计：

```text
8–16 A100 GPU-hours
```

MMLU 单题输出很短，因此通常比长文本 workflow search 更便宜。

---

## 11.3 EC-3 HotpotQA

若做：

```text
AFlow × 3
RPAS × 3
```

1000-example 规模：

预计：

```text
8–14 A100 GPU-hours
```

---

# 12. 总时间

## 只做核心两组

```text
EC-1 HumanEval
+
EC-2 MMLU
```

纯 GPU 计算预计：

```text
20–40 A100 GPU-hours
```

即理论连续时间约：

```text
1–2 天
```

考虑：

- adapter bug
- timeout
- malformed workflow
- inference server restart
- seed 重跑
- Slurm queue
- evaluator / parser 修复

推荐实际排期：

```text
2–3 天
```

---

## 再加 EC-3 HotpotQA

整体预计：

```text
28–54 A100 GPU-hours
```

实际项目排期：

```text
约 3 天左右
```

如果 baseline 第一次接入、调试较多，则预留：

```text
3–4 天
```

更稳妥。

---

# 13. 单张 A100 的正确运行方式

不要启动多个作业各自加载一份模型。

推荐：

```text
AFlow ───────┐
MaAS ────────┤
G-Designer ──┤
RPAS ────────┤
             ↓
      Unified LLM Client
             ↓
        1 × vLLM Server
             ↓
          1 × A100
```

即：

> GPU 上只常驻一个 backbone。

不同方法通过统一 endpoint 请求。

好处：

- 模型只加载一次；
- 更容易 continuous batching；
- token / call accounting 完全统一；
- 降低显存浪费；
- 更公平；
- 更容易复现实验。

---

# 14. 实验执行顺序

## Stage 0：门禁

必须先完成：

- 数据可用；
- benchmark loader 正常；
- evaluator 正常；
- model endpoint 正常；
- telemetry 正常；
- search / test split 固定；
- test leakage 检查；
- 三个 seed 可复现；
- baseline adapter smoke test。

若已有 G1–G9 门禁体系，应全部通过后再进入正式实验。

---

## Stage 1：HumanEval Pilot

先做：

```text
AFlow seed 0
MaAS seed 0
RPAS seed 0
```

目的：

- 验证三种方法都能跑通；
- 校准平均 calls / tokens；
- 冻结正式 search budget；
- 检查 evaluator 一致性。

Pilot 结果不进入正式统计，或者明确标记为 pilot。

---

## Stage 2：HumanEval Formal

执行：

```text
AFlow × seeds 0,1,2
MaAS × seeds 0,1,2
RPAS × seeds 0,1,2
```

完成后：

- aggregate mean ± std；
- plot search curves；
- plot quality–cost curve；
- 固化 EC-1 表。

---

## Stage 3：MMLU Pilot

执行：

```text
G-Designer seed 0
RPAS seed 0
```

检查：

- Agent 数；
- rounds；
- communication token accounting；
- final aggregator；
- subset reproducibility。

---

## Stage 4：MMLU Formal

执行：

```text
G-Designer × seeds 0,1,2
RPAS × seeds 0,1,2
```

完成：

- Accuracy；
- Active edges；
- Communication tokens；
- Total tokens；
- Pareto plot。

---

## Stage 5：决定是否做 HotpotQA

只有当：

- EC-1 已完成；
- EC-2 已完成；
- 主要结果稳定；
- 剩余时间和算力允许；

才进入 EC-3。

不要在 EC-1 / EC-2 尚未闭环时提前扩展新 benchmark。

---

# 15. 实验成功判定标准

外部实验不是要求 RPAS 每个 accuracy 都第一。

真正有价值的结果包括：

### 情况 A

RPAS accuracy 更高，成本相近。

→ 支持性能优势。

### 情况 B

RPAS accuracy 相当，但 tokens / calls 更低。

→ 支持 efficiency advantage。

### 情况 C

RPAS accuracy 略低，但大幅减少 token / communication / search cost。

→ 支持 Pareto / deployment-aware trade-off。

### 情况 D

RPAS 搜索阶段更快达到目标 validation quality。

→ 支持 search efficiency。

### 情况 E

不同方法各有优势，但 RPAS 在 quality–cost Pareto frontier 上提供新的 non-dominated operating point。

→ 同样是很好的顶会结果。

不要为了“必须 Accuracy 第一”而调 test set 或修改实验协议。

---

# 16. 最终论文中的推荐结构

```text
4 Experiments

4.1 Experimental Protocol

4.2 Existing Main Evaluation
    - AIME
    - MASBench
    - GAIA

4.3 External Comparison on HumanEval
    - AFlow
    - MaAS
    - RPAS

4.4 Communication-Efficiency Comparison on MMLU
    - G-Designer
    - RPAS

4.5 Controlled Search Study
    - Random-AS
    - AFlow-style
    - ADAS-style
    - RPAS-Quality
    - RPAS

4.6 Optional Cross-Task Generalization
    - HotpotQA
```

---

# 17. 最终定版

## 必做

### EC-1 HumanEval

```text
AFlow
MaAS
RPAS
× 3 seeds
```

回答：

> RPAS 与现有自动 workflow / MAS architecture search 方法相比表现如何？

---

### EC-2 MMLU

```text
G-Designer
RPAS
× 3 seeds
```

回答：

> RPAS 的优势是否只是 communication topology optimization？

---

## 可选

### EC-3 HotpotQA

```text
AFlow
RPAS
× 3 seeds
```

回答：

> RPAS 是否具有跨任务 workflow generalization？

---

## 不再扩展

除非前两组全部完成，否则：

- 不新增更多 benchmark；
- 不强制跑 GPTSwarm；
- 不优先做 Transfer；
- 不把 style proxy 当作 official external baseline；
- 不用原论文数字直接拼表。

---

# 18. 一句话执行原则

> **HumanEval 打总体 automated agent search，MMLU 打 communication topology；两组实验都用同一 backbone、同一 evaluator、3 seeds、matched realized compute，同时报告 quality、calls、tokens 和 search cost。**

只要这两组实验做扎实，外部对比已经能够形成一个完整、简洁且有说服力的证据链。
