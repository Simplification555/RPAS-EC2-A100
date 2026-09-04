# RPAS 实验协议

> 状态：**核心设计已冻结 FROZEN-CORE v1.0**  
> 冻结日期：2026-08-25  
> 范围：RPAS 最终论文实验  
> 执行状态：第 14 节全部门禁通过前，任何结果都不能标记为正式实验

本文档是英文版 `EXPERIMENT_PROTOCOL.md` 的对应中文版，用于共同检查与执行。若翻译产生歧义，以英文冻结版
为准；任何科学设定的修改必须同时更新两个版本并提升协议版本号。

## 1. 目标与研究范围

本文档在正式实验前统一冻结研究问题、数据划分、搜索空间、对比方法、搜索预算、模型设定、候选选择规则、
指标和报告规范。

RPAS 被定义为一种**无需训练模型权重、面向任务分布的多智能体架构搜索方法**。方法在开发集上搜索出一个
静态架构，再将仅由 selection-validation 选出的架构应用于测试样本。当前论文不声称实现逐 query 动态路由、
模型权重训练或真实 WAN 网络部署。

核心主张为：

> 对于需要分解、验证、并行、工具调用或异构模型分工的任务，反思驱动的 Pareto 搜索能够比固定人工架构、
> 随机搜索和仅质量工作流搜索找到更好的任务质量与资源成本折中。

AIME 用于检验方法能否避免不必要的多智能体编排。MASBench 和 GAIA 是架构自适应优势的主要证据。WAN profile
和异构模型池属于部署分析，不再作为方法唯一的动机。

## 2. 研究问题

- **RQ1：整体有效性。** RPAS 能否在 held-out 测试集上获得优于人工架构和自动工作流搜索 baseline 的质量-
  资源 Pareto 折中？
- **RQ2：搜索机制。** 在相同候选评估预算下，失败样本驱动的 LLM reflection 是否优于随机变异、仅质量进化、
  MCTS 和 meta-agent 搜索？
- **RQ3：任务结构。** RPAS 是否会为数学推理、MASBench 不同结构轴和 GAIA 工具任务选择不同拓扑？
- **RQ4：异构部署。** 当模型能力、价格和部署位置不同时，RPAS 是否能更有效地分配模型、角色、站点和通信策略？
- **RQ5：避免过度编排。** 当额外 agent 不能带来可靠收益时，RPAS 是否会保留简单的单 agent 架构？
- **RQ6：稳健性。** 结论能否在不同搜索种子、数据划分和部署 profile 下保持稳定？

## 3. 实验轨道

### 3.1 主实验 A：同构架构搜索

所有角色使用 Qwen3.5-9B 同构副本，从而隔离拓扑、角色、token 预算和通信策略的影响，避免模型能力差异干扰。

必须覆盖：

- AIME 2025 和 AIME 2026。
- MASBench 的 breadth、depth、horizon、parallel、robustness 五个轴。
- GAIA 2023 validation Level-1，采用第 5.3 节的固定受控划分。

所有方法必须使用共同 executor、角色 prompt、模型池、工具、初始架构库和 validation/test 流程。

### 3.2 主实验 B：异构架构搜索

使用冻结的三档模型池：

- `cheap`：用于规划、分解和常规工具交互的低成本快速模型。
- `standard`：本地 Qwen3.5-9B backbone。
- `strong`：用于困难求解、验证或聚合的更强付费 API 模型。

第一次正式异构实验前，必须在版本化 manifest 中固定具体 provider model ID、API 版本、价格、rate limit 和 endpoint
配置。同一实验矩阵中不允许更换模型。

必须覆盖：

- MASBench parallel、depth 和 robustness。
- GAIA Level-1 受控划分。
- AIME 2025，仅作为过度编排控制。

### 3.3 部署敏感性实验

主 profile 为：

- `lan_homogeneous`。
- `wan_normal`。
- `wan_bandwidth_limited`。
- `wan_edge_hub`。

`wan_degraded` 和 `wan_lossy_unstable` 作为稳健性或附录实验。论文必须将该机制表述为
**trace-based WAN emulation**，不能表述成真实网络 benchmark；当前不注入 sleep 或随机丢包。

必须包含两类分析：

1. Post-hoc trace remapping：同一执行 trace 映射到不同 profile，不重新调用模型。
2. Profile-conditioned search：在 `lan_homogeneous`、`wan_bandwidth_limited` 和 `wan_edge_hub` 上重新搜索，观察
   MASBench-parallel 和 GAIA 的最终架构是否发生变化。

gpu15/gpu7 跨节点实验只作为 sanity check，不作为主要 WAN 结果。

## 4. 搜索单位与数据泄漏约束

架构按 dataset/axis 搜索，而不是针对每个测试 query 单独搜索。每个 benchmark 必须划分为三个互斥集合：

- `D_search`：候选生成、reflection 和父架构选择。
- `D_select`：最终候选重评估、排序和 primary 架构选择。
- `D_test`：一次性的 held-out 测试。

测试标签和测试分数不能参与 reflection、mutation、父架构选择、停止条件、超参数选择或最终架构选择。
`selected_test_rows[0]` 必须始终是查看 test score 之前由 `D_select` 选出的候选。

数据划分固定使用 `data_seed=2026`；搜索随机性使用 `seed=0/1/2`。改变搜索 seed 不能改变任一数据划分。

## 5. 数据集与固定划分

### 5.1 AIME 2025/2026

- 开发数据：本地 `aimo-validation-aime.jsonl`，共 90 条。
- `D_search`：用 `data_seed=2026` 固定选择 60 条。
- `D_select`：剩余 30 条。
- `D_test`：`aime_2025.jsonl` 和 `aime_2026.jsonl` 各自全部 30 条。
- 指标：经过冻结 answer normalizer 后的 exact-match accuracy。

每个方法和搜索 seed 只在共同 AIMO 开发划分上搜索一次。同一个 validation-selected 架构分别评估 AIME 2025
和 AIME 2026，不能为了两个年份分别选择不同架构。

### 5.2 MASBench

- 使用官方 train/test split。
- 覆盖 breadth、depth、horizon、parallel、robustness 五个轴。
- 每个轴的 `D_search`：官方 train 中 24 条。
- 每个轴的 `D_select`：官方 train 中另外 24 条。
- 每个轴的 `D_test`：官方 test 中 60 条，并在可用 axis value 间平衡采样。
- 指标：使用 `masbench_igsm_mod23_v1` 或明确版本化后继协议的结构化 exact match。

必须保存固定样本 ID 和 axis value 数量的 dataset manifest。现有 `masbench_depth_v4_sanity`、
`masbench_dag_parallel_v5_sanity` 等都只属于 calibration。

### 5.3 GAIA

- 使用 GAIA 2023 public validation。
- 主范围为全部可用 Level-1 样本。
- 固定排除损坏的零字节附件 `f918266a-b3e0-4914-865d-4faa564f1aef.py`。
- 预计可用 Level-1 为 52 条，以最终 manifest 检查为准。
- `D_search`：24 条。
- `D_select`：12 条。
- `D_test`：16 条。
- 使用 `data_seed=2026` 固定划分，并尽量按有无附件及附件类型分层。
- 指标：GAIA normalized exact-match accuracy。

正式实验必须尝试全部 52 条可用 Level-1，runtime 需要支持本地文件、Python/code、表格、文档、图像、音频及必要的
web/search。如果 runtime 不能覆盖全部 Level-1，只能报告 `GAIA-L1-supported-subset`，同时公开固定任务 manifest，
不能把它称为官方 GAIA 分数。

现有 24 条 `gaia_adapter_calibration` 仅用于 runtime calibration，其 16.7% 分数和 verifier 行为不能并入正式表格。

## 6. 共同架构搜索空间

所有搜索方法使用相同 typed genome 和 executor。

### 6.1 拓扑

- `single`。
- `self_consistency`。
- `solver_verifier`。
- `planner_solver_verifier`。
- `debate`。
- `dag_decompose`。

### 6.2 可搜索字段

- 拓扑。
- Agent 角色和模型分配。
- 逻辑站点 placement。
- 各角色 `max_tokens`。
- Self-consistency 样本数。
- DAG worker 数量。
- 边通信策略：`full`、`summary`、`final_only`、`critic_brief`。

### 6.3 共享初始架构库

每个自动搜索方法从同样的九个已评估 seed architecture 开始：

1. `single_local`。
2. `single_strong_remote`。
3. `self_consistency_local`。
4. `solver_verifier_local`。
5. `solver_local_verifier_remote_summary`。
6. `solver_local_verifier_remote_final_only`。
7. `planner_solver_verifier_split`。
8. `debate_local_remote`。
9. `dag_decompose_three_workers`。

同一数据划分、profile、模型 manifest 和搜索 seed 下，九个 seed 的评估结果必须跨方法共享缓存。Cache hit 必须复用
完整 rollout 结果，不能重新推理，以消除相同架构在不同 mode 间的噪声。

## 7. 对比方法及论文命名

### 7.1 非搜索执行 baseline

- **Single Agent**：validation 选择的最强单 agent 设置。
- **Self-Consistency**：固定三样本投票架构。
- **Solver-Verifier**：最强 validation-selected 固定 solver-verifier seed。
- **Multi-Agent Debate**：固定 debate seed。
- **DAG Decomposition**：固定 decomposition/parallel-worker/aggregator seed。
- **Best Handcrafted**：使用共同 selector 从全部九个 seed 中选择一个 primary 架构。

### 7.2 自动搜索 baseline

- **Random-AS**：共同搜索空间中的随机 typed architecture 生成和变异。
- **AFlow-style MCTS**：在共同 RPAS executor 和搜索空间上进行 MCTS 候选搜索。
- **ADAS-style Meta-Agent Search**：由 LLM meta-agent 在共同搜索空间中提出可执行架构。

若官方外部仓库可以不改变方法地运行某个 benchmark，可以额外报告 `Official implementation`。共同空间重实现必须
命名为 `AFlow-style` 或 `ADAS-style`，不能声称完全复现官方实现。

### 7.3 RPAS 变体

- **RPAS-Quality**：使用 LLM reflection 和 typed mutation，但父架构与最终选择只考虑质量。
- **RPAS**：失败感知 LLM reflection、多 proposal typed mutation、Pareto parent sampling 和 Pareto 最终选择。

旧代码 mode 与论文名称对应关系：

| 代码 mode | 论文名称 | 类别 |
|---|---|---|
| `baselines` | Best Handcrafted 及固定架构结果 | 非搜索 baseline |
| `random` | Random-AS | 自动搜索 baseline |
| `quality_only` | RPAS-Quality | RPAS 消融 |
| `wan_pareto` | RPAS | 完整方法 |

这四个 mode 不能包装成四个外部方法。

## 8. 公平搜索预算

每个自动搜索方法固定获得：

- 九个共享 seed 评估。
- 二十四个**新的、唯一且可执行的候选架构评估**。
- 最大 archive 为 33 个候选。
- 相同 `D_search` 和评估并发度。
- 相同角色 prompt、工具、模型池、token pool 和候选有效性检查。

重复或无效 proposal 不消耗 24 个新候选预算。Proposer 最多重试三次。Reflection/meta-agent/MCTS controller 调用不计入
候选评估数，但必须单独报告其调用数、token、wall time 和 API cost，称为 **search overhead**。

RPAS 固定使用：

- LLM reflection。
- 每次 reflection 最多三个独立 mutation proposal。
- 最多三个压缩失败样本。
- `pareto_parent_prob=0.5`。
- `parent_score_band=0.05`。
- `parent_top_k=6`。

正式代码必须分别暴露 seed 数和新增候选数。最终实验禁止继续使用 `max_candidates=16`，因为九个 seed 后只剩七个
进化候选。

## 9. 候选评估和有效性

每个候选先在 `D_search` 上执行并记录预测以及完整 model/tool/communication trace。出现以下任一情况时，不允许进入
最终选择：

- 有效执行率低于 99%。
- Context window 或协议错误影响超过 1% 样本。
- 输出截断导致超过 5% 样本无法提取答案。
- 候选违反拓扑或工具 contract。

无效候选保留在诊断日志中，但必须从 Pareto front 和 primary selection 中排除，不能用错误换取低 token 成本。

搜索结束后最多选择八个 shortlist 候选在 `D_select` 上重新评估。若严格 Pareto front 不超过八个则全部进入；否则
保留质量点、效率点和六个覆盖前沿的代表点。只有 `D_select` 指标能够决定最终架构。

## 10. Pareto 目标和最终选择

### 10.1 主要目标

- 最大化任务 score。
- 最小化总模型 token。
- 最小化模型调用数。
- 最小化冻结价格 manifest 下的推理金额成本。

### 10.2 部署目标

- 最小化跨中心 token。
- 最小化 network-only latency estimate。
- 最小化 expected retry overhead。

Observed wall latency 对 server load 和 continuous batching 敏感，因此只报告、不作为主要 Pareto objective。部署目标在
异构和部署实验中参与搜索，在同构主实验 A 中属于辅助诊断。

### 10.3 两个预注册 operating point

每个搜索方法都只能基于 `D_select` 选出两个候选：

- **质量点 Q**：validation score 最高；平分时依次比较总 token、调用数、金额成本、跨中心 token 和 candidate ID。
- **效率点 E**：在质量相对 Q 的绝对下降不超过 `delta=0.05` 的有效 Pareto 候选中最小化总 token；平分时依次
  比较调用数、金额成本、跨中心 token 和 candidate ID。

主表同时报告 Q 和 E，其中 RPAS-E 是主要质量-效率结果。所有搜索方法必须使用同一 Q/E selector。不能根据 test
结果在 Q 和 E 之间切换。

## 11. 指标与统计报告

### 11.1 任务质量

- Exact-match accuracy 和正确题数。
- 各 dataset/axis accuracy。
- GAIA level 和附件类型 breakdown。
- MASBench axis-value breakdown。

### 11.2 推理资源

- 平均模型调用数。
- 平均 prompt、completion 和 total tokens。
- 平均 requested max tokens 和 maxed-call rate。
- 每题金额成本。
- 每题 observed model wall latency。
- GAIA 工具调用、工具失败和工具 latency。

### 11.3 部署资源

- 跨中心 total/message/model tokens。
- Network-only latency estimate。
- Expected retry latency 和 expected failures。
- Emulated end-to-end latency。

### 11.4 搜索资源和架构行为

- 唯一候选评估数。
- Reflection/controller 调用数、token、wall time 和金额成本。
- 有效、无效、重复 proposal 数量。
- Pareto front 大小、dominated hypervolume 和 frontier coverage。
- 最终拓扑、角色-模型分配、站点、token 预算和通信策略。
- 最终选择单 agent 与多 agent 的比例。

### 11.5 重复与不确定性

- 正式搜索 seed：`0/1/2`。
- 数据划分 seed：固定 `2026`。
- 报告搜索 seed 间 mean 和 standard deviation。
- 对 held-out 样本做 10,000 次 paired bootstrap，报告 95% confidence interval。
- 所有方法共享测试样本，采用 paired comparison。
- AIME 额外报告 `correct/30` 和 Wilson binomial interval。

## 12. 主表与图

### 表 1：整体质量与效率

行：Single Agent、Best Handcrafted、Random-AS、AFlow-style MCTS、ADAS-style Meta-Agent Search、RPAS-Quality、RPAS。

每个 benchmark 的列：test score、total tokens、calls、金额成本、observed wall latency。自动搜索方法报告 Q/E 两点。

Benchmark：AIME 2025、AIME 2026、MASBench 五个轴、GAIA Level-1。

### 表 2：异构模型分配

行：Best Handcrafted、Random-AS、AFlow-style MCTS、RPAS-Quality、RPAS。

列：score、各模型档位 token、各模型档位 calls、API cost、最终拓扑和角色-模型分配。

Benchmark：MASBench depth/parallel/robustness 和 GAIA Level-1。

### 图 1：质量-资源 Pareto front

分别画 held-out quality 对 total tokens、calls 和金额成本。AIME、MASBench、GAIA 各一个 panel。不能把不同 test selection
产生的点连接成一条优化轨迹。

### 图 2：架构自适应

展示不同 MASBench axis 和 GAIA 附件/工具类别最终选择的拓扑及角色-模型 placement。

### 图 3：部署敏感性

展示 LAN、带宽受限 WAN 和 edge-hub 下的跨中心 token、trace-emulated network latency 及 profile-conditioned RPAS
是否改变架构。

## 13. 消融实验

在 MASBench depth、MASBench parallel、GAIA Level-1 上运行 seed 0/1/2。AIME 不需要覆盖所有消融。

- **No LLM reflection**：只使用 rule/random typed mutation。
- **No failure examples**：reflector 只能看到聚合指标和架构，看不到失败 rollout。
- **Single proposal**：每次只生成一个 mutation proposal。
- **No Pareto search**：即 RPAS-Quality。
- **No Pareto parent sampling**：父架构只从 high-quality band 选择。
- **Structure-only mutation**：关闭 model、site、communication mutation。
- **No deployment objectives**：异构/WAN 实验中移除跨中心和网络目标。

正文最少必须包含前四个消融；算力不足时后三个可以进入附录。

## 14. 正式实验门禁

以下门禁全部通过前，任何 output 都不能标为 `formal`：

- [ ] **G1 仓库门禁**：初始化 private Git；记录 clean commit；不追踪 output、data、模型、日志、cache、secret 或
  个人绝对路径。
- [ ] **G2 协议门禁**：合作者审核本协议并创建 tag `experiment-protocol-v1.0`。
- [ ] **G3 数据门禁**：manifest 固定 task ID、source hash、split seed 和排除原因。
- [ ] **G4 Executor 门禁**：AIME、MASBench、GAIA 使用共同 genome、trace schema、有效性规则和最终 selector。
- [ ] **G5 GAIA 门禁**：正式 GAIA 样本需要的工具全部可用，不发生 unsupported-tool fallback。
- [ ] **G6 异构门禁**：固定具体模型 ID、API 版本、价格、rate limit 和角色可用性。
- [ ] **G7 Baseline 门禁**：Random-AS、AFlow-style、ADAS-style 通过共同空间公平性测试，统一使用 24 个新候选预算。
- [ ] **G8 可复现门禁**：unit test、lint、各数据集端到端 smoke、cache、resume 和 manifest hash 检查通过。
- [ ] **G9 报告门禁**：结果记录 code commit、协议版本、config hash、dataset/model manifest hash、prompt protocol、
  search seed、data seed 和 selection policy。

## 15. 现有结果的定位

以下结果由于早于本协议或使用 calibration 规模，不能作为最终论文结果：

- 所有 `phase1*` 小模型 GEPA 边界实验。
- `rpas_aime_formal_v1/v2/v3`。
- 所有 `*_sanity`、`*_smoke`、`*_calibration`、`_tmp_*`。
- 现有 MASBench depth/parallel calibration。
- 现有 GAIA readiness、tool、verifier 和 adapter calibration。

它们可以用于工程决策、prompt calibration、runtime 检查和运行时间估计，但不能与 protocol-v1.0 正式结果合并统计。

## 16. 冻结与变更控制

v1.0 冻结以下内容：研究问题、benchmark 范围、三段数据隔离、data seed、search seeds、共同搜索空间、九个 seeds、
24 个新候选预算、必须对比的方法、Q/E 两个 operating point、主要指标、消融和防泄漏规则。

Endpoint URL、GPU ID、API credential 属于基础设施变量，应放在不追踪的环境文件中。具体模型 ID 和价格属于科学变量，
必须冻结到 model manifest。

第一次正式运行后的科学变更必须：

1. 创建新协议版本；兼容澄清使用 `v1.1`，改变 claim、预算、划分或 selector 使用 `v2.0`。
2. 在 changelog 中写明日期、原因和受影响实验 cell。
3. 重跑所有受影响的方法与数据集，禁止只重跑对 RPAS 有利的 cell。
4. 保留旧 manifest 和原协议版本结果。

若 bug fix 会改变 prediction、trace、成本、候选有效性或选择，也必须升级协议并重跑受影响 cell。仅格式或日志修复不需要。

## 17. 正式 artifact 布局

每个正式 run 必须生成：

```text
outputs/formal_v1/<dataset>/<task_or_axis>/<deployment>/<method>/seed_<seed>/
  result.json
  summary.csv
  search_rows.jsonl
  selection_rows.jsonl
  search_checkpoint.json
  proposal_rows.jsonl
  search_overhead_rows.jsonl
  search_overhead.json
  selected_quality_candidate.json
  selected_efficiency_candidate.json
  test_outputs_<candidate_id>.json
  run_manifest.json
```

跨 run 汇总必须由脚本生成，不能人工改表。若 protocol version、config hash、dataset manifest 或 model manifest 混用，
论文表格生成器必须拒绝汇总。

## 18. 冻结协议带来的代码任务

截至 2026-08-27 的实现状态：

1. **已实现，待服务器门禁。** 增加独立 `D_search` 和 `D_select` 执行路径。
2. **已实现，待服务器门禁。** 将 `seed_candidates=9` 与 `new_candidate_budget=24` 分开。
3. **部分实现。** 当前四个 mode 已共享九个 seed archive 和最终 selector；外部 baseline 尚未接入同一 adapter contract。
4. **已实现，待测试/服务器门禁。** Pareto 和 selection 前排除无效候选。
5. **部分实现。** 已输出模型金额成本及 reflection/controller search overhead；仍需冻结正式模型价格和本地 GPU
   成本口径。
6. **已实现，待测试/服务器门禁。** 同时输出 Q/E 两个 operating point 及其候选 artifact。
7. **待实现。** 将 GAIA tool loop 并入共同架构 executor。
8. **待实现。** 实现 AFlow-style MCTS 和 ADAS-style Meta-Agent Search adapter。
9. **部分实现。** 每个 run 已输出规范化 split、model、config、network 和 protocol hash；仍需补 source-file hash、
   强制 clean code commit 和汇总阶段严格门禁。
10. **待实现。** 增加正式汇总、置信区间和协议一致性检查。

这十项完成前，即使样本量很大，新结果仍然只能算 pilot。
