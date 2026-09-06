# RPAS 最新对比实验参数清单（2026-09-06）

本文档用于次日实验讨论和结果审计。参数以
`RPAS_final_benchmark_native_fidelity_protocol_V4.md`、当前 runner、冻结数据 manifest
和 SCIR 实际作业为准。本文不把 one-seed/pilot 结果表述为正式论文结果。

## 0. 明日讨论速览

截至 2026-09-06 21:53（SCIR，UTC+8），最新版代码为 `447ea58`。三组实验的
科学参数已经冻结；当前运行只是在验证 one-seed 的质量、成本和工程稳定性。

| EC | 核心问题 | Benchmark | 方法 | 当前 seed | 冻结 split | 当前状态 |
|---|---|---|---|---:|---|---|
| EC-1B | hard-code stress | LiveCodeBench `release_v6` | Single / AFlow / RPAS-Full | 0 | 20 / 64 / 64 / 256 | 三方法运行中 |
| EC-2 | communication topology | MMLU-57 | Single / Full / Chain / G-Designer / RPAS-Comm | 1 | 57 / 57 / 570 | 三个固定基线完成；两方法运行中 |
| EC-3 | cross-task generalization | HotpotQA distractor | Single / AFlow / RPAS-Full | 0 | 正式 40 / 120 / 80 / 800；pilot 40 / 8 / 8 / 0 | clean gate result 已完成；不解锁 test |

这里的 split 顺序分别是：EC-1B/EC-3 为
`D_calib / D_search / D_select / D_test`，EC-2 为 `D_search / D_select / D_test`。

讨论时需要把三类量严格分开：

1. **冻结科学参数**：模型、数据、解码、方法原生控制流、搜索与选择规则。
2. **运行时参数**：GPU 型号、端口、Slurm 时限和服务 batch；只影响吞吐与 provenance。
3. **已实现成本**：真实 calls、tokens、轮次和 wall-clock；用于公平性分析，不反向修改冻结参数。

## 1. 参数状态约定

| 标签 | 含义 |
|---|---|
| `LOCKED` | 科学比较参数。正式比较中不得按结果调整。 |
| `PILOT_OVERRIDE` | 仅用于快速可执行性/方向判断的缩减配置，必须 `formal_result=false`。 |
| `RUNTIME_ONLY` | GPU、并发、服务端口、Slurm 时限等工程参数；不得改变模型输出契约。 |
| `PENDING_GATE` | 尚未满足正式结果门槛，不能写成论文主结果。 |

## 2. 三个实验的共同参数

| 参数 | 最新值 | 状态/说明 |
|---|---:|---|
| Backbone | `Qwen/Qwen3.5-9B` | `LOCKED`，三个 EC 使用同一模型 |
| 本地模型路径（SCIR） | `/home/jianbaizhao/model/Qwen/Qwen3.5-9B` | `RUNTIME_ONLY` |
| 推理精度 | FP16 | `LOCKED`，非量化 |
| Temperature | `0.0` | `LOCKED` |
| Thinking | disabled | `LOCKED`，`GEPA_QWEN_DISABLE_THINKING=1` |
| API | 本机 OpenAI-compatible HTTP service | 每个 Slurm 作业独立 endpoint |
| 网络 profile | `lan_homogeneous` | 不把同机调用伪装为 WAN 实测；网络延迟为 0 合理 |
| Data seed | `2026` | 三个 EC 的冻结数据抽样种子 |
| Seed 目标 | `0, 1, 2` | 正式聚合要求 3 seeds |
| 当前快速结果 | seed 0 或 seed 1 单 seed | 一律 `formal_result=false` |
| GPU 单元 | 每个方法/seed 独占 1 张 GPU | 作业内不跨卡切分 |
| 最新代码提交 | `447ea58` | GitHub/local 最新；每个作业仍以文件 SHA 和环境清单为最终依据 |
| 记录项 | prompt/completion/total tokens、调用数、轮次、wall time、模型、GPU、split、candidate | 缺项不得进入聚合 |

SCIR 当前可接受的 32GB 以上卡均可运行 9B FP16；实际作业已经使用 A100-40GB、
A100-80GB、A6000。GPU 型号属于 runtime provenance，不应作为方法变量。

## 3. EC-1B：LiveCodeBench release_v6

### 3.1 数据与方法

| 参数 | 值 | 状态 |
|---|---:|---|
| 数据集 | `livecodebench/code_generation_lite` | `LOCKED` |
| Dataset revision | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` | `LOCKED` |
| Release | `release_v6` | `LOCKED` |
| `D_calib` | 20 | `LOCKED` |
| `D_search` | 64 | `LOCKED` |
| `D_select` | 64 | `LOCKED` |
| `D_test` | 256 | `LOCKED` |
| 当前方法 | Single / native AFlow / native RPAS-Full | seed 0 pilot |
| 主指标 | pass@1（共享 LCB evaluator） | 同一题、同一测试器 |

数据 SHA-256：

| Split | SHA-256 |
|---|---|
| calib | `b26d0fbbf6f618afadbb6734b62934a3471f9ec7aed133bba866b792ded68a43` |
| search | `64df9993e51a5b9e980b7a75fde1e79be2e1f6698e88bc663200f21c282f4ec3` |
| select | `952311f5b91ebcfdbc749c49238017005ddd0697d46ffd9adb149fd1931aa4fa` |
| test | `86c41f057bad67ef805bdbba982dcf0f273d59c86de2357438140c703460f337` |
| manifest | `af68ac1bfa55ad1ceec8cd353e1b4f8890649232c2d57c86fb29f370e6b96e99` |

### 3.2 共享生成与执行参数

| 参数 | 值 |
|---|---:|
| Executor max tokens | 2048 |
| Evaluation concurrency | 4 |
| 单次生成 timeout | 300 s |
| Service max batch size | 4 |
| Service batch wait | 25 ms |
| 代码执行 | 独立 evaluator process |
| 搜索反馈 | `D_search` 只允许 public cases |
| Private cases | 仅 `D_select/D_test`，不得进入搜索反馈 |

### 3.3 方法专属参数

| 方法 | Search | 关键参数 |
|---|---|---|
| Single | 无 | 直接对 `D_test` 生成；`search_calls=0` |
| AFlow | 官方 `Optimizer.optimize("Graph")` | pinned commit `3f457218fc716093fe53f6df8a5d5e6379d66346`；`sample=4`；`max_rounds=2`；`validation_rounds=1`；`check_convergence=false` |
| RPAS-Full | 原生 `experiments.phase2_wan_agent_search.run_search` | `mode=wan_pareto`；seed candidates=4；new candidates=3；shortlist=8；test top-k=2 |

RPAS-Full 其余锁定项：

| 参数 | 值 |
|---|---:|
| Reflection mode | LLM |
| Reflection max tokens | 1024 |
| Reflection children | 3 |
| Reflection examples | 3 |
| Rule fallback | 0（强制） |
| Selection strategy | `quality_band_cost` |
| Quality band | 0.05 |
| Pareto parent probability | 0.5 |
| Parent score band | 0.05 |
| Parent top-k | 6 |
| Evaluation cache | disabled |
| Resume | false |

### 3.4 当前运行时配置

| 方法 | Job | GPU | Slurm 时限 | 输出目录 |
|---|---:|---|---:|---|
| Single | `133465_0` | A100-40GB | 24h | `outputs/ec1_livecodebench_v4_seed0/single/seed_0` |
| AFlow | `133463_1` | RTX A6000 48GB | 24h | `outputs/ec1_livecodebench_v4_seed0/aflow/seed_0` |
| RPAS-Full（原作业） | `133486_2` | A100-80GB | 6h | `outputs/ec1_livecodebench_v4_seed0/rpas/seed_0` |
| RPAS-Full（超时兜底） | `133542_2` | A100-80GB | 24h | `outputs/ec1_livecodebench_v4_seed0_rpas_24h/rpas/seed_0` |

兜底作业与原作业使用相同科学参数，只更换输出目录并延长 Slurm wall-time；它依赖
EC-3 clean pilot 完成后启动。若原 RPAS 先完整结束，应取消兜底，不能混合两个 run
的中间产物。GPU 差异只作为 runtime provenance，不能解释为方法变量。

## 4. EC-2：MMLU communication-topology v2

### 4.1 对比问题与数据

本实验只比较 communication topology。除 Single reference 外，所有方法共享六个
worker、角色、消息语义、FinalRefer、轮次和模型。RPAS-Comm 不允许修改模型、角色、
token budget、压缩或 agent 数量。

| 参数 | 值 | 状态 |
|---|---:|---|
| Subjects | 57 | `LOCKED` |
| Data seed | 2026 | `LOCKED` |
| Search per subject | 1（共 57 条池） | G-Designer/RPAS 各实际消耗 40 次 search graph execution |
| Select per subject | 1（共 57） | 独立于 search |
| Test per subject | 10（共 570） | held-out |
| Methods | Single / Fully Connected / Chain / G-Designer / RPAS-Comm | 主矩阵 |
| Secondary | RPAS-Full | 只能作为非 topology-isolated 补充，不可替代 RPAS-Comm |
| Answer parser | strict A/B/C/D | `LOCKED` |

### 4.2 六 agent 公平性控制

| 参数 | 值 |
|---|---|
| Agent count | 6 |
| Roles | Knowlegable Expert / Critic / Mathematician / Psychologist / Historian / Doctor |
| Communication rounds | 1 |
| Worker max tokens | 256 |
| Final aggregator | `GDesigner.FinalRefer` |
| Message policy | 原样转发 official G-Designer messages |
| Compression | 所有方法禁用（官方 G-Designer 无此 primitive） |
| Fixed-method evaluation concurrency | 4 |
| G-Designer commit | `a6efcfa`（完整 HEAD 写入每次 environment/manifest） |

### 4.3 G-Designer 原生训练参数

| 参数 | 值 |
|---|---:|
| Graph | query-conditioned GCN，`optimized_spatial=true` |
| Iterations | 10 |
| Batch size | 4 |
| Learning rate | 0.1 |
| Rounds | 1 |
| Training query budget | 40 |
| D_select | audit-only；官方十轮 loop 不做 checkpoint selection |
| Checkpoints | `gdesigner_initial.pt`、`gdesigner_pretest.pt`，均记录 SHA |

### 4.4 RPAS-Comm 参数

| 参数 | 值 |
|---|---:|
| Candidate space | full-connected / chain / star / layered |
| Seed topology | full-connected |
| New candidates | 3（总候选 4） |
| Search executions | 40（每候选 10，匹配 G-Designer） |
| Reflection | LLM-only typed topology mutation |
| Reflection max tokens | 768（仅 meta-call） |
| Worker/final max tokens | 256，不变 |
| Reflection children | 1 |
| Rule fallback | 0 |
| Selection | 最大化 `D_select` accuracy，再最小化 inter-agent tokens，再按 candidate ID |
| Test | 冻结一个 selected topology 后运行 570 条 |

注意：768 只用于产生可解析拓扑 JSON 的独立 meta-call，不是给 worker 增加回答预算。

### 4.5 当前 seed-1 已完成基线与风险项

所有已完成行使用相同 split manifest：
`e83ebb2d9de3ded9bfd60ee0b66ba6f71ad9b5d8c2025f6c9455b96c4c1efb09`，
均覆盖 570 个唯一 held-out 样本且 model errors 为 0。

| 方法 | Accuracy | Model calls | Total tokens | 256-token cap 命中率 | 状态 |
|---|---:|---:|---:|---:|---|
| Single | 0.8070 | 1,140 | 444,670 | 11.9% | 完成，`formal_result=false` |
| Fully Connected | 0.8439 | 3,990 | 4,414,967 | 74.5% | 完成，`formal_result=false` |
| Chain | 0.8439 | 3,990 | 3,092,918 | 73.0% | 完成，`formal_result=false` |
| G-Designer | 待完成 | 待审计 | 待审计 | 待审计 | `132532` 运行中 |
| RPAS-Comm | 待完成 | 待审计 | 待审计 | 待审计 | `133525_13` 运行中 |

三条已完成基线的高 cap 命中率必须原样报告。当前不能为了改善结果单独提高某个方法的
worker/final token cap；如明天决定做更高 cap 的敏感性分析，必须作为全方法、全 seed
对称的新实验版本，不能覆盖 EC-2 v2 主矩阵。

## 5. EC-3：HotpotQA cross-task generalization

### 5.1 正式协议参数

| 参数 | 值 | 状态 |
|---|---:|---|
| Setting | HotpotQA distractor / provided-context | `LOCKED` |
| Prompt | context + question only；无 external retrieval | `LOCKED` |
| Metric | normalized token F1 + EM | `LOCKED` |
| `D_calib` | 40 | `LOCKED` |
| `D_search` | 120 | `LOCKED` |
| `D_select` | 80 | `LOCKED` |
| `D_test` | 800 | 六状态 unlock 前禁止打开 |
| Split manifest SHA | `1940fbef3792b691f60fe56c604d8e12f81e8f5ac71f444a7239462c0f3dd54c` | `LOCKED` |
| Methods | Single / native AFlow / native RPAS-Full | 3 seeds 才可正式聚合 |
| Executor max tokens | 512 | `LOCKED` |
| Meta max tokens | 4096 | `LOCKED` |
| Evaluation concurrency | 1 | 原生 AFlow/稳定性要求 |

### 5.2 当前 seed-0 最小 pilot override

| 参数 | 正式值 | 当前 pilot |
|---|---:|---:|
| `D_search` | 120 | 前 8 条冻结样本 |
| `D_select` | 80 | 前 8 条冻结样本 |
| `D_calib` | 40 | 完整 40，仅在 Q/E 冻结后诊断 |
| `D_test` | 800 | 不运行 |
| `formal_result` | 三 seed gate 后决定 | false |

### 5.3 RPAS-Full 锁定参数

| 参数 | 值 |
|---|---:|
| Native controller | `experiments.phase2_wan_agent_search.run_search` |
| Mode | `wan_pareto` |
| Seed candidates | 4 个不同原生 workflow |
| New candidates | 3 |
| Selection shortlist | 5 |
| Terminal candidates | Q/E top 2 |
| Reflection | LLM-only，children=3，example limit=3 |
| Rule fallback | 0 |
| Selection | `quality_band_cost`，band=0.05 |
| Pareto parent probability | 0.5 |
| Parent score band / top-k | 0.05 / 6 |
| Cache / resume | disabled / false |

### 5.4 AFlow 锁定参数

| 参数 | 值 |
|---|---:|
| Official commit | `3f457218fc716093fe53f6df8a5d5e6379d66346` |
| Optimizer | 官方 `Optimizer.optimize("Graph")` |
| Operators | Custom / AnswerGenerate / ScEnsemble |
| Sample | 4 |
| Initial round | 1 |
| Max new rounds | 1 |
| Validation rounds | 1 |
| Convergence check | false |
| Selection | 最大化 `D_select` F1，round ID 确定性 tie-break |

### 5.5 Saturation gate（当前已触发）

既有完整 `D_calib=40` 结果：

| 方法/候选 | F1 | EM | Valid rate |
|---|---:|---:|---:|
| Single (`single_local`) | 0.8938 | 0.8750 | 1.000 |
| AFlow round 1 | 0.4516 | 0.4000 | 1.000 |
| AFlow round 2 | 0.7877 | 0.7500 | 1.000 |
| RPAS 第二校准候选 | 0.8938 | 0.8750 | 1.000 |

V4 预注册规则是：Single F1 >= 0.80，且 AFlow/RPAS 均未提升超过 0.02 时，不解锁
HotpotQA `D_test`，转而讨论 MuSiQue amendment。按当前结果，这个条件成立。因此：

1. 当前只继续完成小 pilot 和审计，不运行 HotpotQA `D_test`。
2. 若明天决定保留 HotpotQA，必须形成书面 protocol amendment，不能静默绕过 gate。
3. 若改 MuSiQue，只允许替换 task adapter；AFlow/RPAS 原生 optimizer core 不变。

### 5.6 已完成的 RPAS seed-0 pilot（诊断结果）

| 项 | 值 |
|---|---:|
| SCIR job | 133487 |
| Runtime | 837.96 s（作业总 elapsed 14:58） |
| Search/select/reflection calls | 124 |
| Calibration calls | 40 |
| Total model calls | 164 |
| Search tokens | 201,311 |
| Calibration tokens | 61,200 |
| Total tokens | 262,511 |
| New candidates / mutation logs | 3 / 3 |
| Reflection calls | 4 |
| Rule fallbacks | 0 |
| Pareto archive size | 3 |
| `D_select` selected F1 | 0.7083（8 条 pilot，不可作为正式估计） |
| `D_calib` F1 | 0.8938 |
| Selected topology | single，solver max tokens 128 |
| Formal status | `formal_result=false` |

审计限定：job 133487 所用旧 preflight 曾读取 `D_test` 文件以验证 SHA/ID，但没有用于
生成、选择或评分。因此它只能作为诊断 pilot。最新版 preflight 已改为锁定前只检查
manifest 中的 `D_test` 元数据，并由回归测试证明即使测试文件不存在仍可完成 preflight。

### 5.7 已完成的 clean AFlow seed-0 pilot

| 项 | 值 |
|---|---:|
| SCIR job | `133521` |
| Slurm elapsed | 13:35 |
| Search / select / calibration 样本 | 8 / 8 / 40 |
| Search / select / calibration calls | 25 / 24 / 40 |
| Total model calls | 89 |
| Search / select / calibration tokens | 28,631 / 22,390 / 55,686 |
| Total tokens | 106,707 |
| `D_select` selected F1 | 0.5596 |
| `D_calib` F1 / EM | 0.4516 / 0.4000 |
| Model errors | 0 |
| `D_test` files / access | 0 / false |
| Formal status | `formal_result=false` |

该 run 的 job-root `SHA256SUMS` 已完整通过。clean RPAS 链为 `133531` calibration
后接 `133532` pilot；最终只接受满足 fallback=0、new candidates=3、typed mutations=3、
Pareto/Q/E 均存在且没有 `D_test` 访问的产物。

### 5.8 已完成的 clean RPAS calibration

`133531` 已正常完成（Slurm elapsed 10:55），并通过 job-root `SHA256SUMS` 校验。

| Candidate | Topology | `D_calib` F1 | EM | Calls | Tokens |
|---|---|---:|---:|---:|---:|
| `a0711933a84a` | single | 0.8938 | 0.8750 | 40 | 61,200 |
| `b42836deb980` | self-consistency, 3 samples | 0.8393 | 0.8000 | 120 | 186,215 |
| 合计 | - | - | - | 160 | 247,415 |

两个候选均 valid-answer-rate=1.0、generation-truncation-rate=0、model errors=0，且
`d_test_accessed=false`。该 calibration 只用于冻结候选与验证 gate，不是 held-out test
结果。其 clean pilot `133532` 已由 `afterok` 依赖自动启动并正常完成。

### 5.9 已完成的 clean RPAS seed-0 pilot

| 项 | 值 |
|---|---:|
| SCIR job | `133532` |
| Slurm elapsed / runner wall | 16:33 / 989.74 s |
| Search / select / calibration 样本 | 8 / 8 / 40 |
| Search / select / reflection / calibration calls | 96 / 24 / 6 / 40 |
| Total model calls | 166 |
| Search+select+reflection / calibration tokens | 206,745 / 61,200 |
| Total tokens | 267,945 |
| `D_select` selected F1 / EM | 0.7083 / 0.6250 |
| `D_calib` F1 / EM | 0.8938 / 0.8750 |
| New candidates / accepted typed mutations | 3 / 3 |
| LLM reflections / rule fallbacks | 6 / 0 |
| Q/E candidate | `45c0167f778d` / `45c0167f778d` |
| Model errors / `D_test` access | 0 / false |
| Formal status | `formal_result=false` |

RPAS 的前置 clean calibration 另消耗 160 calls、247,415 tokens，因此端到端链路为
326 calls、515,360 tokens；不能只报告 pilot 内的 166 calls。Pareto front 有 3 个原始
条目但只有 2 个唯一 canonical ID，这是 singleton model/site 下等价 seed workflow
折叠造成的；不影响 Q/E 选择，但不得描述为 3 个不同 Pareto 候选。便携结果包见
`handoff_progress/ec3_seed_0/`。

## 6. 每个结果必须交付的日志字段

每个 method/seed 的 manifest 或 telemetry 至少包含：

| 类别 | 必须字段 |
|---|---|
| Provenance | method、seed、dataset/revision、split SHA、代码 SHA、baseline commit、模型标识、GPU UUID |
| Decoding | temperature、max tokens、thinking、timeout、并发 |
| Search | candidate 数、iterations/rounds、selection policy、reflection/optimizer calls、fallbacks |
| Calls | prompt tokens、completion tokens、total tokens、调用模型、split、candidate、latency |
| Time | run start/end、wall-clock、Slurm elapsed、超时/重试 |
| Communication | active edges、messages、inter-agent tokens、judge input tokens（EC-2） |
| Quality | 每 split 样本数、accuracy/F1/EM/pass@1、valid/executable/truncation rate |
| Gate | `formal_result`、失败原因、是否访问 `D_test`、三 seed 是否齐全 |
| Integrity | `SHA256SUMS`，下载后 `sha256sum -c` 必须全通过 |

建议汇总表固定列：

```text
EC, method, seed, quality, EM/pass@1, total_tokens, search_tokens,
task_calls, optimizer_calls, rounds/iterations, wall_seconds,
active_edges, inter_agent_tokens, GPU, formal_result, gate_status
```

## 7. 当前作业快照（2026-09-06 22:25，SCIR UTC+8）

| EC | Method | Job | 状态 | 最近进度 |
|---|---|---:|---|---:|
| EC-1B | AFlow seed 0 | `133463_1` | RUNNING | 新 workflow round 21/64 |
| EC-1B | Single seed 0 | `133465_0` | RUNNING | 116/256，当前 61 correct |
| EC-1B | RPAS seed 0 | `133486_2` | RUNNING | 首候选评估中 |
| EC-1B | RPAS seed 0 24h 兜底 | `133542_2` | PENDING | 依赖 `afterany:133532` |
| EC-2 | G-Designer seed 1 | `132532`（显示为 `132507_10`） | RUNNING | 最近核验 514/570 test |
| EC-2 | RPAS-Comm seed 1 | `133525_13` | RUNNING | 2/3 LLM reflections 已记录；无 fallback/error |
| EC-3 | AFlow seed 0 clean pilot | `133521` | COMPLETED | 8/8/40；89 calls；106,707 tokens |
| EC-3 | RPAS clean calibration | `133531` | COMPLETED | 160 calls；247,415 tokens；SHA 通过 |
| EC-3 | RPAS seed 0 clean pilot | `133532` | COMPLETED | 166 calls；267,945 tokens；SHA 通过 |
| EC-3 | RPAS seed 0 旧 pilot | `133487` | COMPLETED | 仅诊断；旧 preflight 读取过 test 文件 |

## 8. 明天需要讨论的决策

1. EC-3 是否严格按已触发的 saturation gate 转 MuSiQue；若继续 HotpotQA，必须先签书面 amendment，不能直接解锁 `D_test`。
2. EC-1B seed-0 三方法完整审计后，是否按完全相同参数扩展 seed 1/2；先确认 RPAS 6h 原作业或 24h 兜底只有一个有效结果。
3. EC-2 在 seed-1 五方法矩阵完成后，是直接扩 3 seeds，还是先做全方法对称的 256/512-token cap 敏感性实验。当前主矩阵不得静默改 cap。
4. Search-budget 主表按累计 pre-test tokens 对齐；native iterations/candidates 作为独立列报告，不伪称完全相同。
5. 统计主表只接收三个 protocol-valid seeds；one-seed 只做方向、资源和失败模式判断。

## 9. 禁止性表述

- 不得把 pilot/one-seed 写成 formal result。
- 不得在 EC-3 six-state unlock 前运行或读取 `D_test` 内容。
- 不得称 EC-2 RPAS-Full 严格证明 topology search 优于 G-Designer；该结论只由 RPAS-Comm controlled comparison 支撑。
- 不得仅因平均分更高忽略 invalid output、truncation、fallback 或审计缺项。
- 不得用失败后改过的参数覆盖原 run；必须新 job、新目录、新 manifest。
