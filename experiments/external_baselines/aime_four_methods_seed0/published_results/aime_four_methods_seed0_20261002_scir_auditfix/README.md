# 四种外部方法 AIME seed-0 运行产物

**本次运行未通过质量门禁：0/4 方法正式通过，`aggregate_status=blocked_incomplete`。**
这里发布的是失败/被拒绝尝试的原始产物与审计证据，不能作为合格的四方法正式对比结果。

运行时间（北京时间）：2026-10-02 22:18:42 至 2026-10-04 02:41:02。
公开基线为 `4e5791c4`；实际运行源为 `66ad8287f759e08a5310c190cd0eddb273bbe627`，包含失败分类审计补丁 `9fa62d6` 与源码清单修正。
该补丁未改模型、提示词、数据、划分和搜索预算。本次发布提交是在运行结束后新增产物，不是运行时源码提交。

## 结果状态

| 方法 | 终态 | 原因与测试集访问 |
|---|---|---|
| MaAS | `failed` | D_select 3/30 缺少 `solution_letter`（10%），超过 5%；未打开 D_test |
| ADAS | `failed` | D_select 无满足失败率 ≤5% 的有效候选；未打开 D_test |
| G-Designer | `failed` | D_select 失败率超过 5%；未打开 D_test |
| AFlow | `rejected` | 计算完成，最终质量门禁 FAIL；各年份测试有 2/30 样本超时（6.7%），总截断率 1079/4832（22.3%） |

AFlow 原始 exact-match 分数：AIME 2025 为 17/30（56.67%），AIME 2026 为 20/30（66.67%）。
Q/E 同选第 5 轮，E 复用 Q 结果；不是两次独立实跑。两年答案可解析率均为 22/30（73.33%）。
AFlow 总调用 4,832 次，prompt tokens 11,092,615，completion tokens 8,415,858，total tokens 19,508,473。
API failed_calls=0 不代表样本无失败：本次 AFlow 遥测记录了 17 次 sample timeout。
原始分数保留用于审计，但质量门禁拒绝后不能纳入正式 aggregate；仅 seed 0，不报告跨种子均值/方差。

## 目录与复核入口

- [summary.json](summary.json)：从原始记录导出的状态、token 与限制说明。
- [artifacts/attempt_summary.json](artifacts/attempt_summary.json)：原 launcher 的最终状态。
- [AFlow native_result.json](artifacts/aflow_maas/aflow/seed_0/native_result.json)、[quality_gate.json](artifacts/aflow_maas/aflow/seed_0/quality_gate.json)：完整计算记录和拒绝原因。
- `artifacts/`：运行 manifests、冻结/测试访问记录、逐题 CSV、候选图/提示词、控制器、ADAS 搜索 archive、原生日志及与固定上游不同的运行文件。
- `diagnostics/`：本次四方法 synthetic smoke 的 report、telemetry、worker spec、预测与运行产物；不等同于正式 AIME 测试通过。
- `logs/`：本次 4 套正式服务和 4 套 diagnostic 服务的 vLLM/GPU 日志、session 与模型核验报告。
- `evidence/`：运行时源码 SHA 清单及冻结数据清单。
- [artifact_index.json](artifact_index.json)：每个复制文件的原路径、字节数和 SHA256；原始字节保持不变。
- [SHA256SUMS](SHA256SUMS)：整个发布包的独立字节校验（清单自身除外）。在本目录执行 `sha256sum -c SHA256SUMS`。

上游静态源码、图片、其它任务 workspace 和 Python 缓存不复制；上游可按 manifests 中的固定提交获取。
运行克隆里保留了本次生成文件和相对固定上游的修改文件，原始绝对路径在 artifact index 中可映射。
原始运行目录、模型和数据文件均未修改。

## 必须保留的限制

MaAS 是重建 helper/上下文兼容的**适配器变体**；ADAS 使用 `restricted_ast_v1`；G-Designer 有 role-graph/numerical adapters。
不能把这些称为上游模块逐字复现。MaAS 路径 `maas_repo/workspace/rpas_aime_native/MATH/test` 本次实际用于 **D_select**，不是正式 D_test。

AFlow 第 5 轮 D_select 有 3/30 个不可解析答案（90% 可解析），但显式空输出/执行异常失败率为 0%。
当前 AFlow 门禁检查的 workflow failure 与共享 parser validity 是不同指标，不能称为所有输出都成功。

MaAS gate 异常后没有完整 formal telemetry；ADAS/G-Designer 没有完整逐候选/逐样本 D_select 失败日志。
这里保留实际存在的记录，不补造缺失证据。

旧源码 SHA 清单本身含 CRLF，且 `deployment/hang_recovery_20261001/archive.sha256` 原字节与清单期望只有 CRLF/LF 差异。
规范化换行后哈希一致；本次独立的发布 `SHA256SUMS` 校验的是实际上传原字节。
