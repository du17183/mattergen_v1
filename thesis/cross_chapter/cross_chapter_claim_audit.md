# 第3—5章 Claim–Evidence 跨章审计

本表以冻结 `thesis_release/CLAIMS_AND_LIMITATIONS.md` 为最高解释约束。状态只适用于当前 checkpoint、任务、cohort、指标和代理评价协议。

| ID | 最终主张 | 第3章 | 第4章 | 第5章 | 证据与状态 | 审计结论 |
|---|---|---|---|---|---|---|
| C01 | 合法分支预算内存在属性改善空间 | 3.6 | — | 5.3/5.6 | Branch Oracle test n=12；30.41%；SUPPORTED_MECHANISM | 一致；明确不是可部署效果 |
| C02 | Fixed-K2 优于 C0 | 3.7 | — | 5.2/5.9 | C1-128；24.32%；绝对CI全正；SUPPORTED | 一致；主张限代理属性和2.2×预算 |
| C03 | Fixed-K2 通过冻结代理质量护栏 | 3.7.3 | — | 5.2.2 | E-hull/Stable/NUS改善，Validity不变；SUPPORTED_GUARDRAIL | 一致；Novel下降已披露 |
| C04 | Linear-K2 优于 Fixed-K2 | 3.8 | — | 5.3/5.9 | Phase B与C1均未确认；NOT_SUPPORTED | 一致；未将方向性小样本写成优势 |
| C05 | 在线 Adaptive CFG 稳定有效 | 3.2—3.4 | — | 5.3/5.9 | V1 mixed；V2/V4/V5/Field fail；NOT_SUPPORTED | 一致；仅作为研究演化与负结果 |
| C06 | RC-NFGD 降低 MatterSim MaxF | — | 4.6 | 5.4 | Formal256 29.81%，CI全正；SUPPORTED | 一致；明确同源代理边界 |
| C07 | RC-NFGD 降低 Mean Force 与 RMSD | — | 4.6 | 5.4 | 29.72%/14.17%，CI全正；SUPPORTED | 一致；未扩大为所有结构质量改善 |
| C08 | RC-NFGD 保持 Stable 与 Validity | — | 4.6 | 5.4 | 80.86%/100%均不变；SUPPORTED_GUARDRAIL | 一致；只指两个冻结护栏 |
| C09 | RC-NFGD 改善 Property MAE | — | 4.6 | 5.4/5.9 | 恶化1.14%，不利CI不跨零；NOT_SUPPORTED | 一致；属性代价保留 |
| C10 | CHGNet 支持跨神经势均值方向 | — | 4.7 | 5.5 | MaxF/Mean Force改善；SUPPORTED_INDEPENDENT_SURROGATE | 一致；尾部混合且非DFT |
| C11 | 有界位置修正是性能核心 | — | 4.8.2 | 5.6.2/5.9 | unbounded更强；NOT_SUPPORTED | 一致；仅保留保守工程角色 |
| C12 | 在线 RC-NFGD 优于等预算 POST | — | 4.8.3 | 5.6.3/5.9 | POST代理力更低；NOT_SUPPORTED | 一致；在线反馈可行性与最优性已分开 |
| C13 | 两项创新具有已验证联合协同 | 3.9边界 | 4.8.4/4.9 | 5.8/5.9 | A5低于门槛；NOT_SUPPORTED | 一致；A5并非 Fixed-K2×RC-NFGD 直接验证也已注明 |
| C14 | 方法获得真实物理/DFT验证 | 3.9 | 4.9 | 5.1/5.9 | 无DFT；FALSE | 一致；`DFT_VERIFIED=false` |
| C15 | RC-NFGD 的后期窗口有阶段可靠性依据 | — | 4.2—4.4 | 5.6 | \(t=0.02\)，Spearman=0.925587；SUPPORTED_MECHANISM | 一致；相关性不写成因果最优 |
| C16 | 真实力方向包含有效信息 | — | 4.8.1 | 5.6.1 | G1消融改善30.48%；SUPPORTED_DIRECTION | 一致；路径依赖限制已保留 |

## 审计结果

- 主张总数：16。
- 跨章状态冲突：0。
- 缺失冻结证据的结果主张：0。
- 需要降级或删除的无证据主张：0（章节初稿已使用保守表述）。
- 明确保留的负/边界结论：Linear-K2、在线 Adaptive CFG、Property MAE改善、bound性能核心、online-over-POST、联合协同、DFT验证，共7类。
