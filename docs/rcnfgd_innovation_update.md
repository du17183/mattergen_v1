# 创新点 2 补充证据与论文定位更新

## 当前判定

1. **注入时机**：原方法在 predictor 的末 20 个时间步 `t≤0.02` 注入力；为什么它优于早/中期，在新的配对等力调用实验中尚未得到验证。本次 timing 消融 `NOT_RUN`，不能写入第 4 章结果或摘要贡献。
2. **属性—力 Pareto**：已有 A6 的 32 个配对 seed 足以完成补充性分析。A6 上 C0/F0/POST 的 MatterSim MaxF 均值分别为 0.214179/0.163650/0.043275 eV/Å，Property MAE 均值分别为 0.007900/0.007886/0.007933。点估计前沿包括 F0 和 POST，但 F0 与 POST 的 MAE 差值配对 95% CI 跨零；POST 的 MatterSim MaxF 显著更低。原 A1 Formal256 中 F0 的 MaxF 更低，但 Property MAE 由 0.009757 增至 0.009868，配对 C0−F0 CI 为 [-0.000236,-0.000023]。因此不能用 A6 的极小 MAE 差异声称稳定属性优势，也不能把 A1/A6 跨 cohort 均值混为同一前沿。
3. **评价依赖**：MatterSim 既参与力引导又用于主要终态 MaxF，存在同势偏倚；已有 CHGNet 方向一致是代理间支持，但非 DFT 证据。等预算 POST 在同一 MatterSim MaxF 上优于 RC-NFGD，这是应保留的负对照。

## 推荐表述

目前最准确的标题仍是“面向冻结材料扩散模型的神经力场反馈引导方法（RC-NFGD）”，可在方法描述中注明“后期 predictor 位置分量注入及安全回退”。“后期更合理”只能表述为历史诊断支持的**设计动机**，不能升格为新 timing ablation 已证明的贡献。也不宜改称“物理验证器引导的扩散后期一致性优化方法”而暗示已通过独立 DFT 真值验证。

## 第 4 章与摘要

- 可在第 4 章加入 A6 同 cohort 的 Pareto 表/图和局限性段落，但标为补充/探索性分析；尤其说明 POST 在 MatterSim MaxF 上的更强表现与统计上未确证的属性前沿差异。
- 不能把 timing 未运行的表加入结果章节。恢复原环境并完成 32 对新实验后，无论 early/middle/late 排名如何均应完整报告；只有结果支持时，才把“晚期优选”写成实验性结论。
- 摘要保持“降低代理力误差、CHGNet 方向一致、属性代价很小、未进行 DFT 验证”的已验证边界。不要写“全面优于后处理”“保持或提升属性精度”“真实稳定性已获证明”。

## 是否需要 DFT

如果论文主张限于冻结代理势下的推理增强，当前证据可支撑有边界的硕士论文讨论；若主张真实材料物理稳定性、跨势普适性或拟投稿的强物理结论，仍建议预注册独立小规模 DFT 验证。DFT 不在本次授权实验中，不能把缺失解释为已完成。

资料：`docs/rcnfgd_timing_audit.md`、`results/rcnfgd/timing_ablation/timing_ablation_report.md`、`results/rcnfgd/pareto_analysis/pareto_analysis_report.md`。
