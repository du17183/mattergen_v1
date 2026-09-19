# Chapter 6 Evidence Map

本表将第6章的总结性表述映射到已冻结的章节和发布包证据。Chapter 6 不产生新实验结果；所有结论均受 `thesis_release/CLAIMS_AND_LIMITATIONS.md` 和 `thesis/cross_chapter/master_numbers.md` 约束。

| Section | Conclusion | Source Chapter | Source Result | Status | Limitation |
|---|---|---|---|---|---|
| 6.1 | 本文围绕条件属性控制和局部结构代理一致性两个研究问题展开 | 第1、3、4、5章 | 研究问题、技术路线与统一结论 | SUPPORTED | 两条路线独立，不构成已验证联合模型 |
| 6.2.1 | Fixed-K2 在 C1-128 改善代理 Property MAE | 第3、5章 | 0.034796→0.026332，改善24.32%，绝对CI全正 | SUPPORTED | 2.2×生成预算；限定当前任务和代理协议 |
| 6.2.1 | Fixed-K2 通过冻结代理质量护栏 | 第3、5章 | E-hull、Stable、NUS改善，Validity保持100% | SUPPORTED_GUARDRAIL | Novel下降；不是物理安全保证 |
| 6.2.1 | 学习型 Linear-K2 未确认优于 Fixed-K2 | 第3、5章 | Phase B与C1均未达到优势门槛 | NOT_SUPPORTED | 只约束当前特征、样本和候选库 |
| 6.2.2 | RC-NFGD 在 Formal256 改善 MatterSim 代理力与RMSD | 第4、5章 | MaxF 29.81%、Mean Force 29.72%、RMSD 14.17% | SUPPORTED | MatterSim 同源评价风险；不等于DFT力改善 |
| 6.2.2 | CHGNet 对均值力指标给出同向改善 | 第4、5章 | MaxF 14.36%、Mean Force 9.59% | SUPPORTED_INDEPENDENT_SURROGATE | CHGNet仍是MLIP；尾部结果混合 |
| 6.2.2 | RC-NFGD 不支持目标属性改善主张 | 第4、5章 | Property MAE恶化1.14% | NOT_SUPPORTED_PROPERTY_IMPROVEMENT | 需保留属性—受力权衡 |
| 6.2.3 | 两项创新具有已验证联合协同 | 第5章 | A5未达到双保持门槛；联合状态失败 | NOT_SUPPORTED | 未直接验证Fixed-K2×RC-NFGD完整组合 |
| 6.3 | 当前证据未包含DFT、声子、长时MD或湿实验验证 | 第3、4、5章；发布包 | `DFT_VERIFIED=false` | FALSE | 代理结果不能替代真实物理验证 |
| 6.3 | 在线RC-NFGD优于等预算POST | 第4、5章 | POST在所测MatterSim终态力指标上更低 | NOT_SUPPORTED | 在线方法的结论是可行性，不是支配性 |
| 6.3 | 有界修正是性能核心 | 第4、5章 | unbounded消融未支持该命题 | NOT_SUPPORTED | 保留为保守步长与异常控制机制 |
| 6.4 | 首要后续工作为配对DFT单点力验证 | 第3、4、5章；发布包 | 当前DFT证据缺失 | FUTURE_WORK | 未预估未测量的计算成本或收益 |
| 6.4 | 候选分配、分支成本、跨任务和多物理反馈需要新协议验证 | 第3、4、5章 | Linear-K2、2.2×成本、迁移性和反馈边界 | FUTURE_WORK | 不把未来方向写成当前结果 |
