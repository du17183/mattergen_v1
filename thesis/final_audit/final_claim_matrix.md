# Final Claim–Evidence Matrix

| Claim | Evidence | Status | Allowed wording | Boundary |
|---|---|---|---|---|
| 合法分支预算内存在属性改善空间 | Branch-Compatible Oracle，测试 \(n=12\)，MAE 改善30.41% | SUPPORTED_MECHANISM | Oracle 显示可利用的分支空间 | 不是可部署 Oracle 选择器 |
| Fixed-K2 改善 C0 代理 Property MAE | C1-128，24.32%，CI [0.005817, 0.011391] | SUPPORTED | 当前冻结任务与代理协议下改善 | 约2.2×预算，不保证普适最优 |
| Fixed-K2 通过冻结质量护栏 | E-hull、Stable、NUS 改善，Validity保持100% | SUPPORTED_GUARDRAIL | 注册代理护栏未被破坏 | Novel 单项下降，无物理安全保证 |
| Linear-K2 优于 Fixed-K2 | Phase B 与 C1 均未达到优势门槛 | NOT_SUPPORTED | 当前学习分配未获额外支持 | 不否定未来更强分配器 |
| 在线 Adaptive CFG 稳定有效 | V1混合，V2/V4/V5/Field失败，Stage混合 | NOT_SUPPORTED | 探索结果推动参考保留框架 | 不能写成最终正结果 |
| RC-NFGD 降低 MatterSim MaxF | Formal256，降低29.81%，CI全正 | SUPPORTED | 冻结任务下代理 MaxF 降低 | MatterSim 同源评价 |
| RC-NFGD 降低 Mean Force 与 RMSD | Formal256，29.72%与14.17% | SUPPORTED | 配对代理指标改善 | 不等同全部结构质量改善 |
| CHGNet 支持均值力方向 | MaxF 14.36%，Mean Force 9.59% | SUPPORTED_INDEPENDENT_SURROGATE | 独立 MLIP 代理均值方向一致 | 不是 DFT；尾部混合 |
| RC-NFGD 改善 Property MAE | Formal256 误差恶化1.14% | NOT_SUPPORTED_PROPERTY_IMPROVEMENT | 报告轻微属性代价 | 不能被 Stable/Validity 掩盖 |
| RC-NFGD 优于等预算 POST | POST 的所测终态 MatterSim 力更低 | NOT_SUPPORTED | 在线反馈可行但不支配 POST | 当前势、预算和消融范围 |
| 有界修正是性能核心 | unbounded 消融更强 | NOT_SUPPORTED | 保守步长与异常控制设计 | 非已证实性能来源 |
| 两项创新存在已验证协同 | A5 未达到双保持门槛 | NOT_SUPPORTED | 两项贡献独立报告 | 未直接验证 Fixed-K2×RC-NFGD |
| DFT 验证已完成 | 发布包无 DFT 计算 | NOT_VERIFIED | 尚未开展 DFT 配对验证 | DFT_VERIFIED=false |
