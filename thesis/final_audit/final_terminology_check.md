# Final Terminology Check

| Term | Required form | Check |
|---|---|---|
| Innovation 1 | 参考轨迹保留的预算约束多分支条件引导与安全回退方法；Reference-Preserved Budget-Constrained Multi-Branch Guidance with Safety-Constrained Fallback | PASS |
| Innovation 2 | 基于阶段可靠性校准的后期神经力场引导扩散方法；Reliability-Calibrated Late-stage Neural Force-Guided Diffusion (RC-NFGD) | PASS |
| Baseline | MatterGen 参考基线 C0，固定 CFG=2.0 | PASS |
| Candidate method | 固定双候选分支策略 Fixed-K2 | PASS |
| Negative allocator | 学习型 Linear-K2，不写成已验证优势 | PASS |
| Historical exploration | Adaptive CFG 仅用于探索、机制分析和失败链 | PASS |
| Branch state | 参考轨迹、参考轨迹保留、共享前缀、候选分支、预算约束 | PASS |
| Fallback distinction | 安全回退是整体策略概念；参考回退是返回 exact C0 的具体动作 | PASS |
| Structure estimate | 预测干净结构；与含噪状态区分 | PASS |
| MLIP role | MatterSim 为在线引导与同源代理评价；CHGNet 为未参与在线反馈的独立代理评估器 | PASS |
| Physical boundary | 代理局部结构一致性，不写成真实物理稳定性或 DFT 验证 | PASS |
| Method casing | MatterGen、MatterSim、CHGNet、GemNetT、Fixed-K2、RC-NFGD、Formal256、C1-128 | PASS |

扫描未发现 Mattergen、RC-Nfgd、Fixed K2 或 Fixed K-2 等正文漂移写法。Adaptive CFG 的出现均保留历史探索或边界语义。
