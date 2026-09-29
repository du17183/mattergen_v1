# RC-NFGD 创新点 2 定位建议（当前证据版）

## 推荐标题

**面向冻结材料扩散模型的神经力场反馈引导方法**（Neural-Force-Feedback Guidance for Frozen Material Diffusion Models）。可在副标题或方法小节明确“后期 predictor 位置分量注入及安全回退”。暂不选“物理验证器引导扩散优化方法”作为总标题，以免把 MatterSim 代理评价误读为独立物理真值验证。

## 完成的证据与边界

原 RC-NFGD Formal256 相对 C0 降低 MatterSim MaxF，CHGNet 独立代理评估方向一致；但 Property MAE 从 0.009757 增至 0.009868，未做 DFT。已有 A6 等预算对照中 POST 在同一 MatterSim MaxF 上明显优于 RC-NFGD；两者 Property MAE 差的配对 CI 跨零。`results/rcnfgd/pareto_analysis/pareto_analysis_report.md` 给出同 cohort 前沿与 20,000 次配对 bootstrap 结果。这些证据支持“在冻结生成器内接入力反馈、改善代理力指标”，不支持“全面优于后处理”“属性和物理指标同时稳定改善”或“真实物理稳定性已验证”。

新的 Early/Middle/Late Timing Ablation **未运行**：重建环境虽可运行，但历史 seed 740000 的结构及指标实测门槛 FAIL，见 `docs/rcnfgd_environment_rebuild_20260929.md`。因此不能新增“时机选择经消融验证”的贡献，也不改摘要/第 4 章为已验证结论。若未来通过独立复现门槛并完成新实验，无论结果属于 Late 最优、类似还是 Early/Middle 更优，都应按 `docs/rcnfgd_timing_conclusion.md` 的事先规则更新，而非按论文期望挑选。

## 论文建议

第 4 章可补充已有 Pareto 图和 POST 负对照，明确不同 cohort 不混合、前沿仅为点估计、MatterSim 自洽偏倚及 CHGNet 非 DFT。当前摘要贡献保持“提出神经力反馈推理机制，并在代理力指标上取得改善，同时存在属性代价与验证边界”。不声称与创新点 1 的联合收益。若想把结论提升到真实材料物理层面，需要预注册并实施独立 DFT 验证；本轮没有授权或完成该项。
