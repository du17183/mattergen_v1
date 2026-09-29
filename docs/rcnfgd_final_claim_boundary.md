# 创新点 2 最终贡献边界（截至环境恢复审计）

推荐标题：**面向冻结材料扩散模型的神经力场反馈引导方法（RC-NFGD）**。

已验证的贡献限于：保持 MatterGen 参数冻结，在 predictor 后期的位置分量注入有安全回退的 MatterSim 力反馈；原 Formal256 相对 C0 降低 MatterSim 预测的 MaxF/平均力，CHGNet 独立代理评估方向一致。属性误差有轻微代价，不能写“属性与物理同时稳定提升”。现有 A6 同 cohort Pareto/等预算实验中，POST 的 MatterSim MaxF 比 RC-NFGD 更低；这必须作为重要负对照保留。见 `results/rcnfgd/pareto_analysis/pareto_analysis_report.md`。

Timing Ablation **没有结果**：虽已重建可运行 Python/PyTorch 环境，但历史 seed 740000 的结构和指标复现门槛实测 FAIL，故按协议停止。详情见 `docs/rcnfgd_environment_rebuild_20260929.md`。因此“后期注入更合理”仍只是受历史诊断启发的设计选择，不能写成 Early/Middle/Late 消融已验证的发现。若将来有独立通过门槛的环境，仍应按预先冻结的 `t≤1.0/0.5/0.02` 三臂运行并保留全部结果；不同力调用预算必须如实披露。

论文中避免声称真实物理稳定性（未做 DFT）、全面优于 POST、普适最优注入时机或与 Fixed-K2 的联合增益。更强的“物理验证器引导扩散优化方法”标题目前可能暗示超出证据的物理真实性与比较优势，暂不推荐。若后续完成可信 Timing Ablation 和独立 DFT，再根据结果重新评估标题与贡献。
