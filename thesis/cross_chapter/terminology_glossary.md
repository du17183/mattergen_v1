# 第3—5章术语表

| 规范术语 | 英文/符号 | 首次定义与使用规则 | 避免写法 |
|---|---|---|---|
| 参考轨迹保留的预算约束多分支条件引导与安全回退方法 | Reference-Preserved Budget-Constrained Multi-Branch Guidance with Safety-Constrained Fallback | 创新点1全称；不另造缩写 | 仅写“Adaptive CFG最终方法” |
| 参考轨迹 | reference trajectory | 与候选分支同起点、可精确恢复的 C0 轨迹 | baseline trajectory（未定义） |
| 参考轨迹保留 | reference-preserved | 同时保存完整采样状态与 RNG 状态 | 仅恢复 CFG 数值 |
| 共享前缀 | shared prefix | 分支点前各后缀共用的 C0 采样区间 | common path、共享轨迹（未定义） |
| 候选分支 | candidate branch | 从同一保存状态和 RNG 快照运行的冻结后缀 | sample、alternative（未定义） |
| 预算约束 | budget-constrained | 候选数和 score 调用量预先冻结 | 无限 Best-of-N |
| 固定双候选分支策略 | Fixed-K2 | 创新点1最终确认策略；首次出现中英文并列 | Fixed K2、固定K2 |
| 学习型双候选分支策略 | Linear-K2 | 冻结线性候选分配器，结论为 NOT_SUPPORTED over Fixed-K2 | 已验证学习型分配器 |
| MatterGen参考基线 | C0 | 固定 CFG=2.0；涉及分支时强调精确参考轨迹 | 默认模型、原始结果（未定义） |
| 安全回退 | safety-constrained fallback | 候选不满足冻结终点约束时返回精确 C0；是代理约束，不是物理安全保证 | 保证安全、物理安全回退 |
| 参考回退 | reference fallback | 安全回退的具体动作，即返回精确 C0 | 保底、退回 baseline |
| 分支兼容 Oracle | Branch-Compatible Oracle | 使用事后终点信息的探索性机制上界 | 可部署 Oracle |
| 基于阶段可靠性校准的后期神经力场引导扩散方法 | Reliability-Calibrated Late-stage Neural Force-Guided Diffusion (RC-NFGD) | 创新点2全称；后文用 RC-NFGD | MatterSim优化器、真实力引导 |
| 预测干净结构 | predicted clean structure | 对当前含噪状态恢复的结构级估计；需要区分原子类型、位置与晶胞解析式 | clean-x0、clean estimate、干净晶体（未定义） |
| 含噪状态 | noisy state \(x_t\) | 当前扩散状态；与预测干净结构区分 | raw \(x_t\) |
| 神经势模型 | machine-learning interatomic potential (MLIP) | MatterSim/CHGNet 的模型类别 | DFT模型、真实物理模型 |
| 神经力场 | neural force field | 神经势模型给出的能量梯度力反馈；首次使用时与 MLIP 关系一并说明 | 真实原子力、DFT力 |
| 同源代理评价 | in-loop/same-model surrogate evaluation | MatterSim 既提供 RC-NFGD 力反馈又评价主要力指标 | 独立物理验证 |
| 独立代理评估器 | independent surrogate evaluator | CHGNet 不参与 RC-NFGD 在线力反馈；独立性只相对于引导信号 | 独立 DFT、完全独立验证 |
| 生成后修正 | post-generation correction (POST) | 在生成终态上使用等预算神经势调用的对照 | RC-NFGD后处理（混同在线方法） |
| 先导实验 | P0 | 决定是否进入下一阶段，不承担最终确认 | 正式确认实验 |
| 正式确认实验 | Formal32/Formal256（按协议角色） | Formal256承担创新点2最终主结论；Formal32为独立复现 | 探索调参集 |
| 属性误差 | Property MAE | 冻结代理属性的平均绝对误差，越低越好 | 真实材料属性误差 |
| 代理局部结构一致性 | surrogate local-structure consistency | 对 MaxF、Mean Force 和松弛 RMSD 的谨慎统称 | 已验证物理稳定性 |
| 匹配种子设计 | matched-seed design | 比较方法由同一随机种子启动 | 独立同分布样本（若实际配对） |
| 配对自助法 | paired bootstrap | 核心连续指标使用20 000次确定性重采样 | 非配对 bootstrap |

证据状态固定写法：`SUPPORTED`、`SUPPORTED_MECHANISM`、`SUPPORTED_GUARDRAIL`、`SUPPORTED_INDEPENDENT_SURROGATE`、`DIRECTIONAL_ONLY`、`MIXED`、`NOT_SUPPORTED`、`FAIL`、`FALSE`。状态英文大写保留，中文正文解释其含义。
