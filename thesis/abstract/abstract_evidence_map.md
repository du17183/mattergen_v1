# Abstract Evidence Map

本表逐句记录摘要中的主要事实、数字和边界。摘要数字的唯一主来源为 `thesis/final_audit/abstract_source_sheet.md`；章节文件仅用于核对研究问题和总体边界。

| Sentence ID | Chinese claim | English claim | Source | Status | Boundary |
|---|---|---|---|---|---|
| ZH-01 / EN-01 | 逆向设计从目标性能或约束反推出候选晶体结构，生成模型面临属性控制和局部结构问题。 | Inverse design infers candidate crystals from targets; generative models face property-control and local-structure issues measured by surrogate physical metrics. | `abstract_source_sheet.md` Background; Chapter 1 §1.1 | PASS | 研究背景，不宣称解决全部问题。 |
| ZH-02 / EN-02 | MatterGen 是冻结基础模型，研究针对推理阶段且不改主干和三字段定义。 | MatterGen is the frozen foundation; the work concerns inference without changing the backbone or three-field definition. | `abstract_source_sheet.md` Background; Chapter 1; Chapter 6 §6.1 | PASS | MatterGen 不是本文提出的模型。 |
| ZH-03 / EN-03 | 创新点1保留 CFG=2.0 的 C0、共享前缀，展开两个候选并执行终点选择和精确回退。 | Innovation 1 preserves CFG=2.0 C0, a shared prefix, two candidates, terminal selection, and exact fallback. | `abstract_source_sheet.md` Innovation 1 方法版 | PASS | 不展开 V1/V2、阶段或字段细节。 |
| ZH-04 / EN-04 | C1-128 中 Fixed-K2 的 Property MAE 为 0.034796→0.026332，相对降低24.32%，通过代理护栏，预算约2.2×。 | On C1-128, Fixed-K2 gives 0.034796→0.026332, −24.32%, passes surrogate guardrails, and costs about 2.2×. | `abstract_source_sheet.md` Innovation 1 C1-128；`final_master_numbers.md` Innovation 1 | PASS | 仅限当前 cohort、任务和代理协议。 |
| ZH-05 / EN-05 | Linear-K2 未表现出相对 Fixed-K2 的稳定额外优势。 | Linear-K2 did not show a stable additional advantage over Fixed-K2. | `abstract_source_sheet.md` Innovation 1 100字版；Mandatory boundaries | PASS | 不否定未来其他分配器。 |
| ZH-06 / EN-06 | 创新点2依据后期阶段可靠性使用 MatterSim 力，经周期坐标映射反馈到 position predictor，并有界修正和异常回退。 | Innovation 2 uses late-stage reliability, MatterSim forces, periodic mapping, position-predictor feedback, bounded correction, and fallback. | `abstract_source_sheet.md` Innovation 2 方法版；Chapter 4; Chapter 6 §6.1 | PASS | 不宣称有界修正是已证实性能核心。 |
| ZH-07 / EN-07 | Formal256 中 MatterSim MaxF、Mean Force、RMSD 分别降低29.81%、29.72%、14.17%。 | On Formal256, MatterSim MaxF, mean force, and RMSD decrease by 29.81%, 29.72%, and 14.17%. | `abstract_source_sheet.md` Innovation 2 Formal256；`final_master_numbers.md` Innovation 2 | PASS | 代理结构指标，不等同完整结构质量或 DFT。 |
| ZH-08 / EN-08 | 未参与在线引导的 CHGNet 独立 MLIP 评估中，MaxF 均值降低14.36%。 | Independent CHGNet MLIP evaluation not used online shows a 14.36% mean MaxF reduction. | `abstract_source_sheet.md` Independent surrogate | PASS | 跨代理均值方向，不是独立物理或实验验证。 |
| ZH-09 / EN-09 | RC-NFGD Property MAE 0.009757→0.009868，约恶化1.14%。 | RC-NFGD Property MAE changes 0.009757→0.009868, an approximately 1.14% deterioration. | `abstract_source_sheet.md` Innovation 2 Formal256；`final_master_numbers.md` Innovation 2 | PASS | 不能改写为属性改善。 |
| ZH-10 / EN-10 | 两路线独立，未建立联合协同或优于所有后处理的结论。 | The two routes are independent; no validated joint synergy or universal post-processing superiority is established. | `abstract_source_sheet.md` Mandatory boundaries；Chapter 6 §6.2.3、§6.3 | PASS | `JOINT_SYNERGY=NOT_SUPPORTED`。 |
| ZH-11 / EN-11 | 物理一致性证据限于 MatterSim/CHGNet MLIP 代理，尚未完成 DFT 验证。 | Physical-consistency evidence is limited to MatterSim/CHGNet MLIP surrogates; DFT validation has not been performed. | `abstract_source_sheet.md` Independent surrogate；Mandatory boundaries；Chapter 6 §6.3 | PASS | `DFT_VERIFIED=false`。 |
