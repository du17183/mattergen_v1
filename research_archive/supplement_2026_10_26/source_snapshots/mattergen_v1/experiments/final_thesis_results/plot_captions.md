# 图题与使用边界

以下图均由 make_plots.py 从正式数据导出；坐标轴以图内英文单位为准。对应 CSV 保留完整精度，图上数值仅作显示舍入。

## overall_method_framework.png

两条最终方法主线。Adaptive CFG 与 RC-NFGD 分别从冻结 MatterGen 分出，不要求组成最终串联系统。该图是算法关系，不是实验流程完整性的证明。

## adaptive_cfg_pipeline.png

多字段残差驱动的在线自适应 CFG：先分别计算 atomic/pos/cell 残差 RMS，再取均值，随后更新对应 predictor/corrector 的 EMA，产生应用于所有字段的共享 scale。不得改写为 field-specific CFG。

## rc_nfgd_pipeline.png

阶段可靠性诊断支持的后期神经力引导：late predictor→完整 clean estimate→冻结 MatterSim→有界 Cartesian 位移→fractional/score 映射→继续反向扩散。不是后处理，也不是自适应闭环信赖域控制器。

## innovation1_formal_results.png

创新点一 Formal256 的配对基线与正式结果，seed 20000–20255，target=0.1。上排为均值或比例，下排为原始 A0-C0 配对差及原始95% CI。三个区间均跨零；保留历史方向性门槛确认，不宣称显著优越。

## formal256_maxf_paired.png

RC-NFGD Formal256 的逐 seed 预松弛最大原子力散点；seed 740000–740255，target=0.2，单位 eV/Å。全部256对均展示，虚线表示相同受力；不删除大力尾部样本。

## formal256_maxf_distribution.png

相同 Formal256 原结构 MatterSim MaxF 的经验累计分布，全部观测保留。横坐标采用带线性小值区的 symlog 标度，纵坐标为累计比例。不得把视觉距离当成统计显著性。

## formal256_tail_metrics.png

相同 Formal256 的均值、中位数、P90、P95与最大值。五个子图各自采用零起点纵轴，尺度不同且明确标注；反映不同分布位置，不是五个独立主终点。

## independent_mlip_validation.png

在同一512个原结构上的 MatterSim 与 CHGNet 平均 MaxF。CHGNet 未参与 F0 的力引导，但参与项目属性护栏；不是 DFT 真值，也不保证训练数据独立。其均值改善不能推广为所有分位数改善，完整尾部统计见 distribution_statistics.csv。

## force_direction_ablation.png

32对种子的五组方向消融：G0无修正，G1真实力方向，G2受约束随机方向，G3重排方向，G4反向参考力。左右为 MatterSim/CHGNet 均值，完整CI见统计表。控制组重放G1参考幅度，G4并非自身轨迹实时反梯度。

## bounded_correction_ablation.png

32对种子的三组对照：T0基线，T1冻结F0，T2去除范数相关饱和和硬cap但保留几何安全。T2表现更强，不能用本图证明bound必要或更优；bounded只定位为幅度/安全设计。

## online_vs_post.png

32对同seed C0/F0/POST，F0与POST各实际20次MatterSim调用，且POST由该C0原文件派生。MatterSim MaxF上POST更强；CHGNet直接F0/POST比较CI跨零。均值柱图不构成跨势显著性优越的证明。

## interaction_analysis.png

64对C0/A0/B0/AB交互结果：本图中B0表示RC-NFGD，AB为组合。AB仍有部分指标改善，但未达到冻结双收益保留标准；不救援调参，不强制串联两创新。

## innovation1_minimal_ablation.png（补充）

8个全新配对seed、6个组件对照，目标0.1；本图只在全部生成与评价完成后输出。仅用于补充组件诊断，不能替换或扩充原Formal256；完整表保留所有不利结果。
