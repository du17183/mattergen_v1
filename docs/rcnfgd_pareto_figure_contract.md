# RC-NFGD Pareto 图件契约

- 核心结论：在同一 A6 配对 cohort 中，RC-NFGD 与 POST 在“属性误差—MatterSim MaxF”二维均值上呈不同折中；点估计前沿不等于统计显著优势。
- 结果问题：结构层的分布是什么？方法均值及其不确定性是否支持单一赢家？
- 图型：单面板定量散点图与单面板均值/置信区间前沿图，各一张；不用互不配对的 A1/A6 均值混画前沿。
- 输出：Python Matplotlib、矢量 PDF、目标约 89 mm 单栏宽；全部文本为可选取文字。
- 证据层级：A6 n=32 配对种子的 C0/F0/POST；结构散点为分布证据，均值前沿为描述性证据。
- 统计：横轴 Property MAE（磁密度代理误差），纵轴终态 MatterSim MaxF（eV/Å）；均值横/纵误差棒分别为同一方法 20,000 次 seed bootstrap 95% percentile CI。方法配对差 CI 仅在报告中解释；前沿以点估计定义，不能据此断言差异显著。
- 来源：`/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv` 中 `cohort=A6`，并独立核对原 A6 `per_structure_metrics.csv`。
- 图像完整性：未做平滑、插值、裁剪离群点或人工调整数据。色彩加不同形状编码；PDF 最终检查 glyph 与碰撞。
- 审稿风险：POST 明显降低同一 MatterSim 的 MaxF，RC-NFGD 不能称全面优于 POST。A6 与 Formal256 的属性误差方向不完全一致；无 DFT 验证。
