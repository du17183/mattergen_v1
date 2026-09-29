# 硕士论文最终正文重构报告

日期：2026-09-29。范围：仅修订学校模板预览版 LaTeX 正文、复用已有矢量图并编译新 PDF；未修改生成代码、种子、checkpoint、实验数据、统计产物或旧版 PDF。此报告不将新增实验重新定义为原 C1 的预注册实验。

## 修改文件与产物

- thesis/latex/ustb_2026_polished_preview/contents/abstract.tex：中英文摘要改为预算约束验证选择框架；Fixed-K2 明确为固定双候选实例；同一新 cohort 的独立 Best-of-2 不利点估计和区间跨零事实均予保留；RC-NFGD 的代理改善、Property MAE 代价及无 DFT 边界未删除。
- thesis/latex/ustb_2026_polished_preview/contents/chap1.tex：从冻结模型如何利用额外推理计算、独立采样与共享前缀如何比较、如何验证并回退三个问题引出两类推理增强；创新点1不再写成 CFG 最优调度；保留 Linear-K2 负结果。
- thesis/latex/ustb_2026_polished_preview/contents/chap3.tex：组织为 3.1 推理预算、3.2 单轨迹探索、3.3 验证选择框架、3.4 Fixed-K2 与 C1/Linear、3.5 新强基线及预算分析、3.6 局限。定义状态/动作/转移/验证器/预算/回退；原 C1 表、脉冲参数、SAFE-A 阈值与负结果保留；增加两个独立新 cohort 的表与图。
- thesis/latex/ustb_2026_polished_preview/contents/chap5.tex：更新证据层级、预算账本和 Claim–Evidence 映射；明确 Best-of-2 不支持 Fixed-K2 的相对优势，K=2 是成本折中；明确创新点1目标 0.1 与创新点2目标 0.2 不可直接混合；校正 CHGNet/MatterSim 在创新点1中的代理职责。
- thesis/latex/ustb_2026_polished_preview/contents/chap6.tex：总结为预算约束验证选择与后期物理反馈两类独立机制；保留 Best-of-2 竞争、Linear-K2 失败、RC-NFGD 属性代价、POST 竞争和无 DFT 限制；Adaptive Budget 只写入未来工作。
- thesis/latex/ustb_2026_polished_preview/ustb_mattergen_final_revised.pdf：新 jobname 编译产物，不覆盖此前 PDF。

## 图表与未完成的示意图

第三章已新增独立 Best-of-2 三臂预算/结果表、K=0--4 主要预算/质量表，并直接引用已经完成审计的矢量 PDF：figures/search_framework/method_comparison.pdf 与 figures/search_framework/compute_performance_curve.pdf。两张结果图的图注分别注明新 cohort 的 n=128、均值 bootstrap 95% 区间及与直接配对区间的差别；预算曲线注明横轴仅计 MatterGen score 调用及连线不是拟合的 scaling law。原图 1-1 仍作双路线技术总览，旧 C1 图表和 RC-NFGD 图表保留。

此前规划的“冻结 MatterGen → 参考轨迹 → 有限候选 → 验证器 → 选择/回退”单独框架图，以及“独立 Best-of-2 vs 共享前缀 Fixed-K2”的机制图，在现有 figures/search_framework/ 中没有对应成品。用户要求先检查且不要直接画，本轮没有用已有数值图冒充机制图，也没有生成未经人工审查的新图。若答辩稿需要这两张，应另行按现有图形合同绘制和审查；当前论文已经用算法块、状态公式、结果图与表格完整表达对应关系。

## 风险与事实核对

| 核对项 | 结果 |
|---|---|
| C1-128 历史主结果 | 保留 C0 0.034796 → Fixed-K2 0.026332、24.32% 和配对 95% CI [0.005817, 0.011391]；未与新 cohort 混合。 |
| Linear-K2 | Phase B 与 C1 的额外优势不获支持，保留在第三、五、六章。 |
| Best-of-2 | 新 cohort：C0 0.037623、独立 Best-of-2 0.028097、Fixed-K2 0.029506；Best-of-2−Fixed-K2 配对均值 −0.001410、CI [−0.005039, 0.002167]。明确 2.0×/2 终点与 2.2×/3 终点不等预算。 |
| Budget Scaling | 独立新 cohort：K0--K4 的 MAE 0.035141/0.031636/0.028454/0.028419/0.024669；K2 非最低误差点，嵌套 SAFE-A 的代理误差非增具算法性。 |
| verifier 偏差 | CHGNet 参与属性选择又定义主要 Property MAE；代理收益不能升格为独立 DFT 物性收益。 |
| 状态快照 | 运行时克隆 batch/RNG 并记录分支位置；旧 C1 artifact 未独立序列化条件值和数值 timestep 网格，需冻结配置辅助恢复。 |
| RC-NFGD | MatterSim MaxF/Mean Force/RMSD 改善保留，CHGNet 均值方向一致；Property MAE 0.009757 → 0.009868 的代价、POST 更强的所测终态受力、无 DFT、无最终 Fixed-K2×RC-NFGD 联合确认均保留。 |
| 论文定位 | 适合以边界清晰的优秀硕士论文为目标；当前不能宣称优于 Best-of-N、最优搜索宽度、普适物理稳定或联合协同。 |

## 编译与格式检查

使用项目内 TeX Live 2026/TinyTeX 的学校 ustbthesis 模板编译，修订版为 133 页。编译无致命错误、缺失图件或未定义交叉引用；PDF 文字可提取，新增结果在正文页可检索，抽查新增表格/图注所在页的文字边界未超出纸张。仍有模板及旧表格遗留的少量 overfull hbox（最大约 8.16 pt）和 underfull vbox；当前不能将其说成已经完成逐页视觉终审。封面个人信息占位符和模板字体问题沿用原状态，本轮正文重构没有擅自改动学校模板或封面字段。
