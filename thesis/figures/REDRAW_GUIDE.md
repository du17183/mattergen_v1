# 论文图 1–12 重绘说明

本文件说明每张图为什么要画、使用什么数据、各视觉元素代表什么，以及后期如何在
Matplotlib、Adobe Illustrator、Inkscape、PowerPoint 或其他工具中重新绘制。当前
`generated/` 中的图是**可复现内容草稿**，用于固定数据与叙事，不应被视为最终美术
版本。重绘可以改变字体、间距、线宽和构图，但不得改变数值、证据资格或结论强度。

## 0. 所有图通用的重绘规则

### 0.1 文件优先级

1. 科学事实以 `thesis_archive/` 的冻结数据和报告为准。
2. 逐图绘图数据位于 `thesis/figures/source_data/`。
3. 当前构图逻辑位于 `thesis/figures/source/python/` 和
   `thesis/scripts/generate_*_figures.py`。
4. `generated/svg/` 最适合在 Illustrator/Inkscape 中继续编辑；PDF 用于论文插图，
   PNG 只用于预览。
5. 图注位于 `thesis/figures/captions/zh/` 和 `captions/en/`。图注中的样本量、种子、
   统计量和限制条件不得因重绘而删除。

### 0.2 建议的统一视觉系统

- 双栏图宽度以 7.10 inch 为基准，单栏图以 3.45 inch 为基准；优先矢量 PDF/SVG。
- 正文字号 8–9 pt，坐标轴与图例 7–8 pt，面板标记 A/B 为 9–10 pt。
- 使用色盲安全配色；同一方法跨图保持同色：
  `C0=灰/蓝`、`A0=蓝`、`E3-A=橙`、`E3-G=绿`、`伤害/无效证据=朱红`。
- 颜色不能是唯一编码。Gate-off、无效证据、伤害样本还应分别使用点型、线型或文字。
- 统计图必须保留零效应参考线；不得截断坐标轴以夸大效果。
- 每个轴都标单位。概率/比例统一使用 `%`，比例差统一写 `pp`，力使用
  `eV Å⁻¹`，位移使用 `Å`，E-hull 使用 `eV atom⁻¹`。
- MatterSim 结果必须保留“代理势评价、无 DFT 验证”的图注声明。
- 不新增从汇总均值反推的逐 seed 点，不把两个 64-seed cohort 合并成一个预注册
  128-seed 结果，不把 Mixed 256 写成独立验证。

### 0.3 重绘后的验收清单

- [ ] 缩放到论文实际版芯后，所有文字仍可读。
- [ ] 图例不遮挡点、置信区间或零参考线。
- [ ] 方法、seed 范围、样本量、方向定义和单位与源数据一致。
- [ ] 图中“改善”的正负方向与纵/横轴标签一致。
- [ ] 图注包含证据资格和 MatterSim/DFT 限制。
- [ ] 黑白打印时仍能通过线型、点型和标签区分类别。
- [ ] 导出 PDF/SVG 时文字未栅格化；导出 PNG 时至少 600 dpi。

---

## Figure 1：完整双创新方法架构

**要回答的问题**：两项创新分别位于哪里，数据从目标条件到最终代理评价如何流动？

**推荐图型**：从左到右的两阶段流程图。创新点一用一个虚线大框包围采样过程，创新点二
用另一个虚线大框包围生成后精修；MatterSim 必须画在框外并标成“evaluation only”。

**数据与语义来源**：

- 绘图节点：`source_data/fig01_full_method_architecture.csv`
- Adaptive CFG 配置：`thesis_archive/configs/adaptive_cfg_final.yaml`
- E3-PCR 配置：`thesis_archive/configs/e3_pcr_final.yaml`
- 当前语义源：`source/graphviz/fig01_full_method_architecture.dot`

**建议布局**：

1. 第一行依次放置 `dft_mag_density target → MatterGen Predictor/Corrector →
   Adaptive CFG → generated crystal → Learned Gate`。
2. Gate 后分成上下两路：`Gate-on → E3-PCR` 与 `Gate-off → exact fallback`。
3. 两路重新汇合为 `final crystal`，再用单独箭头连到 `MatterSim-5M surrogate`。
4. 在采样和后处理框上分别写“创新点一”和“创新点二”，不要把 MatterSim 画成训练模块。

**必须保留的标注**：

- Predictor/Corrector 完整保留。
- E3-PCR 只改位置；物种与晶胞不变。
- MatterSim-5M 仅用于代理评价；无 DFT 验证。

**不建议的画法**：不要把所有节点画成同等重要的矩形；可将条件、模型、决策、输出和评价
分别使用输入形、处理框、菱形决策、晶体图标和虚线评价边界，提高层级感。

**单图复现命令**：

```bash
python thesis/figures/source/python/fig01_full_method_architecture.py
```

---

## Figure 2：Adaptive CFG 机制

**要回答的问题**：创新点一如何利用 cell、position、atom 三字段残差在线调整 CFG，同时
不跳过 Predictor/Corrector？

**推荐图型**：双分支汇合的算法流程图，必要时在流程下方增加一行简化公式，不使用结果柱状图。

**数据与语义来源**：

- 阶段顺序：`source_data/fig02_adaptive_cfg_mechanism.csv`
- 冻结参数：`thesis_archive/configs/adaptive_cfg_final.yaml`
- 当前语义源：`source/graphviz/fig02_adaptive_cfg_mechanism.dot`

**建议布局**：

1. 左侧上下放 conditional 与 unconditional 两个 score 分支。
2. 两者汇合为三个并列的小通道：cell、position、atom residual。
3. 三通道依次经过 EMA、residual-driven scale update、`[0, 5]` clamp。
4. 在 CFG fusion 后指向完整 Predictor + Corrector，并用醒目脚注写
   “No Predictor skip / No Corrector skip”。
5. 在图边角列出冻结参数：
   `base=2.0, α=0.50, EMA=0.95, ε=1e−6, range=[0,5]`。

**视觉编码**：

- conditional 用实色，unconditional 用浅色或空心框。
- 三字段用三个小标签，不需要三套颜色；重点是“分别计算、共同更新”。
- EMA 和尺度更新属于状态更新，可用相同颜色；clamp 用边界/限幅图标。

**不能声称**：本图不是 Corrector Gating，也不表示减少物理 forward。

```bash
python thesis/figures/source/python/fig02_adaptive_cfg_mechanism.py
```

---

## Figure 3：Learned-Gated E3-PCR 机制

**要回答的问题**：轻量 Gate 如何决定是否执行受约束的等变位置精修，并在不安全时精确回退？

**推荐图型**：上半部分为 Gate 决策，下半部分为五步安全精修链，右侧汇合输出。

**数据与语义来源**：

- 关键参数：`source_data/fig03_e3pcr_mechanism.csv`
- 完整冻结配置：`thesis_archive/configs/e3_pcr_final.yaml`
- 当前语义源：`source/graphviz/fig03_e3pcr_mechanism.dot`

**建议布局**：

1. `generated crystal → 14-D risk features → 129-parameter Gate → confidence ≥ 0.5?`。
2. 决策节点必须画成菱形或明显分叉，不要继续使用普通处理框。
3. Gate-on 路径依次画：
   `5-step equivariant refinement → per-step radius 0.02 Å →
   cumulative trust region 0.10 Å → max 3 backtracks → safety checks`。
4. Gate-off 和任何拒绝路径都连到 `exact fallback`。
5. 接受路径和 fallback 路径汇合到最终结构。

**必须保留的约束**：

- position-only；
- atomic species unchanged；
- cell unchanged；
- threshold 0.5；
- refinement steps=5；
- `position_eta=0.01`；
- 拒绝时恢复原始结构，而不是输出部分修改结果。

**改进当前视觉的建议**：将“风险特征”和“安全约束”分成输入、决策、执行、验收四个泳道；
用回环箭头表示 backtracking，避免一长串等大小方框。

```bash
python thesis/figures/source/python/fig03_e3pcr_mechanism.py
```

---

## Figure 4：实验与证据血缘

**要回答的问题**：哪些结果是正式、独立复现、补充诊断或无独立结论资格？

**推荐图型**：以 C0/A0 为根节点的证据树，或者按时间排列的分层时间线。证据资格应比具体
数值更突出。

**数据来源**：

- `source_data/fig04_experiment_lineage.csv`
- `thesis_archive/EXPERIMENT_LINEAGE.md`
- `thesis_archive/configs/evaluation_final.yaml`
- 当前语义源：`source/graphviz/fig04_experiment_lineage.dot`

**节点必须包含**：

- Adaptive CFG formal256：20000–20255，Formal；
- E3-PCR formal256：40000–40255，Formal independent；
- Compatibility cohort 1：41000–41063，Independent replication；
- Independent cohort 2：50000–50063，Independent replication；
- Leakage diagnostic：20000–20255，Diagnostic only；
- Mixed 256：INVALID for independent claims。

**视觉编码**：

- 实线=正式，虚线=独立复现，点线=诊断/无效；
- 每个节点直接写证据类别，不能只靠颜色；
- Mixed 256 使用交叉阴影或红色警示框，并写清 `INVALID`，但仍保留其血缘关系。

**改进当前视觉的建议**：如果横向空间不足，改成四列：
`方法基线 → 实验 → seed/n → 证据资格`，每一行是一条完整可追溯记录。

```bash
python thesis/figures/source/python/fig04_experiment_lineage.py
```

---

## Figure 5：Adaptive CFG 正式结果

**要回答的问题**：Adaptive CFG 相对原始 MatterGen 的 E-hull、Stable 和 NUS 是否呈正向趋势，
以及不确定性有多大？

**图型与面板**：

- **A 面板**：256 个逐 seed 配对 `ΔE_hull = A0 − C0` 散点/雨云图；黑色菱形为均值，
  误差线为 paired bootstrap 95% CI；水平零线表示无变化。
- **B 面板**：Stable 和 NUS 的配对率差，单位为百分点；条形/点估计附 bootstrap 95% CI。

**数据来源**：

- 重绘 CSV：`source_data/fig05_adaptive_cfg_results.csv`
- 原始逐 seed：`thesis_archive/data/innovation1/per_seed_metrics.csv`
- 正式统计：`thesis_archive/reports/innovation1/formal_final_report.json`

**字段映射**：

- `ehull_difference_ev_atom = A0 − C0`，负值更好；
- `stable_difference = A0_stable − C0_stable`，正值更好；
- `nus_difference = A0_nus − C0_nus`，正值更好。

**必须标注**：

- n=256，seeds 20000–20255；
- E-hull 均值变化 −0.003435 eV/atom；
- Stable +5.859 pp，NUS +3.516 pp；
- 配对 bootstrap CI 跨零，Holm 校正后不显著。

**推荐重绘改进**：B 面板用“点估计 + CI”替代实心柱，减少小样本效果被面积放大的感觉；
A 面板可用半小提琴 + 原始点 + 均值 CI，但必须展示零线。

```bash
python thesis/figures/source/python/fig05_adaptive_cfg_results.py
```

---

## Figure 6：E3-PCR 三臂正式比较

**要回答的问题**：C0、Always-on E3-A、Learned-gated E3-G 的力分布如何不同，E3-G 的逐 seed
改善是否稳定？

**图型与面板**：

- **A 面板**：三臂分布，y 轴为 log scale 的预松弛最大力；箱线/小提琴表示分布，散点显示
  原始 seed，黑色菱形显示均值。
- **B 面板**：按 `ΔF = E3-G − C0` 从小到大排序的配对效应。负值为改善、正值为伤害；
  零线、均值线和均值 bootstrap 95% CI 带必须同时显示。

**数据来源**：

- `source_data/fig06_e3pcr_force_formal256.csv`
- `thesis_archive/data/innovation2/per_seed_metrics.csv`
- `thesis_archive/reports/innovation2/formal_paired_statistics.csv`

**必须标注**：

- n=256，seeds 40000–40255；
- C0 平均 0.342964，E3-G 平均 0.263107 eV/Å；
- 相对变化 −23.28%；
- 95% CI [−0.144966, −0.032453]；
- Holm-adjusted p=4.19×10⁻¹⁰；
- raw-difference Win/Tie/Loss=163/0/93。

**推荐重绘改进**：A 面板用 raincloud 或 violin+box，避免大量重叠散点；B 面板把改善和伤害
分别用绿色圆点、朱红三角标记，并在图外侧放统计摘要，避免文字挡住数据。

```bash
python thesis/figures/source/python/fig06_e3pcr_force_formal256.py
```

---

## Figure 7：Learned Gate 安全机制消融

**要回答的问题**：Learned Gate 是否通过减少干预覆盖来降低伤害，并付出了多少收益和位移代价？

**图型与面板**：

- **A 面板**：Always-on 与 Learned-gated 的 refinement rate、harm rate、
  low-force harm rate、gain retention 分组横向点图或柱图，单位 `%`。
- **B 面板**：两方法的 mean displacement 和 P95 displacement，单位 `Å`。

**数据来源**：

- `source_data/fig07_gate_safety_ablation.csv`
- `thesis_archive/reports/innovation2/final_summary.json`

**重要限制**：这里的位移是正式汇总值；归档中没有 per-seed mean-displacement，因此不能
给位移条形图伪造逐 seed 误差线或显著性。

**必须标注**：

- n=256；
- refinement 100% → 66.406%；
- harm 25.391% → 18.359%；
- low-force harm 29.688% → 17.969%；
- harm McNemar p=0.000534；
- 保留 80.657% 平均降力收益；
- Gate 不保证每个结构都改善，Always-on 的平均降力更大。

**推荐重绘改进**：A 面板优先使用 dumbbell chart，让“下降多少”比柱面积更清楚；将
gain retention 单独置于右侧或用不同小面板，因为它与 harm 指标的好坏方向相反。

```bash
python thesis/figures/source/python/fig07_gate_safety_ablation.py
```

---

## Figure 8：Gate confidence 与真实降力

**要回答的问题**：Gate 的 confidence 是否与实际最大力改善具有单调关联？

**推荐图型**：二维散点图。

**数据来源**：

- `source_data/fig08_gate_confidence_force_gain.csv`
- `thesis_archive/data/innovation2/per_seed_metrics.csv`

**字段映射**：

- x=`gate_confidence`；
- y=`force_gain = C0_max_force − E3G_max_force`，正值更好；
- `gate_on=True` 用空心圆，`gate_on=False` 用灰色叉号；
- x=0.5 为 Gate 阈值，y=0 为无改善。

**必须标注**：n=256，Spearman ρ=0.375、p=5.44×10⁻¹⁰；拟合线只能写
“descriptive trend”，不能写成校准曲线或因果关系。

**推荐重绘改进**：点较密时使用透明度、hexbin 背景或边缘 rug；不要用高阶多项式拟合。
可在四个象限上轻量标注“gate-on improvement / gate-on harm / fallback”帮助读者理解。

```bash
python thesis/figures/source/python/fig08_gate_confidence_force_gain.py
```

---

## Figure 9：两次独立组合验证 forest plot

**要回答的问题**：A0+E3-G 在两个独立 64-seed cohort 中是否都出现同方向的平均降力？

**推荐图型**：两行 forest plot，不合并总体效应。

**数据来源**：

- `source_data/fig09_combination_replication_forest.csv`
- `thesis_archive/reports/compatibility/paired_statistics.csv`
- `thesis_archive/reports/replication/paired_statistics.csv`

**字段映射**：

- 点=`mean_difference = A0+E3-G − A0`；
- 横线=`bootstrap 95% CI`；
- 竖线 x=0 表示无效应；负值为改善；
- 每行右侧写 seed 范围、n、相对变化和 p 值。

**必须标注**：

- Cohort 1：−27.10%，CI [−0.092341, −0.029754]，p=7.74×10⁻⁵；
- Cohort 2：−19.02%，CI [−0.102213, −0.010696]，p=0.000587；
- 两组独立并列，禁止画 pooled diamond 或合并 128-seed 总效应。

**推荐重绘改进**：使用典型 meta-analysis 排版，左列 cohort，中央 forest，右列 effect/CI；
两行使用同色，避免暗示它们是两种不同方法。

```bash
python thesis/figures/source/python/fig09_combination_replication_forest.py
```

---

## Figure 10：第二独立 cohort 的逐 seed 配对图

**要回答的问题**：在最新独立 64-seed cohort 中，每个 seed 从 A0 到 A0+E3-G 的最大力如何变化？

**推荐图型**：paired slopegraph；若高值压缩多数样本，可使用 log y 轴或在保持全量数据的前提下
增加局部 inset。

**数据来源**：

- `source_data/fig10_independent64_pairplot.csv`
- `thesis_archive/data/compatibility_2/per_seed_metrics.csv`

**视觉编码**：

- 左列 A0，右列 A0+E3-G；
- gate-on 且改善为绿色实线，gate-on 且伤害为朱红实线；
- gate-off exact fallback 为灰色点线/叉号；
- 线连接的是同一个 seed，不能打乱配对。

**必须标注**：

- seeds 50000–50063，n=64；
- algorithmic Win/Tie/Loss=35/18/11；
- 平均最大力相对下降 19.02%；
- Gate-off 样本按算法语义为 exact tie，即使浮点导出存在极小差异。

**推荐重绘改进**：可按 A0 初始力排序后使用 horizontal paired dot plot，通常比 64 条交叉斜线
更清晰；若改图型，仍必须逐 seed 展示并保留 Gate-off 标记。

```bash
python thesis/figures/source/python/fig10_independent64_pairplot.py
```

---

## Figure 11：训练重叠泄漏诊断

**要回答的问题**：训练重叠是否主要夸大平均效果，还是主要夸大 Gate 的安全性？

**图型与面板**：

- **A 面板**：Training-overlap 与 held-out 的逐 seed force gain 分布，显示原始点和均值区间。
- **B 面板**：两组 harm rate，直接标注计数和比例。

**数据来源**：

- `source_data/fig11_leakage_diagnostic.csv`
- `thesis_archive/data/leakage_diagnostic/per_seed_metrics.csv`
- `thesis_archive/reports/leakage_diagnostic/final_summary.json`

**必须标注**：

- overlap n=64、held-out n=192；
- harm：0/64 vs 31/192=16.15%；
- one-sided Fisher p=6.87×10⁻⁵；
- Mixed 256 仅用于诊断，不能作为独立验证；
- 平均降力未显示清晰夸大，但安全性明显被高估。

**推荐重绘改进**：B 面板用“计数/总数 + Wilson CI”的点区间图替代柱图；图标题或顶部横幅
直接写 `Diagnostic only`，防止截图脱离上下文后被误用。

```bash
python thesis/figures/source/python/fig11_leakage_diagnostic.py
```

---

## Figure 12：代表性 No-Go 路线

**要回答的问题**：项目尝试过哪些路线，每条路线的目标、观测收益和停止证据是什么？

**推荐图型**：分组路线矩阵或分层时间线。当前生成版本是表格式草稿，后期不建议继续把
13 行长文字直接塞进一张小图。

**数据来源**：

- `source_data/fig12_negative_routes_summary.csv`
- `thesis_archive/EXPERIMENT_LINEAGE.md`

**推荐分组**：

1. 采样/计算削减：Residual Reuse、Corrector Gating、Budget-aware Gating；
2. 表征/教师监督：FN-PRA、CrystalREPA、CG-TDR；
3. 训练自由物理引导：RP-QTFG；
4. 后处理/排序：Q1、Q2、Q4、Q5、Q6；
5. GPU 执行优化：GPU acceleration routes。

**每一行必须包含**：

- route；
- goal；
- 实测主要收益（如果有）；
- observed stopping evidence；
- status=`No-Go`；
- 可追溯来源。

**不能做的处理**：不构造综合分数，不按主观“失败程度”排名，不从汇总结果虚构逐 seed
分布。若版面拥挤，正文只画 5 个路线类别，13 条完整记录放表 9 或附录。

```bash
python thesis/figures/source/python/fig12_negative_routes_summary.py
```

---

## 13. 后期重绘的推荐工作流

1. 先从本文件确定图的科学问题和禁止改动项。
2. 打开对应 `source_data/*.csv`，核对列、单位和正负方向。
3. 对统计图优先修改 Python；对机制图可从 Graphviz DOT 或 SVG 继续设计。
4. 导出临时 PDF，放入论文模板按最终版芯检查，而不是只在大屏幕查看。
5. 与中英文图注逐项核对 n、seed、CI、p 值和证据资格。
6. 运行：

```bash
python thesis/scripts/validate_outputs.py
git diff --check
```

7. 将新版本的视觉改动与数据改动分开提交；正常重绘不应改动 `source_data`。
