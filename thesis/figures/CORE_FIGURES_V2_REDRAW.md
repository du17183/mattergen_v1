# 七张核心正文图 V2：设计与后续重绘交接

本文件对应 `thesis/scripts/generate_core_figures_v2.py`。V2 只覆盖 Figure 1、2、3、5、6、7、9；
其余图继续作为补充材料草稿。所有 V2 图都复用既有 `source_data/*.csv` 和
`thesis_archive/`，不修改任何冻结结果。

## 一键复现

```bash
python thesis/scripts/generate_all.py
python thesis/scripts/validate_outputs.py
```

只重画核心图：

```bash
python thesis/scripts/generate_core_figures_v2.py
```

只重画指定图：

```bash
python thesis/scripts/generate_core_figures_v2.py fig01 fig05 fig09
```

输出仍写入：

```text
thesis/figures/generated/pdf/
thesis/figures/generated/svg/
thesis/figures/generated/png/
```

## V2 相对原草稿的变化

| 图 | V2 设计 | 解决的问题 | 后续可继续优化 |
|---:|---|---|---|
| 1 | 两个阶段色带、真正的 Gate 菱形、评价边界 | 原图节点层级接近、主线不突出 | 加入简化晶体图标，统一学校字体 |
| 2 | 精确展示三字段 RMS、均值、分阶段 EMA 和公式 | 原图容易误解为三个独立 guidance scale | 用矢量公式排版替换 Matplotlib mathtext |
| 3 | 风险估计、循环精修、安全决策三阶段 | 原图是一长串同级框，backtracking 不直观 | 用 Illustrator 绘制循环和拒绝回退 |
| 5 | E-hull 原始点+violin+CI，Stable/NUS forest | 柱图容易夸大非显著趋势 | 使用 raincloud，并把 CI 数值移到图外 |
| 6 | 三臂 ECDF + 排序配对效应 | 箱线和大量散点重叠 | 增加局部力区间 inset，但不得删高值 |
| 7 | direction-aware dumbbell + 独立收益保留面板 | 不同好坏方向混在同一柱图 | 在最终论文中加入上下箭头图标 |
| 9 | 左标签—中央 forest—右数值的标准布局 | 原图文字压在 forest 轴内 | 按学校版芯微调三列宽度 |

## Figure 1 分层说明

绘图函数：`fig01()`。

建议在 Illustrator/Inkscape 中保留以下图层：

1. `stage_background`：创新点一、创新点二两个浅色阶段色带；
2. `sampling_pipeline`：目标、MatterGen、Adaptive CFG、生成结构；
3. `gate_decision`：Gate 菱形及 Gate-on/Gate-off 标签；
4. `refinement_and_fallback`：E3-PCR 与 exact fallback；
5. `evaluation_boundary`：MatterSim 虚线评价框和无 DFT 声明。

后续改变节点形状和字体时，不得把 MatterSim 移入训练/生成流程，也不得把 E3-PCR 画成
修改原子种类或晶胞的模块。

## Figure 2 分层说明

绘图函数：`fig02()`。

V2 按真实代码表达为：

```text
三字段 conditional–unconditional residual
→ 分字段 RMS
→ 有效字段均值 δ_t
→ predictor/corrector 各自 EMA
→ ratio 与单一 adaptive multiplier
→ 单一 final guidance scale
```

这不是三个字段各自拥有一个 guidance scale。后续重绘必须保留：

```text
multiplier clip=[0.25,4]
final scale clip=[0,5]
base=2.0
alpha=0.50
beta=0.95
epsilon=1e-6
```

## Figure 3 分层说明

绘图函数：`fig03()`。

建议保留四类信息：

- 14 个输入特征分为 size/density、geometry/composition、CHGNet E/F、
  stress/magnetism；
- MLP 结构 `14→8→1`、129 参数、阈值 0.5；
- 五步位置更新、单步 0.02 Å、累计 0.10 Å、最多三次回溯；
- `finite`、最短距离和 CHGNet energy 三类验收，以及任何拒绝后的 exact fallback。

如果图面仍拥挤，可把 14 个特征的完整名称移到表格或附录，只保留四个特征组。

## Figure 5 分层说明

绘图函数：`fig05()`。

- A 面板的每个点是一条真实 seed 配对差值，不得从均值重建。
- violin 只显示分布形状；黑色菱形和误差线才是均值与 paired bootstrap 95% CI。
- B 面板用 forest 而不是柱图，以减少面积造成的视觉夸大。
- 标题必须保留 `directional trends, not statistically significant`。

后期若改成 raincloud，必须保留全部 256 个点、零线和 CI。

## Figure 6 分层说明

绘图函数：`fig06()`。

- A 面板使用 ECDF，所有三臂使用相同 log-x 轴；
- B 面板定义 `ΔF=E3-G−C0`，负值为改善；
- 蓝色水平线为均值，浅蓝带为均值 bootstrap 95% CI；
- 绿色/朱红点分别表示逐 seed 改善/伤害。

后续不应只画三根均值柱，因为这会隐藏分布长尾和 93 个伤害样本。

## Figure 7 分层说明

绘图函数：`fig07()`。

- A 面板只比较 coverage、overall harm、low-force harm；
- B 面板只比较 mean/P95 displacement；
- C 面板独立显示 gain retention，因为其好坏方向与 harm 相反；
- displacement 只有汇总值，禁止添加伪造 CI。

## Figure 9 分层说明

绘图函数：`fig09()`。

三个图层/列分别是：

1. cohort 名称、seed 和 n；
2. 均值差及 95% CI forest；
3. 数值、相对变化和 p 值。

禁止新增 pooled diamond、pooled p-value 或“n=128 预注册验证”。

## 最终人工重绘验收

- [ ] 核心结论在黑白打印下仍能读懂。
- [ ] 真实数据点、均值、CI 和参考线没有被装饰元素遮挡。
- [ ] 图内使用的差值方向与图注完全一致。
- [ ] 任何统计显著性文字都来自冻结报告。
- [ ] Figure 5 仍明确“不显著”，Figure 9 仍明确“不合并”。
- [ ] 代理势/无 DFT 声明存在于图注。
- [ ] SVG/PDF 文字可编辑，PNG 为 600 dpi。
