# 论文表 1–10 重排与重制说明

本文件说明十组表格的用途、字段、格式和后期重排方式。`csv/` 是便于核对的结构化结果，
`markdown/` 用于 GitHub 浏览，`latex/` 用于论文排版，`xlsx/thesis_results.xlsx` 用于
在 Excel 中筛选和人工检查。当前自动生成表格优先保证可复现，不代表最终论文版式。

## 0. 所有表格通用规则

- 数据只能来自 `thesis_archive/` 和 `thesis/scripts/generate_tables.py`。
- E-hull 保留至少 4 位小数；力和 RMSD 通常保留 4–6 位；p 值小于 0.001 时使用科学计数法。
- fraction 在论文中显示为百分比；两个比例之差显示为百分点 `pp`，不能写成 `%` 相对变化。
- 正负方向必须写入列名或表注，例如 `A0 − C0`、`E3-G − C0`。
- 不仅靠粗体或颜色表示“更好”；列名、箭头或注释必须给出好坏方向。
- 所有 MatterSim 力、松弛、E-hull、Stable 结果都在表注写明“代理势，无 DFT 验证”。
- 表头使用物理量和单位，不在每个单元格重复单位。
- 小数位在同一列对齐；缺失值写 `—`，不能写 0。
- 证据资格必须保留为文字：Formal、Independent、Replication、Diagnostic、Invalid。
- 表格过宽时优先拆分或转置，不使用小于 7 pt 的字体硬塞。

## 表 1：实验设置与证据资格

**用途**：给读者一个可追溯的实验清单，回答每个结论使用了哪个基线、seed、checkpoint、
commit、评价器和证据资格。

**文件**：

- `csv/01_experiment_manifest.csv`
- `markdown/01_experiment_manifest.md`
- `latex/01_experiment_manifest.tex`
- Excel sheet：`01_Experiment_Manifest`

**列含义**：

- `method`：实验/创新身份；
- `baseline`：对照方法；
- `seed_range`、`n`：样本血缘；
- `checkpoint`、`commit`：复现锚点；
- `evaluator`：MatterSim-5M；
- `data_qualification`：允许怎样使用该数据。

**推荐排版**：横向表。将 `checkpoint` 和 `commit` 缩写为前 8–12 位并在脚注给出完整值；
按 Formal、Independent replication、Diagnostic 分组。`data_qualification` 不得隐藏。

**验收重点**：Mixed/diagnostic 数据不能被排成与正式结果同等级。

---

## 表 2：创新点一正式结果

**用途**：比较 C0 与 Adaptive CFG A0 的质量指标和配对变化。

**文件**：

- `csv/02_innovation1.csv`
- `markdown/02_innovation1.md`
- `latex/02_innovation1.tex`
- Excel sheet：`02_Innovation1`

**列含义**：`metric`、`C0`、`Adaptive_CFG_A0`、`A0_minus_C0`、`inference`。

**推荐排版**：

1. E-hull 单独作为连续变量，显示 `eV/atom` 和四位以上小数。
2. Stable、NUS、Novel、Unique、validity 统一转成百分比。
3. `A0_minus_C0` 对比例指标显示为 `pp`，对 E-hull 保持绝对差。
4. `inference` 明确区分 `not significant` 与 `descriptive`。

**不能做的处理**：不能只用绿色突出正向变化而省略“不显著”；不能把趋势写成显著提升。

---

## 表 3：创新点二三臂正式比较

**用途**：同时比较 C0、Always-on E3-A、Learned-gated E3-G 的力、RMSD、E-hull 和分类质量。

**文件**：

- `csv/03_innovation2.csv`
- `markdown/03_innovation2.md`
- `latex/03_innovation2.tex`
- Excel sheet：`03_Innovation2`

**列含义**：方法、最大力、RMSD、E-hull、Stable、NUS、Novel、Unique。

**推荐排版**：

- 方法放行，指标放列；分成“几何/物理代理指标”和“生成质量指标”两级表头。
- 最大力、RMSD、E-hull 用小数；其余指标显示百分比。
- 可以粗体标记每列最优值，但表头或脚注必须注明：
  max force/RMSD/E-hull 越低越好，Stable/NUS/Novel/Unique 越高越好。
- 表下另列 E3-G vs C0 的正式配对 CI、Holm p 和 Win/Tie/Loss，避免只比较均值。

**验收重点**：不要把 Always-on 的更大平均降力与 Learned Gate 的安全收益混为一个结论。

---

## 表 4：Learned Gate 机制消融

**用途**：展示 Gate 在覆盖率、伤害率、低力伤害、位移和收益保留之间的取舍。

**文件**：

- `csv/04_gate_ablation.csv`
- `markdown/04_gate_ablation.md`
- `latex/04_gate_ablation.tex`
- Excel sheet：`04_Gate_Ablation`

**推荐排版**：

1. 按 `unit` 分成两个块：比例类和 Å 位移类。
2. 列为 Always-on、Learned-gated、差值或相对变化。
3. harm 与 low-force harm 越低越好；gain retention 越高越好，需用箭头或脚注说明方向。
4. 表注加入 `McNemar p=0.000534` 和 `n=256`。

**重要限制**：位移只有正式汇总值，不得添加伪造的 per-seed CI 或 p 值。

---

## 表 5a：第一组独立组合验证

**用途**：完整记录 Cohort 1（41000–41063，n=64）的 A0 vs A0+E3-G 配对结果。

**文件**：

- `csv/05_compatibility_cohort1.csv`
- `markdown/05_compatibility_cohort1.md`
- `latex/05_compatibility_cohort1.tex`
- Excel sheet：`05_Compatibility_Cohort1`

**推荐排版**：该表只有一行，论文中更适合转置为两列 key–value 表：
seed/n、A0 force、A0+E3-G force、mean difference、relative change、95% CI、p、W/T/L。

**方向定义**：`mean_difference = A0+E3-G − A0`，负值表示降力。

**必须保留**：relative change −27.10%，algorithmic W/T/L=34/19/11；Gate-off 按算法语义为 tie。

---

## 表 5b：第二组独立组合复现

**用途**：完整记录 Cohort 2（50000–50063，n=64）的复现结果。

**文件**：

- `csv/06_compatibility_cohort2.csv`
- `markdown/06_compatibility_cohort2.md`
- `latex/06_compatibility_cohort2.tex`
- Excel sheet：`06_Compatibility_Cohort2`

**推荐排版**：必须与表 5a 使用完全相同的列顺序、小数位和方向定义，便于读者直接比较。

**必须保留**：relative change −19.02%，algorithmic W/T/L=35/18/11；不要因效应小于 Cohort 1
而改变坐标、精度或隐藏置信区间。

---

## 表 5：两次独立组合验证汇总

**用途**：将 Cohort 1 和 Cohort 2 并排展示，以说明方向复现和效应异质性。

**文件**：

- `csv/07_combination_summary.csv`
- `markdown/07_combination_summary.md`
- `latex/07_combination_summary.tex`
- Excel sheet：`07_Combination_Summary`

**推荐排版**：两行 cohort × 九个核心字段。可将 seed/n 合并为一列，将均值差和 95% CI 合并
为 `effect [95% CI]`，以减少宽度。

**必须保留的限制**：两组是独立并列证据，不得增加 pooled 行，不得写“n=128 预注册验证”。

---

## 表 6：训练泄漏诊断

**用途**：比较 training-overlap 与 held-out 的平均 force difference、伤害数量、伤害率和证据资格。

**文件**：

- `csv/08_leakage_diagnostic.csv`
- `markdown/08_leakage_diagnostic.md`
- `latex/08_leakage_diagnostic.tex`
- Excel sheet：`08_Leakage_Diagnostic`

**推荐排版**：

- 两行分别为 training-overlap 和 held-out；
- `harm_count` 写成 `0/64`、`31/192` 的分子/分母形式；
- `valid_for_*` 不用 True/False 裸值，改成人类可读的 `No—diagnostic only`、
  `Supplementary only`；
- `qualification` 使用醒目的文字，颜色只是辅助。

**必须标注**：one-sided Fisher p=6.87×10⁻⁵；Mixed 256 不具备独立结论资格。

---

## 表 7：代表性负面路线

**用途**：说明研究空间如何被系统排除，并保留每条路线的收益、失败原因和可追溯入口。

**文件**：

- `csv/09_negative_results.csv`
- `markdown/09_negative_results.md`
- `latex/09_negative_results.tex`
- Excel sheet：`09_Negative_Results`

**列含义**：route、goal、No-Go reason、status、main benefit、paper usage、branch、commit。

**推荐排版**：

- 正文使用精简版：路线类别、代表方法、主要收益、停止证据；
- 13 条完整记录放附录横向表；
- 按采样削减、表示学习、物理引导、后处理/排序、GPU 优化分组；
- branch/commit 使用等宽字体或脚注链接。

**不能做的处理**：不构造统一失败分数，不按主观强弱排名，不隐藏曾获得的正向工程收益。

---

## 表 8：最终论文结论与资格

**用途**：固定 C1–C6 的结论措辞、样本量和证据资格，防止摘要、正文和答辩材料相互矛盾。

**文件**：

- `csv/10_paper_claims.csv`
- `markdown/10_paper_claims.md`
- `latex/10_paper_claims.tex`
- Excel sheet：`10_Paper_Claims`

**推荐排版**：四列为 Claim ID、可使用的结论、n、qualification；如版面允许，从
`thesis/PAPER_CLAIMS_FINAL.md` 增加“禁止表述”列，但不得改变 CSV 中的冻结结论。

**视觉层级**：

- C1 必须突出“positive trend, not statistically significant”；
- C2–C5 区分 formal independent 与 independent replication；
- C6 明确 `diagnostic only`。

**验收重点**：任何一行都不能暗示 DFT-verified stability 或真实磁属性命中。

---

## 11. 后期制表工作流

1. 运行 `python thesis/scripts/generate_tables.py` 得到结构化基表。
2. 在 CSV/XLSX 中核对数值，不直接手改生成的 Markdown/LaTeX 作为唯一来源。
3. 按本文件选择保留列、显示精度和横竖版式。
4. 将最终表注与 `thesis/tables/captions/` 对齐。
5. 若在 Word/Excel 中美化，保存一份可编辑源；最终论文中优先使用原生表格，不截图。
6. 重排后逐项检查：
   seed、n、单位、差值方向、CI、p 值、W/T/L、证据资格、MatterSim/DFT 限制。
