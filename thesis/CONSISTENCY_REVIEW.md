# 论文结论—章节—图表一致性审查

审查范围：冻结 C1–C6、符号表、第3–6章初稿、Figure 1–12、Table 1–10、正文/附录安排和
一键生成脚本。

## 审查结论

```text
CLAIM_COUNT=6
CORE_CHAPTER_DRAFTS=4
CORE_FIGURES_V2=7
FIGURE_FAMILIES=12
TABLE_FAMILIES=10
TRAINING_OVERLAP_USED_AS_FORMAL=False
POOLED_128_CLAIM_CREATED=False
DFT_CLAIM_CREATED=False
PROPERTY_TARGET_CLAIM_CREATED=False
```

## 已发现并修正的重要问题

### Adaptive CFG 不是三个字段三个 guidance scale

早期符号表将 \(g_k\) 写成“字段尺度”，容易让读者理解为 cell、position、atom 分别使用
不同 guidance。真实实现是：

1. 三字段分别计算 residual RMS；
2. 有效字段 RMS 取平均得到 \(\delta_t\)；
3. Predictor/Corrector 各自维护 EMA；
4. 得到一个共同的 final guidance \(g_t\)；
5. 同一个 \(g_t\) 用于三个字段的 CFG 融合。

现已同步修正：

- `thesis/NOTATION.md`；
- `thesis/chapters/chapter3_outline.md`；
- `thesis/chapters/chapter3_draft.md`；
- Core V2 Figure 2；
- Core V2 重绘交接说明。

## C1 Adaptive CFG 对照

| 项目 | 冻结口径 | 章节/图表状态 |
|---|---|---|
| 方法 | 三字段 residual RMS → 单一在线 scale | 第3章、Figure 2 一致 |
| seeds/n | 20000–20255，n=256 | 第3章、Figure 5、Table 2 一致 |
| E-hull | −0.003435 eV/atom | 一致 |
| Stable | +5.859 pp | 一致 |
| NUS | +3.516 pp | 一致 |
| 显著性 | CI 跨零，Holm p=1.00 | 正文和图标题均保留 |
| 禁止表述 | 统计显著提升 | 未出现 |

## C2/C3 E3-PCR 对照

| 项目 | 冻结口径 | 章节/图表状态 |
|---|---|---|
| Gate | 14→8→1，129 参数，threshold 0.5 | 第4章、Figure 3 一致 |
| Refiner | 5 steps，eta 0.01，0.02 Å/step，0.10 Å cumulative | 一致 |
| fields | position-only，species/cell fixed | 一致 |
| formal seeds/n | 40000–40255，n=256 | 第4章、Figure 6、Table 3 一致 |
| max force | 0.342964→0.263107，−23.28% | 一致 |
| CI/p/W-T-L | [−0.144966,−0.032453]；4.19e−10；163/0/93 | 一致 |
| Gate coverage | 66.406% | 一致 |
| harm | 25.391%→18.359% | 一致 |
| low-force harm | 29.688%→17.969% | 一致 |
| gain retention | 80.657% | 一致 |
| 禁止表述 | Gate 平均降力优于 Always-on / 保证逐结构安全 | 未出现 |

## C4/C5 组合验证对照

| 项目 | Cohort 1 | Cohort 2 | 状态 |
|---|---:|---:|---|
| seeds | 41000–41063 | 50000–50063 | 一致 |
| n | 64 | 64 | 一致 |
| max-force relative change | −27.10% | −19.02% | 一致 |
| 95% CI | [−0.092341,−0.029754] | [−0.102213,−0.010696] | 一致 |
| p | 7.74e−5 | 0.000587 | 一致 |
| algorithmic W/T/L | 34/19/11 | 35/18/11 | 一致 |
| pooled result | 无 | 无 | Figure 9 与第5章明确禁止 |

## C6 泄漏诊断对照

| 项目 | 冻结口径 | 状态 |
|---|---|---|
| overlap | 20000–20063，0/64 harm | 一致 |
| held-out | 20064–20255，31/192=16.15% harm | 一致 |
| Fisher | one-sided p=6.87e−5 | 一致 |
| mean effect | 未见清晰夸大 | 一致 |
| safety | 重叠显著高估安全性 | 一致 |
| Mixed 256 | INVALID for independent claims | 第6章、Figure 11、Table 6 一致 |

## 单位与方向审查

- `E3-G − baseline` 的最大力差：负值为改善；
- `baseline − E3-G` 的 force gain：正值为改善；
- E-hull：eV/atom，越低越好；
- max force：eV/Å，越低越好；
- RMSD/位移：Å；
- Stable/NUS 等比例差：pp；
- 相对最大力变化：%；
- Figure 8 明确使用 force gain 正向定义；
- Figure 6/9 明确使用 selected-minus-baseline 负向定义。

## 评价边界审查

所有核心章节和图表说明均保留：

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

CHGNet 用于 Gate 特征和 E3-PCR 提议/安全检查，MatterSim-5M 用于正式评价；正文未将二者
写成 DFT 或实验真值。

## 表格编号说明

仓库有 10 个表格 family，但论文建议使用 Table 1–8 的正文编号，其中 5a/5b 是两个单独的
cohort 明细，Table 5 是组合汇总。因此“10 个生成 family”和“论文表1–8”并不矛盾。

## 写入学校模板前仍需人工复核

- [ ] 插入正式参考文献编号后，检查交叉引用是否错位；
- [ ] 将工作编号 Figure 1–12 转成学校章节编号；
- [ ] 确认学校模板使用 eV/Å 还是 eV·Å⁻¹；
- [ ] 导师确认 Figure 4 或 Figure 11 哪一张保留正文；
- [ ] 所有摘要和答辩幻灯片继续使用 C1–C6 冻结措辞。
