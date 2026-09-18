# Chapter 3–5 跨章节审计报告

## 范围与约束

审计对象为第3、4、5章正文及各自 evidence map；事实来源为冻结发布包中的 `main_results.csv`、`experiment_status.csv`、`compute_summary.csv`、图索引、正式结果表、最终报告及 `CLAIMS_AND_LIMITATIONS.md`。本轮未运行新实验、未重算统计、未改变 gate、seed、状态或冻结结论。

## 修改摘要

### 第3章

- 章节标题和方法首次定义统一为创新点1完整中英文名称。
- Fixed-K2首次定义为“固定双候选分支策略”；C0定义固定为“固定CFG=2.0的MatterGen参考基线”。
- CFG公式统一使用 \(s_{\mathrm{cond}},s_{\mathrm{uncond}}\)，晶胞字段统一使用 \(H\) 和 \(g_H\)。
- 第3章末补充“属性条件满足不等价于低受力”的第4章过渡。
- 补齐MatterSim文献，evidence map的待引文计数更新为0。

### 第4章

- 开头承接第3章的条件属性结论，结尾自然引向第5章统一比较。
- “clean-x0/clean estimate”等统一为“预测干净结构”，“noisy/raw state”统一为“含噪状态”。
- MatterSim定位为神经势模型/MLIP，CHGNet定位为独立代理评估器；删除任何ground-truth歧义。
- 算法编号统一为算法4-1；图4-5/图4-6按正文顺序重新映射到I2-F6/I2-F5。
- P0、Formal32、Formal256、CHGNet和Property MAE按统一精度展示；负向Property变化、尾部混合、bound与POST边界均保留。

### 第5章

- 开篇补全两项创新标准名称，并定义Fixed-K2和CHGNet独立代理评估角色。
- 统一Formal256百分比、CI、Property MAE和阶段相关系数精度。
- 将“clean-x0”统一为“预测干净结构”，将“有界修正”明确为“有界位置修正”。
- 保留Linear-K2、Property MAE、bound、POST、联合协同和DFT的负/边界结论；未重复第3/4章公式或伪代码。

## 问题计数

| 类别 | 发现 | 修复 | 未解决 | 说明 |
|---|---:|---:|---:|---|
| 实质数值冲突 | 0 | 0 | 0 | 三章关键数值与冻结源语义一致 |
| 数字显示精度漂移类别 | 4 | 4 | 0 | 百分比、相对CI、Property MAE、相关系数 |
| 术语漂移类别 | 7 | 7 | 0 | 创新点全称、Fixed-K2、C0、安全回退、预测干净结构、CHGNet角色、生成后修正 |
| 符号冲突 | 2 | 2 | 0 | \(s_c/s_u\)与晶胞\(L/H\) |
| 图号/出现顺序冲突 | 1 | 1 | 0 | 第4章I2-F5/I2-F6映射 |
| 算法编号冲突 | 1 | 1 | 0 | 算法2改为算法4-1 |
| unresolved citation | 1 | 1 | 0 | 第3章MatterSim |
| 无证据科研主张 | 0 | 0 | 0 | 所有16项审计主张可映射到冻结证据/边界 |
| 逐字重复方法/公式块 | 0 | 0 | 0 | 采用职责分层和交叉引用，不为计数强删 |

## 逻辑与边界结论

章节链条已经统一为：第3章处理条件属性控制并确认Fixed-K2；第4章由“属性不等于低受力”转入RC-NFGD；第5章在统一证据层级下比较收益、成本、消融、独立代理评估和边界。Fixed-K2与RC-NFGD均作为各自问题上的独立贡献，现有证据不支持联合协同。

固定结论保持不变：

```text
Fixed-K2 vs C0 = SUPPORTED
Learned Linear-K2 > Fixed-K2 = NOT_SUPPORTED
RC-NFGD = SUPPORTED
Innovation1 + Innovation2 synergy = NOT_SUPPORTED
DFT_VERIFIED = false
```

## 验收状态

```text
CROSS_CHAPTER_AUDIT = PASS
TERMINOLOGY_CONSISTENCY = PASS
SYMBOL_CONSISTENCY = PASS
NUMBER_CONSISTENCY = PASS
CLAIM_CONSISTENCY = PASS
FIGURE_REFERENCE_CONSISTENCY = PASS
TABLE_REFERENCE_CONSISTENCY = PASS
EVIDENCE_MAP_CONSISTENCY = PASS
UNSUPPORTED_CLAIMS = 0
UNRESOLVED_NUMBER_CONFLICTS = 0
UNRESOLVED_SYMBOL_CONFLICTS = 0
```

## 后续人工排版项

在写第1、2、6章和摘要前无需新增实验。最终进入学校Word/LaTeX模板时仍需人工核对全局参考文献编号、图表分页/字体/公式编号以及学校要求的中英文标点；这些是排版任务，不影响本轮科研一致性PASS。
