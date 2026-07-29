# 正文与附录图表冻结安排

该安排用于避免正文图表过多，并保证两个创新点、组合验证和可信性审计各自有直接证据。

## 正文核心图

| 顺序 | 图 | 放置章节 | 正文任务 | 状态 |
|---:|---|---|---|---|
| 1 | Figure 1 完整架构 | 第1章末或第5章开头 | 连接两项创新与评价边界 | Core V2 |
| 2 | Figure 2 Adaptive CFG | 3.5 | 解释真实控制器公式和完整 P/C | Core V2 |
| 3 | Figure 5 Adaptive CFG 结果 | 3.8 | 展示正向趋势及非显著性 | Core V2 |
| 4 | Figure 3 E3-PCR 机制 | 4.6 | 展示 Gate、循环、安全检查和 fallback | Core V2 |
| 5 | Figure 6 E3-PCR 主效果 | 4.8 | 三臂分布与逐 seed 配对效应 | Core V2 |
| 6 | Figure 7 Gate 消融 | 4.9 | coverage/harm/displacement/retention | Core V2 |
| 7 | Figure 9 组合复现 | 5.5 | 两个 cohort 独立并列、不 pooled | Core V2 |

## 正文可选图

| 图 | 建议位置 | 使用条件 |
|---|---|---|
| Figure 4 证据血缘 | 6.1 | 若正文允许讨论研究可信性则保留 |
| Figure 11 泄漏诊断 | 6.4 | 强烈建议保留；可与 Figure 4 二选一压缩 |

## 附录图

| 图 | 附录用途 |
|---|---|
| Figure 8 | confidence–gain 描述性相关；不是因果或校准证明 |
| Figure 10 | Cohort 2 的 64 条逐 seed 配对与 exact tie |
| Figure 12 | 完整 No-Go 路线；正文只保留类别总结 |

若学校模板允许较多正文图，Figure 11 优先于 Figure 4；Figure 8、10、12 仍建议放附录。

## 正文核心表

| 表 | 章节 | 版式 |
|---|---|---|
| Table 1 实验 manifest | 第2章末/第3章前 | 横向或缩写 checkpoint/commit |
| Table 2 创新点一 | 3.8 | E-hull 与比例指标分组，明确不显著 |
| Table 3 创新点二 | 4.8 | 三臂 × 质量指标 |
| Table 4 Gate 消融 | 4.9 | 比例与位移分块 |
| Table 5 组合汇总 | 5.5 | 两行 cohort，不增加 pooled 行 |

## 附录表

| 表 | 附录用途 |
|---|---|
| Table 5a | Cohort 1 完整单组统计 |
| Table 5b | Cohort 2 完整单组统计 |
| Table 6 | 泄漏诊断与资格 |
| Table 7 | 13 条代表性 No-Go 路线 |
| Table 8 | C1–C6 结论、样本量和证据资格 |

如果导师要求正文强调科研规范，可将 Table 6 放入第6章正文，将 Table 1 的详细 checkpoint/
commit 列移至附录。

## 图表编号落地规则

当前仓库使用 Figure 1–12 / Table 1–10 的工作编号。写入学校模板后建议按章节编号：

| 工作编号 | 建议论文编号 |
|---|---|
| Figure 1 | 图1-1 或图5-1 |
| Figure 2 | 图3-1 |
| Figure 5 | 图3-2 |
| Figure 3 | 图4-1 |
| Figure 6 | 图4-2 |
| Figure 7 | 图4-3 |
| Figure 9 | 图5-2 |
| Figure 4 | 图6-1 |
| Figure 11 | 图6-2 |

正式编号只在 Word/LaTeX 模板中修改，不重命名仓库文件，避免破坏复现脚本。
