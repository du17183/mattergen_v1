# Abstract Audit Report

## Scope

本报告审计 `abstract_zh.md`、`abstract_en.md`、`keywords.md` 和 `abstract_evidence_map.md`。本轮未修改 Chapter 1–6、`thesis_release/`、`release/thesis-final-2026` 或 `archive/thesis-exploration-2026`，也未新增实验、结果或科研结论。

## Required checks

| Check | Status | Evidence |
|---|---|---|
| Chinese abstract structure | PASS | 连续正文，包含背景、问题、MatterGen、两项方法、结果和边界。 |
| English abstract structure | PASS | 与中文摘要逐段对应，采用正式学术英语。 |
| Innovation wording | PASS | 使用冻结的两个正式名称；Fixed-K2 和 RC-NFGD 的范围未扩大。 |
| MatterGen role | PASS | 明确 MatterGen 是冻结基础模型，不写成本文提出。 |
| Surrogate boundary | PASS | MatterSim、CHGNet 均表述为机器学习原子间势代理。 |
| DFT boundary | PASS | 明确 `DFT validation has not been performed` / “尚未完成 DFT 验证”。 |
| Linear-K2 negative result | PASS | 中英文均明确未显示相对 Fixed-K2 的稳定额外优势。 |
| Property MAE trade-off | PASS | 中英文均保留 0.009757→0.009868 和约1.14%恶化。 |
| Joint-synergy boundary | PASS | 中英文均不宣称联合协同。 |
| Citation requirement | PASS | 摘要无参考文献编号，符合摘要独立可读要求。 |
| Keyword consistency | PASS | 中英文各5个关键词，语义一一对应。 |

## Core-number audit

下列数字均来自 `thesis/final_audit/abstract_source_sheet.md`，并在中英文摘要中逐一核对：

```text
0.034796
0.026332
24.32%
2.2×
29.81%
29.72%
14.17%
14.36%
0.009757
0.009868
1.14%
```

## Cross-chapter audit

`ABSTRACT_VS_CHAPTER1 = PASS`：摘要保持 Chapter 1 的研究对象、MatterGen 基础模型定位、属性控制与局部代理物理一致性两个问题，以及无 DFT 验证边界。

`ABSTRACT_VS_CHAPTER6 = PASS`：摘要保持 Chapter 6 的 Fixed-K2 正结果、Linear-K2 未获支持、RC-NFGD 代理指标结果、Property MAE 代价、独立路线和无联合协同结论。

## Length record

```text
CHINESE_ABSTRACT_CHARACTERS = 1159
  统计口径：去除标题和空白，保留标点、数字及英文术语；汉字数量为771。
ENGLISH_ABSTRACT_WORDS = 455
  统计口径：去除标题，按空白分词。
```

## Manual review

`UNSUPPORTED_CLAIMS = 0`：未发现超出冻结证据的科研断言。

`CITATION_NEEDED = 0`：摘要未引入外部文献或需要编号的具体文献主张。

`ENGLISH_MANUAL_POLISH = NOT_REQUIRED`：英文摘要已完成正式语句检查；提交模板时仍可按学校英文字体、大小写和标点规范做排版级润色，不涉及科学内容。

`CHAPTER1_6_MANUAL_REVIEW = NONE`：本轮未发现需要修改 Chapter 1–6 的问题。

## Final status

```text
ABSTRACT_SEMANTIC_CONSISTENCY = PASS
ABSTRACT_NUMBER_CONSISTENCY = PASS
ABSTRACT_INNOVATION_WORDING = PASS
ABSTRACT_DFT_BOUNDARY = PASS
ABSTRACT_SURROGATE_BOUNDARY = PASS
ABSTRACT_VS_CHAPTER1 = PASS
ABSTRACT_VS_CHAPTER6 = PASS
```
