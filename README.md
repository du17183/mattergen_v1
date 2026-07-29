# A0 + E3-G 训练—测试泄漏诊断

> 当前分支：`experiment/a0-e3g-leakage-diagnostic256`

本分支专门量化 seed 重叠对 E3-G 结果的影响。它是负面诊断实验，**不得被描述为独立验证或正式效果数据**。

## 冻结结论

```text
STATUS=DIAGNOSTIC_ONLY — NOT_VALID_AS_INDEPENDENT_VALIDATION
FROZEN_COMMIT=01e9b2c30e5c58e05eaae908ba291c518b977d03
```

| 子集 | 样本数 | Harm |
|---|---:|---:|
| 与 gate 训练重叠：20000–20063 | 64 | 0 / 64 |
| Held-out：20064–20255 | 192 | 31 / 192 |

```text
Fisher exact p=6.87e-5
```

重叠 seeds 上安全性被明显夸大。平均最大力改善没有同等明确的泄漏膨胀，但 harm rate 已足以证明混合 256 结果不能作为独立验证。

## 工作过程

1. 溯源 gate 训练 seeds 与候选验证 seeds。
2. 将原 256 样本严格拆分为 overlap 和 held-out。
3. 分别计算效果、安全性和分布差异。
4. 使用 Fisher exact test 检查 harm rate 差异。
5. 冻结结论，并禁止匿名化或伪装重叠 seeds。

## 正确使用方式

- 可以用于论文的数据治理、威胁有效性和负面诊断章节。
- 不可用于声称 E3-G 已在独立 256 seeds 上验证。
- 真正正式结果应引用 `feature/q3-e3-pcr-formal256`。
- 组合独立复现应引用 `feature/a0-e3g-independent64`。

## 科学边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

完整项目脉络见[论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)。

## 诊断实现

[`research/a0_e3g_leakage256.py`](research/a0_e3g_leakage256.py) 不改变 seed 编码，而是保留原始 seed 并显式构造三个 cohort：

```text
mixed_256        = 20000–20255
train_overlap_64 = 20000–20063
heldout_192      = 20064–20255
```

分析重建 A0 对照指标，将 A0 与 E3-G 按 seed 配对，然后分别计算连续指标、harm rate 和 Fisher exact test。`diagnostic_manifest.json` 保存 cohort 定义，防止后续把 mixed 集合误标为独立验证。

## 实现与测试

| 文件 | 内容 |
|---|---|
| [`research/a0_e3g_leakage256.py`](research/a0_e3g_leakage256.py) | cohort 切分、数据重建、统计和报告 |
| [`tests/test_a0_e3g_leakage256.py`](tests/test_a0_e3g_leakage256.py) | seed 边界、重叠检测和诊断输出测试 |
| [`reports/a0_e3g_leakage256/protocol.md`](reports/a0_e3g_leakage256/protocol.md) | 冻结诊断协议 |

## 数据索引

- [Overlap 64 统计](reports/a0_e3g_leakage256/cohorts/train_overlap_64/paired_statistics.csv)
- [Held-out 192 统计](reports/a0_e3g_leakage256/cohorts/heldout_192/paired_statistics.csv)
- [Mixed 256 统计](reports/a0_e3g_leakage256/cohorts/mixed_256/paired_statistics.csv)
- [泄漏效应摘要](reports/a0_e3g_leakage256/leakage_effect.json)
- [最终诊断报告](reports/a0_e3g_leakage256/final_report.md)

## 复核命令

```bash
python -m pytest tests/test_a0_e3g_leakage256.py -q
python -m research.a0_e3g_leakage256 status
python -m research.a0_e3g_leakage256 analyze
```

`analyze` 依赖分支内已归档的逐 seed 表；不得重命名 seeds、删除 cohort 标签或把诊断输出用于正式独立验证。
