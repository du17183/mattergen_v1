# 最终实验清单

| 实验 | seeds | N | 独立性/用途 | 冻结状态 | 文档提交 |
|---|---:|---:|---|---|---|
| 创新点一：Multi-field Residual-driven Online Adaptive CFG | 20000–20255 | 256 | 独立正式评估；本方法不使用 Q3 Gate 训练数据 | FORMAL_INNOVATION1_CONFIRMED=True | `20255f1a857cd763a7ef2bf2f24c1889c98c4d1c` |
| 创新点二：Learned-Gated E3-PCR 正式 256 | 40000–40255 | 256 | 正式 seeds 与 Q3 训练 seeds 20000–20063 交集为 0 | FINAL_STATE=E3_G_FORMAL_CONFIRMED | `41479015c5c3edc389601c4b7cc44a6db5e115cd` |
| A0 + E3-G 第一次独立兼容性验证 | 41000–41063 | 64 | 与 Q3 训练和全部开发范围交集为 0 | A0_E3G_COMPATIBILITY_GO=True | `e358ee39a8cdd2a061a18bfaddbe88316b455048` |
| A0 + E3-G 第二次完全独立复现 | 50000–50063 | 64 | 预注册全新 seeds；无训练、调参或历史测试交集 | A0_E3G_INDEPENDENT64_GO=True | `85485bc956fce1cf7d01c55baaa92c0b69fd745e` |
| Q3 Gate 训练重叠泄漏诊断 | 20000–20255（训练重叠 64；held-out 192） | 256 | DIAGNOSTIC_ONLY；Mixed 256 非独立 | LEAKAGE_INFLATION_DETECTED=True | `d5bf7d00ab51a2a0b319203443391e3463e7a91b` |
| A0 256 复用资格审计 | 候选 20000–20255 | 256 | 64 个 seeds 与 Q3 训练重叠，故整个旧批次不具正式复用资格 | SOURCE_DATA_INCOMPLETE — NO_EFFECT_ESTIMATE_PRODUCED | `e09858e1b947ff6e14ebc077d61c59f3585f1b55` |
