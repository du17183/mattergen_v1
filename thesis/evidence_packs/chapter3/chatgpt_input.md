# 网页ChatGPT写作入口：第3章

论文学校登记题目固定为《基于深度学习的材料逆向生成》。第3章最终标题为《条件扩散晶体生成基线与评价体系》。本文件是分段写作导航，不直接作为整章一次性生成Prompt。

## 推荐使用顺序

1. 先向网页ChatGPT提供`../WRITING_GUARDRAILS.md`。
2. 使用`chatgpt_input_final.md`或内容相同的`chatgpt_input_part1.md`撰写3.1—3.3。
3. 使用`chatgpt_input_part2.md`撰写3.4—3.6。
4. 使用`chatgpt_input_part3.md`撰写3.7—3.9。
5. 每一部分保存为独立草稿，再由Codex核查项目事实、公式、单位、seed、证据资格和表述边界。

## 冻结定位

- 研究对象：目标属性条件下的周期晶体候选生成。
- 基础方法类别：预训练条件晶体扩散生成模型。
- 实验基线：MatterGen。
- 基线归属：MatterGen是预训练基线、实验框架和实现载体，不是本文提出的方法。
- 创新点一：Multi-field Residual-driven Online Adaptive CFG。
- 创新点二：Learned-Gated E3-PCR。

## 不可删除的边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

不得把MatterGen原有结构写成本文创新，不得把MatterSim或CHGNet写成DFT，不得把`dft_mag_density=0.1`写成真实属性已经命中，不得把Mixed 256写成独立验证。

## 最终章节

- 3.1 材料逆向生成任务定义
- 3.2 条件扩散晶体生成方法
- 3.3 MatterGen基线模型与实验配置
- 3.4 目标属性条件任务
- 3.5 数据划分与实验协议
- 3.6 MatterSim代理评价体系
- 3.7 评价指标与统计方法
- 3.8 数据独立性和真实性控制
- 3.9 本章小结
