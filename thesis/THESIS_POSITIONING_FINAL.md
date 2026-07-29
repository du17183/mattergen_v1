# 论文研究定位（冻结版）

本文件围绕学校登记题目《基于深度学习的材料逆向生成》冻结论文研究对象、基础框架、本文贡献和结论边界。若旧提纲、历史分支 README 或实验代号与本文件冲突，以本文件和 `CHAPTER_NUMBERING_FINAL.md` 为论文写作依据；冻结实验身份仍以 `thesis_archive/` 为准。

## 1. 论文研究主题

本文研究主题为：

> 面向目标属性条件的晶体材料逆向生成。

论文中的“材料逆向生成”具体指：给定目标属性条件，利用深度生成模型产生满足结构约束的周期晶体候选。研究从计算机与深度学习角度关注：

- 属性条件生成；
- 扩散模型推理；
- 自适应条件引导；
- 生成后结构质量优化；
- 实验独立性、证据资格与可复现性。

本文不覆盖所有材料形态，也不声称解决完整的材料发现—合成闭环。实际实验对象为周期晶体材料。

## 2. 基础生成框架

本文使用的基础框架是预训练条件晶体扩散生成模型，具体实现为 MatterGen。

> MatterGen 是本文采用的预训练条件晶体扩散生成基线、实验框架和实现载体，不是本文提出的方法。

基线承担晶体组成、原子位置和晶格的条件扩散生成。本文没有从零构建或训练完整 MatterGen 基础模型，也不得将 MatterGen 原有的晶体表示、扩散主干、Predictor–Corrector 或原始 CFG 写为本文创新。

## 3. 本文贡献

### 创新点一

**Multi-field Residual-driven Online Adaptive CFG**：在完整 Predictor–Corrector 采样流程中，根据 cell、position 和 atomic-number 三字段条件残差的在线统计，自适应调整共享 CFG scale。

### 创新点二

**Learned-Gated E3-PCR**：在生成后使用 14 维风险特征和 129 参数 Learned Gate 决定是否执行安全有界的位置精修，并通过 trust region、backtracking、安全检查和 exact fallback 控制风险。

### 完整方法

```text
预训练条件晶体扩散生成基线
+ Multi-field Residual-driven Online Adaptive CFG
+ Learned-Gated E3-PCR
```

即 A0 生成后连接 E3-G 后处理。两个创新点功能上串联、证据上分别验证；组合效果由两组独立 64-seed cohort 支持，不事后合并成预注册 128-seed 主结论。

## 4. 结论边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

因此论文可以报告统一 MatterSim-5M 代理流程下的方法间相对比较，但不得声称：

- 本文从零构建并训练了完整晶体生成模型；
- 本文提出了 MatterGen；
- 本文完成了真实磁密度验证；
- 本文发现了已由 DFT 证明的新材料；
- 本文证明生成结构具有真实热力学稳定性或实验可合成性。

`dft_mag_density=0.1` 是生成条件输入。现有归档没有独立目标属性真值，因此不得把条件值解释为输出已真实命中目标磁密度。

## 5. 论文叙事主线

1. 从材料逆向生成的目标条件到周期晶体候选生成任务；
2. 以预训练 MatterGen 实现条件晶体扩散生成基线；
3. 识别固定 CFG 无法随字段残差和采样状态在线调节的问题；
4. 提出多字段残差驱动的在线 Adaptive CFG；
5. 识别生成结构预松弛最大力和局部几何风险；
6. 提出 Learned-Gated E3-PCR 进行安全有界的后生成位置优化；
7. 通过正式独立实验、两次组合 cohort、消融和泄漏诊断界定效果与局限。

这条主线以“材料逆向生成中的推理控制与生成后质量优化”为中心，而不是 MatterGen 项目开发过程汇报。
