# MatterGen 命名与归属规则

本规则用于在学校登记题目《基于深度学习的材料逆向生成》下，如实说明 MatterGen 的来源，同时避免把论文写成项目开发报告或把基线结构包装成本文贡献。

## 总原则

> 本文采用预训练 MatterGen 作为条件晶体扩散生成基线。

MatterGen 必须出现、必须正确归属，但不应取代“材料逆向生成”“条件晶体扩散生成基线”或本文两个创新点成为论文叙事中心。

## 第1章

推荐表述：

> 本文选取预训练条件晶体扩散生成模型作为基础框架，并在 MatterGen 实现上完成方法验证。

绪论只需说明框架选择和本文扩展位置，不展开代码路径、类名或工程细节。

## 第2章

MatterGen 作为代表性条件晶体生成相关工作，与 VAE、GAN、Flow、Diffusion 及其他晶体生成模型并列讨论。不得以 MatterGen 作为整章标题，也不得把相关工作贡献写成本文贡献。

## 第3章

必须正式定义：

> 本文采用预训练 MatterGen 作为条件晶体扩散生成基线。

第3章可说明输入输出、晶体表示、属性条件、完整 Predictor–Corrector、原始固定 CFG、`dft_mag_density` 条件任务、冻结 checkpoint 和本文扩展接口。正文保留可复现所需的模型级事实；代码路径、commit 和 SHA256 优先放证据包或附录。

## 第4—6章

首次回指第3章后，优先使用：

- 条件晶体扩散生成基线；
- C0；
- A0；
- E3-A；
- E3-G；
- 完整方法。

只有在说明实现载体、checkpoint 或源码归属时再次使用 MatterGen 名称。

## 图表

图中主模块优先命名为：

```text
条件晶体扩散生成基线
Conditional crystal diffusion baseline
```

图注统一补充：

> 本文采用的条件晶体扩散生成基线由预训练 MatterGen 模型实现。

MatterGen 模块不得使用“本文提出”“创新模块”等视觉标识。Adaptive CFG 与 Learned-Gated E3-PCR 才是本文扩展。

## 推荐与禁止用语

| 场景 | 推荐 | 禁止 |
|---|---|---|
| 基线归属 | 预训练 MatterGen 条件晶体扩散生成基线 | 本文提出的 MatterGen |
| 模型训练 | 复用冻结 checkpoint | 本文训练了完整 MatterGen |
| 本文贡献 | 在基线推理与后生成阶段提出两项扩展 | 改造了 MatterGen 的全部架构 |
| 条件任务 | `dft_mag_density=0.1` 条件生成任务 | 已验证生成材料达到真实磁密度 0.1 |
| 稳定性 | MatterSim-5M 代理相对评价 | DFT 稳定性证明 |

## 归属核查问题

任何正文、图注或答辩材料发布前均检查：

1. 是否明确 MatterGen 是预训练基线？
2. 是否把 MatterGen 原有模块误写为本文贡献？
3. 是否把本文创新准确限定为 Adaptive CFG 与 Learned-Gated E3-PCR？
4. 是否保留 MatterSim、DFT 和目标属性验证边界？
