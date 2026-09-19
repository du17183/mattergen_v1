# Chapter 1 Reference Map

本文件记录第1章研究现状中的外部文献主张、原始来源和本文研究问题之间的对应关系。检索与核对截至 2026-09-19；“VERIFIED”表示题名、作者、年份、发表载体或 arXiv/DOI 信息已核对。本文不把代表性文献列表解释为穷尽式系统综述，也不使用文献数量推导创新性结论。

| Section | Claim | Source | Year | Publication | Primary/Review | Citation status | Notes |
|---|---|---|---:|---|---|---|---|
| 1.1 | 生成式模型可用于晶体材料逆向设计 | Xie et al., CDVAE [2]; Jiao et al., DiffCSP [3] | 2022–2023 | ICLR; NeurIPS | Primary | VERIFIED | 用于背景与问题定义 |
| 1.2.1 | CrystalGAN将GAN用于晶体结构生成与性质搜索 | Nouira et al. [1] | 2018 | arXiv | Primary | VERIFIED | 不将其结果扩展为普适稳定性 |
| 1.2.1 | GNoME结合大规模图模型、主动学习和材料搜索 | Merchant et al. [14] | 2023 | Nature | Primary | VERIFIED | 作为高通量发现路线背景 |
| 1.2.1 | 生成式材料设计的关键环节包括表示、生成架构和属性约束 | Long et al. [21] | 2024 | arXiv review | Review | VERIFIED | 只用于综述性归纳 |
| 1.2.2 | CDVAE采用周期图、变分表示和扩散解码 | Xie et al. [2] | 2022 | ICLR | Primary | VERIFIED | 周期生成代表方法 |
| 1.2.2 | DiffCSP联合生成晶格和坐标并使用周期等变网络 | Jiao et al. [3] | 2023 | NeurIPS | Primary | VERIFIED | 主要面向给定组成的CSP |
| 1.2.2 | DiffCSP++加入空间群和Wyckoff约束 | Jiao et al. [4] | 2024 | ICLR | Primary | VERIFIED | 结构条件生成代表方法 |
| 1.2.2 | UniMat研究可扩展的材料扩散生成 | Jiao et al. [5] | 2024 | ICLR | Primary | VERIFIED | 不展开具体基准数值 |
| 1.2.2 | SLICES提供可逆不变晶体字符串表示 | Xiao et al. [6] | 2023 | Nature Communications | Primary | VERIFIED | 序列化表示代表方法 |
| 1.2.2 | CrystaLLM对CIF文本进行自回归生成 | Antunes et al. [7] | 2024 | Nature Communications | Primary | VERIFIED | 与扩散路线作方法对比 |
| 1.2.3 | MatterGen联合生成原子类型、位置和晶胞并支持属性条件 | Zeni et al. [8] | 2025 | Nature | Primary | VERIFIED | 本文基础模型，非本文创新 |
| 1.2.3 | SCIGEN在扩散采样中注入结构图案约束 | Okabe et al. [9] | 2025 | Nature Materials | Primary | VERIFIED | 与本文的终点回退接口区分 |
| 1.2.3 | PODGen把多个预测器与生成模型结合进行目标导向生成 | Ye et al. [10] | 2026 | npj Computational Materials | Primary | VERIFIED | 与本文的有限分支选择接口区分 |
| 1.2.4 | CFG线性组合条件和无条件预测以进行推理引导 | Ho & Salimans [13] | 2022 | arXiv | Primary | VERIFIED | Chapter 2已给出公式 |
| 1.2.5 | M3GNet、NequIP和MACE代表通用或等变MLIP路线 | Chen et al. [15]; Batzner et al. [16]; Batatia et al. [17] | 2021–2022 | arXiv; Nature Computational Science; Nature Communications; NeurIPS | Primary | VERIFIED | 只说明模型类别与作用 |
| 1.2.5 | CHGNet和MatterSim提供能量、力等原子级代理预测 | Deng et al. [18]; Yang et al. [19] | 2023–2024 | Nature Machine Intelligence; arXiv | Primary | VERIFIED | 本文分别作为属性/力评价角色 |
| 1.2.5 | 力引导扩散已在相邻蛋白质构象任务中出现 | Wang et al. [20] | 2024 | arXiv | Primary | VERIFIED | 不据此声称材料领域首创 |
| 1.2.6 | 本文研究空白限定为冻结MatterGen上的参考保留分支和后期神经力反馈接口 | [1]–[10], [14]–[20]综合判断 | 2018–2026 | 多种正式载体 | Primary synthesis | VERIFIED | 属于本文研究定位，不是某一文献原话 |

## Literature search boundary

- 覆盖的技术线包括晶体VAE/GAN、周期等变扩散、空间群约束扩散、可扩展材料扩散、晶体序列模型、条件生成、结构约束采样以及通用MLIP。
- 检索到的 SCIGEN、PODGen 和相邻领域 force-guided diffusion 与本文接口存在相关性，因此正文不使用“首次提出多分支条件扩散”“首次将原子力用于扩散生成”等表述。
- 当前两个创新点的结论来自本项目冻结实验，而不是由外部文献证明：Fixed-K2 为 SUPPORTED，Linear-K2 为 NOT_SUPPORTED；RC-NFGD 在冻结代理指标上为 SUPPORTED，DFT_VERIFIED=false，JOINT_SYNERGY=NOT_SUPPORTED。
- 本章引用共 21 个编号，当前未发现待核对引用。
