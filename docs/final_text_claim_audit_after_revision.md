# MatterGen 最终正文重构后的 claim audit

状态：导师预审版 READY；不是已消除封面占位符和学校字体差异的最终提交版。审计对象为学校模板目录 thesis/latex/ustb_2026_polished_preview/contents 的双语摘要、第1、3、4、5、6章及本轮新图。日期：2026-09-29。本轮只编辑论文、图件源代码和审计文档；没有新增实验、重算统计、修改 seed/checkpoint/阈值或覆盖冻结结果。

## 章节修改与证据对应

| 文件 | 本轮修改 | 证据边界 |
|---|---|---|
| contents/abstract.tex | 中英文摘要显式确立“冻结材料生成模型的可信推理增强”主线；加入历史 A6 Pareto 与全新 32 对 Timing 结果。 | “可信”限定为预算、回退和代理证据可审计；未写 DFT、联合增益或 Late 最优。 |
| contents/chap1.tex | 研究问题和贡献统一为预算控制、物理反馈两条独立推理路线；补充 POST 与 Timing 的反例。 | 创新点1仍为 Fixed-K2 框架实例；创新点2仍是代理力反馈。 |
| contents/chap3.tex | 保留框架定义、算法、C1-128、Linear-K2 负结果、Independent Best-of-2 和 K0–4；新插入框架图 3-2 与机制差异图 3-8。 | 图 3-8 明示 Best-of-2 为 4,000 score/2 终点、Fixed-K2 为 4,400/3；不称同预算或性能优越。原定量图 3-9 和预算曲线图 3-10 保留。 |
| contents/chap4.tex | 章名改为“面向冻结材料扩散模型的神经力场反馈引导方法”；在原 Formal256、CHGNet、POST、负消融之后加入 A6 Pareto 和 32 对 Timing 两节、三表、两图。 | A6 n=32、Formal256 n=256、Timing n=32 分开报告；Pareto 是方法均值点估计，不是统计支配；Timing 时机与力调用预算混杂。 |
| contents/chap5.tex | 新增跨方法属性–代理力权衡及 Timing 分析；证据总表新增两行，并把不支持的肯定式假设行名改为中性对照名。 | POST 低 MaxF、Late 不具有已确认的跨阶段优势、两创新无已验证联合增益。 |
| contents/chap6.tex | 统一可信推理增强总结；加入 Pareto/Timing 机制边界和后续 DFT 验证要求。 | 保留 Best-of-N 竞争、代理偏差、POST 优势和无 DFT。 |

新增矢量图：figures/thesis_integration/budget_verifier_framework.pdf、best2_vs_fixedk2_mechanism.pdf、timing_paired_maxf.pdf。均有同目录 SVG、600-dpi PNG 与 Python 源 scripts/draw_thesis_integration_figures.py。第4章 Pareto 图复用 figures/rcnfgd/pareto_frontier.pdf；RC-NFGD 流程、Best-of-2 定量对照和预算曲线复用原有矢量 PDF。新图的完整数据合同见 figures/thesis_integration/figure_contract.md。

## 五项禁止性主张逐项核查

| 风险 | 核查结果 |
|---|---|
| Fixed-K2 优于 Best-of-N | 未作为结论。独立 Best-of-2 在新 cohort 的均值 0.028097，Fixed-K2 为 0.029506；直接配对区间跨零，且分别为 2.0×/2 终点与 2.2×/3 终点。C1-128 的 0.034796→0.026332、CI [0.005817,0.011391] 仍只针对同 cohort C0。 |
| RC-NFGD 优于 POST | 未作为结论。历史 A6 同势 MaxF 为 C0 0.214179、RC-NFGD 0.163650、POST 0.043275；POST 终态力更低。在线方法的定位是生成过程反馈。 |
| 两方法联合提升 | 未作为结论。A5 是历史 Adaptive CFG 与力反馈的兼容性检验，不是 Fixed-K2×RC-NFGD 的确认；联合协同仍未支持。 |
| DFT 验证或真实物理稳定性 | 全文仍明确为 MatterSim/CHGNet 代理证据；未进行 DFT 或实验验证。 |
| Late 注入最优 | 未作为结论。Timing 中 Late−C0 MaxF 差 −0.05123 eV/Å、95% CI [−0.06834,−0.03770]，但 Late−Early 与 Late−Middle 的直接 CI 均跨零，且平均力调用约为 20、931.5、500.0 次/样本。 |

其他负结果保留：Linear-K2 未确认增益；K=2 为成本折中而非 K0–4 最低 MAE；Formal256 Property MAE 轻微恶化；CHGNet 尾部混合；unbounded 消融不支持 bound 是性能核心；POST 同势低力更强。第4章新表和正文保留 Early/Middle 不利或不确定结果，没有事后重定义阶段。

## 编译与版面检查

- 采用项目内 TinyTeX/XeLaTeX 及学校 ustbthesis 预览模板，三轮编译成功。PDF：thesis/latex/ustb_2026_polished_preview/ustb_mattergen_final_integrated.pdf；137 个物理页、30 幅图、29 张表。
- 最终日志没有未定义引用、未定义文献、缺字、缺图或 LaTeX 致命错误。新增图引用为图 3-2、图 3-8、图 4-7、图 4-8；新表为表 4-7 至 4-9。新图所在物理页约为 61、69、92、93。
- 目检了第4章新表/图所在物理页 91–93：列距调整后无新表的明显重叠。新增三图的渲染碰撞审计均为 0 FAIL、0 WARN；源码预检 0 FAIL，两个非阻断 WARN 分别是未额外导出 TIFF 和论文版心宽度不等于期刊 89/183 mm 预设。插入终稿 PDF 后，抽查最小图中文字 6.49 pt，超过 5 pt 底线。
- 仍有 8 条小幅 overfull hbox 警告，最大 8.43 pt；本轮新增总表的 20–30 pt 溢出已修正。这些残留警告需在最终学校格式检查时逐处处理，不能称为已完成逐页终审。
- 封面作者、学院、导师、学号等仍为占位符。当前预览类文件仍用 Fandol Song 等替代字体而非已核准的学校正式字体配置；本轮没有擅自修改模板或身份字段。因此这是可交导师核对科学叙事的预审稿，不应直接视为学校格式终版。

PDF SHA256：3fb33cbef19fbd2231853822cb0fbb834f98823c22561ecc139db409b43e735a。若再次编辑正文或图件，必须重编译并重新计算哈希。

