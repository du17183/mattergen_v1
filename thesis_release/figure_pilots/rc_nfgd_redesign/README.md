# RC-NFGD 方法图：独立重设计试画

## 图契约

- 核心结论：RC-NFGD 只在经阶段诊断选定的后期 predictor 中，将冻结神经势的有界力反馈转换为 MatterGen 的位置 score 增量。
- 问题：何时、如何、以哪些护栏把力信号注入逆扩散，而不改变其余字段与 corrector？
- 图类型：单张方法示意图（schematic-led），非定量结果图；不使用旧图素材、构图或绘图脚本。
- 输出：学校论文正文宽 138.4 mm；Python/Matplotlib；PDF/SVG 可编辑矢量图，600 dpi PNG/TIFF；字体下限 5 pt。
- 信息层次：上方时间窗口；中部单次合格 predictor 的三段变换；下方安全回退。没有独立统计面板，也没有暗示 DFT 验证。
- 科学依据：第 4 章方法文字及已冻结的正式配置；示意原子和向量不表示实验样本或实测力值。
- 质检：绘图脚本预检、PDF 字号、碰撞和最终印刷尺寸目测；报告随输出归档。
- 本轮结果：静态审查 20 PASS、1 WARN、0 FAIL（WARN 只因学校版心宽 138.4 mm 不属于 Nature 默认栏宽）；PDF 138.4 × 91.0 mm、文字可选，最小渲染字高 6 pt；严格 PDF 碰撞审计 0 FAIL、0 WARN。单轴方法示意图的面板对齐审计为 NOT APPLICABLE；已在论文宽度下目测检查。
- 范围：仅非覆盖试画；不改论文正文引用，不改变冻结实验结论。

复现：使用已有的 Python 绘图环境运行 `thesis_release/scripts/reproduce_rc_nfgd_skill_redesign.py`。

## AI 辅助的后续候选

按 nature-figure 的 AI 示意图流程另生成两张仅供内部构图参考的概念稿；科学核查后均未直接采用。随后依据其中的三模块构图进行独立、可编辑的矢量重绘，输出 `I2_F1_rc_nfgd_ai_assisted_v3.*`，复现脚本为 `thesis_release/scripts/reproduce_rc_nfgd_ai_assisted_v3.py`。概念稿及生成/审查记录见 `AI_PROVENANCE_AND_QA.md`。v3 并未替换论文正文图，AI 使用与学校提交政策尚待核对。
