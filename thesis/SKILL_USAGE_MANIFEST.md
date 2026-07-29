# Scientific skill usage manifest

读取日期：2026-07-29  
外部技能仓库：`tools/scientific-agent-skills`（仓库外、服务器本地工具目录）  
冻结 commit：`ab2f84ab10597c59fac186ecda6d5edd5dcc8b92`

## 1. scientific-visualization

- 已完整读取：`skills/scientific-visualization/SKILL.md`
- 已读取引用：`references/publication_guidelines.md`、`references/color_palettes.md`
- 落实：Okabe–Ito 色板、颜色外的 marker/线型编码、白底、物理单位、原始点、矢量优先、600 dpi PNG、无 3D/雷达/渐变、图注显式限制。
- 作用文件：Figures 1–12、`paper_style.py`、图注与验证脚本。

## 2. matplotlib

- 已完整读取：`skills/matplotlib/SKILL.md`
- 已读取引用：`references/styling_guide.md`
- 落实：面向对象 API、局部 `rc_context`、`layout="constrained"`、PDF Type 42、SVG 可编辑文字、统一尺寸/轴/图例。
- 作用文件：全部 Python 图源、三个生成脚本和 contact sheet。

## 3. statistical-analysis

- 已完整读取：`skills/statistical-analysis/SKILL.md`
- 已读取引用：`references/reporting_standards.md`
- 落实：配对设计、效应大小、95% CI、精确 p、n、单位、Win/Tie/Loss 口径说明、非显著性不过度解释、训练泄漏资格隔离。
- 作用文件：Figures 5–11、Tables 2–8、`PAPER_CLAIMS_FINAL.*`。

## 4. scientific-schematics

- 已完整读取：`skills/scientific-schematics/SKILL.md`
- 已读取引用：`references/best_practices.md`
- 落实：明确层级/流向/分区、有限节点密度、文本不小于论文可读尺寸、DOT 语义源、SVG/PDF 矢量输出、颜色之外的证据类型编码。
- 作用文件：Figures 1–4 和 Figure 12 的 DOT/Python 源。

## 绘图后端说明

附件明确要求 **Graphviz DOT + Python**。本目录保留 DOT 作为可编辑语义源，并用 CPU-only Matplotlib 导出完全对应的 PDF/SVG/PNG，使没有系统 Graphviz 二进制的笔记本也能一键复现。因此：

```text
FALLBACK_DRAWING_USED=False
MISSING_REQUIRED_SKILL=False
EXTERNAL_IMAGE_GENERATION_USED=False
```

