# 北京科技大学 LaTeX 排版包

本目录保留学校提供的原始模板压缩包
ustb_2026_template_source.zip，并在 ustb_2026_preview/ 中完成 MatterGen
论文的套版。章节 Markdown、实验结果、图像和冻结证据文件不在本目录中修改。

## 已完成的套版内容

- 使用学校模板的硕士封面、授权书、独创性声明、摘要、目录、图表清单、正文、参考文献、致谢、作者简历和学位论文数据集结构；
- 中文和英文摘要完整排入模板，百分号已按 LaTeX 语法转义；
- 22 条全篇唯一参考文献统一为 natbib 引用，包含第 2 章使用的 GemNet；
- 24 个表格和 24 个插图均使用学校模板的 `\\bicaption` 双语题注；长表格采用首页/续页表头分离，避免重复编号和题注，并自动生成图表清单；
- 使用项目内 TinyTeX 和 Fandol 宋体回退字体，等宽字体使用 DejaVu Sans Mono，以避免中文、希腊字母和框线符号缺字；
- 按学校模板统一正文字号、题注样式和三线表结构；对长状态标签、英文方法名和代码标识加入可断行控制，避免破坏页面宽度；
- 提供 build_ustb_preview.sh，执行三次 XeLaTeX 并检查缺字、未定义引用以及图表清单数量。

## 编译

在仓库根目录执行：

    ./thesis/latex/build_ustb_preview.sh

输出文件为：

    thesis/latex/ustb_2026_preview/ustb_mattergen_preview.pdf

当前冻结内容编译结果为 127 页（含封面、摘要、目录、正文及模板后置页；题注和表格分页调整后以构建脚本输出为准）。

## 提交前必须填写

模板要求的作者姓名、英文姓名、学号、学院中英文名、导师中英文名、正式专业名称、
提交日期、分类号、密级和作者简历目前没有出现在冻结研究材料中，因此主文件中仍保留
[待填] 占位符。补齐这些信息后重新运行构建脚本即可；不得根据推测填写。

当前运行环境没有学校指定的 SimSun/Times New Roman 字体，预览使用项目内 Fandol
宋体和 Latin Modern 字体作为可复现回退。若学校要求字体文件级别一致，应在有授权的
本地字体环境中把 ustbthesis.cls 的字体声明替换为对应字体，再复核页码和断行。

GitHub 归档不包含项目内 `.TinyTeX/` 工具树。脚本默认会使用本地的
`.TinyTeX/`（如果存在），也支持通过 `TEXROOT` 指定外部 TeX Live/TinyTeX：

    TEXROOT=/path/to/texlive ./thesis/latex/build_ustb_preview.sh
    TEXROOT=/path/to/texlive ./thesis/latex/build_ustb_polished_preview.sh

其中 `TEXROOT` 应包含 `bin/x86_64-linux/xelatex`，或者系统的 `xelatex` 已经在
`PATH` 中。这样构建脚本不依赖服务器上的绝对路径，也不会把编译环境误当成论文源文件。

## 图表精修预览

`build_ustb_polished_preview.sh` 会从冻结的 `thesis_release` CSV/JSON 重新生成一套
不覆盖原始图件的 `figures_polished/` 矢量图；Chapter 1--2 的概念图由
`reproduce_concept_figures_polished.py` 生成到 `concept_figures_polished/`，并统一以
PDF 矢量主图接入预览。随后脚本编译
`ustb_2026_polished_preview/ustb_mattergen_polished.pdf`。该流程只改变图表呈现：
统一色板、线宽和字体，去除概念图的阴影与 PNG 缩放伪影，重排过宽的流程/证据图，
并为长尾分布保留完整范围和中心区域；不会重新运行模型或修改任何实验数值。

图表设计审计记录在 `thesis_release/FIGURE_TABLE_STYLE_AUDIT.md`。原始
`figures/` 和 `ustb_2026_preview/` 保留，待导师确认 polished 版本后再替换正式稿。

当前 polished 预览已对表 4-4、5-2、5-3、5-4 增加分组表头，对表 5-2
的方法名做局部换行，并移除不一致的非主结果加粗；所有数值、顺序、统计
判定和原始图件均保留。当前构建输出为 125 页，仍有少量历史窄单元格的
Overfull/Underfull 诊断（最大约 8.2 pt），需要在导师确认版式后再决定是否
继续微调，不能将此预览称为已经完全符合学校最终送审格式。
