# 论文定位与第3章真实性验证

验证日期：2026-07-29
基线提交：`48b9974100076643f4263faef0dde6fea7c54473`
验证范围：论文题目、研究定位、MatterGen归属、章节/图表/公式编号、第3章证据包和网页ChatGPT Prompt。

## 最终状态

```text
TITLE_FROZEN=True
BASELINE_ATTRIBUTION_CORRECT=True
CONTRIBUTION_BOUNDARY_CLEAR=True
CHAPTER_NUMBERING_CONSISTENT=True
DATA_MISMATCH_DETECTED=False
```

## 检查结果

| 检查 | 结果 | 证据 |
|---|---:|---|
| 学校中文题目逐字保持“基于深度学习的材料逆向生成” | 通过 | `THESIS_TITLE_FINAL.md/json` |
| 英文题目标为暂定 | 通过 | `ENGLISH_TITLE_STATUS=PROVISIONAL` |
| 研究对象限定为周期晶体材料 | 通过 | `THESIS_POSITIONING_FINAL.md/json` |
| MatterGen明确说明为预训练条件晶体扩散生成基线 | 通过 | 定位文件、命名政策、第3章证据包 |
| MatterGen没有被隐藏 | 通过 | 第1—3章命名规则及所有章节Prompt |
| MatterGen原有结构没有被包装为本文贡献 | 通过 | `MATTERGEN_NAMING_POLICY.md`、`WRITING_GUARDRAILS.md` |
| 本文贡献限定为Adaptive CFG与Learned-Gated E3-PCR | 通过 | 定位文件、术语表和声明矩阵 |
| 第1—7章标题统一 | 通过 | `CHAPTER_NUMBERING_FINAL.md/json` |
| 第3—6章证据包标题与最终编号一致 | 通过 | 四个`CHAPTERX_EVIDENCE_PACK.md/json` |
| 图、表、公式最终编号映射存在 | 通过 | `CHAPTER_NUMBERING_FINAL.md`、`FIGURE_TABLE_CROSSWALK.md` |
| 第3章结构更新为3.1—3.9 | 通过 | `chapter3/section_outline.md`及JSON |
| 第3章分段Prompt可用 | 通过 | part1=3.1—3.3、part2=3.4—3.6、part3=3.7—3.9 |
| 第3章Prompt不存在服务器绝对路径 | 通过 | `/data/dxl`、`/home/ubuntu`和`file://`命中数均为0 |
| 正式实验数据未修改 | 通过 | 相对基线提交的`thesis_archive/` diff为空 |
| 模型、采样、评价与研究源码未修改 | 通过 | `mattergen/`、`research/`和`configs/` diff为空 |
| 证据统计仍与逐seed归档一致 | 通过 | `EVIDENCE_PACKS_VALID=True`、`STATISTICS_VALID=True` |
| 数据不匹配 | 未发现 | `DATA_MISMATCH_DETECTED=False` |

## 证据包复核

```text
SOURCE_INDEX_ENTRIES=37
CLAIMS_CHECKED=10
FORMULAS_CHECKED=20
BROKEN_SOURCE_PATHS=0
CLAIMS_WITH_INCOMPLETE_EVIDENCE=0
SERVER_ABSOLUTE_PATHS_IN_CHAPTER3_PROMPTS=0
FORMAL_DATA_LEAKAGE_FOUND=False
UNSUPPORTED_PROJECT_FACTS_FOUND=False
```

`FORMAL_DATA_LEAKAGE_FOUND=False`只表示正式E3-PCR与两组组合数据没有与Gate训练seed重叠；历史Mixed 256仍按泄漏诊断处理，不能作为独立正式结果。

## 未改变内容

- 未修改学校登记中文题目；
- 未修改正式CSV/JSON、seed、统计口径、模型参数或checkpoint；
- 未修改MatterGen、MatterSim或E3-PCR核心源码；
- 未运行训练、生成、MatterSim、DFT、MLIP或GPU任务；
- 未重命名旧图表产物，避免破坏CPU重绘脚本。

## 历史文档处理

`THESIS_OUTLINE.md`和已有章节草稿保留为历史规划，不在本轮覆盖。最终合稿必须以`CHAPTER_NUMBERING_FINAL.md`为编号依据；README已将旧目录标记为历史参考。

`thesis/evidence_packs/build_evidence_packs.py`是上一轮证据包生成器，本轮按“不得提交代码”要求未修改；在后续专门更新文档生成工具前，不应使用它覆盖本轮已经冻结的定位文档。
