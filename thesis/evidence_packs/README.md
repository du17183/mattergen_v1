# 第3–6章可追溯证据包

本目录面向后续网页ChatGPT论文写作。它将正式commit、源码符号、冻结配置、逐seed数据、报告、图表、公式和允许/禁止结论连接起来；不是新实验，也不是最终论文正文。

学校正式登记题目为《基于深度学习的材料逆向生成》。最终章节编号以`../CHAPTER_NUMBERING_FINAL.md`为准；MatterGen在所有证据包中均定位为预训练条件晶体扩散生成基线，而不是本文提出的方法。

## 入口

- `MASTER_SOURCE_INDEX.md/json`：全部来源和数据资格。
- `CLAIM_TRACEABILITY.md/json`：核心论文claim到数据/代码/图表的映射。
- `FORMULA_REGISTRY.md/json`：公式exact/interpreted资格。
- `CODE_SYMBOL_INDEX.md/json`：正式源码符号。
- `FIGURE_TABLE_CROSSWALK.md`：图表、源数据和章节。
- `WRITING_GUARDRAILS.md`：不可违反的写作边界。
- `EVIDENCE_PACK_VALIDATION.md/json`：CPU重算和真实性验收。
- `chapter3`–`chapter6`：每章完整证据、来源、结构和`chatgpt_input.md`。
- `chapter3/chatgpt_input_final.md`：可直接复制的3.1—3.3最终写作Prompt。
- `chapter3/chatgpt_input_part2.md`、`chatgpt_input_part3.md`：第3章后续分段Prompt。

## 如何交给网页ChatGPT

1. 先复制`WRITING_GUARDRAILS.md`。
2. 第3章依次使用`chatgpt_input_final.md`、`chatgpt_input_part2.md`和`chatgpt_input_part3.md`；其他章使用各自`chatgpt_input.md`。
3. 需要展开时补充同章`CHAPTERX_EVIDENCE_PACK.md`和`formula_notes.md`/`experiment_evidence.md`。
4. 要求网页ChatGPT保留source_id、NOT_SUPPORTED标记和所有限制。

## 如何把正文放回仓库

将返回正文保存为独立草稿文件，不覆盖本证据包；随后让Codex逐段检查claim、公式、seed、n、单位、图表和限制。当前最终目录已经冻结在`../CHAPTER_NUMBERING_FINAL.md`；旧提纲和旧草稿只作历史参考，合稿时统一按最终编号更新交叉引用。

## CPU验证

```bash
python thesis/evidence_packs/validate_evidence_packs.py --write-report
```

验证只读取归档CSV/报告并写本目录验证报告，不启动MatterGen、MatterSim、DFT或GPU。

## 仍需人工确认

- 最终学校模板的排版细节和英文题目格式。
- 参考文献和相关工作来源。
- 图表最终排版和正文交叉引用。
- 部分No-Go原始服务器报告未完整进入GitHub，只能使用当前归档总结。
