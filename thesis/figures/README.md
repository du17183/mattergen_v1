# Figure package

每张 Figure 1–12 都包含：

- `generated/pdf/`：论文优先使用的矢量 PDF；
- `generated/svg/`：可编辑矢量图；
- `generated/png/`：600 dpi 白底预览；
- `source/python/`：独立导出入口；
- `source/graphviz/`：Figures 1–4 和 12 的 DOT 语义源；
- `source_data/`：实际绘图数据；
- `captions/zh/` 与 `captions/en/`：中英文图注；
- `validation/`：自动质量检查。

运行：

```bash
python thesis/scripts/generate_all.py
python thesis/scripts/validate_outputs.py
```

统计图只从 `thesis_archive/` 读取；不从均值重建逐 seed 数据。Figure 7 的 per-seed mean displacement 不存在，因此只使用正式报告中的汇总值并在源代码/docstring 中披露。

