# RC-NFGD 图 4-1 非覆盖试画

- 核心命题：仅在诊断选定的后期 predictor 事件中，将 MatterSim 力反馈转换为位置 score 修正；异常时保持原 score。
- 类型：双路径方法示意图。上方为冻结的 MatterGen 采样主路径，下方为 RC-NFGD 位置反馈路径；不是新的实验结果图。
- 科学依据：第 4 章“RC-NFGD 方法设计”和原图 `innovation2/figures_polished/I2_F1_rc_nfgd_pipeline.pdf`；未改动模型、统计数据或论文结论。
- 复现：`/mnt/datasets-livsyn/dxl/mattergen_nature_figure_env/bin/python thesis_release/scripts/reproduce_pilot_rc_nfgd.py`。旧版绘图源码与输出保留在本目录；该命令生成独立的 v2 文件。
- 输出：同目录 PDF、SVG 矢量文件及 600 dpi PNG/TIFF，宽约 138 mm。当前论文仍引用原图，尚未替换或重新编译正文。
- 比较重点：主/支路径是否比原先六个蛇形步骤更清楚；打印尺寸下字体、连接线和回退说明是否易读。
- v2 质检：静态检查 20 PASS、1 WARN、0 FAIL；唯一 WARN 是技能预设的 Nature 栏宽与本论文的学校版心宽度不同。PDF 字体审计最小 5.285 pt、0 个低于 5 pt；渲染碰撞审计 0 FAIL、0 WARN。诊断 JSON 与 v2 文件同目录；由于没有碰撞，检查器未生成标记 PDF。
- 范围：这仍是非覆盖试画；未启动模型、未变更实验结果、未修改正文的图片引用。
