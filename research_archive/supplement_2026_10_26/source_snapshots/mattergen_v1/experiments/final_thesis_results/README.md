# 最终论文实验材料入口

论文：《基于深度学习的材料逆向生成》。模型探索已关闭；不新增方法或救援调参。

优先阅读 [master_status.md](master_status.md)、[final_experiment_audit.md](final_experiment_audit.md)、[final_report.md](final_report.md) 和 [main_results.md](main_results.md)。只有 final_status.json 中实际出现 THESIS_EXPERIMENTS_COMPLETE=YES 才表示完成；后台启动本身不代表完成。

## 结果位置

- 创新点一原始 Formal256：sources/ 下按 Git commit 导出的原始逐 seed 文件及统计报告；未覆盖旧文件。
- 创新点一新增最小消融：../final_innovation1_audit/minimal_ablation/。8 个全新配对种子、6组，共48次生成。
- F0 Formal256、方向、有界、等预算、交互：来源于 mattergen_v1_matersim_guidance/experiments/ 下已完成实验；本轮检查哈希并复用，不重新抽样。
- 主表、相对改善表、消融表：本目录 CSV/Markdown；所有可得数值由脚本读取，NA 不是0。
- 完整配对分布、均值/中位数/P90/P95/max：paired_statistics.csv 与 distribution_statistics.csv。
- 图：plots/，论文风格 PNG 与 PDF。方法流程图由代码绘制，不使用图像模型生成实验结果。
- 章节：主项目 thesis/final_materials/chapter3_innovation1.md、chapter4_innovation2.md、chapter5_experiments.md。
- 来源哈希与版本：reproducibility_manifest.json；产物哈希：artifact_manifest.json。

## 只重算统计与素材

无需 GPU，不会重跑历史实验：

```bash
cd /mnt/datasets-livsyn/dxl/mattergen_v1
/mnt/datasets-livsyn/dxl/alm/.venv/bin/python experiments/final_thesis_results/collect_results.py
/mnt/datasets-livsyn/dxl/alm/.venv/bin/python experiments/final_thesis_results/verify_evidence.py
MPLCONFIGDIR=/tmp/mattergen-thesis-matplotlib /mnt/datasets-livsyn/dxl/alm/.venv/bin/python experiments/final_thesis_results/make_plots.py
/mnt/datasets-livsyn/dxl/alm/.venv/bin/python experiments/final_thesis_results/make_reports.py
/mnt/datasets-livsyn/dxl/alm/.venv/bin/python experiments/final_thesis_results/final_audit.py
/mnt/datasets-livsyn/dxl/alm/.venv/bin/python experiments/final_thesis_results/refresh_status.py
```

main/.venv/bin/python 是历史占位文件，不能使用；上述 alm 环境才是本次运行环境。

## 最小消融与日志

minimal_ablation.py --register 已在生成前执行；不能重复注册、改seed或覆盖历史结果。注册文件包含完整代码哈希、方法参数和字段控制定义。

minimal_ablation.py --pipeline 采用确定性的 8 个 seed 分片，在当时确认空闲的 GPU 5/6/7 启动，每张卡一个生成任务；生成→CHGNet属性→MatterSim松弛→官方质量→统计串联。既有已完成 seed 不重跑，partial 文件不静默删除。

全部新任务 stdout/stderr 进入本目录 logs/。最终后处理监督器为 finish_pipeline.py，以实际 completion 状态触发；遇到失败会记录原因，不把输出文件存在等同于成功。

若未来复现，应先检查空闲资源并在新的输出目录注册，不能直接重用/覆盖冻结结果。这里记录的历史 GPU 空闲情况不是长期资源使用授权。

## 必须保留的证据边界

创新点一是 shared global CFG，而非 field-specific CFG。原 Formal256 的方向性确认不等于显著提升，原始三个主指标 CI 均跨零。属性/松弛前力缺失字段不得以松弛后力或其他小批结果回填。

RC-NFGD 在 Formal256 上相对本批 C0 的代理力改善获得支持，独立 CHGNet 力验证也获得支持。但 equal-budget POST 的 MatterSim MaxF 更好，独立势直接差异尚不明确；bounding 的必要性没有被证明，简单叠加 Adaptive CFG 未达到双收益保留要求。

E3-G 是强后处理基线，不是当前创新点二；不同 seed cohort/条件目标不直接比较绝对均值。E3 接受步数不等于实际势调用数，松弛耗时不等于生成+精修耗时。

DFT 实际仅完成环境检查，未找到可运行求解器及生产赝势/工作流，单点计算数=0。DFT_VERIFIED=False。所有 MatterSim/CHGNet 结果均为 surrogate，不证明真实物理稳定或可合成性。
