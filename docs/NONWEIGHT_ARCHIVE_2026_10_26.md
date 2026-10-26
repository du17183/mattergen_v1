# 2026-10-26 非权重研究资料补档与换机恢复

本轮补档保存此前仅存在于 h20-1 工作区的执行代码、配置、源码修改、逐样本结果、生成结构、轨迹、日志和生成训练样本。论文 LaTeX、PDF、实验数据、seed、checkpoint 和统计结论均未修改。

## 已归档的范围

- 探索归档分支：`archive/thesis-exploration-2026`。
- 本次数据／源代码快照提交：`c3a0b6fb5c8edfd3211141f3bc27c16a6b9ea638`；后续上传回执另行追加，不改变该快照。
- 浏览与恢复入口：[supplement_2026_10_26](https://github.com/du17183/mattergen_v1/tree/archive/thesis-exploration-2026/research_archive/supplement_2026_10_26)。
- 原始数据下载：[MatterGen non-weight research archive](https://github.com/du17183/mattergen_v1/releases/tag/mattergen-nonweight-20261026)。Release 在全部资产核对 SHA256 后才发布，最终回执为归档分支的 `upload_receipt.json`。
- 覆盖18个源工作区的33,833个文件和10个目录链接；代码、配置及小型报告另有1,944份可直接浏览的 Git 副本。
- 文件去重后分为6个 `.tar.zst` 包，共937,185,557字节。逐文件路径、大小、权限、原提交及 SHA256 见 `files.jsonl`。
- Timing 的生成、评估、统计、流水线、seed 740000 重放和 gate 检查共9个 Python 脚本，以及原始配置、门槛记录和失败尝试日志均在快照中。

本轮同时发布4个原先仅在本地的历史分支：`experiment/calibrated-safe-selection-cfg`、`experiment/field-decoupled-adaptive-cfg`、`experiment/risk-calibrated-adaptive-cfg`、`experiment/stage-calibrated-bounded-cfg`。其原始提交保留，未以事后结果改写历史。

## 换机恢复

1. 下载 Release 中6个数据包、恢复工具包 `mattergen-restore-kit-20261026.tar.gz` 和 `SHA256SUMS.txt`。
2. 解压恢复工具包，安装 Python 3.10+ 和 `zstd`。运行 `python3 restore-kit/restore_snapshot.py --assets-dir /绝对路径/下载目录 --verify-only`，验证压缩包及其内部全部文件内容。
3. 根据工具包 `worktrees.json` 选择工作区，在单独的 Git 克隆中检出该工作区记录的原始 `commit`。例如 RC-NFGD 工作区名为 `mattergen_v1_matersim_guidance`。
4. 执行 `python3 restore-kit/restore_snapshot.py --assets-dir /绝对路径/下载目录 --worktree mattergen_v1_matersim_guidance --destination /绝对路径/干净克隆目录`。工具按清单还原文件和安全目录别名，遇到不同内容的既有文件会停止。
5. 单独恢复有许可的 MatterGen、MatterSim 和 CHGNet 权重，核对冻结哈希，并按[原环境与门槛说明](REPRODUCE_ON_NEW_MACHINE.md)配置运行环境。任何新的 RC-NFGD 实验前仍需通过历史 seed 740000 门槛。

快照是原始 Git 提交上的完整研究文件覆盖层。清单中的原服务器绝对路径只用于溯源，恢复文件不需要挂载原服务器。历史执行脚本中的绝对运行路径仍需在独立复现目录配置；归档文件通过哈希验证不等于新 GPU 环境已通过实验复现门槛。

## 验证与保留边界

已完成6个压缩包及全部内部 blob 校验、3项归档边界测试，并在临时干净克隆中实际恢复 RC-NFGD，确认9个 Timing／重放脚本哈希匹配。未运行生成、松弛、评价或任何新 seed 实验。

训练网络权重、环境目录、包／字节码缓存、运行时外部参考数据库和私有配置未上传，排除记录见 `excluded.jsonl`。指定的扩散／RNG 状态和教师／DAgger 张量样本属于非权重实验数据，已保留。原工作区及其数据均未删除。
