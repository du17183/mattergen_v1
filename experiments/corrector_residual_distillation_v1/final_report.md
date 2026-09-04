# 轻量级残差蒸馏与不确定性回退：第一轮完整实验报告

日期：2026-09-04

任务：`dft_mag_density=0.1` 条件生成

分支：`experiment/corrector-residual-distillation-v1`

## 结论摘要

第一轮证明了两个不同层面的事实：

1. Corrector 前后 score residual **具有有限但不充分的可预测性**。2,661 参数 Adapter 在独立 test teacher 数据上，相对直接 Reuse 将 `atomic_numbers/cell/pos` residual MSE 分别降低 7.95%/6.93%/14.27%；但 cosine 只有 0.282/0.266/0.379，且 position 的困难主要集中在 late stage。
2. 无回退的 Reuse、Adapter 以及直接 Skip 均不能满足质量底线。validation 校准的 fallback 明显改善了 always-on 方法，其中 `Adapter+Fallback@50` 是本轮最有希望的点：实际 coverage 53.99%，MatterGen forward 减少 27.00%，配对 speedup 1.267×；在 32 个独立 seed 的描述性结果中，E-hull、Stable、NUS 优于 C0，但 RMSD 和 pre-relaxation force 变差。因此它是**值得复验的候选点，不是已经成立的 Pareto 或 formal256 结论**。

建议暂不直接进入 formal256。先冻结当前 test，不再用其调参；改进 on-policy 风险校准后，对 `Fallback@25/@50` 使用一组全新 64 seeds 做预注册式复验。只有能同时复现能量/稳定性趋势，并把 RMSD/force 拉回 C0 附近，才进入正式 256 seeds。

## 1. 环境与权重恢复

恢复成功。运行环境、缓存、权重和实验产物均位于 `/mnt/lis-wam-data/dxl/mattergen_v1`：

- 项目虚拟环境：`.venv`，Python 3.10.20；只读复用服务器既有 CUDA/PyTorch 基础层，任务新增包安装在 `.venv`。
- GPU：8 × NVIDIA H20，单卡 97,871 MiB；Driver 535.161.08。
- PyTorch 2.4.1+cu121、CUDA runtime 12.1、MatterSim 1.1.2。
- MatterGen checkpoint：511,777,278 bytes，SHA-256 `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`。
- MatterSim checkpoint：91,176,875 bytes，SHA-256 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`。
- Alex-MP reference：873,410,170 bytes，SHA-256 `c722f72c7d0cd398fc382870f0b731425b54bd4492b089ca36e3119e49f469b5`；项目内 LMDB 共审计 845,997 条记录。
- 最终 Adapter：17,684 bytes，SHA-256 `fae136412448be0b82267dd1a40d0168cf4847130d41cfa4295c8def715108c4`。

激活脚本将 pip、Hugging Face、Torch 和临时缓存统一定向到项目内 `.cache`。Git LFS 可执行文件也位于项目内 `.tools/git-lfs`。没有把任务权重或环境写入 `/root`。

完整版本与路径见 `environment.md`。

## 2. 实验协议与数据隔离

- `N=1000`，`n_steps_corrector=1`，batch size 1，CFG scale 2.0，deterministic，关闭 trajectory 输出。
- baseline recovery：seeds 60000–60007。
- train：61000–61015；validation：62000–62007；Stage-C test：63000–63031。
- train/validation/test 互斥，且与仓库历史正式 seed 区间无交集；`teacher_data_manifest.json` 中两类 overlap 均为空。
- 模型、epoch 和 coverage 阈值只根据 train/validation 确定，没有根据 Stage-C 质量指标调阈值。
- Adaptive CFG 的冻结实现和 formal256 结果未修改；新模块默认关闭。

早期错误的 `Skip` 空字典 override 被发现为 OmegaConf 递归合并，实际上仍执行了 Corrector。该 32-run 集已可恢复地隔离到 `generation/_invalid/Skip_wrong_empty_dict_override`，未进入聚合；原因和判据记录在 `config/run_exclusions.json`。正确的 Skip 使用显式 Hydra deletion overrides 后在相同 seeds 重跑。

## 3. Baseline 与实际 forward 数

原始采样每个 diffusion step 的执行为：

```text
score_before = MatterGen(x_t, t)
x_after      = Corrector(x_t, score_before)
score_after  = MatterGen(x_after, t)
x_{t-1}      = Predictor(x_after, score_after)
```

因此每个样本是 `1000 × 2 = 2000` 次逻辑 MatterGen score call。一次 classifier-free guidance 调用在模型 batch 内联合计算 conditional/unconditional，不在这里重复计为两个逻辑 call。position 与 cell 有 Corrector；atomic_numbers 在该默认配置中由 Predictor 更新。Adaptive CFG 只改变 guidance scale schedule，本实验测得仍为 2000 calls。

Stage-A 的 8-seed 恢复结果：8/8 成功，`117.127 ± 2.902 s/sample`，30.752 samples/hour，2000 calls/sample，峰值显存 `306.689 ± 14.718 MB`。

Stage-C 使用同一独立 32-seed 集重新运行不含 recorder I/O 的纯 C0：32/32 成功，`115.935 ± 3.220 s/sample`，31.074 samples/hour，2000 calls/sample，平均峰值显存 295.3 MB。纯 C0 与 teacher-C0 的 32 个输出逐一比较，原子序数、位置和晶胞最大差异均为 0，说明 recorder 是只读的。

Stage-C C0 质量为：E-hull 0.0886 eV/atom、Stable 59.38%、NUS 40.62%、Novel 75.00%、Unique 100%、RMSD 0.0227 Å、pre-relaxation max-force mean/max 0.246/1.491 eV/Å。

## 4. Teacher 数据规模

共采集 56 seeds × 1000 steps = **56,000 records**：train 16,000、validation 8,000、test 32,000。数据为 CPU bfloat16、shard size 64，共约 896 shards、360,067,968 bytes。

每条记录保存 seed/step/t/progress/num_atoms，以及三字段的 Corrector 前后状态、`score_before`、`score_after` 和 residual。所有 run manifest 和 shard 都带 SHA-256；最终审计重新核对了 56 个 run manifest。

## 5. 最终 Adapter 结构与训练

MatterGen 主干完全冻结。最终 Adapter 共 **2,661 参数**：

- 10 个旋转不变的全局特征：t、progress、atom count、position/cell Corrector displacement 的 RMS/max，以及三字段 `score_before` RMS。
- shared context：`10 → 32 → 16`，SiLU。
- position head：预测两个 sample-wise scalar，残差为 `a·Δx + b·s_before`，不会引入优选 Cartesian 方向。
- cell head：同样预测 `a·Δcell + b·s_before`。
- atomic_numbers head：每原子使用 context、原子 score RMS 和位移，经 32 维层输出 scale 与 rank-8 系数，再乘 `8 × 101` basis；屏蔽 logits 保持不变。
- 三个输出头零初始化，因此未训练模型严格等价于 Reuse。

最终训练命令参数为：20 epochs、batch records 64、learning rate `5e-4`、weight decay `1e-5`、hidden/context `32/16`、atomic rank 8、seed 20260904；按 validation normalized MSE 选择 epoch 16。完整机器可读配置见 `config/final_adapter_training.json`。同时尝试了 2,661 参数/12 epochs、7,877 参数大模型和低学习率版本；大模型没有显示收益，最终选择低学习率小模型。

## 6. Residual 可学习性

以下是完全独立的 Stage-C teacher test（32,000 records），数值来自 `evaluation_metrics.csv`：

| 模型 | atomic MSE / cosine | cell MSE / cosine | pos MSE / cosine |
|---|---:|---:|---:|
| Zero residual / Reuse | 4.1248 / 0.000 | 7.8883 / 0.000 | 2385.0477 / 0.000 |
| Linear residual | 4.0741 / 0.111 | 7.5743 / 0.201 | 2071.2939 / 0.363 |
| Lightweight Adapter | **3.7969 / 0.282** | **7.3420 / 0.266** | **2044.7548 / 0.379** |

Adapter 相对 Reuse 的 MSE 改善为 atomic 7.95%、cell 6.93%、position 14.27%；相对 linear 分别为 6.80%、3.07%、1.28%。这支持“residual 可学习”，但改善幅度不足以支持无条件替换 exact score。

分阶段诊断进一步说明问题：position residual MSE 在 early/middle/late 为 0.0333/93.839/6028.429；Reuse 为 0.00270/115.226/7025.978。Adapter 在 early position 的绝对误差反而高于 Reuse，在 middle/late 才产生收益。cell 在 early/middle 很容易预测（MSE 0.0120/0.0494，cosine 0.970/0.882），late 明显困难（MSE 21.921）；atomic cosine 总体也只有 0.282。风险模型观测到最高误差集中在最后一个 timestep decile。

因此直接 Reuse 不可靠：Corrector 改变了 Predictor 的输入状态，尤其在 late-stage position/cell 上 `s_after - s_before` 不再可忽略。Learned residual 在离线误差上优于 Reuse，但 rollout 会累积误差并造成分布漂移，所以离线 MSE 改善没有自动转化为 always-on 质量改善。

## 7. Uncertainty 与 fallback 规则

风险特征为上述 10 个 context 特征加三字段 predicted residual RMS，共 13 维。validation 上用标准化线性模型预测
`log1p(mean field-normalized residual RMSE)`，相关系数为 0.82985。

运行时顺序为：

1. Adapter 先预测三字段 residual。
2. Adapter 输出含 NaN/Inf、风险含 NaN/Inf、校准字段缺失或显式 `force_fallback` 时，无条件调用原始 MatterGen。
3. 否则计算 batch 最大风险；风险高于 validation 阈值时调用原始 MatterGen，低于阈值才使用 Adapter。
4. 所有回退都执行原始 `MatterGen(x_after_corrector, t)`，Corrector 从未被新方法省略。

validation 的 coverage 目标/阈值为：25%/0.295166、50%/0.342227、75%/0.405957、90%/0.536762。独立 Stage-C 的实际 coverage 分别为 33.00%、53.99%、76.09%、91.14%，说明 75/90 较准，25/50 有一定 calibration shift。

## 8. Stage-C 计算效率

所有 speedup 都是相同 seed 对纯 C0 的配对比值。`time`、显存和 speedup 对生成成功样本统计；尝试数与失败率单独完整保留。MB 为十进制 MB。

| 方法 | 成功/尝试 | score calls | coverage | forward reduction | time/sample (s) | speedup | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 32/32 | 2000.0 | 0.00% | 0.00% | 115.935 | 1.000× | 295.3 |
| A0 | 32/32 | 2000.0 | 0.00% | 0.00% | 116.482 | 0.996× | 295.3 |
| Skip | 24/32 | 1000.0 | 0.00% | 50.00% | 59.638 | 1.946× | 296.2 |
| Reuse | 32/32 | 1000.0 | 100.00% | 50.00% | 62.854 | 1.847× | 294.7 |
| Adapter | 32/32 | 1000.0 | 100.00% | 50.00% | 67.513 | 1.719× | 294.8 |
| Adapter+Fallback@25 | 32/32 | 1670.0 | 33.00% | 16.50% | 104.622 | 1.123× | 294.9 |
| Adapter+Fallback@50 | 32/32 | 1460.1 | 53.99% | 27.00% | 92.784 | 1.267× | 294.7 |
| Adapter+Fallback@75 | 32/32 | 1239.1 | 76.09% | 38.04% | 81.043 | 1.439× | 294.7 |
| Adapter+Fallback@90 | 32/32 | 1088.6 | 91.14% | 45.57% | 72.940 | 1.593× | 294.7 |
| A0+Adapter+Fallback@75 | 32/32 | 1242.1 | 75.79% | 37.89% | 81.281 | 1.433× | 294.6 |

Adapter 本身每 1000 次调用约增加 4.66 s（Adapter 总计代表值约 4.45 s），远低于被替代的 GemNet score 开销。显存基本不变。对 fallback 方法，`score calls = 1000 次 Corrector 前 exact + fallback calls`，所以 forward reduction 等于 `coverage/2`。

## 9. Stage-C 质量指标

使用项目内 MatterSim 5M 松弛与 Alex-MP reference 计算；所有值都是代理模型评估，不是 DFT。`quality n` 是实际生成并进入质量评估的结构数。

| 方法 | quality n | E-hull | Stable | NUS | Novel | Unique | RMSD | force mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 32 | 0.0886 | 59.38% | 40.62% | 75.00% | 100.00% | 0.0227 | 0.246/1.491 |
| A0 | 32 | 0.0984 | 59.38% | 21.88% | 56.25% | 96.88% | 0.1139 | 0.228/1.320 |
| Skip | 24 | 0.0995 | 66.67% | 25.00% | 54.17% | 100.00% | 2.5948 | 3.321/68.826 |
| Reuse | 32 | 0.1500 | 59.38% | 31.25% | 65.62% | 100.00% | 0.0806 | 0.579/4.291 |
| Adapter | 32 | 0.1852 | 43.75% | 25.00% | 75.00% | 100.00% | 0.0854 | 0.650/2.215 |
| Adapter+Fallback@25 | 32 | 0.0902 | 65.62% | 37.50% | 71.88% | 100.00% | 0.0370 | 0.240/1.000 |
| Adapter+Fallback@50 | 32 | 0.0815 | 75.00% | 53.12% | 78.12% | 100.00% | 0.0493 | 0.295/1.918 |
| Adapter+Fallback@75 | 32 | 0.2320 | 53.12% | 28.12% | 71.88% | 100.00% | 0.0578 | 0.320/1.685 |
| Adapter+Fallback@90 | 32 | 0.1757 | 43.75% | 25.00% | 75.00% | 100.00% | 0.0340 | 0.332/1.191 |
| A0+Adapter+Fallback@75 | 32 | 0.1218 | 56.25% | 21.88% | 59.38% | 100.00% | 0.0722 | 0.546/9.240 |

单位：E-hull 为 eV/atom，RMSD 为 Å，force 为 eV/Å。

## 10. 哪些指标变好、持平或下降

相对 C0：

- `Fallback@25`：Stable +6.25 pp，mean/max force -0.005/-0.492；E-hull +0.00163，接近 C0；但 NUS/Novel 各 -3.125 pp，RMSD +0.01427。它是更保守的速度点（1.123×）。
- `Fallback@50`：E-hull -0.00716、Stable +15.625 pp、NUS +12.50 pp、Novel +3.125 pp、Unique 持平；但 RMSD +0.02662、mean/max force +0.049/+0.427。它是当前首选候选，但不能表述为“所有质量不下降”。
- `Fallback@75/@90`：尽管达到 1.439×/1.593×，E-hull 分别恶化 +0.14334/+0.08706，Stable -6.25/-15.625 pp，NUS -12.50/-15.625 pp。coverage—quality 不是单调平滑关系，显示风险分数对 on-policy rollout 的排序还不够稳健。
- Always-on Adapter：相对 Reuse 离线 MSE 更好，但在线 E-hull 0.1852、Stable 43.75%、NUS 25.00%，没有转化为质量收益。这是必须保留的负结果。
- Reuse：速度 1.847×，但 E-hull +0.06137、NUS -9.375 pp、RMSD +0.05786、mean force +0.333，说明直接复用不成立。
- Skip：仅 24/32 生成成功；成功结构的 RMSD 2.5948、mean/max force 3.321/68.826。它虽然最快，却明显不稳定，不能作为合格 Pareto 点。
- `A0+Fallback@75`：速度 1.433×。相对本轮 A0，NUS 持平、Novel/Unique 各 +3.125 pp、RMSD -0.0417；但 E-hull +0.0234、Stable -3.125 pp、mean/max force +0.318/+7.920。因此本轮没有证明两项创新在质量上兼容。

本轮 A0 的 32-seed 小样本相对 C0 是 E-hull +0.00977、Stable 持平、NUS -18.75 pp，没有复现仓库冻结 formal256 的方向。冻结 formal256 仍是 E-hull -0.003435、Stable +5.859 pp、NUS +3.516 pp；本实验没有改写、重调或替代该结论。两组规模与用途不同，不能用本轮 32 seeds 推翻或覆盖正式结果。

## 11. 与旧 Corrector Gating 的区别

旧 G3 通过跳过 Corrector/相关 score evaluation 获得约 35.368% forward reduction 和 1.506× speedup，但正式结果为 E-hull +0.022423、Stable -9.766 pp、NUS -9.375 pp。

本方法始终保留 Corrector，只近似 Corrector 后、Predictor 前的第二次 MatterGen score；风险高时恢复 exact score。因此它不是换条件的 Corrector skip。`Fallback@50` 的 forward reduction 只有 27.00%、speedup 1.267×，速度低于 G3，但当前 32-seed 描述性 E-hull/Stable/NUS 方向更好。由于实验规模不同，这只是继续研究的证据，不是与 G3 的正式显著性比较。

## 12. 第一创新兼容性

实现层面兼容：Adaptive CFG 与 residual controller 是正交开关，组合配置可正常运行，32/32 生成成功，新功能默认 off，也没有改动冻结的 Adaptive CFG 代码路径。

质量层面尚未证明兼容：`A0+Fallback@75` 出现较高最大力 9.240 eV/Å，且相对 A0 的 E-hull/Stable 变差。下一轮不应同时调两项创新；应先在 C0 guidance 下把 residual fallback 的 on-policy calibration 稳定，再用预注册阈值复验组合。

## 13. 当前最大问题与局限

最大问题是**离线 one-step residual 拟合与在线 closed-loop 质量之间存在明显鸿沟**：Adapter 的离线 MSE 稳定优于 Reuse，但 always-on rollout 质量更差；当前线性风险头虽在 validation 上相关性 0.830，却不能可靠排序 75%/90% coverage 下的在线累积风险。

其他局限：

- Stage-C 仅 32 seeds，百分率步长为 3.125 pp，当前改善/下降均为描述性结果，没有 formal256 的统计把握。
- MatterSim 5M 是 surrogate；没有 DFT relaxation/energy 验证，也没有验证除 `dft_mag_density=0.1` 之外的属性目标。
- batch size 为 1；尚未验证大 batch throughput、DDP 或不同 GPU/driver 的时序稳定性。
- atomic_numbers residual cosine 偏低，离散分支仍是瓶颈之一；late-stage position/cell residual 与 off-policy 漂移更关键。
- Skip 的 8 个失败源于极端结构导致 GemNet 邻居/triplet 为空；失败被保留，没有从分母中删除。效率数字对成功样本统计，不能忽略 75% generation success。

## 14. 是否进入 formal256 与下一步三项修改

当前决策：**值得继续，但不直接进入 formal256**。

最值得尝试的三项修改：

1. **On-policy / DAgger 式再蒸馏**：用当前 Adapter+Fallback rollout 收集偏离 teacher 轨迹后的状态，补充 late 20–30% timestep，并对累计轨迹偏差或最终结构 proxy 加权；仍保持主干冻结和小参数头。
2. **分字段风险与部分 exact 回退**：分别校准 position/cell/atomic risk；优先让高风险 late position/cell 使用 exact score，探索 atomic 分支独立回退或直接 exact，避免单一 batch 最大风险丢失字段差异。
3. **预注册的新 64-seed 复验**：只保留 `Fallback@25/@50`，在不再查看 63000–63031 的前提下冻结阈值、模型和分析脚本；报告 bootstrap CI/paired 指标，并把 RMSD/force 作为与 E-hull/Stable/NUS 同等的准入条件。通过后才运行 formal256，并为候选结构补充 DFT 子集验证。

## 15. 测试、审计、branch 与 commit

最终定向测试：

```text
22 passed, 5 warnings in 35.82s
```

覆盖默认行为不变、Adapter off 数值等价、forced fallback 等价原始 MatterGen、NaN 自动回退、变 atom 数 shape、teacher round trip、seed 无泄漏与可复现、Skip corrector 删除后的 call 计数。

最终审计 `research/corrector_distillation/audit_results.py` 返回 `PASS`：320 benchmark attempts、10 methods、312 successful generations、8 retained failures、10 quality rows、56 teacher runs/56,000 records，且四个项目内资产的 SHA-256 全部匹配。

当前 branch：`experiment/corrector-residual-distillation-v1`。最终 commit 以提交本报告后的 `git rev-parse HEAD` 为准；完整分阶段提交历史由 `git log --oneline` 保留。

## 复现命令

```bash
cd /mnt/lis-wam-data/dxl/mattergen_v1
source experiments/corrector_residual_distillation_v1/activate.sh

# 最终 Adapter 训练；输出目录必须不存在，脚本拒绝覆盖。
CUDA_VISIBLE_DEVICES=0 python -m research.corrector_distillation.train_adapter \
  --train-root experiments/corrector_residual_distillation_v1/teacher_data/train \
  --validation-root experiments/corrector_residual_distillation_v1/teacher_data/validation \
  --output-dir experiments/corrector_residual_distillation_v1/adapter_reproduction \
  --epochs 20 --batch-records 64 --hidden-dim 32 --context-dim 16 \
  --atomic-rank 8 --learning-rate 5e-4 --device cuda

# Stage-C 10 方法矩阵；指定 8 卡只是 seed 级并行，不改变模型结构。
python -m research.corrector_distillation.run_benchmark_matrix \
  --checkpoint-root checkpoints/official/hf_mattergen/checkpoints/dft_mag_density \
  --adapter-checkpoint experiments/corrector_residual_distillation_v1/adapter_final/residual_adapter.pt \
  --output-root experiments/corrector_residual_distillation_v1/generation \
  --seed-start 63000 --seed-end 63031 --num-gpus 8

# 定向测试；-s 避免本服务器文件系统禁止 pytest 临时捕获文件 unlink。
PYTHONDONTWRITEBYTECODE=1 python -m pytest -s \
  mattergen/diffusion/tests/test_residual_distillation.py \
  mattergen/diffusion/tests/test_guidance_schedule.py -q

# 对 CSV、失败行、teacher manifest 和项目内资产哈希做完整审计。
PYTHONDONTWRITEBYTECODE=1 python research/corrector_distillation/audit_results.py
```

大型权重、teacher tensors、生成结构、日志和 relaxation 轨迹仅保留在项目目录并由 `.gitignore` 排除；Git 只记录源码、配置、小型表格、manifest/checksum 和报告。
