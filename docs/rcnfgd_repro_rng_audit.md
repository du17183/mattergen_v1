# RC-NFGD seed 740000 随机状态审计

结论：历史 Formal256 **记录 seed，但不保存可恢复的 RNG 状态**。这使跨环境轨迹分叉无法定位；不过“未保存 RNG”本身**不是**本次失败的已证实原因——若执行路径和数值环境完全相同，从同一 seed 重新运行理论上仍可能一致。

## 历史生成路径

`mattergen_v1_matersim_guidance/experiments/innovation2_finalization/run_generation.py` 在每个 worker 设置 `CUDA_VISIBLE_DEVICES`，历史 seed 740000 位于 GPU 5；`generate_frozen_pair.py` 调用冻结的 `generate_formal32_pair.py`。后者在 C0 和 F0 各自生成前调用 `generator._seed_sampling_rngs()`。`mattergen/generator.py:395–403` 依次执行 `random.seed(seed)`、`np.random.seed(seed)`、`torch.manual_seed(seed)`、`torch.cuda.manual_seed(seed)`、`torch.cuda.manual_seed_all(seed)`。F0 sampler 的 `sample_seed` 字段仅用于 trace 元数据，并不替代上述 RNG 初始化。

| 状态 | 历史运行是否初始化 | seed 740000 历史制品是否保存完整状态 | 说明 |
|---|---|---|---|
| Python `random` | 是 | 否 | 没有 `random.getstate()` 序列化。 |
| NumPy 全局 RNG | 是 | 否 | `NumAtomsCrystalDataset.from_num_atoms_distribution()` 用 `np.random.choice` 抽原子数；没有 `np.random.get_state()`。 |
| PyTorch CPU RNG | 是 | 否 | DataLoader 默认 `shuffle=True`，未传显式 `torch.Generator`；即使 `num_workers=0`，迭代器/采样器仍可能消耗全局流。没有 `torch.get_rng_state()`。 |
| PyTorch CUDA RNG | 是（含 `manual_seed_all`） | 否 | prior、predictor/corrector 的随机抽样依赖 CUDA/CPU 随机流；没有 `torch.cuda.get_rng_state_all()`，也没有单设备快照。 |
| DataLoader worker seed | 生成路径未设置 worker，默认 0 | 不适用 | 训练用 `datamodule.worker_init_fn` 不在本次生成 `condition_factory.py` 的 DataLoader 路径上。 |
| 显式 `torch.Generator` | 生成路径未发现 | 否 | `condition_factory.py` 与冻结 sampler 均未传独立 generator；依赖全局 RNG。 |

归档 `generation/{C0,F0}/740000/` 仅有终态 extxyz/CIF、run summary；F0 另有末 20 步 force trace 和调用计数。未发现 CPU/CUDA/NumPy/Python RNG 字节快照、抽到的 condition tensor、初始 prior 或任何逐步状态。seed 是重跑入口，不是可从中间恢复的状态。

## 与 Fixed-K2 的区别

`mattergen_v1_linear_k2_confirm/.../confirmatory_sampler.py:144–175` 在共享前缀后保存 `batch.clone()`、`branch_point=400`、状态 digest 与 `rng_state`。其 `_rng_snapshot()`（field-CFG 的 `risk_calibrated_cfg.py:334–346`）覆盖 Python、NumPy、Torch CPU 和**当前 CUDA 设备**，但不是 `get_rng_state_all()`；也未单独序列化 scheduler 配置或数值 timestep。它服务于**同一运行内**的分支重放，不能证明 RC-NFGD Formal256 跨环境再生，也不能拿 Fixed-K2 的 checkpoint 当作 RC-NFGD 的历史状态。

## 对失败的解释边界

从源代码看 C0/F0 均会重新设定相同全局 seed，且 `MultiCorruption` 对字段名排序后抽 prior；因此不能简单断言“没保存 RNG 导致 Gd4”。历史 worker 未显式设置 `PYTHONHASHSEED`，此前失败的 replay wrapper 却设为 740000；历史还设置三个 CPU 线程变量和 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。这是需要控制的执行差异，但没有证据证明其为主因。

本轮仅对**历史 seed 740000**做无完整轨迹的最小测试：以相同历史源码入口、相同进程变量，分别在重建 2.4.1/cu121 与现存 ALM 2.9.0/cu128 环境中，记录 reseed 后、condition 后、prior 后、首次 score 后的 Python/NumPy/Torch CPU/CUDA RNG 摘要。四个检查点的摘要两边完全一致；condition 与初始 prior 的逐字段 SHA256 也一致。故当前两环境的**首次可测分叉不是随机流初始化或 prior 抽样**，而是首次模型 score 数值。它不等于历史运行的 RNG 状态比较，因为原运行没有保存这些指纹。不得据此改 seed、补造历史状态或放宽原门槛。
