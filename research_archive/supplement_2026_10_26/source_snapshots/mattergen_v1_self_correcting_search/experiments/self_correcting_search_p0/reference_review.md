# Hidden-State Probe + Self-Correcting Search P0：参考实现审阅

审阅日期：2026-09-11。固定任务为官方 `dft_mag_density` Full FT、目标
`0.2 Å^-3`、constant CFG 2.0、1000 steps。本轮只允许单 proposal 的概率接受/拒绝；
不使用 probe gradient、hidden-state steering、best-of-K、硬上下界或任何模型再训练。

## 1. Goodfire 原始方法

Goodfire 的公开文章（2026-04-01）给出的核心定义是：probe 从 MatterGen 去噪过程中
的内部 activation 预测最终材料属性；若 proposal 的预测目标距离更好则接受，否则以随
恶化程度下降的概率接受。文章把算法称为 MALA，但没有公开 proposal transition-density
校正项、精确能量函数、温度、具体 GemNet 层、pooling 或 probe 网络定义。因此本项目只称
其为 **probabilistic self-correcting search**，不作严格 MALA 或复现 Goodfire 私有实现的
声明。

来源：<https://www.goodfire.com/research/self-correcting-search>

## 2. `probe-gradient-guidance` 仓库与 MatterGen reproduction

审阅仓库 commit：`921ad30`。

来源：<https://github.com/Dynamical-Systems-Research/probe-gradient-guidance>

### `mattergen_repro/frontier_v2.py`

- v2 是 single-proposal 实验编排器，向 MatterGen CLI 传入 `self_correct=true`。
- 固定公开参数为 `max_scoring_t=0.05`、`center_weight=0.35`、
  `in_window_weight=0.5`、`window_exit_penalty=1.0`。
- 文件本身不包含 proposal scorer、hidden activation 抽取、probe 定义或接受公式；它假设
  被挂载的 MatterGen 树已经包含这些未随该仓库发布的改动。

源码：<https://github.com/Dynamical-Systems-Research/probe-gradient-guidance/blob/main/mattergen_repro/frontier_v2.py>

### `mattergen_repro/frontier_v3.py`

- v3 在 v2 基础上加入 `hard_floor=2.0`、`hard_ceiling=8.0` 和
  `num_proposals=3`。
- 这些是 hard constraint + best-of-3，不属于本 P0；本轮明确禁止移植。

源码：<https://github.com/Dynamical-Systems-Research/probe-gradient-guidance/blob/main/mattergen_repro/frontier_v3.py>

### `mattergen_repro/sampler_patch.py`

- patch 只把原单 proposal 路径扩展为可选 `choose_best_of_k`：第一份 candidate 沿用原
  corrector proposal，再额外抽取 `K-1` 份；`K=1` 时仍调用原 `choose`。
- patch 不包含 `choose`/`choose_best_of_k` 的 scorer 实现，也不定义 probe 输入或概率。
- 因本 P0 禁止 best-of-3，此 patch 不应用到项目。

源码：<https://github.com/Dynamical-Systems-Research/probe-gradient-guidance/blob/main/mattergen_repro/sampler_patch.py>

## 3. 公开可核对的接受/拒绝公式

同一仓库公开的 `scripts/metropolis.py` 给出通用 Metropolis 形式。令能量差
`ΔE = E(proposal) - E(current)`，则

```text
P(accept) = min(1, exp(-ΔE / temperature))
```

即更优 proposal（`ΔE <= 0`）必然接受，变差 proposal 仍以非零概率接受。该脚本还含有
窗口能量、hard floor/ceiling 和 best-of-K，但它针对 Crystalite，不足以证明未发布的
MatterGen scorer 与其逐行一致。

本 P0 在预注册后采用最小、无硬约束的点目标版本：

```text
d(z) = |probe(z, t) - 0.2|
ΔE = d(proposal) - d(current)
P(accept) = min(1, exp(-ΔE / 0.01))
```

`0.01` 与冻结的命中容差 `tau` 相同，是量纲匹配的预注册温度，不做 search 后调参。
不含 proposal density ratio，所以不称严格 MALA。

源码：<https://github.com/Dynamical-Systems-Research/probe-gradient-guidance/blob/main/scripts/metropolis.py>

## 4. Hidden activation 与 probe 输入的证据边界

- Goodfire 公开文章只说明输入来自 MatterGen 内部 activation，未披露层与 pooling。
- reproduction 的 v2/v3/patch 未发布 MatterGen activation/probe 实现。
- 同仓库的 Crystalite 实验使用指定 Transformer layer 的原子 hidden state，去除非原子
  token 后对真实原子 mean-pool，并向 probe 传入 timestep/sigma；这只能作为设计参考，
  不能冒充 Goodfire 的 MatterGen 细节。
- 本项目按研究任务独立定义：官方冻结 GemNet-T 四个 interaction block 各自输出
  `[N_atoms, 512]`；取 **conditional CFG branch**，按结构 mean-pool 为 `[512]`，拼接
  归一化 timestep 得到 `[513]`。训练标签只取同一条 trajectory 最终 clean `x0` 的冻结
  CHGNet 磁密度，不用 noisy structure 的 CHGNet 值。

## 5. P0 与参考路线的对应关系

| 项目 | Goodfire 公开描述 | reproduction v2 | reproduction v3 | 本 P0 |
|---|---|---|---|---|
| verifier | hidden-state probe | checkpoint 参数，内部实现缺失 | 同 v2 | Linear / Tiny MLP 二选一 |
| proposal 数 | 未明确公开 | single | best-of-3 | **single** |
| 接受方式 | 更优必收、变差概率收 | scorer 未发布 | scorer 未发布 | `exp(-ΔE/0.01)` |
| hard bounds | 未公开 | 无 CLI hard bounds | floor + ceiling | **无** |
| probe gradient | 未声称 | 无 | 无 | **禁止** |
| 严格 MALA 声明 | 文章称 MALA | 无 transition proof | 无 transition proof | **不声明** |

结论：本实验复现的是公开思想和可审计的概率接受形式，并按指定磁密度任务重新训练独立
probe；不是对 Goodfire 私有 MatterGen scorer 的逐行复现。
