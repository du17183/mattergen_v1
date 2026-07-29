# 第3章 Adaptive CFG 写作提纲

## 3.1 基线

定义 C0、目标条件、固定 guidance=2.0、完整 Predictor/Corrector、FP32、batch=1。

## 3.2 三字段条件残差

对 cell、position、atom 分别定义 \(r_k=s_k^{cond}-s_k^{uncond}\) 和字段 RMS
\(\delta_k\)，说明不可直接拼接张量统一归一化；三个 RMS 取有效字段均值后驱动一个共同
guidance scale，而不是三个字段各有一个 scale。

## 3.3 在线自适应

- predictor/corrector 分别维护 EMA：\(m_{t,p}\leftarrow\beta m_{t-1,p}+(1-\beta)\delta_t\)。
- \(q_t=\delta_t/(m_{t,p}+\epsilon)\)，multiplier 限制在 [0.25,4]。
- 最终共同 guidance scale 限制在 [0,5]。
- \(\alpha=0.50,\beta=0.95,\epsilon=10^{-6}\)。

## 3.4 集成与边界

- Figure 2。
- Predictor/Corrector 不删减。
- 不是 Corrector Gating；不声称推理加速。
- 相同 seed 确定性与三字段日志。

## 3.5 实验

- seeds 20000–20255，n=256，C0/A0 配对。
- MatterSim-5M 代理松弛与质量评价。
- Table 1/2。

## 3.6 结果

- Figure 5。
- E-hull −0.003435 eV/atom；Stable +5.859 pp；NUS +3.516 pp。
- 给出 CI 与 Holm p=1.0；使用“正向趋势”。

## 3.7 讨论

说明方向一致但方差大、效应未显著；贡献落在在线多字段机制与完整采样兼容性，而非夸大质量结论。

