图2 Adaptive CFG 机制。条件与无条件分支形成 cell、position、atom 三字段残差，经 EMA 平滑、残差驱动尺度更新及 [0,5] 限幅后完成 CFG 融合；Predictor 和 Corrector 均不跳过。本方法不是 Corrector Gating。
